# 0. Import dependencies
import json
import numpy as np
import torch
import torchaudio
import xtalkdemixhelpers as xdmx
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from pathlib import Path
import matplotlib.pyplot as plt

''' NotImplementedError: The operator 'aten::sinc.out' is not currently implemented for the MPS device.
If you want this op to be added in priority during the prototype phase of this feature,
please comment on https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set
the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op.
WARNING: this will be slower than running natively on MPS. '''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'



from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/xtalkdemix_6')




CHUNK = 4 #duration of chunks of audio, in seconds, from the dataset that will be used in model training
FEATUREVECSIZE = 547
LENDEMIXMODEL = 48 #number of parameters in model of DEMIX transfer function
TARGETDELAY = 1. #millisecond delay of training input in order to promote causal filter result of neural network training

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device for model training")

# 1. Define classes and methods
class FeaturesCalculator(torch.nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, signal: torch.Tensor) -> torch.Tensor:        
        featvec = xdmx.featureanalysis(signal)
        return featvec

class MixMatrixInverter(torch.nn.Module):
    def __init__(self,
                Fs=44100,
                LENFILT=1999,
                fc=16000,
                order=4):
        super().__init__()
        self.Fs = Fs
        self.LENFILT = LENFILT
        self.fc = fc
        self.order = order

    def forward(self, hL: torch.Tensor, hR: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gl, gr = xdmx.mixmatrix_to_demix3(hL.numpy(), hR.numpy(),LENFILT=self.LENFILT,Fs=self.Fs,fc=self.fc,order=self.order)
        gl = torch.from_numpy(gl)
        gr = torch.from_numpy(gr)
        return gl, gr

class PCAreconstructor(torch.nn.Module):
    def __init__(self,
                 basisfilename="svd_demixirs_len1999",
                 basispath=os.getcwd(),
                 ):
        super().__init__()
        fname = Path(basispath,basisfilename)
        basis = torch.load(fname) #'U', 'S', 'V', 'select95', 'select98', 'select99', 'Fs'
        self.eigspace = basis["V"].to(device)

    def forward(self, gproj: torch.Tensor) -> torch.Tensor:
        '''
        input argument gproj is B x Q x M tensor of projections on Q eigenvectors of M data vectors for B batches
        
        returns B x N x M tensor of M data vectors reconstructed to N features from stored (N x L) eigenvector basis
        where L >= Q
        '''

        batches, eigs, channels = gproj.shape
        features, _ = self.eigspace.shape

        #grecon = torch.zeros((batches,channels,features),device=device)
        grecon = torch.matmul(self.eigspace[:,:eigs],gproj)
        
        return grecon

class AtMicCadenzaICASSP2024_2(Dataset):    
    def __init__(self,
                 chunk = CHUNK, #default duration of audio chunks in seconds 
                 atmicdir = "/Users/sridhar/Documents/Projects/clarity/clarity/cad_icassp_2024/audio/at_mic_music",
                 audiodir = "/Users/sridhar/Documents/Projects/clarity/clarity/cad_icassp_2024/audio/music",
                 metadatadir = "/Users/sridhar/Documents/Projects/clarity/clarity/cad_icassp_2024/metadata",
                 transform=None,
                 target_transform=None):

        split = "train"
        print(f"Initializing dataset")
        # Load the scenes metadata
        scenes_metafile = Path(metadatadir,f"at_mic_music.{split}.json")
        with open(scenes_metafile, encoding = "utf-8") as f:
            scenes_metadata = json.load(f)
        self.scenes_metadata = scenes_metadata
        self.scenes_names = list(self.scenes_metadata.keys())    
         
        # Define location of audio files & hrtf files
        self.atmicdir = atmicdir
        self.audiodir = audiodir
        
        # Divide database of audio files into CHUNK (10) second pieces, each labeled by the spatial configuration of the recording
        # With this initialized list, easily extract each CHUNK-second audio chunk in the __getitem__() method
        self.chunkidx_metadata = {} #initialize chunk metadata dictionary
        chunkidx = 0
        print(f"Processing metadata of {len(scenes_metadata)} audio files ...")

        for idx in range(len(scenes_metadata)):
            # scene name, path, and audio file information
            idx_scenemetadata = scenes_metadata[f"{self.scenes_names[idx]}"]
            idx_scenepath = idx_scenemetadata["Path"]
            idx_musicpath = f"{split}/{idx_scenemetadata['Original Track Name']}"
            idx_atmicname = Path(self.atmicdir,idx_scenepath,'mixture.wav')
            idx_audname = Path(self.audiodir,idx_musicpath,'mixture.wav')
            finf = torchaudio.info(idx_atmicname)
            nchunks = np.floor(finf.num_frames / finf.sample_rate / chunk).astype(int) #leave out the leftover chunk that is less than CHUNK long

            # create metadata entry identifying audio "chunk"
            for chunki in range(nchunks):
                self.chunkidx_metadata.update({f"{chunkidx}":(idx_atmicname, idx_audname, chunki*finf.sample_rate*chunk, finf.sample_rate*chunk)})
                chunkidx += 1

            self.totalchunks = chunkidx

        print(f"{self.totalchunks} chunks metadata initialized!!!")


        # Define object methods
        if 0:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        print(f"Using {self.device} device for feature extraction")

        self.calculator = FeaturesCalculator()
        self.calculator.to(self.device)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.totalchunks

    def __getitem__(self, idx):

        chunk_metadat = self.chunkidx_metadata[f"{idx}"]
        audinput, _ = torchaudio.load(chunk_metadat[0], frame_offset=chunk_metadat[2], num_frames=chunk_metadat[3])
        target,_ = torchaudio.load(chunk_metadat[1], frame_offset=chunk_metadat[2], num_frames=chunk_metadat[3])
        with torch.no_grad():
            audinput = audinput.to(torch.float32).to(self.device)
            features = self.calculator(audinput)

        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            target = self.target_transform(target)
        return features, audinput, target

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(FEATUREVECSIZE, FEATUREVECSIZE),
            nn.ReLU(),
            nn.Linear(FEATUREVECSIZE, 380),
            nn.ReLU(),
            nn.Linear(380, LENDEMIXMODEL*4),
        )
    def forward(self, x):
        x = self.flatten(x)
        pred = self.linear_relu_stack(x)
        return pred

class DemixLoss(nn.Module):
    def __init__(self, wavindel: torch.tensor, samplerate=44100):
        super(DemixLoss, self).__init__()
        self.delay = wavindel
        self.samplerate = samplerate

    def forward(self, netpred, xin, target, globind):

        # netpred (i.e., the output of the neural network) is a flattened torch tensor 48*4 points long, where each 48 points represents a demix filter
        # the first 47 points are PCA coefficients of the linear-phase component of the demix transfer function
        # 48th point is delay in milleseconds of all-pass filter component of the demix transfer function
        # Of the 4 segments -- segment 0: glL, segment 1: glR, segment 2: grL, segment 3: grR

        gmix = torch.reshape(netpred,(-1, LENDEMIXMODEL, 4))
        glpca = gmix[:,:-1,:2]  # model of demix linear phase transfer function between left ear and Left and Right loudspeakers respectively
        grpca = gmix[:,:-1,-2:] # model of demix linear phase transfer function between right ear and Left and Right loudspeakers respectively
        dell = gmix[:,-1,:2] # model of demix all-pass delay transfer function between left ear and Left and Right loudspeakers respectively
        delr = gmix[:,-1,-2:] # model of demix all-pass delay transfer function between right ear and Left and Right loudspeakers respectively

        # reconstruct linear-phase filters from PCA coefficients
        reconstructor = PCAreconstructor()
        reconstructor.to(device)
        glhat = reconstructor(glpca).swapaxes(1,2) # model of demix linear phase transfer function between left ear and Left and Right loudspeakers respectively
        grhat = reconstructor(grpca).swapaxes(1,2) # model of demix linear phase transfer function between right ear and Left and Right loudspeakers respectively

        # construct all-pass filters from delay
        delayline = xdmx.AllPassDelayFilter(LENFILTER = 128,device=device)
        delayline.to(device)
        gldelL = delayline(dell[:,0])
        gldelR = delayline(dell[:,1])
        gldel = torch.stack((gldelL,gldelR),dim=1)
        grdelL = delayline(delr[:,0])
        grdelR = delayline(delr[:,1])
        grdel = torch.stack((grdelL,grdelR),dim=1)

        # filter xin by cascaded linear phase and all-pass models to predict loudspeaker signals
        fftconvolver = torchaudio.transforms.FFTConvolve(mode='same')
        fftconvolver.to(device)
        fftconvolver.to('cpu')
        predLl = fftconvolver(fftconvolver(xin[:,0,:].cpu(),glhat[:,0,:].cpu()),gldel[:,0,:].cpu())
        predLr = fftconvolver(fftconvolver(xin[:,1,:].cpu(),grhat[:,0,:].cpu()),grdel[:,0,:].cpu())
        predRl = fftconvolver(fftconvolver(xin[:,0,:].cpu(),glhat[:,1,:].cpu()),gldel[:,1,:].cpu())
        predRr = fftconvolver(fftconvolver(xin[:,1,:].cpu(),grhat[:,1,:].cpu()),grdel[:,1,:].cpu())
        xLhat = predLl + predLr
        xRhat = predRl + predRr

        # compare loudspeaker reconstructions with delayed input. Delaying the input allows the demix filters to be sufficiently delayed to be possible to apply causally
        # Delaying by ~10 milliseconds accounts for sound propagation between loudspeaker and micrphone of approximately 10 feet.
        #delaysamples = torch.tensor(self.delay/1000*self.samplerate).to(torch.float32)
        lenfilter = 2**(2+torch.ceil(torch.log2(self.delay/1000*self.samplerate)).int().item()) #pick a long enough filter so that the delay filter can have better all-pass characteristics
        if lenfilter == 0:
            targetdel = target.clone().cpu()
        else:
            wavdelayer = xdmx.AllPassDelayFilter(LENFILTER=lenfilter,device=device)
            wavdelayer.to(device)
            delayfilter = wavdelayer(self.delay)
            targetdel = fftconvolver(target.cpu(),torch.unsqueeze(delayfilter,dim=1).cpu())

        # %% visualize predictions
        samps = torch.arange(3000)+1000
        tm = samps/44100
        fig = plt.figure()
        #plt.plot(tm,xLhat[2,10000:13000].detach().numpy(),'k')
        #plt.plot(target[2,1,10000:13000].detach().cpu().numpy(),'b-')
        #plt.plot(targetdel[2,1,10000:13000].detach().cpu().numpy(),'r--')
        plt.subplot(2,1,1)
        plt.plot(tm,xLhat[2,samps].detach().numpy(),'r',
                tm,target[2,1,samps].detach().cpu().numpy(),'b',
                tm,targetdel[2,1,samps].detach().cpu().numpy(),'k--')
        plt.subplot(2,1,2)
        ftxL = torch.fft.fft(xLhat,n=1024).detach().cpu()
        fttarget = torch.fft.fft(target[2,1,samps].detach().cpu(),n=1024)
        fttargetdel = torch.fft.fft(target[2,1,samps].detach().cpu(),n=1024)
        fx = torch.fft.fftfreq(n=1024)
        plt.plot(fx[:512],torch.squeeze(20*torch.log10(torch.abs(ftxL[2,:512]))),'r',
                 fx[:512],20*torch.log10(torch.abs(fttarget[:512])),'b',
                 fx[:512],20*torch.log10(torch.abs(fttargetdel[:512])),'k--')
        plt.show

        mseloss = 1/2*(torch.mean((xLhat - targetdel[:,0])**2) + torch.mean((xRhat - targetdel[:,1])**2))
        mseref = 1/2*(torch.mean((xin[:,0,:].cpu() - targetdel[:,0])**2) + torch.mean((xin[:,1,:].cpu() - targetdel[:,1])**2))
        writer.add_figure(f"Prediction quality {mseloss}, {mseref}, {mseloss/mseref}",fig,global_step=globind)


        # %% return from calculation
        return mseloss, mseref

def train(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    print(f"Size data set = {size}, epoch = {epoch}, batches = {len(dataloader)}")
    model.train()
    for batch, (X, xin, y) in enumerate(dataloader):
        print(f"Batch # {batch}")
        #if device == 'mps':
        #    X, y = X.to(torch.float32), y.to(torch.float32)
        X, xin, y = X.to(torch.float32), xin.to(torch.float32), y.to(torch.float32)
        X, xin, y = X.to(device), xin.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss, ref = loss_fn(pred, xin, y, (epoch-1)*size+batch)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 1 == 0:
            loss, refloss, current = loss.item(), ref.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}, ref: {ref:>7f} [{current:>5d}/{size:>5d}]")
            writer.add_scalar(f"training loss / epoch {epoch}",loss,batch)
            writer.add_scalar(f"ref loss / epoch {epoch}",refloss,batch)
            writer.add_scalar(f"fracloss/ epoch {epoch}",loss/refloss,batch)


def test(dataloader, model, loss_fn, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    print(f"Size data set = {size} and num batches = {num_batches}")
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch, (X, xin, y) in enumerate(dataloader):
            X, xin, y = X.to(torch.float32), xin.to(torch.float32), y.to(torch.float32)
            X, xin, y = X.to(device), xin.to(device), y.to(device)
            pred = model(X)
            if 0:
                test_loss += loss_fn(pred, xin, y, (epoch-1)*size+batch).item()
            else:
                loss, ref = loss_fn(pred, xin, y, (epoch-1)*size+batch)
                test_loss += loss.item()
            print(f"loss: {loss.item()}, ref: {ref.item()} [{batch:>5d}/{size:>5d}]") 
            writer.add_scalar(f"Test loss / epoch {epoch}", test_loss, batch)
            writer.add_scalar(f"Avg test loss / epoch {epoch}",test_loss / (batch+1),batch)


    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


# 2. Define and load training and validation data sets
# Initialize audio, hrtf, and metadata directories
if 1:
    metadatadir = "/Users/sridhar/Documents/Projects/clarity/clarity/cad_icassp_2024/metadata"
    atmic_dir = "/Users/sridhar/Documents/Projects/clarity/clarity/cad_icassp_2024/audio/at_mic_music"
    audio_dir = "/Users/sridhar/Documents/Projects/clarity/clarity/cad_icassp_2024/audio/music"
BATCH = 200
EPOCHS = 1
PROPORTIONTEST = 0.2

dataset = AtMicCadenzaICASSP2024_2(
    metadatadir = metadatadir,
    atmicdir = atmic_dir,
    audiodir = audio_dir,
)

dataset_size = len(dataset)
test_size = int(PROPORTIONTEST * dataset_size)  # 20% for testing
train_size = dataset_size - test_size  # 80% for training

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=BATCH,shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH,shuffle=True)


# Move neural network model and loss computing object to computational engine
model = NeuralNetwork().to(device)
if 1: #warm start model
    model.load_state_dict(torch.load("xtalkdemix_model_weights_epoch2.pth"), strict=False)
print(model)
loss_fn = DemixLoss(torch.tensor(TARGETDELAY).to(device)).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

if 0:
    # Visualize features, input, neural network, and target
    train_features, train_input, train_target = next(iter(train_dataloader))
    writer.add_graph(model, train_features.to(torch.float32).to(device))

    test_features, test_input, test_target = next(iter(test_dataloader))
    print(f"Features batch shape: {test_features.size()}")
    print(f"Targets batch shape: {test_target.size()}")

# Perform training and testing run
for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, t+1)
    test(test_dataloader, model, loss_fn, t+1)
    torch.save(model.state_dict(),f"xtalkdemix_model_weights_epoch{t+1}.pth")

writer.close()
print("Done!")