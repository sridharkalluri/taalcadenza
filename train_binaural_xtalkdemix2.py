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

''' NotImplementedError: The operator 'aten::sinc.out' is not currently implemented for the MPS device.
If you want this op to be added in priority during the prototype phase of this feature,
please comment on https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set
the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op.
WARNING: this will be slower than running natively on MPS. '''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


CHUNK = 4 #duration of chunks of audio, in seconds, from the dataset that will be used in model training
FEATUREVECSIZE = 547
LENDEMIXMODEL = 48 #number of parameters in model of DEMIX transfer function
TARGETDELAY = 4. #millisecond delay of training input in order to promote causal filter result of neural network training

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
            nn.Linear(FEATUREVECSIZE, FEATUREVECSIZE),
            nn.ReLU(),
            nn.Linear(FEATUREVECSIZE, LENDEMIXMODEL*4),
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

    def forward(self, netpred, xin, target):

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
        if 0:
            predLl = torchaudio.functional.fftconvolve(torchaudio.functional.fftconvolve(xin[:,0,:],glhat[:,0,:],mode='same'),gldel[:,0,:],mode='same')
            predLr = torchaudio.functional.fftconvolve(torchaudio.functional.fftconvolve(xin[:,1,:],grhat[:,0,:],mode='same'),grdel[:,0,:],mode='same')
            predRl = torchaudio.functional.fftconvolve(torchaudio.functional.fftconvolve(xin[:,0,:],glhat[:,1,:],mode='same'),gldel[:,1,:],mode='same')
            predRr = torchaudio.functional.fftconvolve(torchaudio.functional.fftconvolve(xin[:,1,:],grhat[:,1,:],mode='same'),grdel[:,1,:],mode='same')
        elif 1:
            fftconvolver.to('cpu')
            predLl = fftconvolver(fftconvolver(xin[:,0,:].cpu(),glhat[:,0,:].cpu()),gldel[:,0,:].cpu())
            predLr = fftconvolver(fftconvolver(xin[:,1,:].cpu(),grhat[:,0,:].cpu()),grdel[:,0,:].cpu())
            predRl = fftconvolver(fftconvolver(xin[:,0,:].cpu(),glhat[:,1,:].cpu()),gldel[:,1,:].cpu())
            predRr = fftconvolver(fftconvolver(xin[:,1,:].cpu(),grhat[:,1,:].cpu()),grdel[:,1,:].cpu())
        xLhat = predLl + predLr
        xRhat = predRl + predRr

        # compare loudspeaker reconstructions with delayed input. Delaying the input allows the demix filters to be sufficiently delayed to be possible to apply causally
        # Delaying by ~10 milliseconds accounts for sound propagation between loudspeaker and micrphone of approximately 10 feet.
        delaysamples = torch.tensor(self.delay/1000*self.samplerate)
        lenfilter = 2**(np.ceil(np.log2(delaysamples.numpy()))+1) #pick a long enough filter so that the delay filter can have better all-pass characteristics
        wavdelayer = xdmx.AllPassDelayFilter(LENFILTER=lenfilter.astype(int),device=device)
        wavdelayer.to(device)
        delayfilter = wavdelayer(self.delay)
        if 0:
            targetdel = torchaudio.functional.fftconvolve(target,torch.unsqueeze(delayfilter,dim=1),mode='same')
        else:
            targetdel = fftconvolver(target.cpu(),torch.unsqueeze(delayfilter,dim=1).cpu())

        # mean squared error
        mseloss = 1/2*(torch.mean((xLhat - targetdel[:,0])**2) + torch.mean((xRhat - targetdel[:,1])**2))

        return mseloss

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    print(f"Size data set = {size}, batches = {len(dataloader)}")
    model.train()
    for batch, (X, xin, y) in enumerate(dataloader):
        print(f"Batch # {batch}")
        #if device == 'mps':
        #    X, y = X.to(torch.float32), y.to(torch.float32)
        X, xin, y = X.to(torch.float32), xin.to(torch.float32), y.to(torch.float32)
        X, xin, y = X.to(device), xin.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, xin, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 1 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    print(f"Size data set = {size} and num batches = {num_batches}")
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            if device == 'mps':
                X, y = X.to(torch.float32), y.to(torch.float32)
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


# 2. Define and load training and validation data sets
# Initialize audio, hrtf, and metadata directories
if 1:
    metadatadir = "/Users/sridhar/Documents/Projects/clarity/clarity/cad_icassp_2024/metadata"
    atmic_dir = "/Users/sridhar/Documents/Projects/clarity/clarity/cad_icassp_2024/audio/at_mic_music"
    audio_dir = "/Users/sridhar/Documents/Projects/clarity/clarity/cad_icassp_2024/audio/music"
BATCH = 50
EPOCHS = 1
PROPORTIONTEST = 0.2

dataset = AtMicCadenzaICASSP2024_2(
    metadatadir = metadatadir,
    atmicdir = atmic_dir,
    audiodir = audio_dir,
)

dataset_size = len(dataset)
train_size = int(PROPORTIONTEST * dataset_size)  # 80% for training
test_size = dataset_size - train_size  # 20% for testing

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=BATCH,shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH,shuffle=True)


# Move model to computational engine
model = NeuralNetwork().to(device)
print(model)
loss_fn = DemixLoss(torch.tensor(TARGETDELAY)).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

if 0:
    # Review features and target
    train_features, train_target = next(iter(train_dataloader))
    print(f"Features batch shape: {train_features.size()}")
    print(f"Targets batch shape: {train_target.size()}")
    test_features, test_target = next(iter(test_dataloader))
    print(f"Features batch shape: {test_features.size()}")
    print(f"Targets batch shape: {test_target.size()}")

    # Compute model predictions in the computational engine
    # "MPS" device does not accept float64, hence use float32
    #pred = model(train_features.to(torch.float32).to(device))
elif 1:
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        #test(test_dataloader, model, loss_fn)

print("Done!")