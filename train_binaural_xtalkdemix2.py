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
                 basispath=os.getcwd()):
        super().__init__()
        fname = Path(basispath,basisfilename)
        basis = torch.load(fname) #'U', 'S', 'V', 'select95', 'select98', 'select99', 'Fs'
        self.eigspace = basis["V"]

    def forward(self, gproj: torch.Tensor) -> torch.Tensor:
        '''
        input argument gproj is Q x M tensor of projections on Q eigenvectors of M data vectors
        
        returns N x M tensor of M data vectors reconstructed to N features from stored (N x L) eigenvector basis
        where L >= Q
        '''

        neigs,_ = gproj.shape
        grecon = torch.matmul(self.eigspace[:,:neigs],gproj)
        return grecon

class AtMicCadenzaICASSP2024(Dataset):    
    def __init__(self, 
                 split = "train", #or "valid"
                 audiodir = "/Users/sridhar/Documents/Projects/clarity/clarity/cad_icassp_2024/audio/at_mic_music",
                 hrtfdir = "/Users/sridhar/Documents/Projects/clarity/clarity/cad_icassp_2024/audio/hrtf",
                 metadatadir = "/Users/sridhar/Documents/Projects/clarity/clarity/cad_icassp_2024/metadata",
                 transform=None,
                 target_transform=None):

        # Determine if training or validation split
        self.split = split
        print(f"Initializing {self.split} dataset")
        
        # Load the scenes metadata
        scenes_metafile = Path(metadatadir,f"at_mic_music.{self.split}.json")
        with open(scenes_metafile, encoding = "utf-8") as f:
            scenes_metadata = json.load(f)
        self.scenes_metadata = scenes_metadata
        self.scenes_names = list(self.scenes_metadata.keys())
    
        # Load the spatial configurations metadata
        spatconfigs_file = Path(metadatadir,"head_loudspeaker_positions.json")
        with open(spatconfigs_file, encoding="utf-8") as f:
            spatconfigs_metadata = json.load(f)
        self.spatconfigs_metadata = spatconfigs_metadata
        self.spatconfigslabels = list(spatconfigs_metadata.keys())
            
        # Define location of audio files & hrtf files
        self.audiodir = audiodir
        self.hrtfdir = hrtfdir
        
        # Divide database of audio files into CHUNK (10) second pieces, each labeled by the spatial configuration of the recording
        # With this initialized list, easily extract each CHUNK-second audio chunk in the __getitem__() method
        CHUNK = 10 #seconds
        MINCHUNK = 1 #seconds
        self.chunkidx_metadata = {} #initialize chunk metadata dictionary
        chunkidx = 0
        print(f"Processing {len(scenes_metadata)} audio files ...")
        for idx in range(len(scenes_metadata)):
            # scene name, path, and audio file information
            idx_scenemetadata = scenes_metadata[f"{self.scenes_names[idx]}"]
            idx_scenepath = idx_scenemetadata["Path"]
            idx_audname = Path(self.audiodir,idx_scenepath,'mixture.wav')
            finf = torchaudio.info(idx_audname)
            nchunks = np.floor(finf.num_frames / finf.sample_rate / CHUNK).astype(int)
            trailchunkdur = np.remainder(finf.num_frames / finf.sample_rate, CHUNK)
            if trailchunkdur >= MINCHUNK: #if leftover chunk is greater than MINCHUNK long, then include in the list of chunks
                nchunks += 1

            # dictionary with details on spatial configuration of the scene
            #idxitem_scenedict = head_positions_metadata[f"{idx_scenemetadata['Head Position']}"]
            # get dict describing spatial configuration of the scene
            if self.split == 'train':
                idxitem_scenedict = self.spatconfigs_metadata[f"{idx_scenemetadata['Head Position']}"]
            elif self.split == 'valid':
                extract = xdmx.extract_text_after_pattern(idx_scenemetadata['Track Name'],r"hlp")
                idx_headpos = f"hlp{extract}"
                idxitem_scenedict = self.spatconfigs_metadata[idx_headpos]                
            else:
                idxitem_scenedict = []

            # create metadata entry identifying audio "chunk" and corresponding spatial configuration metadata 
            for chunki in range(nchunks):
                self.chunkidx_metadata.update({f"{chunkidx}":(idx_audname, chunki*finf.sample_rate*CHUNK, finf.sample_rate*CHUNK, idxitem_scenedict)})
                chunkidx += 1

        self.totalchunks = chunkidx
        print(f"{self.totalchunks} {self.split} chunks metadata initialized!!!")

        # Define object methods
        if 0:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.calculator = FeaturesCalculator()
        self.calculator.to(self.device)
        self.demixmatrix = MixMatrixInverter()
        self.demixmatrix.to(self.device)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.totalchunks

    def __getitem__(self, idx):

        if 0:                         
            #get single scene name 
            item_scenemetadata = self.scenes_metadata[f"{self.scenes_names[idx]}"]
            item_scenepath = item_scenemetadata["Path"]
            item_audname = Path(self.audiodir,item_scenepath,'mixture.wav')
            # read the binaural audio of the recorded scene from disk
            audscene, sample_rate = torchaudio.load(item_audname)
            with torch.no_grad():
                audscene = audscene.to(torch.float32).to(self.device)
                features = self.calculator(audscene)
                                 
            # get dict describing spatial configuration of the scene
            if self.split == 'train':
                idxitem_scenedict = self.spatconfigs_metadata[f"{item_scenemetadata['Head Position']}"]
            elif self.split == 'valid':
                extract = xdmx.extract_text_after_pattern(item_scenemetadata['Track Name'],r"hlp")
                item_headpos = f"hlp{extract}"
                idxitem_scenedict = self.spatconfigs_metadata[item_headpos]
            else:
                idxitem_scenedict = []
            
            if not idxitem_scenedict:
                target = torch.zeros(1)        
            else:
                # get hrtfs corresponding to the spatial configuration of the scene, where
                # hL is the binaural pair of impulse responses for stimulus at Left loudspeaker
                # hR is the binaural pair of impulse responses for stimulus at right loudspeaker
                hL, hR = xdmx.load_hrtf_signals(self.hrtfdir,idxitem_scenedict)
                # gl is the pair of demix impulse responses to apply to left ear
                # gr is the pair of demix impulse responses to apply to right ear
                if 0:
                    gl, gr = xdmx.mixmatrix_to_demix(hL, hR)
                    target = np.concatenate((np.reshape(gl,(len(gl)*2, 1),'F'),np.reshape(gr,(len(gr)*2,1),'F')),axis=0)
                    target = torch.from_numpy(target)
                else:
                    with torch.no_grad():
                        hL = torch.from_numpy(hL).to(torch.float32).to(self.device)
                        hR = torch.from_numpy(hR).to(torch.float32).to(self.device)
                        gl, gr = self.demixmatrix(hL, hR)
                    target = torch.cat((torch.reshape(gl,(len(gl)*2, 1)),torch.reshape(gr,(len(gr)*2,1))),axis=0)
        else:
            chunk_metadat = self.chunkidx_metadata[f"{idx}"]
            audscene, sample_rate = torchaudio.load(chunk_metadat[0], frame_offset=chunk_metadat[1], num_frames=chunk_metadat[2])
            with torch.no_grad():
                audscene = audscene.to(torch.float32).to(self.device)
                features = self.calculator(audscene)

            if not chunk_metadat[3]:
                target = torch.zeros(1)
            else:
                # get hrtfs corresponding to the spatial configuration of the scene, where
                # hL is the binaural pair of impulse responses for stimulus at Left loudspeaker
                # hR is the binaural pair of impulse responses for stimulus at right loudspeaker
                hL, hR = xdmx.load_hrtf_signals(self.hrtfdir,chunk_metadat[3])
                # gl is the pair of demix impulse responses to apply to left ear
                # gr is the pair of demix impulse responses to apply to right ear
                with torch.no_grad():
                    hL = torch.from_numpy(hL).to(torch.float32).to(self.device)
                    hR = torch.from_numpy(hR).to(torch.float32).to(self.device)
                    gl, gr = self.demixmatrix(hL, hR)
                target = torch.cat((torch.reshape(gl,(len(gl)*2, 1)),torch.reshape(gr,(len(gr)*2,1))),axis=0)


        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            target = self.target_transform(target)
        return features, target

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(547, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 235*4),
        )
    def forward(self, x):
        x = self.flatten(x)
        pred = self.linear_relu_stack(x)
        return pred

class DemixLoss(nn.Module):
    def __init__(self, wavindel, samplerate=44100):
        super(DemixLoss, self).__init__()
        self.delay = wavindel
        self.samplerate = samplerate

    def forward(self, input, target, inwav):

        # input is flattened torch tensor 48*4 points long, where each 48 points represents a demix filter
        # the first 47 points are coefficients of PCA expansion of and 48th point is delay in milleseconds of all-pass filter 

        # 
        gmix = torch.reshape(input,(48, 4))
        glpca = gmix[:47,0:1]
        grpca = gmix[:47,2:3]
        dell = gmix[-1,0:1]
        delr = gmix[-1,2:3]
        # reconstruct linear-phase filters from PCA coefficients
        reconstructor = PCAreconstructor()
        glhat = reconstructor(glpca)
        grhat = reconstructor(grpca)

        # construct all-pass filters from delay
        delayline = xdmx.AllPassDelayFilter(LENFILTER = 128)
        gldelL = delayline(dell[0])
        gldelR = delayline(dell[1])
        grdelL = delayline(delr[0])
        grdelR = delayline(delr[1])

        # apply filters in cascade (linear phase and all-pass) to inwav to predict loudspeaker signals
        predLl = torchaudio.functional.fftconvolve(torchaudio.functional.fftconvolve(inwav[:,0],glhat[:,0],mode='same'),gldelL,mode='same')
        predLr = torchaudio.functional.fftconvolve(torchaudio.functional.fftconvolve(inwav[:,1],grhat[:,0],mode='same'),grdelL,mode='same')
        predRl = torchaudio.functional.fftconvolve(torchaudio.functional.fftconvolve(inwav[:,0],glhat[:,1],mode='same'),gldelR,mode='same')
        predRr = torchaudio.functional.fftconvolve(torchaudio.functional.fftconvolve(inwav[:,1],grhat[:,1],mode='same'),grdelR,mode='same')
        xLhat = predLl + predLr
        xRhat = predRl + predRr

        # compare loudspeaker reconstructions with delayed input. Delaying the input allows the demix filters to be sufficiently delayed to be possible to apply causally
        # Delaying by ~10 milliseconds accounts for sound propagation between loudspeaker and micrphone of approximately 10 feet.
        delaysamples = self.delay/1000*self.samplerate
        lenfilter = 2**(torch.ceil(torch.log2(delaysamples))+1) #pick a long enough filter so that the delay filter can have better all-pass characteristics
        wavdelayer = xdmx.AllPassDelayFilter(LENFILTER=lenfilter.astype(int))
        targetdel = wavdelayer(target)
        mseloss = 1/2*(torch.mean((xLhat - targetdel[:,0])**2) + torch.mean((xRhat - targetdel[:,1])**2))

        return mseloss

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    print(f"Size data set = {size}, batches = {len(dataloader)}")
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        print(f"Batch # {batch}")
        if device == 'mps':
            X, y = X.to(torch.float32), y.to(torch.float32)
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, torch.squeeze(y))

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
if 0:
    metadatadir = "/Users/sridhar/Documents/Projects/clarity/clarity/cad_icassp_2024/metadata"
    hrtf_dir = "/Users/sridhar/Documents/Projects/clarity/clarity/cad_icassp_2024/audio/hrtf"
    audio_dir = "/Users/sridhar/Documents/Projects/clarity/clarity/cad_icassp_2024/audio/at_mic_music"
BATCH = 50
EPOCHS = 1

training_data = AtMicCadenzaICASSP2024(
    split="train",
    #metadatadir = metadatadir,
    #hrtfdir = hrtf_dir,
    #audiodir = audio_dir,
)

test_data = AtMicCadenzaICASSP2024(
    split="valid",
    #metadatadir = metadatadir,
    #hrtfdir = hrtf_dir,
    #audiodir = audio_dir,
)

train_dataloader = DataLoader(training_data, batch_size=BATCH, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH, shuffle=True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Move model to computational engine
model = NeuralNetwork().to(device)
print(model)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


if 0:
    # Review features and target
    train_features, train_target = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_target.size()}")
    #test_features, test_target = next(iter(train_dataloader))
    #print(f"Feature batch shape: {train_features.size()}")
    #print(f"Labels batch shape: {train_target.size()}")

    # Compute model predictions in the computational engine
    # "MPS" device does not accept float64, hence use float32
    pred = model(train_features.to(torch.float32).to(device))
elif 1:
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)

print("Done!")