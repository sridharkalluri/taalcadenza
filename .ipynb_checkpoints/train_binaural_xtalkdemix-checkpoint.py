##0. Import dependencies
import json
import numpy as np
import torch
import torchaudio
import xtalkdemixhelpers as xdmx
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path


##1. Define classes and methods
class FeaturesCalculator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.spec_centroid = torch.nn.Sequential(
        #    #torchaudio.transforms.Resample(input_freq, resample_freq),
        #    torchaudio.transforms.SpectralCentroid(sample_rate = resample_freq, n_fft= n_fft, hop_length = n_hop),
        #)


    def forward(self, signal: torch.Tensor) -> torch.Tensor:        
        featvec = xdmx.featureanalysis(signal)
        return featvec

class AtMicCadenzaICASSP2024(Dataset):    
    def __init__(self, 
                 split = "train", #or "valid"
                 audiodir = "/Users/sridhar/Documents/Projects/clarity/clarity/cad_icassp_2024/audio/at_mic_music",
                 hrtfdir = "/Users/sridhar/Documents/Projects/clarity/clarity/cad_icassp_2024/audio/hrtf",
                 metadatadir = "/Users/sridhar/Documents/Projects/clarity/clarity/cad_icassp_2024/metadata",
                 transform=None,
                 target_transform=None):

        #Determine if training or validation split
        self.split = split
        
        # Load the scenes metadata
        scenes_metafile = Path(metadatadir,f"at_mic_music.{split}.json")
        with open(scenes_metafile, encoding = "utf-8") as f:
            scenes_metadata = json.load(f)
        self.scenes_metadata = scenes_metadata
        self.scenes_names = list(scenes_metadata.keys())
        
        # Load the spatial configurations metadata
        spatconfigs_file = Path(metadatadir,"head_loudspeaker_positions.json")
        with open(spatconfigs_file, encoding="utf-8") as f:
            spatconfigs_metadata = json.load(f)
        self.spatconfigs_metadata = spatconfigs_metadata
        self.spatconfigslabels = list(spatconfigs_metadata.keys())
            
        # Define location of audio files & hrtf files
        self.audiodir = audiodir
        self.hrtfdir = hrtfdir
        
        # Define object methods
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.calculator = FeaturesCalculator()
        self.calculator.to(device)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.scenes_metadata)

    def __getitem__(self, idx):
                                 
        #get single scene name 
        item_scenemetadata = self.scenes_metadata[f"{self.scenes_names[idx]}"]
        item_scenepath = item_scenemetadata["Path"]
        item_audname = Path(self.audiodir,item_scenepath,'mixture.wav')
        #read the binaural audio of the recorded scene from disk
        audscene, sample_rate = torchaudio.load(item_audname)
        features = self.calculator(audscene)
                                 
        #get dict describing spatial configuration of the scene        
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
            #get hrtfs corresponding to the spatial configuration of the scene, where
            #hL is the binaural pair of impulse responses for stimulus at Left loudspeaker
            #hR is the binaural pair of impulse responses for stimulus at right loudspeaker
            hL, hR = xdmx.load_hrtf_signals(self.hrtfdir,idxitem_scenedict)
            #gl is the pair of demix impulse responses to apply to left ear
            #gr is the pair of demix impulse responses to apply to right ear
            gl, gr = xdmx.mixmatrix_to_demix(hL, hR)
            target = np.concatenate((np.reshape(gl,(len(gl)*2, 1),'F'),np.reshape(gr,(len(gr)*2,1),'F')),axis=0)
            target = torch.from_numpy(target)
        
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


##1. Define and load training and validation data sets
#Initialize audio, hrtf, and metadata directories
metadatadir = "/Users/sridhar/Documents/Projects/clarity/clarity/cad_icassp_2024/metadata"
hrtf_dir = "/Users/sridhar/Documents/Projects/clarity/clarity/cad_icassp_2024/audio/hrtf"
audio_dir = "/Users/sridhar/Documents/Projects/clarity/clarity/cad_icassp_2024/audio/at_mic_music"

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

train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True)

# Review features and target
train_features, train_target = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_target.size()}")
#test_features, test_target = next(iter(train_dataloader))
#print(f"Feature batch shape: {train_features.size()}")
#print(f"Labels batch shape: {train_target.size()}")

