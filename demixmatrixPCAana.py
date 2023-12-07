# %%
# 0. Import dependencies and define classes
import json
import numpy as np
import scipy
import torch
import torchaudio
import xtalkdemixhelpers as xdmx
import matplotlib.pyplot as plt
from pathlib import Path

class MixMatrixInverter(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hL: torch.Tensor, hR: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gl, gr = xdmx.mixmatrix_to_demix(hL.numpy(), hR.numpy())
        gl = torch.from_numpy(gl)
        gr = torch.from_numpy(gr)
        return gl, gr

# %%

# 1. Define data directories and read metadata of HRTF database

metadatadir = "/Users/sridhar/Documents/Projects/clarity/clarity/cad_icassp_2024/metadata"
hrtf_dir = "/Users/sridhar/Documents/Projects/clarity/clarity/cad_icassp_2024/audio/hrtf"
audio_dir = "/Users/sridhar/Documents/Projects/clarity/clarity/cad_icassp_2024/audio/at_mic_music"

# Load the spatial configurations metadata
spatconfigs_file = Path(metadatadir,"head_loudspeaker_positions.json")
with open(spatconfigs_file, encoding="utf-8") as f:
    spatconfigs_metadata = json.load(f)
spatconfigslabels = list(spatconfigs_metadata.keys())

# %%
#device = (
#    "cuda"
#    if torch.cuda.is_available()
#    else "mps"
#    if torch.backends.mps.is_available()
#    else "cpu"
#)
device = "cpu"
demixmatrix = MixMatrixInverter()
demixmatrix.to(device)
glall = torch.zeros((235, 2, len(spatconfigslabels)))
grall = torch.zeros_like(glall)
for idx, label in enumerate(spatconfigslabels):
    configdict = spatconfigs_metadata[f"{label}"]

# %%
    # get hrtfs corresponding to the spatial configuration of the scene, where
    # hL is the binaural pair of impulse responses for stimulus at Left loudspeaker
    # hR is the binaural pair of impulse responses for stimulus at right loudspeaker
    hL, hR = xdmx.load_hrtf_signals(hrtf_dir,configdict)
    # gl is the pair of demix impulse responses to apply to left ear
    # gr is the pair of demix impulse responses to apply to right ear
    with torch.no_grad():
        hL = torch.from_numpy(hL).to(torch.float32)
        hR = torch.from_numpy(hR).to(torch.float32)
    if 1: #for testing, eliminate any mixing so that demixing filter is directly the inverse HRIR 
        hL[:,1] = 0
        hR[:,0] = 0
        


# %%
    gl, gr = demixmatrix(hL.to(device), hR.to(device))
    glall[:,:,idx] = gl
    grall[:,:,idx] = gr

# %%
#gln, grn = xdmx.mixmatrix_to_demix(hL.numpy(),hR.numpy())
#gl, gr = xdmx.mixmatrix_to_demix2(hL.numpy(),hR.numpy())
glrs, grrs = xdmx.mixmatrix_to_demix3(hL.numpy(),hR.numpy())

# %%
tm = 1000/44100*np.arange(235)
plt.subplot(2,1,1)
plt.plot(tm,np.real(gln[:,0]),'b',tm,np.real(gl[:,0]),'r',tm,np.real(glrs[:,0]),'k')
plt.plot(tm,hL[:,0],'m')
plt.subplot(2,1,2)
plt.plot(tm,scipy.signal.fftconvolve(hL[:,0],gl[:,0],mode='same'),'b')
plt.plot(tm,scipy.signal.fftconvolve(hL[:,0],gln[:,0],mode='same'),'r')
plt.plot(tm,scipy.signal.fftconvolve(hL[:,0],glrs[:,0],mode='same'),'k')

# %%
w, HLl = scipy.signal.freqz(hL[:,0],fs=44100)

w, GLl = scipy.signal.freqz(gl[:,0],fs=44100)
w, GLln = scipy.signal.freqz(gln[:,0],fs=44100)
w, GLlrs = scipy.signal.freqz(glrs[:,0],fs=44100)
plt.subplot(2,1,1)
plt.plot(w,20*np.log10(np.abs(HLl)),'k',w,20*np.log10(np.abs(GLl)),'b',w,20*np.log10(np.abs(GLln)),'r')
plt.subplot(2,1,2)
plt.plot(w,np.unwrap(np.angle(HLl)),'k',w,np.unwrap(np.angle(GLl)),'b',w,np.unwrap(np.angle(GLln)),'r')

# %%
plt.subplot(2,1,1)
plt.plot(torch.real(hL))
plt.subplot(2,1,2)
plt.plot(torch.real(gl))

# %%

gl.shape

# %%
winmat = np.tile(win,[1, 2, 2])

# %%
winmat.shape

# %%
win

# %%
win.shape

# %%



