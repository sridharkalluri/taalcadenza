import json
import numpy as np
import scipy
import torch
import torchaudio
import re
import librosa
import python_auditory_toolbox.auditory_toolbox as pat
import pandas as pd
import os
from clarity.utils.file_io import read_signal
from numpy import ndarray
from pathlib import Path
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


''' NotImplementedError: The operator 'aten::sinc.out' is not currently implemented for the MPS device.
If you want this op to be added in priority during the prototype phase of this feature,
please comment on https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set
the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op.
WARNING: this will be slower than running natively on MPS. '''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def load_hrtf_signals(hrtf_path: str, hp: dict) -> tuple[ndarray, ndarray]:
    """Loads the HRTF signals for a given head position.

    Args:
        hrtf_path (str): Path to the HRTF signals.
        hp (dict): Head position.

    Returns:
        tuple(ndarray, ndarray): Left and right HRTF signals.
    """


    '''
    Requires pathlib, json, numpy, scipy, clarity.utils.file_io, numpy.ndarray
    '''

    hp_left_path = (
            Path(hrtf_path) / f"{hp['mic']}-{hp['subject']}-n{abs(hp['left_angle'])}.wav"
    )
    hp_right_path = (
            Path(hrtf_path) / f"{hp['mic']}-{hp['subject']}-p{abs(hp['right_angle'])}.wav"
    )

    hp_left_signal = read_signal(hp_left_path)
    hp_right_signal = read_signal(hp_right_path)

    return hp_left_signal, hp_right_signal


def mixmatrix_to_demix(lefthrir: ndarray, righthrir: ndarray) -> tuple[ndarray, ndarray]:
    """
    Compute demix impulse responses from mix impulse responses

    Args:
        lefthrir: binaural impulse response (left ear in col 0, right ear in col 1) for signal from left loudspeaker
        righthrir: binaural impulse response (left ear in col 0, right ear in col 1) for signal from right loudspeaker

    Returns:
        tuple(ndarray, ndarray): demix impulse response to reconstruct left loudspeaker and right loudspeaker

    """

    '''
    Requires numpy, numpy.ndarray, scipy.fft
    '''

    ## do demix in the frequency domain, and then invert to get back to time-domain impulse responses
    irlen = len(lefthrir)

    # compute freq domain representations of head related impulse responses
    NFFT = 2 ** np.ceil(np.log2(irlen))
    leftHRTF = fft(lefthrir, 256, axis=0)
    rightHRTF = fft(righthrir, 256, axis=0)

    # assemble mixing matrix, (experiment later with assembling only 1/2 the matrix and
    # assembling the other half by complex conjugation)
    # H = [leftHRTF[:,0] rightHRTF[:,0]
    #     leftHRTF[:,1] rightHRTF[:,0]]
    Hw = np.concatenate((leftHRTF[:, :, np.newaxis], rightHRTF[:, :, np.newaxis]), axis=2)

    # de-mix matrix at every frequency is the matrix inverse of the mix matrix
    # numpy.linalg.inv separately inverts each row of the NFFTx2x2 nd-array and returns the result
    # in a NFFTx2x2 nd-array
    # ***** N.B. the inverted transfer function / impulse response does not behave well at high frequencies
    # and low frequencies, so may need to smooth the spectrum before inverting *****

    Gw = np.linalg.inv(Hw)
    demixir = ifft(Gw, axis=0)
    lefteardemix = demixir[:irlen, 0, :]
    righteardemix = demixir[:irlen, 1, :]

    return lefteardemix, righteardemix

def mixmatrix_to_demix3(lefthrir: ndarray, righthrir: ndarray, Fs=44100, LENFILT=1023, fc=16000, order=4) -> tuple[ndarray, ndarray]:
    """
    Compute demix impulse responses from mix impulse responses

    Args:
        lefthrir: binaural impulse response (left ear in col 0, right ear in col 1) for signal from left loudspeaker
        righthrir: binaural impulse response (left ear in col 0, right ear in col 1) for signal from right loudspeaker

    Returns:
        tuple(ndarray, ndarray): demix impulse responses to reconstruct left loudspeaker and right loudspeaker

    """

    '''
    Requires numpy, numpy.ndarray, scipy.fft
    '''

    #Fs = 44100
    #LENFILT = 1023
    #fc = 16000
    #order = 4

    # compute freq domain representations of head related impulse responses
    irlen = len(lefthrir) #235 samples in impulse respone
    NFFT = 4*(2 ** np.ceil(np.log2(irlen))) 
    NFFT = NFFT.astype(int)
    NYQ = NFFT/2
    NYQ = NYQ.astype(int)

    leftHRTF = fft(lefthrir, NFFT, axis=0) #zero-pad to NFFT points
    rightHRTF = fft(righthrir, NFFT, axis=0)
    
    '''
    plt.figure
    fq = np.arange(NYQ)/NFFT*44100
    plt.subplot(2,1,1)
    plt.plot(fq,20*np.log10(np.abs(leftHRTF[:NYQ,:])))
    plt.subplot(2,1,2)
    plt.plot(fq,20*np.log10(np.abs(rightHRTF[:NYQ,:])))
    '''

    # assemble mixing matrix, (experiment later with assembling only 1/2 the matrix and
    # assembling the other half by complex conjugation)
    # H = [leftHRTF[:,0] rightHRTF[:,0]
    #     leftHRTF[:,1] rightHRTF[:,0]]
    Hw = np.concatenate((leftHRTF[:, :, np.newaxis], rightHRTF[:, :, np.newaxis]), axis=2)
    #win = np.hanning(NFFT)
    #Hwwin = Hw*win[:,np.newaxis,np.newaxis]


    # de-mix matrix at every frequency is the matrix inverse of the mix matrix
    # numpy.linalg.inv separately inverts each row of the NFFTx2x2 nd-array and returns the result
    # in a NFFTx2x2 nd-array

    Gw = np.linalg.inv(Hw)

    if 0:
        demixir = ifft(Gw, axis=0)
        lefteardemix = demixir[:irlen, 0, :]
        righteardemix = demixir[:irlen, 1, :]
    else:
        #design linear filters using scipy.signal.firls (least-squares)
        ghat = np.zeros((LENFILT,2,2)) #allocate storage for the least-squares filters
        fax = np.append(np.append(0.,np.repeat(np.arange(1,NYQ)/NYQ,2)),1.) #freq axis for filter design function
        #define rolloff of filter magnitude above fc Hz
        fc = fc / (Fs/2)
        if 0:
            magmod = np.ones_like(fax)
        else:
            magmod = 1/2+1/2*np.tanh(-order * np.pi*(fax - fc))
        
        #define magnitude response of each filter and call function for computing the least-squares filter
        for idx1 in range(2):
            for idx2 in range(2):
                mag = np.append(np.append(Gw[0,idx1,idx2],np.repeat(np.squeeze(Gw[1:NYQ,idx1,idx2]),2)),Gw[0,idx1,idx2])
                #set gain for frequencies > 12 kHz to zero
                ghat[:,idx1,idx2] = scipy.signal.firls(LENFILT,fax,np.abs(mag)*magmod)
                #plt.subplot(2,2,idx1*2+idx2+1)
                #wG, Ghat = scipy.signal.freqz(ghat[:,idx1,idx2],fs=2)
                #plt.plot(fax,20*np.log10(mag),'k',np.arange(NYQ)/NYQ,20*np.log10(np.abs(Hw[:NYQ,idx1,idx2])),'r')
                #plt.plot(wG,20*np.log10(np.abs(Ghat)),'b')
        lefteardemix = ghat[:,0,:]
        righteardemix = ghat[:,1,:]

    return lefteardemix, righteardemix

def bernsteinenvelopecompression(multichansig, sample_rate):
    '''
    halfwave rectify then full envelope compression ...
   
    The envelope compression itself is from Bernsten, van de Par
    and Trahiotis (1996, especially the Appendix). The
    lowpass filtering is from Berstein and Trahiotis (1996,
    especially eq 2 on page 3781). 
    '''

    '''
    Requires numpy, scipy
    '''

    # envelope compression using Weiss/Rose lowpass filter
    compress1 = 0.23
    compress2 = 2.0

    # define lowpass filter
    cutoff = 425  # Hz
    order = 4

    lpf = np.linspace(0, sample_rate / 2, 10000)
    f0 = cutoff * (1 / (2 ** (1 / order) - 1) ** 0.5)
    lpmag = 1 / (1 + (lpf / f0) ** 2) ** (order / 2)
    lpf = lpf / (sample_rate / 2)
    f = lpf
    m = lpmag
    m[-1] = 0
    lowpassfiltercoeffs = scipy.signal.firwin2(256, f, m, window='hamming')

    # compress each filter! 
    envelope = np.abs(scipy.signal.hilbert(multichansig))  # hilbert envelope
    compressedenvelope = (envelope ** (compress1 - 1)) * multichansig  # power-law compression
    rectifiedenvelope = np.maximum(compressedenvelope, 0) ** compress2  # half-wave rectification and raise to power2
    multichanout = scipy.signal.oaconvolve(rectifiedenvelope, lowpassfiltercoeffs[np.newaxis, :], mode='same', axes=1)

    '''
    %Akeroyd's MATLAB code
    nfilters= size(multichansig,1)
    for filter in 1:nfilters
    % get envelope
        envelope = abs(hilbert(multichanneldata(filter,:)));
        % compress the envelope to a power of compression1, while maintaining
        % the fine structure. 
        compressedenvelope = (envelope.^(compress1 - 1)).*multichanneldata(filter,:);
        % rectify that compressed envelope 
        rectifiedenvelope = compressedenvelope;
        findoutput = find(compressedenvelope<0);
        rectifiedenvelope(findoutput) = zeros(size(findoutput));
        % raise to power of compress2
        rectifiedenvelope = rectifiedenvelope.^compress2;
        % overlap-add FIR filter using the fft
        multichanneldata2(filter,:) = fftfilt(lowpassfiltercoefficients, rectifiedenvelope);
    end
    '''
    return multichanout


def pairwisecov(x, y):
    '''
    pairwisecov(x,y) computes covariance matrices of corresponding rows in 2D arrays x and y
    x, y have same dimensions, with different variables on each row and repeated observations on each column and can include NaN entries
    '''

    '''
    Requires numpy, pandas
    '''

    # Initialize an empty list to store the covariance matrices
    cov_matrices = []

    # Compute pairwise covariance matrices
    for i in range(x.shape[0]):
        # Convert the i-th rows of x and y into pandas Series
        s1 = pd.Series(x[i])
        s2 = pd.Series(y[i])

        # Compute the covariance matrix and handle NaN entries
        cov_matrix = np.array([[s1.cov(s1, min_periods=1), s1.cov(s2, min_periods=1)],
                               [s2.cov(s1, min_periods=1), s2.cov(s2, min_periods=1)]])

        cov_matrices.append(cov_matrix)

    # Convert the list of covariance matrices into a 3D numpy array
    cov_matrices = np.array(cov_matrices)

    return cov_matrices


def binauralanalysis(scenefbankleft, scenefbankright):
    '''
    Requires numpy, librosa, scipy, pandas
    '''

    MAXITD = 1  # milliseconds
    MAXILD = 10  # dB
    COHTHRESH = 0.9  # threshold interaural coherence
    SAMPLE_RATE = 44100  # Hz

    WINDOW = 20  # milliseconds, 10 ms a la Faller and Merimaa
    minframelen = WINDOW / 1000 * SAMPLE_RATE
    framelen = 2 ** np.ceil(np.log2(minframelen))
    hop_len = framelen / 2  # the relatively large hop_len is a form of smoothing akin to Faller and Merimaa

    if 1:
        compleft = bernsteinenvelopecompression(scenefbankleft, SAMPLE_RATE)
        compright = bernsteinenvelopecompression(scenefbankright, SAMPLE_RATE)

        leftframes = librosa.util.frame(compleft, frame_length=framelen.astype(int), hop_length=hop_len.astype(int),
                                        axis=1)
        rightframes = librosa.util.frame(compright, frame_length=framelen.astype(int), hop_length=hop_len.astype(int),
                                         axis=1)
    else:
        leftframes = librosa.util.frame(scenefbankleft, frame_length=framelen.astype(int),
                                        hop_length=hop_len.astype(int), axis=1)
        rightframes = np.copy(leftframes)

    nchans, nframes, _ = leftframes.shape

    '''
    Compute ITD, ILD, ICC akin to Faller and Merimaa.
    Current code does not smooth the ITD, ILD, and ICC estimates in the manner of Faller and Merimaa. They smoothed the estimates
    using a first-order IIR filter. Might be something to incorporate down the line.
    '''

    ildest = 10 * np.log10(
        np.mean(leftframes ** 2, axis=-1) / (np.mean(rightframes ** 2, axis=-1) + np.finfo(np.float64).eps))
    # prefer scipy.signal.fftconvolve over scipy.signal.correlate due to better control over matrix axes over which computation is performed
    binxcorr = scipy.signal.fftconvolve(leftframes, np.flip(rightframes, axis=-1), mode='full',
                                        axes=-1)  # compute cross-correlation in FFT domain, by filtering left frame with reversed right frame
    # leftxcorr = scipy.signal.fftconvolve(leftframes,np.flip(leftframes,axis=-1),mode='full',axes=-1) #autocorrelation
    # rightxcorr = scipy.signal.fftconvolve(rightframes,np.flip(rightframes,axis=-1),mode='full',axes=-1) #autocorrelation
    leftxcorrlag0 = np.sum(leftframes ** 2, axis=-1, keepdims=True)
    rightxcorrlag0 = np.sum(rightframes ** 2, axis=-1, keepdims=True)
    cohlags = scipy.signal.correlation_lags(framelen, framelen)  # correlation lag in samples
    iacoh = binxcorr / (np.tile(leftxcorrlag0, len(cohlags)) * np.tile(rightxcorrlag0, len(cohlags)) + np.finfo(
        np.float64).eps) ** 0.5  # normalized interaural coherence (need to check if normalization is done appropriately)

    lagsms = cohlags * 1 / SAMPLE_RATE * 1000  # correlation lag in milliseconds
    validlags = np.logical_and(lagsms >= -MAXITD, lagsms <= MAXITD)
    iamaxcohest = np.max(iacoh[:, :, validlags],
                         axis=-1)  # in each frame, IAC is the maximum coherence in the range of valid correlation lags
    itdindices = np.min(np.argwhere(validlags)) + np.argmax(iacoh[:, :, validlags],
                                                            axis=-1)  # in each frame, ITD is the lag at which IAC is maximum in the range of valid correlation lags
    itdest = lagsms[itdindices]  # convert lags in units of samples into units of milliseconds

    # suppress ITD, ILD estimates in frames where normalized interaural coherence < COHTHRESH
    cohthreshind = iamaxcohest < 0.9
    # invalind = np.logical_or(itdest < -MAXITD, itdest > MAXITD)
    itd = np.copy(itdest)
    ild = np.copy(ildest)
    # itd[np.logical_or(invalind,cohthreshind)] = np.NaN
    itd[cohthreshind] = np.NaN
    ild[cohthreshind] = np.NaN

    # compute descriptive frequency statistics of ITD, ILD, and ICC
    prophighcoh = 1 - np.sum(cohthreshind, axis=-1) / nframes
    meancoh = np.mean(iamaxcohest, axis=-1)
    # itd stats
    meanselitd = np.nanmean(itd, axis=-1)
    # stdselitd = np.nanstd(itd,axis=-1) #already
    medselitd = np.nanmedian(itd, axis=-1)
    itdpercentiles = np.nanpercentile(itd, [10, 25, 75, 90], axis=-1)

    # ild stats
    meanselild = np.nanmean(ild, axis=-1)
    # stdselitd = np.nanstd(itd,axis=-1)
    medselild = np.nanmedian(ild, axis=-1)
    ildpercentiles = np.nanpercentile(ild, [10, 25, 75, 90], axis=-1)

    # joint itd, ild stats
    covmats = pairwisecov(itd, ild)
    unselcovmats = pairwisecov(itdest, ildest)

    # aggregate stats into single feature vector
    # meancoh, meanselitd, medselitd, itdpercentiles, meanselild, medselild, ildpercentiles, covmats, unselcovmats
    binfeatvec = np.concatenate([prophighcoh, meancoh, meanselitd, medselitd, np.ndarray.flatten(itdpercentiles),
                                 meanselild, medselild, np.ndarray.flatten(ildpercentiles),
                                 np.ndarray.flatten(covmats), np.ndarray.flatten(unselcovmats)])

    return binfeatvec


def featureanalysis(signal):
    '''
    Requires numpy, librosa, scipy, pandas, python_auditory_toolbox, torch
    '''

    N_FFT = 512
    N_HOP = 64
    SAMPLE_RATE = 44100
    N_BINS = 32
    NUM_CHAN = 38  # need to figure out number and spacing of channels such that each channel is 1-ERB wide
    LOW_FREQ = 100

    # monaural features
    # numpy() calculations are not supported on "MPS" device. Possibly there's no advantage of sending "signal" tensor 
    # to the "MPS" device as it must be moved back to the cpu anyway
    stft = librosa.stft(y=signal.numpy(), n_fft=N_FFT, hop_length=N_HOP)
    mstft, pstft = librosa.magphase(stft)
    # compute spectral centroid, spectral contrast, spectral bandwidth
    # librosa routines keep the two ears separated
    centroid = librosa.feature.spectral_centroid(S=mstft, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=N_HOP)
  
    bandwidth = librosa.feature.spectral_bandwidth(S=mstft, sr=SAMPLE_RATE, n_fft=N_FFT,
                                                   hop_length=N_HOP, centroid=centroid)
    contrast = librosa.feature.spectral_contrast(S=mstft, sr=SAMPLE_RATE, n_fft=N_FFT,
                                                 hop_length=N_HOP)

    # probability density of spectral centroid, spectral contrast, spectral bandwidth
    # (down the line, include other music features like rhythm features, etc)
    # The two ears get pooled into a single histogram by numpy.histogram
    def histproportion(x, nbins):
        dens, be = np.histogram(x, bins=N_BINS, density=True)
        return dens * np.diff(be), be

    dcentroid, bec = histproportion(centroid, N_BINS)
    dbandwidth, beb = histproportion(bandwidth, N_BINS)
    dcontrast, beco = histproportion(contrast, N_BINS)
    #divide by 1000 in order to report audio frequency in units of kHz
    monfeats = np.concatenate([dcentroid, bec/1000, dbandwidth, beb/1000, dcontrast, beco/1000])

    # binaural features
    fcoefs = pat.MakeErbFilters(SAMPLE_RATE, NUM_CHAN, LOW_FREQ)
    scenefbankleft = pat.ErbFilterBank(signal[0].numpy(), fcoefs)
    scenefbankright = pat.ErbFilterBank(signal[1].numpy(), fcoefs)
    binfeats = binauralanalysis(scenefbankleft, scenefbankright)

    featvec = torch.from_numpy(np.concatenate([monfeats, binfeats]))

    # ****need to check that the feature vector has good properties like 0-mean, unit-std *****
    # featvec = np.torch([avgslpcnt, avgiacnt.unsqueeze(dim=0), avgslpiacnt.unsqueeze(dim=0)])

    return featvec


def extract_text_after_pattern(input_string, pattern):
    '''
    Requires Re package (regular expressions)
    '''

    match = re.search(pattern, input_string)
    if match:
        return input_string[match.end():]
    else:
        return None


def PCAproject(impresp: torch.tensor, topeigs: torch.tensor, basisfilename="svd_demixirs_len1999", basispath=os.getcwd()) -> torch.tensor:
    fname = Path(basispath,basisfilename)
    basis = torch.load(fname) #'U', 'S', 'V', 'select95', 'select98', 'select99', 'Fs'
    eigspace = basis["V"]
    _,numeigs = eigspace.shape
    if topeigs > numeigs:
        topeigs = numeigs
    gproj = torch.matmul(torch.swapaxes(impresp,0,2),eigspace[:,:topeigs])
    return gproj

def invPCAproject(gproj: torch.tensor, basisfilename="svd_demixirs_len1999", basispath=os.getcwd()) -> torch.tensor:
    '''
    input argument gproj is Q x M tensor of projections on Q eigenvectors of M data vectors
    returns N x M tensor of M data vectors reconstructed to N features from stored (N x L) eigenvector basis
    '''

    fname = Path(basispath,basisfilename)
    basis = torch.load(fname) #'U', 'S', 'V', 'select95', 'select98', 'select99', 'Fs'
    eigspace = basis["V"]
    neigs,_ = gproj.shape
    recon = torch.matmul(eigspace[:,:neigs],gproj)
    return recon

def mysinc(x):
    # Handle the case where x is 0
    x = torch.where(x == 0, torch.tensor([1.], device=x.device), x)
    # Define the 
    return torch.sin(torch.pi*x) / (torch.pi*x)

class AllPassDelayFilter(torch.nn.Module):

    def __init__(self,
                 LENFILTER=64,
                 Fs = 44100,
                 device = 'cpu',
                 ):
        super().__init__()
        self.LENFILTER = LENFILTER
        self.Fs = Fs
        self.device = device


    def forward(self, delay: torch.Tensor) -> torch.Tensor:
        '''
        input argument delay is in ms, with batches in axis=0
        
        return taps of all-pass FIR delay filter as a torch tensor B x NTAPS
        '''
        
        if delay.dim() == 0:
            batches = 1
            delay = torch.unsqueeze(delay,dim=0)
        else:
            batches = len(delay)

        n = torch.arange(self.LENFILTER).to(torch.float32).to(self.device)
        htaps = torch.zeros(batches,self.LENFILTER).to(self.device)
        wind = torch.signal.windows.kaiser(self.LENFILTER).to(torch.float32).to(self.device)
        for batch in torch.arange(batches):
            tau = delay[batch]*1e-3*self.Fs # Delay [in samples].
            # Compute sinc filter.
            h = mysinc(n - (self.LENFILTER - 1) / 2 - tau)
            # Multiply sinc filter by window
            h *= wind
            # Normalize to get unity gain.
            h /= torch.sum(h)
            htaps[batch,:] = h

        return htaps
    
class AllPassIntegerDelayFilter(torch.nn.Module):

    def __init__(self,
             LENFILTER=1024,
             Fs = 44100,
             device = 'cpu',
             ):
        super().__init__()
        self.LENFILTER = LENFILTER
        self.Fs = Fs
        self.device = device


    def forward(self, delay: torch.Tensor) -> torch.Tensor:
        '''
        input argument delay is in ms, with batches in axis=0
        
        return taps of all-pass FIR delay filter as a torch tensor B x NTAPS
        '''
        
        if delay.dim() == 0:
            batches = 1
            delay = torch.unsqueeze(delay,dim=0)
        else:
            batches = len(delay)

        n = torch.arange(self.LENFILTER).to(torch.float32).to(self.device)
        htaps = torch.zeros(batches,self.LENFILTER).to(self.device)
        for batch in torch.arange(batches):
            tau = torch.round(delay[batch]*1e-3*self.Fs)
            #place impulse at requisite delay
            htaps[batch.int(),tau.int()] = 1
        #reorder filter taps so that the filter is centered around N//2 by using the fft.fftshift function to do the shift
        htaps = torch.fft.fftshift(htaps,dim=1)

        return htaps
    

class FractionalDelayFiltering(torch.nn.Module):

    def __init__(self,
                 delay: torch.tensor, #milliseconds
                 Fs = 44100,
                 device = 'cpu',
                 ):
        super().__init__()
        self.delay = delay
        self.Fs = Fs
        self.device = device


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        input is [B batches x ... x N samples] tensor of audio waveforms
        
        returns [B batches x ... x N samples] torch tensor of delayed audio waveforms
        '''
        
        if delay.dim() == 0:
            batches = 1
            delay = torch.unsqueeze(delay,dim=0)
        else:
            batches = len(delay)

        intpart = torch.floor(self.delay/1000*self.Fs)


        n = torch.arange(self.LENFILTER).to(torch.float32).to(self.device)
        htaps = torch.zeros(batches,self.LENFILTER).to(self.device)
        wind = torch.signal.windows.kaiser(self.LENFILTER).to(torch.float32).to(self.device)
        for batch in torch.arange(batches):
            tau = delay[batch]*1e-3*self.Fs # Delay [in samples].
            # Compute sinc filter.
            h = mysinc(n - (self.LENFILTER - 1) / 2 - tau)
            # Multiply sinc filter by window
            h *= wind
            # Normalize to get unity gain.
            h /= torch.sum(h)
            htaps[batch,:] = h

        return htaps
    

class FeaturesCalculator(torch.nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, signal: torch.Tensor) -> torch.Tensor:        
        featvec = featureanalysis(signal)
        return featvec

class AtMicCadenzaICASSP2024_3(Dataset):    
    def __init__(self,
                 chunk = 4, #default duration of audio chunks in seconds 
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

        # Load the spatial configurations metadata
        spatconfigs_file = Path(metadatadir,"head_loudspeaker_positions.json")
        with open(spatconfigs_file, encoding="utf-8") as f:
            spatconfigs_metadata = json.load(f)
        self.spatconfigs_metadata = spatconfigs_metadata
        self.spatconfigslabels = list(spatconfigs_metadata.keys())
         
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
            idxitem_scenedict = self.spatconfigs_metadata[f"{idx_scenemetadata['Head Position']}"]
            finf = torchaudio.info(idx_atmicname)
            nchunks = np.floor(finf.num_frames / finf.sample_rate / chunk).astype(int) #leave out the leftover chunk that is less than CHUNK long

            # create metadata entry identifying audio "chunk", and include the spatial configuration metadata
            for chunki in range(nchunks):
                self.chunkidx_metadata.update({f"{chunkidx}":
                                               (idx_atmicname,
                                                idx_audname,
                                                chunki*finf.sample_rate*chunk,
                                                finf.sample_rate*chunk,
                                                idxitem_scenedict,
                                                f"{idxitem_scenedict['subject']}, {idxitem_scenedict['left_angle']}, {idxitem_scenedict['right_angle']}")
                                                })
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
        spatconfig = chunk_metadat[5] #chunk_metadat[4] contains the same information in a dict
        with torch.no_grad():
            audinput = audinput.to(torch.float32).to(self.device)
            features = self.calculator(audinput)

        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            target = self.target_transform(target)
        return features, audinput, spatconfig, target
    

class AtMicCadenzaICASSP2024_2(Dataset):    
    def __init__(self,
                 chunk = 4, #default duration of audio chunks in seconds 
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