import json
import numpy as np
import scipy
import torch
import re
import librosa
import python_auditory_toolbox.auditory_toolbox as pat
import pandas as pd
from clarity.utils.file_io import read_signal
from numpy import ndarray
from pathlib import Path
from scipy.fft import fft, ifft

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

def mixmatrix_to_demix(lefthrir: ndarray, righthrir: ndarray) -> tuple[ndarray, ndarray, ndarray, ndarray]:
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
    
    #compute freq domain representations of head related impulse responses
    NFFT = 2**np.ceil(np.log2(irlen))
    leftHRTF = fft(lefthrir,256,axis=0)
    rightHRTF = fft(righthrir,256,axis=0)
    
    #assemble mixing matrix, (experiment later with assembling only 1/2 the matrix and
    #assembling the other half by complex conjugation)
    #H = [leftHRTF[:,0] rightHRTF[:,0]
    #     leftHRTF[:,1] rightHRTF[:,0]]
    Hw = np.concatenate((leftHRTF[:,:,np.newaxis],rightHRTF[:,:,np.newaxis]),axis=2)

    #de-mix matrix at every frequency is the matrix inverse of the mix matrix 
    #numpy.linalg.inv separately inverts each row of the NFFTx2x2 nd-array and returns the result
    #in a NFFTx2x2 nd-array
    # ***** N.B. the inverted transfer function / impulse response does not behave well at high frequencies
    # and low frequencies, so may need to smooth the spectrum before inverting *****
    
    Gw = np.linalg.inv(Hw)
    demixir = ifft(Gw,axis=0)
    lefteardemix = demixir[:irlen,0,:]
    righteardemix = demixir[:irlen,1,:]

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


    #envelope compression using Weiss/Rose lowpass filter
    compress1 = 0.23;
    compress2 = 2.0;
 
    # define lowpass filter
    cutoff = 425; #Hz
    order = 4;

    lpf = np.linspace(0, sample_rate/2, 10000)
    f0 = cutoff * (1/ (2**(1/order)-1)**0.5)
    lpmag = 1 / (1+(lpf/f0)**2) ** (order/2)
    lpf=lpf / (sample_rate/2);
    f=lpf
    m=lpmag
    m[-1]=0
    lowpassfiltercoeffs = scipy.signal.firwin2(256, f, m, window='hamming')
    
    # compress each filter! 
    envelope = np.abs(scipy.signal.hilbert(multichansig)) #hilbert envelope
    compressedenvelope = (envelope**(compress1-1))*multichansig #power-law compression
    rectifiedenvelope = np.maximum(compressedenvelope, 0)**compress2 #half-wave rectification and raise to power2
    multichanout = scipy.signal.oaconvolve(rectifiedenvelope,lowpassfiltercoeffs[np.newaxis,:],mode='same',axes=1)
    
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

def pairwisecov(x,y):
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


def binauralanalysis(scenefbankleft,scenefbankright):    
    '''
    Requires numpy, librosa, scipy, pandas
    '''
    
    MAXITD = 1 #milliseconds
    MAXILD = 10 #dB
    COHTHRESH = 0.9 #threshold interaural coherence
    SAMPLE_RATE = 44100 #Hz
    
    WINDOW = 20 #milliseconds, 10 ms a la Faller and Merimaa
    minframelen = WINDOW/1000 * SAMPLE_RATE
    framelen =  2**np.ceil(np.log2(minframelen))
    hop_len = framelen/2 #the relatively large hop_len is a form of smoothing akin to Faller and Merimaa
    
    if 1:
        compleft = bernsteinenvelopecompression(scenefbankleft, SAMPLE_RATE)
        compright = bernsteinenvelopecompression(scenefbankright,SAMPLE_RATE)
    
        leftframes = librosa.util.frame(compleft,frame_length = framelen.astype(int), hop_length = hop_len.astype(int), axis = 1)
        rightframes = librosa.util.frame(compright,frame_length = framelen.astype(int), hop_length = hop_len.astype(int), axis = 1)
    else:
        leftframes = librosa.util.frame(scenefbankleft,frame_length = framelen.astype(int), hop_length = hop_len.astype(int), axis = 1)
        rightframes = np.copy(leftframes)
    
    nchans, nframes, _ = leftframes.shape
    
    '''
    Compute ITD, ILD, ICC akin to Faller and Merimaa.
    Current code does not smooth the ITD, ILD, and ICC estimates in the manner of Faller and Merimaa. They smoothed the estimates
    using a first-order IIR filter. Might be something to incorporate down the line.
    '''
 
    #######
    ildest = 10*np.log10(np.mean(leftframes**2,axis = -1) / np.mean(rightframes**2,axis=-1))
    #prefer scipy.signal.fftconvolve over scipy.signal.correlate due to better control over matrix axes over which computation is performed
    binxcorr = scipy.signal.fftconvolve(leftframes,np.flip(rightframes,axis=-1),mode='full',axes=-1) #compute cross-correlation in FFT domain, by filtering left frame with reversed right frame
    #leftxcorr = scipy.signal.fftconvolve(leftframes,np.flip(leftframes,axis=-1),mode='full',axes=-1) #autocorrelation
    #rightxcorr = scipy.signal.fftconvolve(rightframes,np.flip(rightframes,axis=-1),mode='full',axes=-1) #autocorrelation
    leftxcorrlag0 = np.sum(leftframes**2,axis=-1,keepdims=True)
    rightxcorrlag0 = np.sum(rightframes**2,axis=-1,keepdims=True)
    cohlags = scipy.signal.correlation_lags(framelen,framelen) #correlation lag in samples
    iacoh = binxcorr / (np.tile(leftxcorrlag0,len(cohlags))*np.tile(rightxcorrlag0,len(cohlags)) + np.finfo(np.float64).eps)**0.5 #normalized interaural coherence (need to check if normalization is done appropriately) 
      
    lagsms = cohlags*1/SAMPLE_RATE*1000 #correlation lag in milliseconds
    validlags = np.logical_and(lagsms >= -MAXITD, lagsms <= MAXITD)
    iamaxcohest = np.max(iacoh[:,:,validlags],axis=-1) #in each frame, IAC is the maximum coherence in the range of valid correlation lags
    itdindices = np.min(np.argwhere(validlags)) + np.argmax(iacoh[:,:,validlags],axis=-1) #in each frame, ITD is the lag at which IAC is maximum in the range of valid correlation lags
    itdest = lagsms[itdindices] #convert lags in units of samples into units of milliseconds

    #suppress ITD, ILD estimates in frames where normalized interaural coherence < COHTHRESH
    cohthreshind = iamaxcohest < 0.9
    #invalind = np.logical_or(itdest < -MAXITD, itdest > MAXITD)
    itd = np.copy(itdest)
    ild = np.copy(ildest)
    #itd[np.logical_or(invalind,cohthreshind)] = np.NaN
    itd[cohthreshind] = np.NaN
    ild[cohthreshind] = np.NaN

    #compute descriptive frequency statistics of ITD, ILD, and ICC
    prophighcoh = 1 - np.sum(cohthreshind,axis=-1) / nframes
    meancoh = np.mean(iamaxcohest,axis=-1)
    #itd stats
    meanselitd = np.nanmean(itd,axis=-1)
    #stdselitd = np.nanstd(itd,axis=-1) #already 
    medselitd = np.nanmedian(itd,axis=-1)
    itdpercentiles = np.nanpercentile(itd,[10, 25, 75, 90],axis=-1)

    #ild stats
    meanselild = np.nanmean(ild,axis=-1)
    #stdselitd = np.nanstd(itd,axis=-1)
    medselild = np.nanmedian(ild,axis=-1)
    ildpercentiles = np.nanpercentile(ild,[10, 25, 75, 90],axis=-1)

    #joint itd, ild stats
    covmats = pairwisecov(itd,ild)
    unselcovmats = pairwisecov(itdest,ildest)

    #aggregate stats into single feature vector
    #meancoh, meanselitd, medselitd, itdpercentiles, meanselild, medselild, ildpercentiles, covmats, unselcovmats
    binfeatvec = np.concatenate([prophighcoh, meancoh, meanselitd, medselitd, np.ndarray.flatten(itdpercentiles),
                                 meanselild, medselild, np.ndarray.flatten(ildpercentiles), 
                                 np.ndarray.flatten(covmats), np.ndarray.flatten(unselcovmats)])
       
    return binfeatvec

def featureanalysis(signal):   
    '''
    Requires numpy, librosa, scipy, pandas, python_auditory_toolbox, torch
    '''
    
    N_FFT = 512
    N_HOP = 16
    SAMPLE_RATE = 44100
    N_BINS = 32
    NUM_CHAN = 16 #need to figure out number and spacing of channels such that each channel is 1-ERB wide
    LOW_FREQ = 100
    
    #monaural features
    stft = librosa.stft(y = signal.numpy(),n_fft=N_FFT,hop_length=N_HOP)
    mstft, pstft = librosa.magphase(stft)
    #compute spectral centroid, spectral contrast, spectral bandwidth
    #librosa routines keep the two ears separated
    centroid = librosa.feature.spectral_centroid(S = mstft, sr = SAMPLE_RATE, n_fft = N_FFT, hop_length = N_HOP)
    bandwidth = librosa.feature.spectral_bandwidth(S = mstft, sr = SAMPLE_RATE, n_fft = N_FFT,
                                                    hop_length = N_HOP, centroid = centroid)
    contrast = librosa.feature.spectral_contrast(S = mstft, sr = SAMPLE_RATE, n_fft = N_FFT,
                                                    hop_length = N_HOP)
    #probability density of spectral centroid, spectral contrast, spectral bandwidth
    #(down the line, include other music features like rhythm features, etc)
    #The two ears get pooled into a single histogram by numpy.histogram
    def histproportion(x,nbins):
        dens, be = np.histogram(x,bins=N_BINS,density=True)
        return dens*np.diff(be), be

    dcentroid, bec = histproportion(centroid,N_BINS)
    dbandwidth, beb = histproportion(bandwidth,N_BINS)
    dcontrast, beco = histproportion(contrast,N_BINS)
    #featvec = np.torch([avgslpcnt, avgiacnt.unsqueeze(dim=0), avgslpiacnt.unsqueeze(dim=0)])    
    monfeats = np.concatenate([dcentroid, bec, dbandwidth, beb, dcontrast, beco])
    
    #binaural features
    fcoefs = pat.MakeErbFilters(SAMPLE_RATE, NUM_CHAN, LOW_FREQ)
    scenefbankleft = pat.ErbFilterBank(signal.numpy()[0], fcoefs)
    scenefbankright = pat.ErbFilterBank(signal.numpy()[1], fcoefs)    
    binfeats = binauralanalysis(scenefbankleft,scenefbankright)
    
    featvec = torch.from_numpy(np.concatenate([monfeats, binfeats]))
    
    # ****need to check that the feature vector has good properties like 0-mean, unit-std *****
    #featvec = np.torch([avgslpcnt, avgiacnt.unsqueeze(dim=0), avgslpiacnt.unsqueeze(dim=0)])
    
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