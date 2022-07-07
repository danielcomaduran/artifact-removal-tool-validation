"""
    EEG preprocessing
    -----------------
        These functions are used to preprocess the EEG data
"""

## Import libraries
import numpy as np
import scipy.signal as signal

def filter(type:str, x:np.ndarray, fc:list[int], srate:int, order:int = 4):
    """
        This function applies Butterworth filters of the desired type, cut-off frequency, and order

        Parameters
        ----------
            Type: str
                Filter type, select form the following
                    'lowp': Low-pass filter
                    'high': High-pass filter
                    'band': Band-pass filter
                    'notch': Notch filter
            x: array_like
        
        Returns
        -------
            Filtered signal x
    """
    ftype = {'low':'lowpass', 'high':"highpass", 'band':'bandpass', 'notch':'bandstop'}

    sos = signal.butter(N=order, Wn=fc, btype=ftype[type], fs=srate, output='sos')
    x_filt = signal.sosfiltfilt(sos, x)

    return x_filt

def auc(psd, srate:int, fstim:float, fband:int, harmonics:int = 1):
    """
        Calculate Area Under the Curve of te Power Spectral Density
        
    """

    # Make sure you have a row matrix
    data_shape = np.shape(psd)
    data_length = data_shape[-1]

    if data_shape[0] > data_shape[1]:
        psd = psd.T
        data_length = data_shape[0]     

    auc = 0 # Initialize AUC
    f = np.linspace(0, srate/2, data_length)    # Frequency vector [Hz]

    # Calculate AUC for each harmonic
    for h in range(1,harmonics):
        fmin = fstim*h - fband
        fmax = fstim*h + fband
        fidx = (f>=fmin) & (f<=fmax)                    # Frequencies indeces
        auc += np.trapz(psd[:,fidx], f[fidx], axis=1)    # Additive AUC

    return  auc