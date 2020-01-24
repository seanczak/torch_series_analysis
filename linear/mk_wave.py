import numpy as np
import pandas as pd

def mk_wave(freq, amp= 1, noise_std = 0.1, signal_mean = 0, 
            fs = 10, total_time=100, phase_shift = False):
    '''
    Parameters
    ----------
    freq
        frequency in Hz
    amp (Default:  1)
        amplitude in desired units
    noise_std (Default:  0.1)
        st. deviation of the noise term (noise is drawn from normal dist)
    signal_mean (Default:  0)
        mean of the signal in desired units
    fs
        sampling frequency in Hz
    total_time
        length of time array and signal

    Returns
    -------
        tuple of form (time, signal) as prescribed by the parameters passed
        The signal also includes a random phase shift
    '''
    
    # create time array
    N = int(total_time * fs)
    t = np.linspace(0, total_time, N)
    
    # induce random phase shift
    phase = np.random.uniform(high=np.pi/2,low=-np.pi/2) if phase_shift else 0
    
    # create noise and induce signal mean
    noise = np.random.normal(signal_mean,noise_std,N)
    
    # signal
    s =  amp * np.sin(freq * 2 * np.pi * t + phase) + noise 
        
    return t,s