import numpy as np


'''Created by sampling 100000 random frequencies between [20,40] 
(uniform dist). Same amplitude (1), sampling frequency (1000Hz), 
total time (1 sec), offset (0), 10% of amp for noise (0.1). 
I chose these frequencies to give enough periods in a 1 second 
window for the thing to learn from. They are also spread enough 
that really I want to see if this thing can back out the frequency 
from just a few data points. Also limiting the values between -1 
and 1 (more or less) cuts down on some preprocessing (for now)'''


# filenames
sinewave_filename = "single_freq_sines.npy" # row: example, cols: data with time
freqs_file = "single_freq_frequencies.npy" # corresponding freqs

# sines in `data` and corresponding freqs used in creation in `freqs`
sines = np.load(sinewave_filename)
freqs = np.load(freqs_file)

## Time Vector
total_time = 1 # sec
fs         = 1000 # Hz
N          = int(total_time * fs) + 1
t          = np.linspace(0, total_time, N)


