
# coding: utf-8

# In[1]:

import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
import cmath
from scipy.io.wavfile import write, read

def wav_write(file_name,sample_rate,signal):
    write(file_name,sample_rate,np.array(signal))
    
def wav_read(file_name):
    return read(file_name)


# In[2]:

data_read = wav_read('qbhexamples.wav')

srate = data_read[0]
data = data_read[1]
data = data[:srate*10]

print srate, len(data), data
ipd.Audio(data, rate=srate)


# In[33]:

##split up original data into grains (aka analysis frames)
grains = []           ##array of all the grains
grain_len = srate/10  ##number of amples in each grain (here = 100ms)
Ha = srate/40       ##analysis hopsize (here 25ms, will be evaluated as Hs/alpha)

num_grains = len(data)/Ha-1

for grain in range(num_grains):
    if(grain*Ha <= (len(data)-grain_len)):
        grains.append(data[grain*Ha:(grain*Ha)+grain_len])


# In[34]:

##copy grains back into a single data array using the synthesis hopsize
Hs = grain_len/2      ##synthesis hopsize (specific value, grain_len/2 or grain_len/4)
stretch_factor = float(Hs)/float(Ha)
print Hs, Ha, stretch_factor
data_synth = [0]*(len(grains)*Hs+Hs)

for x in range(len(grains)):
    grain = grains[x]
    data_synth[x*Hs:x*Hs+grain_len] += grain


# In[35]:

ipd.Audio(data_synth, rate=srate)


# In[30]:




# In[ ]:




# In[ ]:



