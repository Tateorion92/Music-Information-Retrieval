
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

##applying OLA to synthesis frames 

#hann window function, apllied to each grain before synthesis
def hann(grain):
    N = len(grain)
    hgrain = [0]*N
    for i,x in enumerate(grain):
        w = 0.5*(1-np.cos((2.0*np.pi*(i+N/2.0))/(N-1)))
        #print w*x, x
        hgrain[i] = int(w*x)
    return hgrain


# In[3]:

data_read = wav_read('qbhexamples.wav')

srate = data_read[0]
data = data_read[1]
data = data[:srate*10]

grains = []           ##array of all the grains
grain_len = srate/10  ##number of samples in each grain 
Ha = srate/20         ##analysis hopsize (will be evaluated as Hs/alpha?)
print Ha
Hs = grain_len/2      ##synthesis hopsize (specific value, grain_len/2 or grain_len/4)
stretch_factor = float(Hs)/float(Ha)


# In[4]:

##split up original data into grains (aka analysis frames)

num_grains = len(data)/Ha-1
shift_range = 512

for grain in range(num_grains):
    if(grain*Ha <= (len(data)-grain_len)):
        if(grain==0):
            grains.append(data[grain*Ha:(grain*Ha)+grain_len])
        else:
            prv_g = grain-1
            des_i = prv_g*Ha+Hs     ##starting index of natural progression from previous grain
            previous = grains[prv_g]
            natural = data[des_i:des_i+grain_len]
            
            #target_region = data[grain*Ha-shift_index:grain*Ha+grain_len+shift_index+1]
            
            corr = 0
            max_corr = 0
            shift_index = 0
            #loop to find the maximally optimal succeeding grain
            for shift in range(-shift_range, shift_range+1):
                target = data[grain*Ha+shift:grain*Ha+shift+grain_len]
                corr = np.correlate(natural, target)
                if(corr[0] > max_corr):
                    max_corr = corr[0]
                    shift_index = shift
            target = data[grain*Ha+shift_index:grain*Ha+grain_len]
            grains.append(target)
            


# In[5]:

##copy grains back into a single data array using the synthesis hopsize

def hann_grains(grains):
    hgrains = []
    for x in range(len(grains)):
        grain = grains[x]
        grain = hann(grain)
        hgrains.append(grain)
    return hgrains

def synthesize(grains, Hs):
    synth = [0]*(len(grains)*Hs+Hs+1)
    for synhop, grain in enumerate(grains):
        for i in range(len(grain)):
            synth[synhop*Hs+i] = synth[synhop*Hs+i]+grain[i]
    return synth

hgrains = hann_grains(grains)
data_synth = synthesize(hgrains, Hs)


# In[6]:

ipd.Audio(data_synth, rate=srate)


# In[ ]:

one = np.array(range(10))
two = np.array(range(10,20))
print two
x = np.correlate(one, two)
k =0
for i in one:
    k += i    
print x
print x[0]


# In[ ]:



