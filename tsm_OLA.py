
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


# In[4]:

##split up original data into grains (aka analysis frames)
grains = []           ##array of all the grains
grain_len = srate/10  ##number of samples in each grain 
Ha = srate/20      ##analysis hopsize (will be evaluated as Hs/alpha?)
print Ha
num_grains = len(data)/Ha-1

for grain in range(num_grains):
    if(grain*Ha <= (len(data)-grain_len)):
        grains.append(data[grain*Ha:(grain*Ha)+grain_len])


# In[5]:

##copy grains back into a single data array using the synthesis hopsize
Hs = grain_len/2      ##synthesis hopsize (specific value, grain_len/2 or grain_len/4)
stretch_factor = float(Hs)/float(Ha)

def hann_grains(grains):
    hgrains = []
    for x in range(len(grains)):
        grain = grains[x]
        #print grain
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
#print np.array(hgrains[:5])

data_synth = synthesize(hgrains, Hs)
#print len(data_synth), len(data)
#print np.array(data_synth[5000:5100])
#print np.array(data[5000:5100])


# In[6]:

ipd.Audio(data_synth, rate=srate)


# In[150]:

plt.plot(data_synth[5000:5100])
plt.figure()
plt.plot(data[5000:5100])
plt.show()


# In[ ]:

test = [  3   ,2, -14, -19, -15, -50, -34, -40, -27, -48]
print test
htest = hann(test)
print htest
print test[-len(test)/2:len(test)/2-1]


# In[137]:

wav_write('synth.wav',22500,data_synth)


# In[ ]:



