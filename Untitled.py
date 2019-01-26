
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt

import numpy as np
import IPython.display as ipd

srate = 44100
duration = 5 
freq1 = 220 
amp = 0.5 

# create srate*duration time points 
t = np.linspace(0,duration,srate*duration)
# create a 220Hz sine wave at 44100 sampling and amplitude of 0.5 
data = amp * np.sin(2*np.pi*220*t);
# play back the audio 
ipd.Audio(data,rate=srate)




# In[2]:




# generate a sine wave with specified frequency, duration, sampling 
# rate and amplitude. You can modify to this to add an argument for phase 

def generate_sin(freq, duration, srate=44100.0, amp=1.0): 
    t = np.linspace(0,duration,int(srate*duration))
    data = amp * np.sin(2*np.pi*freq *t)
    #print(data)
    #plt.plot(t,data)
    #plt.show()
    return data

# similar function for generating a cosine wave 
def generate_cos(freq, duration, srate=44100.0, amp=1.0): 
    t = np.linspace(0,duration,int(srate*duration))
    data = amp * np.cos(2*np.pi*freq *t)
    return data

# the frequencies of 3 notes 
c_freq = 523.0 
d_freq = 587.0 
e_freq = 659.0 

# generate the sine waves
c_data = generate_sin(c_freq, 0.5, amp=2.0)
d_data = generate_sin(d_freq, 0.5, amp=2.0)
e_data = generate_sin(e_freq, 0.5, amp=2.0)

# a simple melody with 3 notes 
data = np.hstack([c_data, d_data, e_data, c_data, 
                  c_data, d_data, e_data, c_data])

ipd.Audio(data,rate=srate)


# In[17]:

test_sin = generate_sin(1, 1, srate=8.0, amp=1.0)

fft_size = 4096
complex_spectrum = np.fft.fft(test_sin[0:fft_size])
print(complex_spectrum)
magnitude_spectrum = np.abs(complex_spectrum)
plt.plot(magnitude_spectrum)



# In[4]:

#sin1 = generate_sin(440, 1)
#sin2 = generate_sin(440, 1)

#dot_product = np.dot(sin1, sin2)

#print(dot_product)
#print(len(sin1))
#for i in range(0, len(sin1)):
 #  print sin1[i], '+', sin2[i], '=', sin1[i]+sin2[i]


# In[24]:

test_sin = generate_sin(100, 1, srate=1000, amp=1.0)

def dft(sin):
    N = len(sin)
    freq_bin = np.zeros(N)
    for k in range(N):
        for n in range(N):
            angle = 2*np.pi*n*k/N
            freq_bin[k] += sin[n] * (np.cos(-angle) + np.sin(-angle))
        
        #freq_bin.append(complex_sum)
        #print freq_bin
    
    return freq_bin
        
complex_spectrum = dft(test_sin)

magnitude_spectrum = np.abs(complex_spectrum)

plt.plot(magnitude_spectrum)


# In[6]:

test_sin = generate_sin(1, 1, srate=8.0, amp=1.0)
print(test_sin)


# In[ ]:



