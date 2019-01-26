
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

def build_sin(freq, duration, phase=0, sample_rate=44100, amp=1.0):
    t = np.linspace(0, duration, int(duration*sample_rate))
    sin = amp * np.sin(freq * 2 * np.pi * t + phase)
    return sin

def dotAmplitude(sin1, sin2): #returns dot product amplitude of two input signals
    dot_product = np.dot(sin1, sin2)
    return 2*(dot_product/len(sin1))

def innerProduct(sin_target, freq):
    sin_probe = build_sin(freq, 1, amp=1.0)
    return dotAmplitude(sin_target, sin_probe)

def generate_half_mag(data, fft_end, fft_start=0):
    complex_spectrum = np.fft.fft(data[fft_start:fft_end])
    magnitude_spectrum = np.abs(complex_spectrum)
    half_magnitude_spectrum = magnitude_spectrum[0: int(len(magnitude_spectrum)/2)]
    return half_magnitude_spectrum

def freq_map(w, fs=44100.0):
    T = w/fs
    k = np.arange(int(w))
    m = k/T
    return m


# In[2]:

#read qbhexamples.wav and convert to 44100 sampling rate
qbh_original = wav_read('qbhexamples.wav')
wav_write('qbh_44100.wav', 44100, qbh_original[1])
qbh_44100 = wav_read('qbh_44100.wav')

srate_qbh = qbh_44100[0]
data_qbh = qbh_44100[1]

print srate_qbh, len(data_qbh)


# In[53]:

frere_jaques = wav_read('frere_jaques.wav')
srate_fj = frere_jaques[0]
data_fj = frere_jaques[1]

print srate_fj, len(data_fj)


# In[54]:

data_300 = build_sin(300, 1)
ipd.Audio(data_300, rate=44100)


# In[56]:

def freq_from_mag(data, window_size):
    num_windows = len(data)/window_size
    f0_data = []
    freq_res = 44100.0/window_size
    
    for window in range(num_windows):
        half_mag_spec = generate_half_mag(data, fft_start=window*window_size, fft_end=(window+1)*window_size)
        max_mag = 0.0
        max_freq_bin= 0
        for freq_bin,mag in enumerate(half_mag_spec):
                if(mag > max_mag):
                    max_mag = mag
                    max_freq_bin = freq_bin
        f0_data.append(max_freq_bin*freq_res)
    
    return f0_data

fj_mag_est = freq_from_mag(data_fj, 2048)
plt.plot(fj_mag_est)
plt.show()

qbh_mag_est = freq_from_mag(data_qbh, 2048)
plt.plot(qbh_mag_est)
plt.show()



# In[57]:

def get_peaks(data):
    high_peak = 0.0 
    hpeak_index = 0 
    peak_array = [0]*len(data)
    for i in range(len(data)):
        if(data[i] > high_peak): 
            high_peak=data[i] 
            hpeak_index = i
            peak_array[i] = hpeak_index
    return peak_array

def get_intervals(data):
    time_data = [0.0]*len(data)
    for t in range(len(time_data)): 
        time_data[t] = t*1.0/44100;
    return time_data

def autocorrelate(data):
    ac = np.correlate(data, data, mode='full')
    return ac[len(ac)/2:]

def freq_from_lags(data, window_size):
    num_windows = len(data)/window_size 
    f0_data = [0.0]*int(math.ceil(num_windows))
    for window in range(len(f0_data)):
        ac = autocorrelate(data[window_size*window:window_size*(window+1)])
        rev_ac = list(reversed(ac))
        index_data = get_peaks(rev_ac) 
        time_data = get_intervals(data)
        p_data = []
        t_data = []
        for i in range(len(index_data)-1):
            if(index_data[i+1] == 0 and index_data[i] != 0): 
                p_data.append(index_data[i])
        for p in range(len(p_data)-1): 
            t_data.append(time_data[p_data[p+1]] - time_data[p_data[p]])
        if(sum(t_data) != 0):
            freq = 1/float(sum(t_data)/len(t_data))
            f0_data[window] = freq
    
    return f0_data
    
fj_lag_est = freq_from_lags(data_fj, 2048)
plt.plot(fj_lag_est)
plt.show()

qbh_lag_est = freq_from_lags(data_qbh, 2048)
plt.plot(qbh_lag_est)
plt.show()


# In[64]:

#fj_mag_est = freq_from_mag(data_qbh, 2048)

#fj_lag_est = freq_from_lags(data_qbh, 2048)

fj_sum = [m + l for m, l in zip(qbh_mag_est, qbh_lag_est)]

plt.plot(fj_sum)
plt.show()


# In[70]:

num_windows = len(data_fj)/2048
centroid_data = []

for window in range(num_windows):
    complex_spectrum = np.fft.fft(data_fj[window*2048: window*2048+2048])
    magnitude_spectrum = np.abs(complex_spectrum)
    half_magnitude_spectrum = magnitude_spectrum[0: int(len(magnitude_spectrum)/2)]
    
    centroid_numer = 0
    centroid_denom = 0
    for freq,mag in enumerate(half_magnitude_spectrum):
        centroid_numer += freq*mag
        centroid_denom += mag
        
    centroid_data.append(centroid_numer/centroid_denom)
    
plt.plot(centroid_data)
plt.show()


# In[61]:

orig_sin = build_sin(, 1)
for freq in centroid_data:
    centroid_sin = build_sin(freq, 1)
    for j in range(len(orig_sin)):
        orig_sin[j] += centroid_sin[j]


# In[62]:

plt.plot(orig_sin)
plt.show()


# In[63]:

ipd.Audio(orig_sin, rate=44100)


# In[ ]:



