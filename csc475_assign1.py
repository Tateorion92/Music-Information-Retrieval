
# coding: utf-8

# In[16]:

import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
import cmath
from scipy.io.wavfile import write, read

freq_1 = 220 #Hz cycles per second
amp = 2.0
duration = 2 #number of seconds in sample
sample_rate = 44100 #samples per second

#create array of time points
t = np.linspace(0, duration, duration*sample_rate)
print(t)

#transform array to sine wave
sin = amp * np.sin(2*np.pi*freq_1*t)
#play the frequency
ipd.Audio(sin, rate = sample_rate)


# In[22]:

#functions for generating sin/cos waves with specified frequency, duration,
#sample rate, and amplitude (as shown in csc475_sinusoids.ipynb by George Tzanetakis)

def build_sin(freq, duration, phase=0, sample_rate=44100, amp=1.0):
    t = np.linspace(0, duration, int(duration*sample_rate))
    sin = amp * np.sin(freq * 2 * np.pi * t + phase)
    return sin

def build_cos(freq, duration, phase=0, sample_rate=44100, amp=1.0):
    t = np.linspace(0, duration, int(duration*sample_rate))
    cos = amp * np.cos(freq * 2 * np.pi * t + phase)
    return cos
    
    
#frequencies of three notes: c, d, e:
freq_c = 523.0
freq_d = 587.0
freq_e = 659.0

#build the sin waves for each of the notes
sin_c = build_sin(freq_c, 0.5, amp=2.0)
sin_d = build_sin(freq_d, 0.5, amp=2.0)
sin_e = build_sin(freq_e, 0.5, amp=2.0)

#for assignment 2
sin_3 = build_sin(3, 0.5, amp=2.0)

#create a simple melody with the three notes by playing them in sequence
#with np.hstack
sin = np.hstack([sin_c, sin_d, sin_e, sin_c, sin_c, sin_d, sin_e, sin_c])

#for assignment 2
wav_write('frere_jaques.wav',44100,np.array(sin))

ipd.Audio(sin, rate=sample_rate)
#plt.plot(sin)
#plt.show()


# In[4]:

#play the three notes simultaneously
sin = np.vstack([sin_c, sin_d, sin_e])
ipd.Audio(sin, rate=sample_rate)


# In[6]:

#Q1.1
#functions for estimating amplitude of a given sinusoid
#-referenced from George Tzanetakis
def peakAmplitude(sin): #returns the peak amplitude of an input signal
    return max(sin)

def rmsAmplitude(sin): #returns the RMS amplitude of an input signal
    square_sum = 0
    for i in sin:
        square_sum += i * i
    mean_square = square_sum/len(sin)
    root_mean_square = np.sqrt(mean_square) * np.sqrt(2.0)
    return root_mean_square

def dotAmplitude(sin1, sin2): #returns dot product amplitude of two input signals
    dot_product = np.dot(sin1, sin2)
    return 2*(dot_product/len(sin1))

freq = 300
sin = build_sin(freq, 0.5, amp=4.0)
probe_1 = build_sin(freq, 0.5, amp=1.0)
noise = np.random.normal(0, 0.5, len(sin))
sin_corrupt = sin + noise

(round(peakAmplitude(sin_corrupt), 3), round(rmsAmplitude(sin_corrupt), 3), 
round(dotAmplitude(sin_corrupt, probe_1), 3))


# In[7]:

#Q1.2
#function that mixes harmonic frequencies. Option to pass specific amplitudes and phases.

def harmonic(freq, time_to_plot=1, srate=1000, amp_1=1.0, amp_2=0.5, amp_3=0.25, phase_1=0, phase_2=0,
                                                                phase_3=0):
    sin_1 = build_sin(freq, duration=time_to_plot, phase=phase_1, sample_rate=srate, amp=amp_1)
    sin_2 = build_sin(2*freq, duration=time_to_plot, phase=phase_2, sample_rate=srate, amp=amp_2)
    sin_3 = build_sin(3*freq, duration=time_to_plot, phase=phase_3, sample_rate=srate, amp=amp_3)
    
    return [s1 + s2 + s3 for s1,s2,s3 in zip(sin_1,sin_2,sin_3)]


sin_ph0 = harmonic(440, srate=1000, amp_1=1.0, amp_2=0.5, amp_3=0.33)

phase_1 = np.random.normal()
phase_2 = np.random.normal()
phase_3 = np.random.normal()
sin_phrand = harmonic(440, srate=1000, amp_1=1.0, amp_2=0.5, amp_3=0.33, phase_1=phase_1, phase_2=phase_2, phase_3=phase_3)

input_1 = build_sin(440, 1, sample_rate=1000)
input_2 = build_sin(880, 1, sample_rate=1000)
input_3 = build_sin(1320, 1, sample_rate=1000)

plt.figure()
plt.plot(sin_ph0)
plt.show()
plt.figure()
plt.plot(sin_phrand)
plt.show()


# In[ ]:




# In[8]:

#Q1.3

sin_ph0 = harmonic(440, srate=1000, amp_1=1.0, amp_2=0.5, amp_3=0.33)

phase_1 = np.random.normal()
phase_2 = np.random.normal()
phase_3 = np.random.normal()
sin_phrand = harmonic(440, srate=1000, amp_1=1.0, amp_2=0.5, amp_3=0.33, phase_1=phase_1, phase_2=phase_2, phase_3=phase_3)

plt.plot(input_1) 
plt.xlabel('Frequency 440Hz')
plt.show()
plt.figure()
plt.plot(input_2)
plt.xlabel('Frequency 880Hz')
plt.show()
plt.figure()
plt.plot(input_3)
plt.xlabel('Frequency 1320Hz')
plt.show()
plt.plot(sin_ph0)
plt.xlabel('Harmonic Frequency Phase Zero')
plt.show()
plt.figure()
plt.plot(sin_phrand)
plt.xlabel('Harmonic Frequency Phase Random')
plt.show()

#sin_test = build_sin(220, 1)
write('wave_1-3.wav', 1000, np.array(sin_ph0))
write('wave_1-3rand.wav', 1000, np.array(sin_phrand))

ipd.Audio(sin_ph0, rate=1000)



# In[9]:

#Q1.4

def noiseWave(freq, snr, time=1):
    sin = build_sin(440, duration=time)
    sin_rms = rmsAmplitude(sin)
    #print'rms of 440Hz sine wave: ',sin_rms
    
    noise_rms = (math.pow(10,snr/20))/sin_rms
    #print 'rms of noise wave: ',noise_rms
    #print
    
    noise = np.random.normal(0, noise_rms, len(sin))

    return [s1 + s2 for s1, s2 in zip(sin, noise)]

noisy = noiseWave(440, 20, time=2)
write('wave_snr100', 44100, np.array(noisy))
ipd.Audio(noisy, rate=44100)


# In[10]:

#Q1.5 Inner Product Estimation of Amplitude 

def innerProduct(sin_target, freq):
    sin_probe = build_sin(freq, 1, amp=1.0)
    return dotAmplitude(sin_target, sin_probe)

sin = build_sin(440, 1, amp=3.0)
innerProduct(sin, 440)

#The inner product of a sinusoid and a correspponding probe returns the amplitude
#of the sinusoid.


# In[11]:

#Q1.6 Comparing amplitude estimation functions on a 440Hz sinusoid with 
#varrying levels of noise (SNR = 1, 2, 5, 10, 20)

def estimateAmplitudes(sin):
    print 'Peak: ', peakAmplitude(sin)
    print 'RMS: ', rmsAmplitude(sin)
    print 'Dot-product: ', innerProduct(sin, 440)
    print
    
def displayAmplitudes(snr, peak, rms, dot):
    for x in range(0, len(snr)):
        print 'SNR: ', snr[x]
        print 'Peak: ', peak[x]
        print 'RMS: ', rms[x]
        print 'Dot-product: ', dot[x]
        print
        
noisy_1 = noiseWave(440, 1)
noisy_2 = noiseWave(440, 2)
noisy_5 = noiseWave(440, 5)
noisy_10 = noiseWave(440, 10)
noisy_20 = noiseWave(440, 20)

snr_list = [1, 2, 5, 10, 20]
peak_list = [peakAmplitude(noisy_1), peakAmplitude(noisy_2), 
            peakAmplitude(noisy_5), peakAmplitude(noisy_10), 
            peakAmplitude(noisy_20)]
rms_list = [rmsAmplitude(noisy_1), rmsAmplitude(noisy_2), 
           rmsAmplitude(noisy_5), rmsAmplitude(noisy_10),
           rmsAmplitude(noisy_20)]
dot_list = [innerProduct(noisy_1, 440), innerProduct(noisy_2, 440),
           innerProduct(noisy_5, 440), innerProduct(noisy_10, 440),
           innerProduct(noisy_20, 440)]

#displayAmplitudes(snr_list, peak_list, rms_list, dot_list)

plt.plot(snr_list, peak_list, 'p')
plt.plot(snr_list, rms_list, 'p')
plt.plot(snr_list, dot_list, 'p')
plt.xlabel('SNR')
plt.ylabel('Estimate of Amplitude')
plt.show()

#Dot product is the most robust estimation. Although peak  provides a good estimate without any noise, 
#it loses accuracy even with a small amount of noise. RMS maintains accuracy with 
#low amounts of noise, but similarly falls off as the amount of noise increases. 
#Dot product is slightly less accurate than the other two on sinusoids with no noise, 
#but maintains the same level of accuracy regardless of the amount of noise present.


# In[12]:

#Q1.7

def iterativeProbing(sin, freq):
    sin_probe1 = build_sin(freq, 1, amp=1.0)
    sin_probe2 = build_sin(2*freq, 1, amp=1.0)
    sin_probe3 = build_sin(3*freq, 1, amp=1.0)
    
    sin_probe0 = build_sin(523, 1, amp=1.0) #probe for frequency not present in the mixture
    
    amp_1 = dotAmplitude(sin, sin_probe1)
    amp_2 = dotAmplitude(sin, sin_probe2)
    amp_3 = dotAmplitude(sin, sin_probe3)
    
    amp_0 = dotAmplitude(sin, sin_probe0)
    
    print amp_1, amp_2, amp_3, amp_0

sin_harm = harmonic(440, srate=44100, amp_1=1.0, amp_2=0.5, amp_3=0.33)

iterativeProbing(sin_harm, 440)

#The inner products return the amplitudes of the 3 harmonic waves. For a mixture 
#of 4 sinusoids with noise, as long as the distinct frequencies are known we can 
#create a probing sinusoid for each frequency and compute the inner product to 
#identify the amplitudes of the sinusoids in the mixture.


# In[13]:

#Q1.8 --INCOMPLETE--

sin_harm = harmonic(440, srate=44100, amp_1=1.0, amp_2=0.5, amp_3=0.33)

iterativeProbing(sin_harm, 440)


def iterativePhase(sin, freq):
    probe_phase = 0
    t = np.arange(linspace(0, 2*np.pi, len(sin)))
    for i in t:
        sin_probe1 = build_sin(freq, 1, phase=sin[i], amp=1.0)
    
    amp_1 = dotAmplitude(sin, sin_probe1)


# In[8]:

#Q.2.1

def wav_write(file_name,sample_rate,signal):
    write(file_name,sample_rate,np.array(signal))
    
def wav_read(file_name):
    return read(file_name)

sin = build_sin(440, 1, amp=4)
wav_write('wav_test.wav',44100,np.array(sin))
test_rate=wav_read('qbhexamples.wav')

print(test_rate)


# In[15]:

#Q2.2 
#Referenced from George Tzanetakis 

#data = build_sin(10, 1, sample_rate=10)
data = harmonic(220, time_to_plot=1, srate=1000)

fft_size = 1000
complex_spectrum = np.fft.fft(data[0:fft_size])
magnitude_spectrum = np.abs(complex_spectrum)
phase_spectrum = np.angle(complex_spectrum)
#print(complex_spectrum)
#print(magnitude_spectrum)
plt.plot(magnitude_spectrum)
plt.show()
#plt.figure()
#plt.plot(phase_spectrum)
#half_magnitude_spectrum = magnitude_spectrum[0: int(len(magnitude_spectrum)/2)]

#plt.figure()
#plt.plot(half_magnitude_spectrum)

#magnitude_spectrum = np.zeros(len(magnitude_spectrum))
#fft_bin = 100
#magnitude_spectrum[fft_bin] = 1 
#magnitude_spectrum[len(magnitude_spectrum)-fft_bin] = 1
#real_spectrum = magnitude_spectrum * np.cos(phase_spectrum)
#imag_spectrum = magnitude_spectrum * np.sin(phase_spectrum)

#back_to_time_domain = np.fft.ifft(real_spectrum + 1j * imag_spectrum)
#plt.figure()
#plt.plot(np.real(back_to_time_domain[1:1000]))
#plt.plot(data[0:fft_size])
#plt.show()


# In[ ]:




# In[ ]:




# In[16]:

#Q2.3 - Implementing DFT

#test_sin = build_sin(10, 1, sample_rate=10)
test_sin = harmonic(220, time_to_plot=1, srate=1000)
#print test_sin

def dft(sin):
    N = len(sin)
    freq_bin = []
    bin_val = []
    for k in range(N):
        r =0
        i=0
        for n in range(N):
            r = r + sin[n]*(np.cos(-(2*np.pi*n*k/N)))        #the real part
            i = i + sin[n]*(np.sin(-(2*np.pi*n*k/N)))       #the imaginary part
 
        freq_bin.append(np.sqrt(r*r + i*i))
    return freq_bin
        
magnitude_spectrum = dft(test_sin)

#print(magnitude_spectrum)
plt.plot(magnitude_spectrum)
plt.show()

#half_magnitude_spectrum = magnitude_spectrum[0: int(len(magnitude_spectrum)/2)]

#plt.figure()
#plt.plot(half_magnitude_spectrum)
#plt.show()


# In[56]:

#Q2.5

#The magnitude spectrum shows a spike at the bin corresponding to the freequency of the input signal.
#Increasing the amplitude of the frequency causes the magnitude at that frequency bin to increase
#accordingly (ie changing amplitude from 1 to 4 will result in 4x the original magnitude). Changing
#the phase has no effect on the magnitude spectrum.

test_sin = build_sin(10, 1, sample_rate=1000, amp=4)

magnitude_spectrum = dft(test_sin)


#print(magnitude_spectrum)
plt.plot(magnitude_spectrum)
plt.show()


# In[61]:

sin = build_sin(10, 1, sample_rate=1000)
cos = build_cos(10, 1, sample_rate=1000)

product = np.multiply(sin, cos)

plt.plot(sin)
plt.figure()
plt.plot(cos)
plt.figure()
plt.plot(product)
plt.show()


# In[ ]:



