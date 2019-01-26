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

def main():
	i = read('qbhexamples.wav')

	data = i[1]
	srate = i[0]

	write('out.wav', srate, data)

main()
