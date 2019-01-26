#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import wave
import os
import struct
import matplotlib.pyplot as plt

def np_to_int16_bytes(x):
    x = np.int16(x * 2**(16-1))
    return struct.pack(str(len(x))+'h', *x)


def int16_bytes_to_np(x, num, sw):
    x = np.array(struct.unpack(str(num)+'h', x))
    x = x.astype(np.float) / 2**(16-1)
    return x


class Signal(object):

    def __init__(self, Fs=44100, duration=None, data=[]):
        self._duration = duration
        self._Fs, self._Ts = Fs, 1./Fs
        self._data = data
        if duration:
            self._n = np.arange(0, duration, self._Ts)

    def plot(self):
        if len(self._data):
            plt.plot(self._n, self._data)
            plt.show()

    def estimate_amplitude(self, base):
        amp = (np.dot(self._data, base._data) /
               np.dot(base._data, base._data) )
        return amp

    def magnitude(self, win_size = 1):
        return np.abs(self.dft(win_size = win_size))

    def largest_freq(self, win_size = 1, n = 1, s_factor = 10):
        m = self.magnitude(win_size = win_size)
        neg = np.median(m)
        m = [i if i > neg else 0 for i in m]
        m_smooth = [np.sum(m[i-s_factor:i])/s_factor for i in range(s_factor,len(m))]
        freqs, up = [], (True if m[1] > m[0] else False)
        for bin in range(len(m_smooth) - 1):
            if m_smooth[bin] > m_smooth[bin + 1] and up:
                up = False
                freqs.append((bin, m[bin]))
            elif m_smooth[bin] < m_smooth[bin + 1] and not up:
                up = True
        freqs.sort(key=lambda tup: tup[1], reverse = True)
        return freqs[:n]

    def decibel(self, win_size = 1):
        m = self.magnitude(win_size = win_size)
        max = np.max(m)
        return [20 * np.log10(a / max) for a in m]

    def phase(self, win_size = 1):
        complex = self.dft(win_size = win_size)
        return [np.arctan2(p.imag,p.real) for p in complex]

    def add_signal(self, signal, start_time = 0):
        assert(signal.params[1] == self._Fs)
        if start_time + signal._duration > self._duration:
            self._data = [self.data[i] if i < len(self.data) else 0
                          for i in range(int((start_time + signal._duration)*self._Fs))]
            self._duration = signal._duration + start_time
            self._n = np.arange(0, self._duration, self._Ts)
        start, stop = int(start_time*self._Fs), int((start_time + signal._duration)*self._Fs)
        for i in range(start, stop):
            self._data[i] += signal._data[i - start]

    def multiply_signal(self, signal):
        assert(len(signal) == len(self._data))
        self._data = [self.data[i]*signal[i] for i in range(len(self._data))]


    def dft(self, win_size = 1):
        if len(self._data):
            data = [self._data[i] if i%(self._Fs*self.duration/win_size) == 0 else 0
                    for i in range(len(self._data))]
            data = np.fft.fft(data)/len(data)
            return data[:len(data)/2]

    def wav_write(self, outfile, Nch=1, Sw=2, normalize=True):
        if len(self._data):
            x = self._data
            x = x / max(x) if normalize else x
            dst = wave.open(outfile, 'wb')
            dst.setparams((Nch, Sw, self._Fs, len(x), 'NONE', 'not_compressed'))
            dst.writeframes(np_to_int16_bytes(x))
            dst.close()

    def wav_read(self, in_file):
        assert(os.path.exists(in_file))
        src = wave.open(in_file, 'rb')
        nch, sw, fs, nframes, _, _ = src.getparams()
        self.__init__(Fs=fs, duration=nframes/fs)
        assert(nch == 1), "wav must be 1 ch"
        self._data = int16_bytes_to_np(src.readframes(nframes), nframes, sw)
        src.close()

    @property
    def data(self):
        return self._data

    @property
    def duration(self):
        return self._duration

    @property
    def params(self):
        return self._duration, self._Fs, self._Ts


class Sinusoid(Signal):

    def __init__(self, duration=1, Fs=44100.0, amp=1.0, freq=440.0, phase=0):
        super(self.__class__, self).__init__(duration=duration, Fs=Fs)
        self.A, self.f, self.phi = amp, freq, phase
        self._w = 2 * np.pi * self.f
        self.__make()

    def __make(self):
        self._data = self.A * np.sin(self._w * self._n + self.phi)

    def power(self):
        return self.A**2/2.0

    def add_noise(self, snr):
        sigma2 = self.power()/(10**(snr/10.0))
        noise = np.random.normal(0, np.sqrt(sigma2), len(self._data))
        self._data += noise

    def remove_noise(self):
        self.__make()

    def shift(self, phi):
        self._phi = phi
        self.__make()

s1, s2, s3 = Sinusoid(freq=440), Sinusoid(freq=880), Sinusoid(freq=1320)

s1.add_signal(s2)
s1.add_signal(s3)
s1.wav_write('test.wav')
s1.plot()
