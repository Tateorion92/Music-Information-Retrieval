from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter

with WavReader('qbhexamples.wav') as reader:
    with WavWriter('qbh_half.wav', reader.channels, reader.samplerate) as writer:
        tsm = phasevocoder(reader.channels, speed=0.5)
        tsm.run(reader, writer)
