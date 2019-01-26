
# coding: utf-8

# In[11]:

from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter

with WavReader('qbhexamples.wav') as reader:
    print reader.channels, reader.samplerate
    with WavWriter('qbh_half.wav', reader.channels, reader.samplerate) as writer:
        tsm = phasevocoder(reader.channels, speed=0.5)
        tsm.run(reader, writer)
        print "Finished, closing files."
        close(reader)
        close(writer)


# In[ ]:



