
from Tkinter import *
from PIL import ImageTk,Image
from tkFileDialog import askopenfilename
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
import cmath
from scipy.io.wavfile import write, read
import pygame


srate=0
Ha=0
Hs=0
grain_len=0
data=[]
grains=[]
root = Tk()
#from tkinter import filedialog

#####################################################################
def wav_write(file_name,sample_rate,signal):
    write(file_name,sample_rate,np.array(signal))
    
def wav_read(file_name):
    return read(file_name)

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

def hann_grains(grains):
    hgrains=[]
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
#####################################################################

def open_masker():
    global audio_file_name
    audio_file_name = askopenfilename(filetypes=(("Audio Files", ".wav .ogg"),   ("All Files", "*.*")))
    return
   
def masker_screen():  
    # we will also use the audio_file_name global variable
    global m_screen, audio_file_name  
    global srate,data, Ha,grain_len,Hs
    if audio_file_name: # play sound if just not an empty string
	
	read_wavefile=read(audio_file_name)
	srate = read_wavefile[0]
	#print srate
	data = read_wavefile[1]
	Ha = srate/20
	 
	grain_len = srate/10
	Hs = grain_len/2 
        print("BEFORE ADJUST", data)
	#get_speed(0)
	
        noise = pygame.mixer.music.load(audio_file_name)
	#print audio_file_name
        pygame.mixer.music.play(-1, 0)
	
	create_grains()
	return

def stop_music():
	pygame.mixer.music.stop()
def pause_music():
	pygame.mixer.music.pause()

def unpause_music():
	pygame.mixer.music.unpause()
def create_grains():
	
	global Hs,grain_len,grains,data,Ha
	
	grains=[]
	print("HSSSSS",Hs)
	print("HAAAAAA",Ha)
	print("grain_lennnn",grain_len)
	print("HERERE IS DAATTTTAA",data)
	    					
	num_grains = len(data)/Ha-1
	#grain_len = srate/10
	for grain in range(num_grains):
	    if(grain*Ha <= (len(data)-grain_len)):
		grains.append(data[grain*Ha:(grain*Ha)+grain_len])

	##print (grains)
	return

def get_speed(speed):
	print ("YOYOYO",speed)
	speed =int(speed)
	if(speed==0):
		return
	
	global srate,Ha,data,grains


	##array of all the grains
	print ("srate",srate)
	
	#print("grain_len",grain_len)

	##adjust speed according to slider
	if(speed==5):
		Ha = int(srate/20*.5) 
		     
	elif(speed==4):
		Ha = int(srate/20*.6)	
	elif(speed==3):
		Ha = int(srate/20*.7)
	elif(speed==2):
		Ha = int(srate/20*.8)         
	elif(speed==1):
		#print "NOT IN HERRRRRRRR"
		Ha = int(srate/20*2)
		#print ("HAHAHAHAHAHHAHA",Ha)
	
	elif(speed==0): 
		Ha = srate/20   
	elif(speed==-1):
		Ha = int(srate/20*.5)   
	elif(speed==-2): 
		Ha = int(srate/20*1.4)   
	elif(speed==-3): 
		Ha = int(srate/20*1.6)   
	elif(speed==-4): 
		Ha = int(srate/20*1.8)      
	elif(speed==-5):
		Ha = int(srate/20*2.0)   

	#split up original data into grains (aka analysis frames)
	#copy grains back into a single data array using the synthesis hopsize
	print("sizeOF GRAINS:before create", len(grains))
	#print("SIZE OF HGRAINS:BEFORE", len(hgrains))
	create_grains()
	print("sizeOF GRAINS:before creat2e", len(grains))
	#print("SIZE OF HGRAINS:BEF2ORE", len(hgrains))
	hgrains=hann_grains(grains)
	print("sizeOF GRAINS:before cr3eate", len(grains))
	print("SIZE OF HGRAINS:BEFO3RE", len(hgrains))
	#print ("HGRAINS",hgrains)
	print("HS",Hs)
	
	data_synth=synthesize(hgrains,Hs)
	print("sizeOF GRAINS:before cr4eate", len(grains))
	print("SIZE OF HGRAINS:BEFORE4", len(hgrains))
	print len(data_synth)
	data_synth = np.asarray(data_synth,dtype='int16')
	print len(data_synth)
	print "APRES DATA SYBNTH"
	#print("HERE IS DATA SYNTH",data_synth)
	wav_write("temp1.wav",srate,data_synth)
	print len(data_synth)
	stop_music()

	return

def play_again():
	pygame.mixer.music.load('temp1.wav')
	pygame.mixer.music.play(-1,0)

def main():	
	audio_file_name = ''
	pygame.mixer.init() # initializing the mixer

	
	b1 = Button(root, text = 'open file',command = open_masker)
	
	# does not make sense to call pack directly 
	# and stored the result in a variable, which would be None
	b1.pack(anchor=CENTER)


	Button(root, text = 'play', command = masker_screen).pack(anchor = E)
	Button(root, text='MODplay', command=play_again).pack(anchor = E)
	Button(root, text='Pause', command=pause_music).pack(anchor=E)
	Button(root, text='Unpause', command=unpause_music).pack(anchor=E)
	Button(root, text='Stop', command=stop_music).pack(anchor=E)

	w=Scale(root, from_=-5,to=5,tickinterval=1,command = get_speed, background ='white', width= 30, length=300, troughcolor='black', font='Arial', orient=HORIZONTAL)
	w.set(0)
	w.pack()
	############
	


	root.configure(background='white')
	#load logo
	img = ImageTk.PhotoImage(Image.open("/home/jared/Documents/csc475/KeepUp/keepuplogo.png"))
	panel = Label(root,image =img)
	panel.pack(side = "bottom",fill = "both", expand = "yes")




	root.mainloop()
main()

