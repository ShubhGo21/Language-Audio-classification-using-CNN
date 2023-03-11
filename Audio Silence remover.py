# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 17:53:26 2022

@author: shubh
"""
import os

from pydub import AudioSegment
from pydub.silence import split_on_silence

for i in range (2717):
    files_path = "F:/Project 2/Trimmed_Audios/New folder/"
    di="F:/Project 2/spectograms/3 sec specto"
    file1_name = os.listdir(files_path)
    o = files_path+file1_name[i]
    print(o)
    
    #file_name = o.split('/')[-1]
    audio_format = "wav"
    
    # Reading and splitting the audio file into chunks
    sound = AudioSegment.from_file(o, format = audio_format) 
    audio_chunks = split_on_silence(sound,min_silence_len = 200,silence_thresh = -30,keep_silence = 200)
    
    # Putting the file back together
    combined = AudioSegment.empty()
    for chunk in audio_chunks:
        combined += chunk
        
    name=file1_name[i].split(".")[0]
    combined.export(di+"\\"+name+".wav", format="wav")