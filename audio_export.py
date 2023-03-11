# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 07:54:34 2022

@author: shubh
"""
from pydub import AudioSegment

files_path = 'E:/mr_in_female'
file_name = '/mrt_03349_00834370656'

startMin = 0
startSec = 1

endMin = 0
endSec = 6

# Time to miliseconds
startTime = startMin*60*1000+startSec*1000
endTime = endMin*60*1000+endSec*1000

# Opening file and extracting segment
song = AudioSegment.from_wav( files_path+file_name+'.wav' )
extract = song[startTime:endTime]

# Saving
# extract.export( files_path+file_name+'_extract.mp3', format="mp3")
extract.export( 'extract.wav', format="wav")