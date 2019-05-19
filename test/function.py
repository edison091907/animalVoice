import os
import struct
import wave

import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks
from scipy.io import wavfile

path = r"C:/Users/user/Desktop/test/animal/cat/cutend/"
filesNameArray = os.listdir(path)
filesNameArray = [path + "\\" + f for f in filesNameArray if f.endswith('.wav')]

def cutSoundForMs (filesNameArray):
    for i in range(len(filesNameArray)):
        fileName = filesNameArray[i]
        myaudio = AudioSegment.from_file(fileName , "wav") 
        chunk_length_ms = 1000 # seg ms
        chunks = make_chunks(myaudio, chunk_length_ms) #将文件切割成1秒每块

        for i, chunk in enumerate(chunks):
            chunk_name = fileName.replace('.wav', '') + "_cutNumber{}s.wav".format(i)
            print ("exporting", chunk_name)
            chunk.export(chunk_name, format="wav")

def trans_mp3_to_wav (filepath):
    song = AudioSegment.from_mp3(filepath)
    song.export("C:/Users/user/Desktop/test/animal/sheep/wav/sheep10.wav", format="wav")
#path = "C:/Users/user/Desktop/test/animal/sheep/sheep10.mp3"
#trans_mp3_to_wav(path)

def splitChannel (filesNameArray):#有問題待研究
    for i in range(len(filesNameArray)):
        fileName = filesNameArray[i]
        samepleRate, musicData = wavfile.read(fileName)
        left = []
        right = []

        for item in musicData:
            left.append(item[0])
            right.append(item[1])

        wavfile.write(fileName.replace('.wav', '') + "_left.wav", samepleRate, np.array(left))
        wavfile.write(fileName.replace('.wav', '') + "_right.wav", samepleRate, np.array(right))

def removeOneChannel (filesNameArray):
    for i in range(len(filesNameArray)):
        fileName = filesNameArray[i]
        f = wave.open(fileName,'rb')
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        strData = f.readframes(nframes)#读取音频，字符串格式
        waveData = np.fromstring(strData,dtype=np.int16)#将字符串转化为int
        waveData = waveData*1.0/(max(abs(waveData)))#wave幅值归一化
        f.close()
        #wav文件写入
        outData = waveData#待写入wav的数据，这里仍然取waveData数据
        outwave = wave.open(fileName.replace('.wav', '') + "_oneChannel.wav", 'wb')#定义存储路径以及文件名
        nchannels = 1
        sampwidth = 2
        fs = 16000
        data_size = len(outData)
        framerate = framerate
        nframes = data_size
        comptype = "NONE"
        compname = "not compressed"
        outwave.setparams((nchannels, sampwidth, framerate, nframes, comptype, compname))
        
        for v in outData:
                outwave.writeframes(struct.pack('h', int(v * 64000 / 2)))#outData:16位，-32767~32767，注意不要溢出
        outwave.close()

#####################################################################################################################
#removeOneChannel (filesNameArray)