import os
import struct
import wave
from os import listdir
from os.path import isfile, join

import matplotlib.mlab
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile

import librosa
from pydub import AudioSegment
from pydub.utils import make_chunks

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
def processAudio(bpm,samplingRate,mypath):
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	classes = len(onlyfiles)

	dataList = []
	labelList = []
	for ix,audioFile in enumerate(onlyfiles):
			audData = scipy.io.wavfile.read(mypath+audioFile)
			seconds = audData[1][:,1].shape[0]/samplingRate
			samples = (seconds/60) * bpm
			audData = np.reshape(audData[1][:,1][0:samples*((seconds*samplingRate)/samples)],[samples,(seconds*samplingRate)/samples])
			for data in audData:
					dataList.append(data)
			labelList.append(np.ones([samples,1])*ix)

	Ys = np.concatenate(labelList)

	specX = np.zeros([len(dataList),1024])
	xindex = 0
	for x in dataList:
			work = matplotlib.mlab.specgram(x)[0]
			worka = work[0:60,:]
			worka = scipy.misc.imresize(worka,[32,32])
			worka = np.reshape(worka,[1,1024])
			specX[xindex,:] = worka
			xindex +=1

	split1 = specX.shape[0] - specX.shape[0]/20
	split2 = (specX.shape[0] - split1) / 2

	formatToUse = specX
	Data = np.concatenate((formatToUse,Ys),axis=1)
	DataShuffled = np.random.permutation(Data)
	newX,newY = np.hsplit(DataShuffled,[-1])
	trainX,otherX = np.split(newX,[split1])
	trainYa,otherY = np.split(newY,[split1])
	valX, testX = np.split(otherX,[split2])
	valYa,testYa = np.split(otherY,[split2])
	trainY = oneHotIt(trainYa)
	testY = oneHotIt(testYa)
	valY = oneHotIt(valYa)
	return classes,trainX,trainYa,valX,valY,testX,testY
'''
bpm = 240
samplingRate = 44100
mypath = r"C:/Users/user/Desktop/test/animal/cat/cutend/"
classes,trainX,trainYa,valX,valY,testX,testY = processAudio(bpm,samplingRate,mypath)
'''

path = r"C:/Users/User/Google 雲端硬碟/筆電桌面/animalVoice/test/animal/cat/OneChannel/"
filesNameArray = os.listdir(path)
filesNameArray = [path + "\\" + f for f in filesNameArray if f.endswith('.wav')]

for i in range(len(filesNameArray)):
	fileName = filesNameArray[i]
	sr = 8000
	y, s = librosa.load(fileName, sr=sr) # Downsample origin to sr = you want sample rate
	librosa.output.write_wav(fileName.replace('.wav', '') + "sr16000.wav", y, sr, norm=False)

print('!!!complete!!!')

