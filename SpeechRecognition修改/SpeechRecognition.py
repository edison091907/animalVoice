# 導入函式庫
import os
import time as time
from os import listdir
from os.path import isdir, isfile, join

import keras
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras import initializers
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential, load_model
from keras.utils import to_categorical

from preprocess import *

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count={'GPU':0, 'CPU':8}))
keras.backend.set_session(sess)

# 載入 data 資料夾的訓練資料，並自動分為『訓練組』及『測試組』
X_train, X_test, y_train, y_test = get_train_test()
X_train = X_train.reshape(X_train.shape[0], 20, 11, 1)
X_test = X_test.reshape(X_test.shape[0], 20, 11, 1)

# 類別變數轉為one-hot encoding
y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)
print("X_train.shape=", X_train.shape)

class EarlyStoppingThreshold(keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', value=0):
        super(EarlyStoppingThreshold, self).__init__()
        self.monitor=monitor
        self.value=value
    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            return
        if(current < self.value):
            self.stopped_epoch = epoch
            self.model.stop_training = True

class RestoreBestWeightsFinal(keras.callbacks.Callback):
    def __init__(self,
                 min_delta=0,
                 mode='auto',
                 baseline=None):
        super(RestoreBestWeightsFinal, self).__init__()
        self.min_delta = min_delta
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        
    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)

    def on_epoch_end(self, epoch, logs=None):
        val_current = logs.get('val_loss')
        if val_current is None:
            return

        if self.monitor_op(val_current - self.min_delta, self.best):
            self.best = val_current
            self.best_weights = self.model.get_weights()

batch_size = 10
epochs = 50
callbacks = []
#callbacks.append(keras.callbacks.EarlyStopping(monitor='val_acc', patience=3))
callbacks.append(RestoreBestWeightsFinal())
#callbacks.append(EarlyStoppingThreshold(monitor='loss', value=0.05))


activation = 'relu'
activationEnd = 'softmax'
#activation = 'elu'
#activation = 'hard_sigmoid'
#activation = 'linear'
#activation = 'relu'
#activation = 'selu'
#activation = 'sigmoid'
#activation = 'softmax'
#activation = 'softplus'
#activation = 'softsign'
#activation = 'tanh'

loss = keras.losses.categorical_crossentropy
#optimizer = keras.optimizers.Adadelta()
#optimizer = keras.optimizers.SGD()
#optimizer = keras.optimizers.RMSprop()
#optimizer = keras.optimizers.Adagrad()
optimizer = keras.optimizers.Adam()
#optimizer = keras.optimizers.Adamax()
#optimizer = keras.optimizers.Nadam()

#kernel_initializer = initializers.normal(mean=0, stddev=0.5, seed=None)
#kernel_initializer = initializers.uniform(minval=-0.05, maxval=0.05, seed=None)
#kernel_initializer = initializers.truncated_normal(mean=0.0, stddev=0.05, seed=None)
#kernel_initializer = initializers.orthogonal(gain=1, seed=None)
#kernel_initializer = initializers.identity(gain=1)
#kernel_initializer = initializers.he_uniform()
#kernel_initializer = initializers.glorot_uniform()

#bias_initializer = initializers.normal(mean=0, stddev=0.5, seed=None)
#bias_initializer = initializers.uniform(minval=-0.05, maxval=0.05, seed=None)
#bias_initializer = initializers.truncated_normal(mean=0.0, stddev=0.05, seed=None)
#bias_initializer = initializers.orthogonal(gain=1, seed=None)
#bias_initializer = initializers.identity(gain=1)
#bias_initializer = initializers.he_uniform()
#bias_initializer = initializers.glorot_uniform()
#bias_initializer = initializers.zeros()

# 建立簡單的線性執行的模型
model = Sequential()
# 建立卷積層，filter=32,即 output size, Kernal Size: 2x2, activation function激活函式 採用 relu
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation=activation, input_shape=(20, 11, 1)))
# 建立池化層，池化大小=2x2，取最大值
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout層隨機斷開輸入神經元，用於防止過度擬合，斷開比例:0.25
#model.add(Dropout(0.25))
# Flatten層把多維的輸入一維化，常用在從卷積層到全連接層的過渡。
model.add(Flatten())
# 全連接層: 128個output
unit = 256
model.add(Dense(unit, activation=activation))
model.add(Dropout(0.5))
# Add output layer
model.add(Dense(2, activation=activationEnd))
# 編譯: 選擇損失函數、優化方法及成效衡量方式
model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])
model.summary()

# 進行訓練, 訓練過程會存在 train_history 變數中
t = time.time()
train_history=model.fit(X_train, y_train_hot, 
						batch_size=batch_size, 
						epochs=epochs, verbose=1, 
						callbacks = callbacks,
						validation_data=(X_test, y_test_hot))
elapsed = time.time() - t

history = pd.DataFrame(train_history.history)

# 模型存檔
model.save('ASR.h5')  # creates a HDF5 file 'model.h5'

#print("labels=", get_labels())

# 預測(prediction)
mypath = "./SpeechRecognition修改/test/pig/"
files = listdir(mypath)
list1 = [0,0]
list2 = [0,0]
print("\nTest 0...")
for f in files:
	# 產生檔案的絕對路徑
	fullpath = join(mypath, f)
	#print("測試檔案：", fullpath)
	mfcc = wav2mfcc(fullpath)
	mfcc_reshaped = mfcc.reshape(1, 20, 11, 1)
	#print("predict=", np.argmax(model.predict(mfcc_reshaped)))
	if np.argmax(model.predict(mfcc_reshaped))==0:
		list1[0]= list1[0]+1
	elif np.argmax(model.predict(mfcc_reshaped))==1:
		list1[1]= list1[1]+1

mypath = "./SpeechRecognition修改/test/sheep/"
files = listdir(mypath)
list2 = [0,0]

print("Test cat...")
for f in files:
	# 產生檔案的絕對路徑
	fullpath = join(mypath, f)
	#print("測試檔案：", fullpath)
	mfcc = wav2mfcc(fullpath)
	mfcc_reshaped = mfcc.reshape(1, 20, 11, 1)
	#print("predict=", np.argmax(model.predict(mfcc_reshaped)))
	if np.argmax(model.predict(mfcc_reshaped))==1:
		list2[0]= list2[0]+1
	elif np.argmax(model.predict(mfcc_reshaped))==0:
		list2[1]= list2[1]+1

print("Finish\n")		

print("\n0")
print("----------------------------------------")
print("0 Level All Data    :", list1[0]+list1[1], "\t個音檔")
print("Classification For 0:", list1[0], "\t個音檔")
print("Classification For cat:", list1[1], "\t個音檔")
print("----------------------------------------")
print("CORRECT:", (list1[0]/(list1[0]+list1[1])), "  ", "ERROR:", (list1[1]/(list1[0]+list1[1])))

print("\n\ncat")
print("----------------------------------------")
print("cat All Data    :", list2[0]+list2[1], "\t個音檔")
print("Classification For cat:", list2[0], "\t個音檔")
print("Classification For 0:", list2[1], "\t個音檔")
print("----------------------------------------")
print("CORRECT:", (list2[0]/(list2[0]+list2[1])), "  ", "ERROR:", (list2[1]/(list2[0]+list2[1])))

print(' elapsed : ' + str(elapsed))
print("\n\nTHE END")

f = plt.figure(figsize=(10,10))
plt.plot(history['loss'])
plt.grid()
plt.plot(history['val_loss'])
plt.title('Model loss= ' + str(history['loss'].values[-1]) + ',val_loss = ' + str(history['val_loss'].values[-1]))
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.ylim(0,np.array(history['loss']).mean()*2)
plt.xlim(left=0)
plt.legend(['Train', 'valid'], loc='upper left')
plt.show()