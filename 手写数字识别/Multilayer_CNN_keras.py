# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:57:14 2019

@author: mayong
"""

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D
from keras.layers import MaxPool2D,Flatten,Dropout,ZeroPadding2D,BatchNormalization
from keras.utils import np_utils
from keras.models import save_model,load_model
from keras.models import Model


#tf.disable_v2_behavior()
#tf.enable_eager_execution()

# 加载数据
df = pd.read_csv('D:/数据挖掘项目练习/手写数字识别/data/train.csv')

# 打乱数据顺序
data = df.as_matrix()
df = None
np.random.shuffle(data)

#分割特征并改变格式
x_train = data[:,1:]
x_train = x_train.reshape(data.shape[0],28,28,1).astype('float32')
x_train = x_train/255.0

print(x_train.shape)

#分割label并one-hot
y_train = np_utils.to_categorical(data[:,0],10).astype('float32')
print(y_train.shape)

#batch_size
batch_size = 64

# 卷积滤镜
n_filters = 32

#池化核
pool_size = (2,2)

cnn_net = Sequential()

#第一层:卷积池化
cnn_net.add(Conv2D(32,kernel_size=(3,3),strides=(1,1),input_shape=(28,28,1)))
cnn_net.add(Activation('relu'))
cnn_net.add(BatchNormalization(epsilon=1e-6,axis=1))
cnn_net.add(MaxPool2D(pool_size=pool_size))

# 第二层：卷积池化
cnn_net.add(ZeroPadding2D((1,1)))
cnn_net.add(Conv2D(64,kernel_size=(2,2)))
cnn_net.add(Activation('relu'))
cnn_net.add(BatchNormalization(epsilon=1e-6,axis=1))
cnn_net.add(MaxPool2D(pool_size=pool_size))

# 第三层：卷积池化
cnn_net.add(ZeroPadding2D((1,1)))
cnn_net.add(Conv2D(64,kernel_size=(2,2)))
cnn_net.add(Activation('relu'))
cnn_net.add(BatchNormalization(epsilon=1e-6,axis=1))
cnn_net.add(MaxPool2D(pool_size=pool_size))

# 第四层：全连接层
cnn_net.add(Dropout(0.25))
cnn_net.add(Flatten())

cnn_net.add(Dense(3168))
cnn_net.add(Activation('relu'))

cnn_net.add(Dense(10))
cnn_net.add(Activation('softmax'))

#查看网络结构，可视化模型
cnn_net.summary()

#from keras.utils.vis_utils import model_to_dot
#from IPython.display import SVG

# 可视化模型
#SVG(model_to_dot(cnn_net).create(prog='dot',format='svg'))

# 训练模型、保存和载入模型
#cnn_net = load_model('cnn_net_model.h5')
cnn_net.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
cnn_net.fit(x_train,y_train,batch_size = batch_size,epochs=50,verbose=1,validation_split=0.2)
save_model(cnn_net,'cnn_net_model.h5')

#使用训练好并且保存好的模型进行预测
df_t = pd.read_csv('D:/数据挖掘项目练习/手写数字识别/data/test.csv')
data_t = df_t.as_matrix()
df_t = None
x_test = data[:,1:]
x_test = x_test.reshape(data_t.shape[0],28,28,1).astype('float32')
x_test = x_test/255.0
yPred = cnn_net.predict_classes(x_test,batch_size=32,verbose=1)
np.savetxt('Digit3.csv',np.c_[range(1,len(yPred)+1),yPred],delimiter=',',header='ImageID,Label',comments='',fmt='%d')