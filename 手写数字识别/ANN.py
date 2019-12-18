# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

#1、加载数据，把特征与结果标签分开
data = pd.read_csv('./data/train.csv')
images = data.iloc[:,1:].values
labels = data.iloc[:,0].values

#2、对数据特征进行处理
images = images.astype(np.float)
images = np.multiply(images,1.0/255.0)

print('输入数据量：(%g,%g)' %(images.shape))

images_size = images.shape[1]
print('输入数据维度=>{0}'.format(images_size))

images_width = images_height = np.ceil(np.sqrt(images_size)).astype(np.uint8)
print('图片的长=>{0}\n图片的高=>{1}'.format(images_width,images_height))

x = tf.placeholder('float',shape=[None,784])

# 3、对标签进行处理
labels_count = np.unique(labels).shape[0]
y = tf.placeholder('float', shape=[None,10])
print('标签种类=>{0}'.format(labels_count))

#4、进行One-hot编码
def dense_to_one_hot(labels_dense,num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels,num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

labels_flat = dense_to_one_hot(labels,labels_count)
labels_flat = labels_flat.astype(np.uint8)

print('标签数量:({0[0]},{0[1]})'.format(labels_flat.shape))

#5、把数据分为训练数据集和测试数据集，其中训练数据集为42000，测序数据集为18000
VALIDATION_SIZE = 18000
validation_images = images[:VALIDATION_SIZE]
validation_labels = labels_flat[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = labels_flat[VALIDATION_SIZE:]

#5、对训练数据集分批
batch_size = 100
n_batch = len(train_images)/batch_size

#6、创建一个简单的神经网络用来对图片中数字的识别
weights = tf.Variable(tf.zeros([784,10]))
biases = tf.Variable(tf.zeros([10]))
result = tf.matmul(x,weights) + biases
prediction = tf.nn.softmax(result)

#7、创建损失函数，以交叉熵的平均值为衡量标准
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y,logits = prediction))

#8、用梯度下降法优化参数
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#9、初始化变量
init = tf.global_variables_initializer()
#10、计算准确度
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    #初始化
    sess.run(init)
    #循环50轮
    for epoch in range(200):
        for batch in range(int(n_batch)):
            batch_x = train_images[batch*batch_size:(batch+1)*batch_size]
            batch_y = train_labels[batch*batch_size:(batch+1)*batch_size]
            sess.run(train_step,feed_dict={x:batch_x,y:batch_y})
            
        accuracy_n = sess.run(accuracy,feed_dict = {x:validation_images,y:validation_labels})
        print('第'+str(epoch+1) + '轮,准确度:'+str(accuracy_n))
    
