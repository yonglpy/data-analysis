# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 13:35:56 2019

@author: mayong
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

def add_layer(inputs,in_size,out_size,activation_funciton=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs,Weights) + biases
        if activation_funciton is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_funciton(Wx_plus_b)
        return outputs
x_data = np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data = np.square(x_data)-0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name='x_in')
    ys = tf.placeholder(tf.float32,[None,1],name='y_in')

with tf.variable_scope('Net'):
    L1 = add_layer(xs,1,10,activation_funciton=tf.nn.relu)
    prediction = add_layer(L1,10,1,activation_funciton=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
writer = tf.summary.FileWriter('logs/',sess.graph)
sess.run(init)
sess.close()

sess = tf.Session()
sess.run(init)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i % 50 == 0:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        prediction_values = sess.run(prediction,feed_dict={xs:x_data})
        lines = ax.plot(x_data,prediction_values,'r-',lw=5)
        plt.pause(1)