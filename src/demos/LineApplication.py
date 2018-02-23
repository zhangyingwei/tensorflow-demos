#!/usr/local/soft/python-env/py3-env/bin/python
# encoding: utf-8

"""
@version: v1.0
@author: zhangyw
@site: http://blog.zhangyingwei.com
@software: PyCharm
@file: LineApplication.py
@time: 2018/2/23 17:22
"""

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

print(tf.VERSION)

train_X = numpy.asarray([1.1, 1.8, 3.2, 4.7, 5.9, 6.7])
train_Y = numpy.asarray([1.2, 2.1, 3.1, 4.6, 5.5, 6.9])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(-1., name="weight")
b = tf.Variable(-1., name="bias")

activation = tf.add(tf.multiply(X, W), b)

learning_rate = 0.01

cost = tf.reduce_sum(tf.pow(activation - Y, 2))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

training_epochs = 2000
display_step = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=",
                  "{:.9f}".format(sess.run(cost, feed_dict={X: train_X, Y: train_Y})), "W=", sess.run(W), "b=",
                  sess.run(b))
    print("Optimization Finished!")
    print("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}), "W=", sess.run(W), "b=", sess.run(b))

    writer = tf.summary.FileWriter('./graphs', sess.graph)

    plt.scatter(train_X, train_Y, color='red', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), color='blue', label='Fitted line')
    plt.show()
writer.close()