#!/usr/bin/env python
# encoding: utf-8
"""
@version: v1.0
@author: zhangyw
@site: http://blog.zhangyingwei.com
@software: vscode
@file: NumClassify.py
"""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import numpy
import matplotlib.pyplot as plt

# 设置随机数种子
seed = 7

numpy.random.seed(seed)

(x_train,y_train),(x_test,y_test) = mnist.load_data()


def showImg():
    plt.subplot(221)
    plt.imshow(x_train[0], cmap=plt.get_cmap('gray'))
    plt.subplot(222)
    plt.imshow(x_train[1], cmap=plt.get_cmap('gray'))
    plt.subplot(223)
    plt.imshow(x_train[2], cmap=plt.get_cmap('gray'))
    plt.subplot(224)
    plt.imshow(x_train[3], cmap=plt.get_cmap('gray'))
    # show the plot
    plt.show()

# showImg()

num_pixels = x_train.shape[1]*x_train.shape[2]

x_train = x_train.reshape(x_train.shape[0],num_pixels).astype("float32")
x_test = x_test.reshape(x_test.shape[0],num_pixels).astype("float32")

x_train = x_train/255
x_test = x_test/255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]

print(y_test)
print("shape",x_test.shape)
print("num class ",num_classes)

def baseline_model():
    model = Sequential()
    model.add(Dense(num_pixels,input_dim=num_pixels,kernel_initializer="normal",activation="relu"))
    model.add(Dense(num_classes,kernel_initializer="normal",activation="softmax"))

    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
    return model


model = baseline_model()

model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size=200,verbose=2)

scores = model.evaluate(x_test,y_test,verbose=0)

print(scores)

print("baseline error: %.2f%%" % (100-scores[1]*100))


