#!/usr/bin/env python  
# encoding: utf-8  

"""
@version: v1.0 
@author: zhangyw
@site: http://blog.zhangyingwei.com
@software: PyCharm 
@file: DgaApplication.py 
@time: 2018/2/24 14:04 
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split

def load_dga_dataset():
    with open("../../resources/dga/dga.txt") as dgaFile:
        lines = dgaFile.readlines()
        lines = [line.split("\t")[1] for line in lines]
    return lines,np.zeros(len(lines))

def load_domains_dataset():
    with open("../../resources/dga/w_domain.txt") as domainFile:
        lines = domainFile.readlines()
    return lines, np.ones(len(lines))

def formate_domain(domain):
    domain = domain.strip()
    return [
        len(domain)/10,
        len([cha for cha in domain if cha in "aeiou"])/5,
        len(set(domain))/len(domain)
    ]

dgas,labels = load_dga_dataset()
domains,wlabels = load_domains_dataset()

dgas = [formate_domain(domain) for domain in dgas]
domains = [formate_domain(domain) for domain in domains]


pixels = len(dgas[0])

def build_model():
    model = Sequential()
    model.add(Dense(pixels, input_dim=pixels, kernel_initializer="normal", activation="relu"))
    model.add(Dense(2, kernel_initializer="normal", activation="softmax"))
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
    return model

model = build_model()

dgas = np.array(dgas)
domains = np.array(domains)

x_train = np.concatenate((dgas,domains)) #dgas+domains
y_train = np.append(labels,wlabels)

train_x,test_x,train_y,test_y=train_test_split(x_train,y_train,test_size=0.3)

print(train_y[:10])
print(test_y[:10])

train_y = np_utils.to_categorical(train_y)
test_y = np_utils.to_categorical(test_y)

for a in train_y:
    if(a[0] == 0):
        print(a)

# print(test_y)

# print(len(test_y))
# print(len(test_x))

model.fit(train_x,train_y,epochs=10,batch_size=1000,verbose=2)
#
scores = model.evaluate(test_x,test_y,verbose=0)

print(scores)

print("baseline error: %.2f%%" % (100-scores[1]*100))

# model.save("./dga-keras.model")

print(model.predict_classes(dgas))

print(model.predict_classes(domains))
