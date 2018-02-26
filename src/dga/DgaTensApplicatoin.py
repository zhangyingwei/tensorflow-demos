#!/usr/bin/env python  
# encoding: utf-8  

"""
@version: v1.0 
@author: zhangyw
@site: http://blog.zhangyingwei.com
@software: PyCharm 
@file: DgaTensApplicatoin.py 
@time: 2018/2/26 14:55 
"""
import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

# tf.device('/gpu:1')

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

dgas = [formate_domain(dga) for dga in dgas]
domains = [formate_domain(domain) for domain in domains]

print(dgas[:10])
print(labels[:10])

print(domains[:10])
print(wlabels[:10])

data = np.concatenate((dgas,domains),axis=0)
targets = np.append(labels,wlabels)

print(len(data))
print(len(targets))
print(data[1073920:1074920])
print(targets[1073920:1074920])

train_x,test_x,train_y,test_y = train_test_split(data,targets,test_size=0.3)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=3)]
classifier = tf.contrib.learn.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10,20,10],
    n_classes=2,
    model_dir="tmp/dga_model"
)

def get_train_input():
    x = tf.constant(train_x,np.float32)
    y = tf.constant(train_y,np.float32)
    return x,y

def get_test_input():
    x = tf.constant(test_x,np.float32)
    y = tf.constant(test_y,np.float32)
    return x,y

# classifier.fit(input_fn=get_train_input,steps=2000)
for i in range(2):
    classifier.fit(input_fn=get_train_input,steps=2)
    accuracy_score = classifier.evaluate(input_fn=get_test_input, steps=1)["accuracy"]
    print("nTest Accuracy: {0:f}n".format(accuracy_score))

# def get_test_set():
#     return np.array([
#         formate_domain("baidu.com"),
#         formate_domain("zhangyingwei.com"),
#         formate_domain("sghuhuddbxqcftm.ga"),
#         formate_domain("fnxqfobcjwpmlhxobmq.mn")
#     ])

result = classifier.predict(input_fn=get_test_input)
result = list(result)
for index,y in enumerate(result):
    print("res is:"+y+" and acc is:"+test_y[index])

