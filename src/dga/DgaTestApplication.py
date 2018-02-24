#!/usr/bin/env python  
# encoding: utf-8  

"""
@version: v1.0 
@author: zhangyw
@site: http://blog.zhangyingwei.com
@software: PyCharm 
@file: DgaTestApplication.py 
@time: 2018/2/24 15:08 
"""

from keras.models import load_model
import numpy as np

def load_dga_dataset():
    with open("../../resources/dga/dga.txt") as dgaFile:
        lines = dgaFile.readlines()
        lines = [line.split("\t")[1] for line in lines]
    return lines

def load_domains_dataset():
    with open("../../resources/dga/w_domain.txt") as domainFile:
        lines = domainFile.readlines()
    return lines

def formate_domain(domain):
    return [
        len(domain)/10,
        len([cha for cha in domain if cha in "aeiou"])/5,
        len(set(domain))/len(domain)
    ]

model = load_model("./dga-keras.model")

def predict(domains):
    return model.predict_classes(np.array([formate_domain(domain) for domain in domains]),verbose=1)

if __name__ == '__main__':
    # print(predict([
    #     "zhangyingwei.com",
    #     "google.com",
    #     "jmewmdum.net",
    #     "bank.pingan.com",
    #     "fnxqfobcjwpmlhxobmq.mn"
    # ]))
    print(predict(load_domains_dataset()))