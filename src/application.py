#!/usr/bin/env python  
# encoding: utf-8  

"""
@version: v1.0 
@author: zhangyw
@site: http://blog.zhangyingwei.com
@software: PyCharm 
@file: application.py 
@time: 2018/1/12 13:51
"""

import tensorflow as tf

if __name__ == '__main__':
    print("hello tensorflow")
    hello = tf.constant("hello world tensorflow")
    session = tf.Session()
    result = session.run(hello)
    print(result)
    session.close()
    pass