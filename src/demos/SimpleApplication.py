#!/usr/bin/env python  
# encoding: utf-8  

"""
@version: v1.0 
@author: zhangyw
@site: http://blog.zhangyingwei.com
@software: PyCharm 
@file: simpleApplication.py 
@time: 2018/2/23 16:25 
"""

import tensorflow as tf

print("常量")
a,b,c,d = tf.constant(1),tf.constant(2),tf.constant(3),tf.constant(4)
# +
add = tf.add(a,b)
# *
mul = tf.multiply(add,c)
# /
sub = tf.subtract(mul,d)
with tf.Session() as session:
    writer = tf.summary.FileWriter("./graph",session.graph)
    res = session.run(sub)
    print(res)
    writer.close()

print("var")
e = tf.Variable(1)
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    print(session.run(e))