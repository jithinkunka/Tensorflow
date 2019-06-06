# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 11:08:36 2019

@author: jkunka
"""

import tensorflow as tf

session = tf.Session()

hello = tf.constant("Hello from tensorflow")

print(session.run(hello))

a = tf.constant(20)
b = tf.constant(10)

print("a+b = {0}".format(session.run(a+b)))
