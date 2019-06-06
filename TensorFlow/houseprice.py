# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 11:25:10 2019

@author: jkunka
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation

num_houses = 160
np.random.seed(42)

house_size = np.random.randint(low=1000,high=3500,size=num_houses)

np.random.seed(42)
house_price = house_size*100.0 + np.random.randint(low=20000,high=70000,size=num_houses)

plt.plot(house_size,house_price,"bx")
plt.xlabel("house size")
plt.ylabel("Price")
plt.show()

def normalise(array):
    return (array-array.mean())/array.std()

num_train_samples = math.floor(num_houses*0.7)

train_house_size = np.asarray(house_size[:num_train_samples])
train_house_price = np.asarray(house_price[:num_train_samples])

train_house_size_norm = normalise(train_house_size)
train_house_price_norm = normalise(train_house_price)

test_house_size = np.asarray(house_size[num_train_samples:])
test_house_price = np.asarray(house_price[num_train_samples:])

test_house_size_norm = normalise(test_house_size)
test_house_price_norm = normalise(test_house_price)

tf_house_size = tf.placeholder("float",name="house_size")
tf_house_price = tf.placeholder("float",name="house_price")

tf_size_factor = tf.Variable(np.random.randn(),name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(),name="price_offset")

tf_price_pred = tf.add(tf.multiply(tf_size_factor,tf_house_size),tf_price_offset)

tf_cost = tf.reduce_sum(tf.pow(tf_price_pred-tf_house_price,2))/(2*num_train_samples)

learning_rate = 0.1

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    display_every = 2 
    num_iterations = 50
    
    fit_num_plots = math.floor(num_iterations/display_every)
    fit_size_factor = np.zeros(fit_num_plots)
    fit_price_offset = np.zeros(fit_num_plots)
    fit_plot_idx = 0
    
    for iteration in range(num_iterations):
        
        for x,y in zip(train_house_size_norm,train_house_price_norm):
            sess.run(optimizer,feed_dict={tf_house_size:x,tf_house_price:y})
            
        if (iteration+1) % display_every == 0:
            c = sess.run(tf_cost,feed_dict={tf_house_size:train_house_size_norm,tf_house_price:train_house_price_norm})
            print("Iteration #:",'%04d' % (iteration+1),"cost =",'{:.9f}'.format(c), \
                  "size_factor=",sess.run(tf_size_factor),"price_offset=",sess.run(tf_price_offset))
            fit_size_factor[fit_plot_idx] = sess.run(tf_size_factor)
            fit_price_offset[fit_plot_idx] = sess.run(tf_price_offset)
            fit_plot_idx +=1
            
    print("Optimization finished")
    training_cost = sess.run(tf_cost,feed_dict={tf_house_size:train_house_size_norm,tf_house_price:train_house_price_norm})
    print("Trained Cost =",training_cost,"size factor=",sess.run(tf_size_factor),"price offset=",sess.run(tf_price_offset),'/n')
    
    
    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()
    
    train_house_price_mean = train_house_price.mean()
    train_house_price_std = train_house_price.std()
    
    fig,ax = plt.subplots()
    line, = ax.plot(house_size,house_price)
    
    
    plt.rcParams["figure.figsize"]=(10,8)
    plt.title("Gradient Descent Fitting Regression Line")
    plt.ylabel("Price")
    plt.xlabel("Size (sq.ft)")
    plt.plot(train_house_size, train_house_price, 'go', label='Training data')
    plt.plot(test_house_size, test_house_price, 'mo', label='Testing data')
    
    def animate(i):
        line.set_xdata(train_house_size_norm * train_house_size_std + train_house_size_mean)  # update the data
        line.set_ydata((fit_size_factor[i] * train_house_size_norm + fit_price_offset[i]) * train_house_price_std + train_house_price_mean)  # update the data
        return line,
    
    def initAnim():
        line.set_ydata(np.zeros(shape=house_price.shape[0])) # set y's to 0
        return line,
    
    ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, fit_plot_idx), init_func=initAnim,
                                 interval=1000, blit=True)

    plt.show()  
    
    


                 
            
    



    