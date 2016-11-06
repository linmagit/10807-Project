#! /usr/bin/env python

'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function
import pdb

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import datetime
import os

#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
d_prefix='/home/junjuew/deep_learning/data'
fns = [
'data_mat_10k.pkl',
'label_mat_10k.pkl',    
'data_mat_1k_val.pkl',
'label_mat_1k_val.pkl',
]
data=[]
for idx, fn in enumerate(fns):
    data.append(np.load(os.path.join(d_prefix, fn)))
    
trd=(data[0], data[1])
ted=(data[2], data[3])

print('training #: {} testing #: {}'.format(len(trd[0]), len(ted[0])))

def get_batch(data, s_idx, e_idx):
    return (data[0][s_idx:e_idx,:], data[1][s_idx:e_idx,:])

def get_batch_num(data, batch_size):
    return int(data[0].shape[0]/ float(batch_size))

# Parameters
learning_rate = 0.001
training_epochs = 1000
batch_size = 1000
display_step = 1
save_step = 100
dropout = 0.75 # Dropout, probability to keep units

# Network Parameters
n_hidden_1 = 4000 # 1st layer number of features
n_input = trd[0].shape[1] 
n_classes = 1 if len(trd[1].shape)==1 else trd[1].shape[1]

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create model
def multilayer_perceptron(x, weights, biases, dropout):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, dropout)
    
    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases, keep_prob)

# Define loss and optimizer
# cost = tf.reduce_mean(tf.nn.l2_loss(pred-y))
#cost = tf.reduce_mean(np.absolute(pred-y) /(y+1))
cost = tf.reduce_mean(np.absolute(pred-y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
#correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# te_loss = tf.reduce_mean(tf.nn.l2_loss(pred-y))
#te_error = tf.reduce_mean( np.absolute(pred-y) )
# te_error = tf.reduce_mean( tf.logical_or(
#     tf.logical_and(tf.less_equal(pred, y*1.2), tf.greater_equal(pred, y*0.8)),
#     tf.logical_and(tf.less_equal(pred, y+20), tf.greater_equal(pred, y-20))
# ))


crt_1=tf.logical_and(
    tf.less_equal(pred, y*1.5),
    tf.greater_equal(pred, y*0.5),
)
crt_2=tf.logical_and(        
    tf.less_equal(pred-y, 50),
    tf.greater_equal(pred-y, -50.0),
)

te_error=tf.where(
    tf.logical_not(
        tf.logical_or(
            crt_1,
            crt_2
        )
    )
)

y_shape=tf.shape(y)[0]

te_avg_error = tf.reduce_mean(tf.shape(te_error)[0])
#/ np.float32(tf.shape(y)[0])

#cost = tf.reduce_mean(np.absolute(pred-y))

# Initializing the variables
init = tf.initialize_all_variables()

saver = tf.train.Saver()

# with tf.Session() as sess:
#     # Restore variables from disk.
#     saver.restore(sess, "/home/junjuew/deep_learning/baseline/model-100.ckpt")
#     y_pred=sess.run(pred, feed_dict={x: ted[0], keep_prob: 1.})
#     result=[(y_pred[idx][0], ted[1][idx][0]) for idx, _ in enumerate(y_pred)]
#     for idx, _ in enumerate(y_pred):
#         print('{:7.1f} : {}'.format(result[idx][0], result[idx][1]))

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = get_batch_num(trd, batch_size)
        # Loop over all batches
        for i in range(total_batch):
            # train next_batch returns a tuple
            # first item is x of shape (batch_size, input_shape)
            # second item is y (batch_size, output_shape)
            batch_x, batch_y = get_batch(trd, batch_size*i, batch_size*(i+1))
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y,
                                                          keep_prob: dropout
                                                      })
            # Compute average loss
            avg_cost += c / total_batch

        # print("Epoch:", '%04d' % (epoch+1), "training loss=", \
        #       "{:.9f}".format(avg_cost))
        # Display logs per epoch step
        if epoch % display_step == 0:
            # print("crt_1: {}".format(
            #       sess.run(crt_1, feed_dict={x: ted[0],
            #                                     y: ted[1],
            #                                     keep_prob: 1.})
            #     ))
            
            # print("Testing error: {}".format(
            #       sess.run(te_error, feed_dict={x: ted[0],
            #                                     y: ted[1],
            #                                     keep_prob: 1.})
            #     ))
            # print('y shape: {}'.format(
            #     sess.run(y.shape, feed_dict={x: trd[0],
            #                                     y: trd[1],
            #                                     keep_prob: 1.}))
            # out=sess.run(y_shape, feed_dict={x: trd[0],
            #                                     y: trd[1],
            #                                     keep_prob: 1.})

            # print('tf type: {}, val:{}'.format(type(out), out))
            
            print("training error: {}, training loss: {}, Testing error: {}, loss: {}".format(
                  sess.run(te_avg_error, feed_dict={x: trd[0],
                                                y: trd[1],
                                                keep_prob: 1.}),
                 avg_cost,
                
                  sess.run(te_avg_error, feed_dict={x: ted[0],
                                                y: ted[1],
                                                keep_prob: 1.}),
                  sess.run(cost, feed_dict={x: ted[0],
                                                y: ted[1],
                                                keep_prob: 1.})
                ))
            
        if epoch !=0 and epoch % save_step == 0:
            saver.save(sess, 'model-{}.ckpt'.format(epoch))
            
    print("Optimization Finished!")


                


    
    # # Test model
    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # # Calculate accuracy
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
