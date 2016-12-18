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
import sys

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import datetime
import os

#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
d_prefix='/home/avnishs/data/updated_log'
o_prefix = '/home/avnishs/results'
if len(sys.argv) == 1:
    fns = [
    'tr_bow_10k.pkl',
    'tr_label_bs20_10k.pkl',    
    'val_bow_1k.pkl',
    'val_label_bs20_1k.pkl',
    ]
    out_fn = 'bow_bs20.txt'
else:
    fns = [
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    ]
    out_fn = sys.argv[5]

data=[]
for idx, fn in enumerate(fns):
    data.append(np.load(os.path.join(d_prefix, fn)))
out_path = os.path.join(o_prefix, out_fn)

print('Writing to:',out_path)

trd=(data[0], data[1])
ted=(data[2], data[3])

print('training #: {} testing #: {}'.format(len(trd[0]), len(ted[0])))

def get_batch(data, s_idx, e_idx):
    return (data[0][s_idx:e_idx,:], data[1][s_idx:e_idx,:])

def get_batch_num(data, batch_size):
    return int(data[0].shape[0]/ float(batch_size))

# Parameters
learning_rate = 0.001
training_epochs = 500
batch_size = 1000
display_step = 1
save_step = 100
dropout = 0.5 # Dropout, probability to keep units

# Network Parameters
n_hidden_1 = 4000 # 1st layer number of features
n_input = trd[0].shape[1] 
n_classes = 1 if len(trd[1].shape)==1 else trd[1].shape[1]

print('Number of Classes =',n_classes)

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
    out_layer = tf.nn.softmax(out_layer)
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}
print_weight = tf.Print(weights['out'], [weights['out']], summarize=100)
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
print_bias = tf.Print(biases['out'], [biases['out']], summarize=30)

# Construct model
pred = multilayer_perceptron(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
te_error = 1 - accuracy
y_shape=tf.shape(y)[0]
te_avg_error = te_error


# Initializing the variables
init = tf.initialize_all_variables()

saver = tf.train.Saver()

with open(out_path,'w') as o_file:
    with tf.Session() as sess:
        sess.run(init)
        print('Running')
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


            # Display logs per epoch step
            if epoch % display_step == 0:
       
                status_line = "Epoch: {}, training error: {}, training loss: {}, Testing error: {}, loss: {}".format(
                  epoch,
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
                    )
                print(status_line)
                o_file.write(status_line)
                o_file.write('\n')
          
            if epoch !=0 and epoch % save_step == 0:
                saver.save(sess, 'model-{}.ckpt'.format(epoch))

            # sess.run(print_weight)
            # sess.run(print_bias)
            
        print("Optimization Finished!")
o_file.close()

              


    
    # # Test model
    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # # Calculate accuracy
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
