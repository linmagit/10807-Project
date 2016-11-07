#Source code with the blog post at http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
import numpy as np
import random
from random import shuffle
import tensorflow as tf
import os
import pdb
d_prefix='/home/junjuew/deep_learning/data'
fns = [
'char_rnn_data_mat_10k.pkl',
'char_rnn_label_mat_10k.pkl',       
'char_rnn_data_mat_1k_val.pkl', 
'char_rnn_label_mat_1k_val.pkl'
]
data=[]
for idx, fn in enumerate(fns):
    data.append(np.load(os.path.join(d_prefix, fn)))
    
trd=(data[0], data[1])
ted=(data[2], data[3])

print('training #: {} testing #: {}'.format(len(trd[0]), len(ted[0])))

def get_batch(data, s_idx, e_idx):
    return (data[0][s_idx:e_idx,:,:], data[1][s_idx:e_idx,:,])

def get_batch_num(data, batch_size):
    return int(data[0].shape[0]/ float(batch_size))


print "test and training data loaded"

input_dim=256
seq_length=100
learning_rate = 0.001
save_step = 500
display_step = 5

#Number of examples, sequence length, dimension of each input
data = tf.placeholder(tf.float32, [None, seq_length, input_dim]) 
target = tf.placeholder(tf.float32, [None, 1])
num_hidden=10
cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
# transpose so that the val is [sequence_idx, batch_idx, feature_dim_idx]
val = tf.transpose(val, [1, 0, 2])
# last is  a tensor of shape (batch_size, num_hidden)
#last = tf.gather(val, int(val.get_shape()[0]) - 1)
last = val[seq_length-1, :, :]
# weight of shape (num_hidden, output dim)
weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.random_normal([int(target.get_shape()[1])]))
prediction = tf.add(tf.matmul(last, weight), bias)
loss = tf.reduce_mean(np.absolute(target-prediction))
#loss = tf.reduce_mean(tf.nn.l2_loss(target-prediction))
#cross_entropy = -tf.reduce_sum(target * tf.log(prediction))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#minimize = optimizer.minimize(cross_entropy)
minimize = optimizer.minimize(loss)

crt_1=tf.logical_and(
    tf.less_equal(prediction, target*1.5),
    tf.greater_equal(prediction, target*0.5),
)
crt_2=tf.logical_and(        
    tf.less_equal(prediction-target, 50),
    tf.greater_equal(prediction-target, -50.0),
)

te_error=tf.where(
    tf.logical_not(
        tf.logical_or(
            crt_1,
            crt_2
        )
    )
)

mistakes = tf.reduce_mean(tf.shape(te_error)[0])
#/ tf.constant(float(tf.shape(target)[0].item()))

#mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
#error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

init_op = tf.initialize_all_variables()
saver = tf.train.Saver()

sess = tf.Session()
sess.run(init_op)
batch_size = 500
total_batch = get_batch_num(trd, batch_size)
epoch = 5000
for i in range(epoch):
    for j in range(total_batch):
        batch_x, batch_y = get_batch(trd, batch_size*i, batch_size*(i+1)) 
        sess.run(minimize,{data: batch_x, target: batch_y})

    if i !=0 and i % display_step == 0:        
        tr_e = sess.run(mistakes, feed_dict={data: trd[0], target: trd[1]})
        tr_l = sess.run(loss, feed_dict={data: trd[0], target: trd[1]})    
        te_e = sess.run(mistakes, feed_dict={data: ted[0], target: ted[1]})
        te_l = sess.run(loss, feed_dict={data: ted[0], target: ted[1]})    
        print "{}: training error: {}, training loss: {}, Testing error: {}, loss: {}".format(i, tr_e, tr_l, te_e, te_l)

    if i !=0 and i % save_step == 0:
        saver.save(sess, 'model-{}.ckpt'.format(i))
sess.close()

# incorrect = sess.run(error,{data: trd[0], target: trd[1]})
# print sess.run(prediction,{data: [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]})
# print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))

