#Source code with the blog post at http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
import numpy as np
import random
from random import shuffle
import tensorflow as tf
import os
import pdb
import math

d_prefix='/home/junjuew/deep_learning/data'
fns = [
'/home/linm/data/char_rnn_digit_data_mat_10k.pkl',
'/home/linm/data/char_rnn_digit_label_mat_10k.pkl',       
'/home/linm/data/char_rnn_digit_data_mat_1k_val.pkl', 
'/home/linm/data/char_rnn_digit_label_mat_1k_val.pkl'
]
data=[]
for idx, fn in enumerate(fns):
    data.append(np.load(os.path.join(d_prefix, fn)))
    
trd=(data[0], data[1])
ted=(data[2], data[3])

print('training #: {} testing #: {}'.format(len(trd[0]), len(ted[0])))

def get_batch(data, s_idx, e_idx):
    return (data[0][s_idx:e_idx,:,:], data[1][s_idx:e_idx,:,:])

def get_batch_num(data, batch_size):
    return int(data[0].shape[0]/ float(batch_size))


print "test and training data loaded"

input_dim=256
seq_length=100
learning_rate = 0.001
save_step = 100
display_step = 1
# output digit number
odn=6
# output digit dimension (one hot representation for digits)
odd=10
batch_size=1000
lstm_hidden=100

def RNN(x, seq_length, input_dim, n_hidden):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, input_dim])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, seq_length, x)
    # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Get lstm cell output
    outputs, states = tf.nn.rnn(lstm_cell, x, dtype=tf.float32)
    # Linear activation, using rnn inner loop last output
    return outputs[-1]

def output_layer(output_rnn, o_weight, o_bias):
    # digits now is a list len 6 of (batchsize, 10) tensor
    digits = [tf.matmul(output_rnn, o_weight[i,:,:]) + o_bias[i,:] for i in range(odn)]
    # [digit_idx, batch idx, one hot dimension]
    digits = tf.pack(digits)
    # [batch idx, digit_idx, one hot dimension]    
    digits = tf.transpose(digits, [1,0,2])
    return digits

def calc_loss(pred, label):
    # list 6 of (batch size,)
    batch_digits_val_loss=[math.pow(10.0, i) * tf.nn.softmax_cross_entropy_with_logits(pred[:,i,:], label[:,i,: ]) for i in range(odn)]
    batch_digits_val_loss_tensor=tf.pack(batch_digits_val_loss, axis=1)
    batch_val_loss = tf.reduce_sum(batch_digits_val_loss_tensor, 1)
    avg_batch_val_loss = tf.reduce_mean(batch_val_loss)
    return batch_val_loss, avg_batch_val_loss

    # digits_val_loss=[math.pow(10.0, i) * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred[i], label[i,:,: ])) for i in range(odn) ]
    # digits_loss=tf.pack(digits_loss)
    # loss = tf.reduce_mean(digits_loss) 
    
# def output_one_hot_to_val(pred, odn):
#     # list 6 of (batch size,)
#     pred_val_list=[math.pow(10.0, i) * tf.argmax(pred[:,i,:], 0) for i in range(odn)]
#     pred_val_list=tf.pack(pred_val_list)
#     return pred_val_list

# [examples_idx, sequence length (100), input_dimension (256: ASIC representation) ]
data = tf.placeholder(tf.float32, [None, seq_length, input_dim])
# [examples_idx, output digit idx, output digit dimension (10: one hot digit) ]    
target = tf.placeholder(tf.float32, [None, odn, odd])
#    target_t = tf.transpose(target, [1,0,2])
# [output digit idx, lstm hidden units, output digit dimension]
o_weight = tf.Variable(tf.truncated_normal([odn, lstm_hidden, odd]))
# [output digit idx, output digit dimension]    
o_bias = tf.Variable(tf.random_normal([odn, odd]))
# [batch_idx, lstm hidden idx] 
output_rnn = RNN(data, seq_length, input_dim, lstm_hidden)
# [batch_idx, digit idx, one hot dimension]
pred = output_layer(output_rnn, o_weight, o_bias)
batch_val_loss, loss = calc_loss(pred, target)

digit_wts=tf.reshape(tf.constant([math.pow(10.0,i) for i in reversed(range(odn))], dtype=tf.float32), [odn,1])
pred_text_rep=tf.to_float(tf.argmax(pred, 2))
# batch_idx
pred_val = tf.reshape(tf.matmul(pred_text_rep, digit_wts), [-1])
target_text_rep=tf.to_float(tf.argmax(target, 2))
target_val = tf.reshape(tf.matmul(target_text_rep, digit_wts), [-1])

#l1_norm=tf.reduce_mean(np.absolute(pred_val - target_val))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)

#loss = tf.reduce_mean(tf.nn.l2_loss(target-prediction))
#cross_entropy = -tf.reduce_sum(target * tf.log(prediction))

crt_1=tf.logical_and(
    tf.less_equal(pred_val, target_val*1.5),
    tf.greater_equal(pred_val, target_val*0.5),
)
crt_2=tf.logical_and(        
    tf.less_equal(pred_val-target_val, 50),
    tf.greater_equal(pred_val-target_val, -50.0),
)

te_error=tf.where(
    tf.logical_not(
        tf.logical_or(
            crt_1,
            crt_2
        )
    )
)

mistakes = tf.shape(te_error)[0]
#/ tf.constant(float(tf.shape(target)[0].item()))

#mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
#error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

init_op = tf.initialize_all_variables()
saver = tf.train.Saver()

sess = tf.Session()
sess.run(init_op)

total_batch = get_batch_num(trd, batch_size)
epoch = 5000
for i in range(epoch):
    for j in range(total_batch):
        batch_x, batch_y = get_batch(trd, batch_size*j, batch_size*(j+1))
        #        batch_y=np.transpose(batch_y, [1,0,2])
        sess.run(train_op, feed_dict={data: batch_x, target:batch_y})

    if i !=0 and i % display_step == 0:        
        tr_e = sess.run(mistakes, feed_dict={data: trd[0], target: trd[1]})
        tr_l = sess.run(loss, feed_dict={data: trd[0], target: trd[1]})
        te_e = sess.run(mistakes, feed_dict={data: ted[0], target: ted[1]})
        te_l = sess.run(loss, feed_dict={data: ted[0], target: ted[1]})
        print "{}: training loss: {}, error:{} Testing loss: {}, error:{}".format(i, tr_l, tr_e, te_l, te_e)

    if i !=0 and i % save_step == 0:
        saver.save(sess, 'model-{}.ckpt'.format(i))
sess.close()

# incorrect = sess.run(error,{data: trd[0], target: trd[1]})
# print sess.run(prediction,{data: [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]})
# print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))

