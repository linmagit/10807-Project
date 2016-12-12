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
#'/home/avnishs/data/updated_log/tr_data_2col_30k.pkl',
'/home/avnishs/data/updated_log/tr_data_10k.pkl',
#'/home/avnishs/data/updated_log/tr_label_2col_bs20_30k.pkl',
'/home/avnishs/data/updated_log/tr_label_bs50_10k.pkl',
#'/home/avnishs/data/updated_log/val_data_2col_8k.pkl',
'/home/avnishs/data/updated_log/val_data_1k.pkl',
#'/home/avnishs/data/updated_log/val_label_2col_bs20_8k.pkl',
'/home/avnishs/data/updated_log/val_label_bs50_1k.pkl',
#'/home/linm/data/tr_data_mat_10k.pkl',
#'/home/linm/data/tr_label_mat_10k.pkl',       
#'/home/linm/data/val_data_mat_10k.pkl', 
#'/home/linm/data/val_label_mat_10k.pkl'
]
data=[]
for idx, fn in enumerate(fns):
    data.append(np.load(os.path.join(d_prefix, fn)))
    
trd=(data[0], data[1])
ted=(data[2], data[3])
#trd=(data[0][0:100], data[1][0:100])
#ted=(data[2][0:100], data[3][0:100])

print data[0].shape
print data[1].shape
print data[2].shape
print data[3].shape

print('training #: {} testing #: {}'.format(len(trd[0]), len(ted[0])))

def get_batch(data, s_idx, e_idx):
    return (data[0][s_idx:e_idx,:,:], data[1][s_idx:e_idx,:])

def get_batch_num(data, batch_size):
    return int(data[0].shape[0]/ float(batch_size))


print "test and training data loaded"

input_dim=164
seq_length=15
learning_rate = 0.01
save_step = 100
display_step = 1
# classification output dimension (one hot representation classification result)
output_dim=24
# the number of non-overlapping regions of equal size for pooling
pooling_region_num=5
batch_size=1000
lstm_hidden=50
hidden_layers_num=1
dropout_keep_prob=0.5
# number of directions (one-direction/bidirection)
direction_num=2
epoch = 220
out_dir = "log_output"
os.system("rm -rf " + out_dir)

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
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, activation=tf.nn.relu)
    # Define dropout wrapper about lstm cell
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob = dropout_keep_prob)
    multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * hidden_layers_num)
    # Get lstm cell output
    lstm_cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, activation=tf.nn.relu)
    lstm_cell2 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob = dropout_keep_prob)
    multi_lstm_cell2 = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * hidden_layers_num)
    outputs, state, _ = tf.nn.bidirectional_rnn(multi_lstm_cell, multi_lstm_cell2, x, dtype=tf.float32)
    #outputs, state = tf.nn.rnn(multi_lstm_cell, x, dtype=tf.float32)
    # Output is a list of 'seq_length' tensors of shape (batch_size, lstm_hidden)
    return outputs, state

def output_layer(output_rnn, o_weight, o_bias):
    #pooling_result = output_rnn[-1]
    # Concat and reshape to (seq_length, batch_size, lstm_hidden)
    x = tf.concat(0, output_rnn)
    x = tf.reshape(x, [seq_length, -1, lstm_hidden])
    # Reshape to (batch_size, seq_length, lstm_hidden)
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, seq_length, lstm_hidden, 1])
    # pooling_result is in shape (batch_size, pooling_region_num, lstm_hidden, 1)
    pooling_result = tf.nn.max_pool(x, 
                                    ksize = [1, seq_length / pooling_region_num, 1, 1],
                                    strides = [1, seq_length / pooling_region_num, 1, 1],
                                    padding = 'SAME') 
    # Rehape to (batch_size, pooling_region_num * lstm_hidden)
    pooling_result = tf.reshape(pooling_result, [-1, pooling_region_num * lstm_hidden * direction_num])
    # Result in shape (batch_size, output_dim)
    result = tf.matmul(pooling_result, o_weight) + o_bias
    # Apply softmax to the last dimension
    result = tf.nn.softmax(result)
    return result

def calc_loss(pred, label):
    # Result in shape (batch_size, output_dim)
    batch_val_loss= tf.nn.softmax_cross_entropy_with_logits(pred, label)
    # A 1-D Tensor of length batch_size of the softmax cross entropy loss
    avg_batch_val_loss = tf.reduce_mean(batch_val_loss)
    return batch_val_loss, avg_batch_val_loss

# [examples_idx, sequence length (100), input_dimension (256: ASIC representation) ]
data = tf.placeholder(tf.float32, [None, seq_length, input_dim])
# [examples_idx, output digit idx, output digit dimension (10: one hot digit) ]    
target = tf.placeholder(tf.float32, [None, output_dim])
#    target_t = tf.transpose(target, [1,0,2])
# [lstm hidden units * pooling regions, classification output dimension]
o_weight = tf.Variable(tf.truncated_normal([pooling_region_num * lstm_hidden * direction_num, output_dim]), name = 'output_weights')
# [output digit idx, output digit dimension]    
o_bias = tf.Variable(tf.random_normal([output_dim]), name = 'output_biases')
# [batch_idx, lstm hidden idx] 
output_rnn, memory_state = RNN(data, seq_length, input_dim, lstm_hidden)
# [batch_idx, output_dim]
pred = output_layer(output_rnn, o_weight, o_bias)
batch_val_loss, loss = calc_loss(pred, target)

pred_text_rep=tf.argmax(pred, 1)
# batch_idx
target_text_rep=tf.argmax(target, 1)

#l1_norm=tf.reduce_mean(np.absolute(pred_val - target_val))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
grads_and_vars = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(grads_and_vars)


# Keep track of gradient values and sparsity (optional)
grad_summaries = []
for g, v in grads_and_vars:
    if g is not None:
        grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
        sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
        grad_summaries.append(grad_hist_summary)
        grad_summaries.append(sparsity_summary)
grad_summaries_merged = tf.merge_summary(grad_summaries)

# Keep track of the final memory state
memory_summary = tf.histogram_summary("memory_state", memory_state)

# error criterion
te_error = tf.not_equal(pred_text_rep, target_text_rep)

mistakes = tf.reduce_mean(tf.cast(te_error, "float"))
#/ tf.constant(float(tf.shape(target)[0].item()))

#mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
#error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

init_op = tf.initialize_all_variables()
saver = tf.train.Saver()

sess = tf.Session()
sess.run(init_op)

# Summaries for loss and accuracy
loss_summary = tf.scalar_summary("loss", loss)
err_summary = tf.scalar_summary("error", mistakes)

# Train Summaries
train_summary_op = tf.merge_summary([loss_summary, err_summary])
train_summary_dir = os.path.join(out_dir, "summaries", "train")
train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

# Gradient Summaries
gradient_summary_op = tf.merge_summary([grad_summaries_merged, memory_summary])
gradient_summary_dir = os.path.join(out_dir, "summaries", "gradient")
gradient_summary_writer = tf.train.SummaryWriter(gradient_summary_dir, sess.graph)

# Dev summaries
dev_summary_op = tf.merge_summary([loss_summary, err_summary])
dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

total_batch = get_batch_num(trd, batch_size)
for i in range(epoch):

    if i !=0 and i % display_step == 0:        
        tr_e, tr_l, tr_summary = sess.run([mistakes, loss, train_summary_op], feed_dict={data: trd[0], target: trd[1]})
        te_e, te_l, te_summary = sess.run([mistakes, loss, dev_summary_op], feed_dict={data: ted[0], target: ted[1]})
        print "{}: training loss: {}, error:{} Testing loss: {}, error:{}".format(i, tr_l, tr_e, te_l, te_e)

        train_summary_writer.add_summary(tr_summary, i)
        train_summary_writer.flush()
        dev_summary_writer.add_summary(te_summary, i)
        dev_summary_writer.flush()

    for j in range(total_batch):
        batch_x, batch_y = get_batch(trd, batch_size*j, batch_size*(j+1))
        #        batch_y=np.transpose(batch_y, [1,0,2])
        _, summaries = sess.run([train_op, gradient_summary_op], feed_dict={data: batch_x, target:batch_y})
        gradient_summary_writer.add_summary(summaries, i * total_batch + j)
        gradient_summary_writer.flush()

    if i !=0 and i % save_step == 0:
        saver.save(sess, 'model-{}.ckpt'.format(i))
sess.close()

# incorrect = sess.run(error,{data: trd[0], target: trd[1]})
# print sess.run(prediction,{data: [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]})
# print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))

