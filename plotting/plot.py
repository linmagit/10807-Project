#!/usr/bin/python

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib as mpl
import sys
import os

mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [
        #r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
        #r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
        r'\usepackage{helvet}',    # set the normal font here
        r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
        r'\sansmath' # <- tricky! -- # gotta actually # tell tex to use!
        ]  
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20

#input_path = "baseline_training_log"
#input_path = "lstm_training.log"
#input_path = "lstm_regression_training.log"
input_path = sys.argv[1]
result_dir = "../graphs/"
total_num = 200

axis = []
train_loss = []
train_error = []
test_loss = []
test_error = []

with open(input_path) as input_file:
    for line in input_file:
        terms = line.replace(':', ' ').replace(',', ' ').split()
        print terms
        # depending on the output format of your training log, you may want to modify the term indices below
        if terms[1] == "training":
            axis.append(int(terms[0]))
            train_loss.append(float(terms[3]))
            train_error.append(float(terms[5]))
            test_loss.append(float(terms[8]))
            test_error.append(float(terms[10]))


def plot(data1, data2, legend1, legend2, y_legend, result_prefix):
    fig = plt.figure()
    fig.set_size_inches(10, 2)
    ax = fig.add_subplot(111)
    plt.gca().set_xlim([0, total_num])
    ax.grid()
    ax.set_ylabel(y_legend,fontsize=18,weight='bold')
    ax.set_xlabel(r"\textbf{Training Epoch}",fontsize=18,weight='bold')
    ax.plot(axis, data1, 'r-', label = legend1, linewidth = 2.0)
    ax.plot(axis, data2, 'b-',
        label = legend2, linewidth = 2.0)
    ax.legend(ncol=2, bbox_to_anchor=(1.01, 1.2))
    plt.savefig(result_dir + "%s_%s.pdf" % (result_prefix, input_path), bbox_inches='tight')

plot(train_loss, test_loss, r"\textbf{Training Loss}", r"\textbf{Development Loss}", r"\textbf{L1-loss}", "loss")
plot(train_error, test_error, r"\textbf{Training Error}", r"\textbf{Development Error}",
        r"\textbf{Prediction Error}", "error")
