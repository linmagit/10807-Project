from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.manifold import TSNE
import math


def plot_curve(series_list, axis_labels = None, legend_labels = None, filename = None):
    for i,series in enumerate(series_list):
        idxs = range(1,len(series)+1)
        if legend_labels is None:
            plt.plot(idxs, series)
        else:
            plt.plot(idxs, series,label=legend_labels[i])
    if axis_labels is not None:
        plt.xlabel(axis_labels[0])
        plt.ylabel(axis_labels[1])

    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def read_file(filename, num_rows = None):
    tr_error, tr_loss, val_error, val_loss = [],[],[],[]
    with open(filename,'r') as f:
        for line in f:
            elems = line.strip().split(',')
            if len(elems) < 5: raise Exception
            tr_error.append(float(elems[1].split(':')[1].strip()))
            tr_loss.append(float(elems[2].split(':')[1].strip()))
            val_error.append(float(elems[3].split(':')[1].strip()))
            val_loss.append(float(elems[4].split(':')[1].strip()))
    if num_rows is None or num_rows >= len(tr_error):
        return tr_error, tr_loss, val_error, val_loss
    else:
        return tr_error[:num_rows], tr_loss[:num_rows], val_error[:num_rows], val_loss[:num_rows]


def main():
    filename = 'bow_bs20_5'
    tr_error, tr_loss, val_error, val_loss = read_file('results/{}.txt'.format(filename), num_rows=200)
    plot_curve([tr_error,val_error],axis_labels=['Epoch','Error'],legend_labels=['Training Error','Validation Error'], filename='results/{}_error.png'.format(filename))
    plot_curve([tr_loss,val_loss],axis_labels=['Epoch','Loss'],legend_labels=['Training Loss','Validation Loss'], filename='results/{}_loss.png'.format(filename))
if __name__ == '__main__':
    main()