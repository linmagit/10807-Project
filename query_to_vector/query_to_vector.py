from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import math
import sys
import re

TOKENIZER = re.compile('[^ ,.]+').findall
EPS = 10e-5
np.random.seed(0)

def read_data_file(filename, label_col=2, skip_lines = 2):
    """
    :param filename: name of the file to be read
    :param skip_lines: number of lines to skip at the start of the file
    :return: list of queries converted to lowercase
    """
    queries = []
    labels = []
    with open(filename,'r') as f:
        count = 0
        for line in f:
            count += 1
            if count <= skip_lines: continue
            cols = line.split('|')
            if len(cols) <= 1: continue
            query = cols[0].strip().lower()
            query = re.sub(r'([a-zA-Z0-9]*)([\W]+)([a-zA-Z0-9]*)', r'\1 \2 \3',query)
            queries.append(query)
            labels.append(cols[label_col].strip())
    return queries,labels


def vectorize_by_char(queries, num_features):
    """
    Converts each query to a matrix where row i is a one hot representation
    of the (i+1)th character in the query.

    :param queries: list of queries
    :param num_features: size of feature vector
    :return: Row-wise data matrix. #rows = #queries and #cols = num_features
             and each element is a one hot vector of length 256
    """
    X = []
    for query in queries:
        vector = []
        for c in query:
            vector.append(ord(c))
        if len(vector) <= num_features:
            vector += [0]*(num_features-len(vector))
        else:
            vector = vector[:num_features]
        one_hot_mat = np.zeros((len(vector),256))
        one_hot_mat[np.arange(len(vector)),np.array(vector)] = 1
        X.append(one_hot_mat)

    return np.array(X)


def vectorize_by_term(queries,num_features,vocab = None):
    """
    Converts each query to a matrix where row i is a one hot representation
    of the (i+1)th term in the query.

    :param queries: list of queries
    :param num_features: size of feature vector
    :return: Row-wise data matrix. #rows = #queries and #cols = num_features
    """
    vectorizer = CountVectorizer(min_df=0,tokenizer=TOKENIZER,vocabulary=vocab)
    vectorizer.fit(queries)
    tokenize = vectorizer.build_tokenizer()
    X = []
    lengths = []
    for query in queries:
        vector = []
        for token in tokenize(query):
            vector.append(np.argmax(vectorizer.transform([token]).todense()))
        if len(vector) <= num_features:
            vector += [0]*(num_features-len(vector))
        else:
            vector = vector[:num_features]
        one_hot_mat = np.zeros((len(vector),len(vectorizer.vocabulary_)))
        one_hot_mat[np.arange(len(vector)),np.array(vector)] = 1
        X.append(one_hot_mat)
    return np.array(X),vectorizer.vocabulary_


def get_bag_of_words(queries, vocab = None):
    """
    Converts each query to an array of counts where index i represents the
    number of times the (i+1)th term in the vocabulary occurs in the query

    :param queries: list of queries
    :return: Row-wise data matrix. #rows = #queries and #cols = vocabulary_size
             Vocabulary
    """
    vectorizer = CountVectorizer(min_df=0,tokenizer=TOKENIZER,strip_accents='unicode',vocabulary=vocab)
    X = vectorizer.fit_transform(queries).todense()
    return X,vectorizer.vocabulary_


def get_closest_value_from_vocab(num, vocab):
    """
    return the closest integer in the seen vocabulary to the given number
    :param num: The given number
    :param vocab: The seen vocabulary
    :return: the closest number from vocab
    """
    min_diff = sys.maxint
    closest_num = 0
    for token in vocab:
        if token.isnumeric() and min_diff > abs(int(token) - num):
            min_diff = abs(int(token) - num)
            closest_num = int(token)
    return closest_num


def map_ints_to_closest_vocab(queries,vocab):
    """
    Convert numbers in the queries of test/val set
    to the closest number seen in the vocabulary
    :param queries: list of queries from test/val set
    :param vocab: vocabulary of the train set
    :return: list of modified queries
    """
    for i in range(len(queries)):
        tokens = queries[i].split()
        for j in range(len(tokens)):
            if tokens[j].isdigit():
                tokens[j] = str(get_closest_value_from_vocab(int(tokens[j]), vocab))
        queries[i] = " ".join(tokens)


def bucket_ints(list_of_strings, bucket_size = 1000, log_base = None):
    """
    Bucket numbers in the queries/label of test/val set
    to the closest bucket based on bucket size.
    :param list_of_strings: list of queries/labels from test/val set
    :param bucket_sizet: Size of buckets
    :param log_base: base w.r.t which log is calculated
    :return: list of modified queries/labels
    """
    for i in range(len(list_of_strings)):
        tokens = TOKENIZER(list_of_strings[i])
        for j in range(len(tokens)):
            if tokens[j].isdigit():
                old = tokens[j]
                if log_base:
                    # tokens[j] = str(bucket_size * int(((math.log(int(tokens[j]) + EPS,log_base) + (bucket_size / 2)) / bucket_size)))
                    tokens[j] = str(int(math.log(max(int(tokens[j]) + EPS,bucket_size)/bucket_size,log_base)))
                else:
                    tokens[j] = str(bucket_size*((int(tokens[j])+(bucket_size/2))/bucket_size))
                # print old,tokens[j]
        list_of_strings[i] = " ".join(tokens)


def random_sample(queries, labels, val_set_size = 1000):
    idx = np.array(range(len(queries)))
    np.random.shuffle(idx)
    tr_idx,val_idx = idx[:-val_set_size],idx[-val_set_size:]
    tr_queries, tr_labels = queries[tr_idx],labels[tr_idx]
    val_queries,val_labels = queries[val_idx],labels[val_idx]

    return tr_queries, tr_labels, val_queries, val_labels
    # return list(tr_queries), list(tr_labels), list(val_queries), list(val_labels)

if __name__ == '__main__':
    num_features = 21
    bucket_size = 20
    log_base = 1.2

    path = 'C:/Users/Avnish Saraf/Desktop/Deep Learning/Project/'
    output_data_filename = 'data_2col_bs{}'.format(bucket_size)
    ouput_label_filename = 'label_2col_bs{}'.format(bucket_size)

    #One-col query
    # # read list of query strings and label strings
    # tr_queries, tr_labels = read_data_file(path + 'data_10k.txt', skip_lines=2)
    # val_queries, val_labels = read_data_file(path + 'data_10k_val.txt', skip_lines=2)
    # # random sample from queries and labels
    # tr_queries, tr_labels, val_queries, val_labels = random_sample(np.array(tr_queries + val_queries), np.array(tr_labels + val_labels))

    #Two-col query
    queries, labels = read_data_file(path + 'two_columns_data.txt', skip_lines=0)
    # random sample from queries and labels
    tr_queries, tr_labels, val_queries, val_labels = random_sample(np.array(queries), np.array(labels),val_set_size=8470)

    # bucket the integers in queries and labels
    bucket_ints(tr_queries)
    bucket_ints(tr_labels, bucket_size=bucket_size, log_base=log_base)

    # convert queries to term level one-hot vectors
    X,vocab = vectorize_by_term(tr_queries,num_features)

    # convert queries to BOW vectors
    # X,vocab = get_bag_of_words(tr_queries)


    tr_labels = np.array(map(int, tr_labels))
    # one hot vector length = largest label bucket + 1
    one_hot_dim = max(tr_labels) + 1
    one_hot_mat = np.zeros((len(tr_labels), one_hot_dim))
    # convert to one hot representation
    one_hot_mat[np.arange(len(tr_labels)), tr_labels] = 1

    print 'X ={}, labels = {}, one-hot labels = {}'.format(X.shape, tr_labels.shape, one_hot_mat.shape)
    print map(int,np.sum(one_hot_mat,axis=0))

    X.dump(open('tr_{}_30k.pkl'.format(output_data_filename),'wb'))
    one_hot_mat.dump(open('tr_{}_30k.pkl'.format(ouput_label_filename),'wb'))

    # bucket the integers in queries and labels
    bucket_ints(val_queries)
    bucket_ints(val_labels, bucket_size=bucket_size, log_base=log_base)
    # map the Out of Vocab integers in the validation data to the closest integer in training data
    map_ints_to_closest_vocab(val_queries, vocab)

    # convert queries to term level one-hot vectors
    X,_ = vectorize_by_term(val_queries,num_features, vocab)

    # convert queries to BOW vectors
    # X,_ = get_bag_of_words(val_queries,vocab)

    val_labels = np.array(map(int, val_labels))
    # one hot vector length should be same as the training label one-hot vector
    one_hot_mat = np.zeros((len(val_labels), one_hot_dim))
    one_hot_mat[np.arange(len(val_labels)), val_labels] = 1

    print 'X ={}, labels = {}, one-hot labels = {}'.format(X.shape, val_labels.shape, one_hot_mat.shape)
    print map(int,np.sum(one_hot_mat,axis=0))

    X.dump(open('val_{}_8k.pkl'.format(output_data_filename),'wb'))
    one_hot_mat.dump(open('val_{}_8k.pkl'.format(ouput_label_filename),'wb'))


