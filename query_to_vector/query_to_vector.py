from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import sys
import re

TOKENIZER = re.compile('[^ ,.]+').findall


def read_data_file(filename,skip_lines = 2):
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
            labels.append(int(cols[2].strip()))
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


def bucket_ints(queries, bucket_size = 1000):
    """
    Bucket numbers in the queries of test/val set
    to the closest bucket based on bucket size.
    :param queries: list of queries from test/val set
    :param bucket_sizet: Size of buckets
    :return: list of modified queries
    """
    for i in range(len(queries)):
        tokens = TOKENIZER(queries[i])
        for j in range(len(tokens)):
            if tokens[j].isdigit():
                tokens[j] = str(bucket_size*((int(tokens[j])+(bucket_size/2))/bucket_size))
        queries[i] = " ".join(tokens)


if __name__ == '__main__':
    num_features = 100
    path = 'C:/Users/Avnish Saraf/Desktop/Deep Learning/Project/'
    queries, labels = read_data_file(path+'data_10k.txt', skip_lines=2)
    bucket_ints(queries)
    X,vocab = vectorize_by_term(queries, num_features)

    print X.shape,len(labels)

    # X.dump(open('data_mat_10k.pkl','wb'))
    # labels = np.array(labels).reshape((len(labels), 1))
    # labels.dump(open('label_mat_10k.pkl', 'wb'))

    queries, labels = read_data_file(path+'data_10k_val.txt', skip_lines=2)
    bucket_ints(queries)
    map_ints_to_closest_vocab(queries,vocab)
    X,_ = vectorize_by_term(queries,num_features,vocab)

    print X.shape,len(labels)

    # X.dump(open('data_mat_1k_val.pkl','wb'))
    # labels = np.array(labels).reshape((len(labels), 1))
    # labels.dump(open('label_mat_1k_val.pkl', 'wb'))

