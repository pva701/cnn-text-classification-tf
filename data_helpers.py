import re

import numpy as np
from gensim.models import word2vec
import json
import pickle


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def load_word2vec(fname, vocab_dict):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    model = word2vec.KeyedVectors.load_word2vec_format(fname, binary=True)
    word_vecs = {}
    for word in vocab_dict:
        if word in model:
            word_vecs[word] = model[word]
    return word_vecs


def dump_json_word_vecs_np(fname, word_vecs):
    word2vec_list = {}
    for k, v in word_vecs.items():
        word2vec_list[k] = v.tolist()

    with open(fname, 'w') as f:
        json.dump(word2vec_list, f)


def load_json_word_vecs_np(fname):
    with open(fname, 'r') as f:
        word2vec_list = json.load(f)
        word2vec_np = {}
        for k, v in word2vec_list.items():
            word2vec_np[k] = np.array(v, dtype=np.float32)
        return word2vec_np


def dump_pickle_word_vecs_np(fname, word_vecs):
    with open(fname, 'wb') as f:
        pickle.dump(word_vecs, f)


def load_pickle_word_vecs_np(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)
        # word_vecs64 = {}
        # for k, v in word_vecs.items():
        #     word_vecs64[k] = v.astype(np.float64)
        # print(list(word_vecs64.items())[0][1].dtype)


def add_unknown_words(word_vecs, vocab_dict, bound=0.1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab_dict:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-bound, bound, k)


def load_glove_model(glove_file, vocab_dict):
    f = open(glove_file, 'r')
    word_vecs = {}
    for line in f:
        split_line = line.split()
        word = split_line[0].lower()
        vec = split_line[1:]
        if len(vec) != 300:
            continue
        if word in vocab_dict:
            word_vecs[word] = [float(val) for val in vec]
    print("Done. ", len(word_vecs), " words loaded!")
    return word_vecs
