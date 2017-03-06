import tensorflow as tf
import keras
import numpy as np
import os


def loader(data_path):
    """
    load the dataset from the original path
    assumes separate train, valid, and train directory locations
    :param data_path: the master math
    :return: train, valid, and test set of ndarray format
    """

    # define the path
    path_train = os.path.join(data_path, 'train')
    path_valid = os.path.join(data_path, 'valid')
    path_test = os.path.join(data_path, 'test')

    # load data
    # currently assumes one file named train_merged.txt
    # fix later
    temp_train = os.path.join(path_train, 'train_merged.txt')
    temp_valid = os.path.join(path_valid, 'valid_merged.txt')
    temp_test = os.path.join(path_test, 'test_merged.txt')

    with open(temp_train, 'r') as f:
        data_train = f.readlines()
        data_train = np.asarray([line.strip().split() for line in data_train])

    with open(temp_valid, 'r') as f:
        data_valid = f.readlines()
        data_valid = np.asarray([line.strip().split() for line in data_valid])

    with open(temp_test, 'r') as f:
        data_test = f.readlines()
        data_test = np.asarray([line.strip().split() for line in data_test])

    return data_train, data_valid, data_test


def generate_tokens(tokenizers, data_discrete):
    """
    generate token matrix, with one tokenizer per feature column
    :param tokenizers: list of tokenizer objects
    :param data_discrete: original data containing discrete features
    :return: tokenized matrix data_tokenized
    """

    # initialize tokenized matrix
    # tokens must be int
    data_tokenized = np.zeros((len(data_discrete), len(data_discrete[0])), dtype='int64')

    # tokenizing loop
    for i in xrange(len(data_discrete[0])):
        tokenizers[i].fit_on_texts(data_discrete[:, i])
        tokens = tokenizers[i].texts_to_sequences(data_discrete[:, i])
        tokens = np.asarray(tokens).transpose()
        data_tokenized[:, i] = tokens

    return data_tokenized