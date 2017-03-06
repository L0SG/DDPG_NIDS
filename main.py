import tensorflow as tf
import keras
import numpy as np
import os
import datamodule
import keras.preprocessing.text as kt

# dimension of input feature
INPUT_SIZE = 13

# idx of discrete and continuous features
idx_discrete = [0, 1, 3, 6, 8]
idx_continuous = [2, 4, 5, 7, 9, 10, 11, 12]

""" not used
# token for elements not in the training vocabulary
unknown_token = '-1'
"""

# absolute path of the dataset
# currently assumes a single file of train, valid, and test data
DATA_PATH = '/home/tkdrlf9202/Datasets/SK_Infosec/input_merged'

# grab the raw data
# all elements are strings for now
data_train, data_valid, data_test = datamodule.loader(DATA_PATH)

""" not necessary
# sort the columns to [ (discrete features) , (continuous features)] format
order_idx = np.array(idx_discrete + idx_continuous)
data_train = data_train[:,order_idx]
data_valid = data_valid[:,order_idx]
data_test = data_test[:,order_idx]
"""

# load discrete & continuous columns separately
data_train_discrete = data_train[:, idx_discrete]
data_train_continuous = data_train[:, idx_continuous]

# generate tokenizer, one per discrete column
tokenizers = [kt.Tokenizer(nb_words=None) for i in xrange(len(idx_discrete))]

# tokenize discrete columns
data_train_tokenized = datamodule.generate_tokens(tokenizers, data_train_discrete)

print 'fuck'