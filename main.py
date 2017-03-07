import tensorflow as tf
import keras
import keras.preprocessing.text as kt
from keras.layers import Input, Embedding, LSTM, MaxoutDense, Dense, Dropout, merge
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import numpy as np
import os
import datamodule
from sklearn.preprocessing import normalize
from keras.layers.core import Lambda
from keras.utils.np_utils import to_categorical
from keras.backend.tensorflow_backend import set_session

# VRAM limit, typical
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

# dimension of input feature
INPUT_SIZE = 13

# idx of discrete and continuous features
idx_discrete = [0, 1, 3, 6, 8, 10]
idx_continuous = [2, 4, 5, 7, 9, 11, 12]

# batch size == seq_length is explicitly 1 for stateful LSTM
batch_size = 1

print('Input feature column size: ' + str(INPUT_SIZE))
print('Idx of discrete features: ' + str(idx_discrete))
print('Idx of continuous features: ' + str(idx_continuous))

# absolute path of the dataset
# currently assumes a single file of train, valid, and test data
DATA_PATH = '/home/tkdrlf9202/Datasets/SK_Infosec/input_merged'

# grab the raw data
# all elements are strings for now
print('loading raw data...')
data_train, data_valid, data_test = datamodule.loader(DATA_PATH)

# load discrete & continuous columns separately
data_train_discrete = data_train[:, idx_discrete]
data_train_continuous = data_train[:, idx_continuous]

# TEMPORARY HACK: REMOVE DOTS IN IP ADDRESS
# dots IN THE IP make the tokenizer API split the IP address to 4 elements
# so just remove them
IDX_IP = idx_discrete.index(10)
data_train_discrete[:, IDX_IP] = np.char.replace(data_train_discrete[:, IDX_IP], '.', '')

# cast to float and normalize continuous features per column
# storing the mean is necessary for the valid and test set
# it'd be better to normalize the data prior to the code
data_train_continuous = data_train_continuous.astype('float32')
data_train_continuous = normalize(data_train_continuous, axis=0)

# generate tokenizer, one per discrete column
print('tokenizing discrete features...')
tokenizers = [kt.Tokenizer(nb_words=None) for i in xrange(len(idx_discrete))]

# tokenize discrete columns
data_train_tokenized = datamodule.generate_tokens(tokenizers, data_train_discrete)
vocab_size = [len(tokenizers[i].word_index) for i in xrange(len(idx_discrete))]
print('vocab size of discrete features from training set : ' + str(vocab_size))

############################################# model construction

# discrete feature input vector generation
input_discrete = [None for i in xrange(len(idx_discrete))]
input_embed = [None for i in xrange(len(idx_discrete))]

# embedding per column
for i in xrange(len(idx_discrete)):
    input_discrete[i] = Input(batch_shape=(batch_size, 1),
                              dtype='int64', name='input_discrete'+str(i))
    input_embed[i] = Embedding(output_dim=10, input_dim=len(tokenizers[i].word_index),
                               input_length=1, name='input_embedded'+str(i))(input_discrete[i])

# continuous feature input vector generation
input_continuous = Input(batch_shape=(batch_size, len(idx_continuous)),
                         dtype='float32', name='input_continuous')

# expand continuous input vector dimension from (None, x) to (None, 1, x)
# this is because input_embed has shape of (None, 1, x), where 1 is input_length
# LSTM and dense layers also assumes this
input_continuous_expanded = Lambda(datamodule.expand_dims,
                                   datamodule.expand_dims_output_shape)(input_continuous)

# merge embedded vectors and continuous vector
x_embed = merge([input_embed[i] for i in xrange(len(idx_discrete))], mode='concat')
x = merge([x_embed, input_continuous_expanded], mode='concat')

# apply the vectors to the LSTM layers
y = LSTM(128, batch_input_shape=(batch_size, x.shape[1], x.shape[2]), stateful=True, return_sequences=True)(x)
y = BatchNormalization()(y)
y = Dropout(0.2)(y)
y = LSTM(128, stateful=True)(y)
y = BatchNormalization()(y)
y = Dropout(0.2)(y)

# some more layers
y = MaxoutDense(512)(y)
y = BatchNormalization()(y)
y = Dropout(0.5)(y)
y = MaxoutDense(512)(y)
y = BatchNormalization()(y)
y = Dropout(0.5)(y)
y = MaxoutDense(512)(y)
y = BatchNormalization()(y)
y = Dropout(0.5)(y)

# discrete output vector generation
output_discrete = [None for i in xrange(len(idx_discrete))]
for i in xrange(len(idx_discrete)):
    output_discrete[i] = Dense(output_dim=len(tokenizers[i].word_index),
                               activation='softmax', name='output_discrete'+str(i))(y)

# continuous output vector logistic regression
# add in several more dense layers for the regression perhaps?
output_continuous = Dense(output_dim=len(idx_continuous),
                          activation='sigmoid', name='output_continuous')(y)

model = Model(input=input_discrete+[input_continuous],
              output=output_discrete+[output_continuous])

loss_dict = {'output_discrete'+str(i): 'kld' for i in xrange(len(idx_discrete))}
loss_dict.update({'output_continuous': 'kld'})

model.compile(optimizer='adam',
              loss=loss_dict)
model.summary()

####################### model construction end, modulate for cleaner code?

####################### input data formulation

# formulate x and y for model.fit, corresponding to the defined input above
# should split the data to (# of discrete feature) + 1 continuous feature array = # + 1
# example: 5 discrete, 8 continuous => split to 5 + 1 = 6 subarrays
feed_x_discrete = [data_train_tokenized[:-1, i] for i in xrange(len(idx_discrete))]
feed_x_continuous = [data_train_continuous[:-1, :]]
"""
# reshape to (nb_samples, seq_length=1, nb_features)
for i in xrange(len(idx_discrete)):
    feed_x_discrete[i] = np.reshape(feed_x_discrete[i], (feed_x_discrete[i].shape[0], 1))
# continuous feed has just one element of ndarray
# damn it's getting ugly, better ways?
feed_x_continuous[0] = np.reshape(feed_x_continuous[0], (feed_x_continuous[0].shape[0], feed_x_continuous[0].shape[1]))
"""
# merge
feed_x = feed_x_discrete + feed_x_continuous

# same goes to y, but should convert to one hot encoding for discrete features
feed_y_discrete = [data_train_tokenized[1:, i] for i in xrange(len(idx_discrete))]
feed_y_continuous = [data_train_continuous[1:, :]]
for i in xrange(len(idx_discrete)):
    # vocabulary is one-based, and to_categorical is zero-based (fuck)
    feed_y_discrete[i] = to_categorical(np.add(feed_y_discrete[i], -1), nb_classes=len(tokenizers[i].word_index))
    #feed_y_discrete[i] = np.reshape(feed_y_discrete[i], (feed_y_discrete[i].shape[0], feed_y_discrete[i].shape[1]))
#feed_y_continuous[0] = np.reshape(feed_y_continuous[0], (feed_y_continuous[0].shape[0], feed_y_continuous[0].shape[1]))
feed_y = feed_y_discrete + feed_y_continuous

# softmax from discrete output is zero-based
# tokenizer is one-based
# be cautious when inference
for i in xrange(1000):
    model.fit(x=feed_x,
              y=feed_y,
              batch_size=batch_size,
              nb_epoch=1,
              verbose=1,
              shuffle=False)
    model.reset_states()

