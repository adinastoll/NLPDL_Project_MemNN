from __future__ import print_function

from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout
from keras.layers import add, dot, concatenate
from keras.layers import LSTM
import numpy as np

from dataproc_utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

test_size = 0.2
random_state = 42
batch_size = 64
epochs = 25
body_size = 20 
claim_size = 12
embedding_dim = 100
output_size = 4  # size of the output vector


# load data and labels
data = load_proc_data('processed_data\\train_bodies.txt', 'processed_data\\train_claims.txt', split_pars=False)
labels = [label for body, claim, label in data]
y = np.array(labels)

print('First input tuple (body, claim, stance):\n', data[0])


# train/validation/test split
train_data, val_data, train_labels, val_labels = train_test_split(data, y, test_size=.2, random_state=random_state)


# create a vocabulary dict from train data
word2freq = make_word_freq_V(train_data)
word2index = word2idx(word2freq)

vocab_size = len(word2index) + 1
print('Vocab size:', vocab_size, 'unique words in the train set')

# vectorize input words (turn each word into its index from the word2index dict)
# for new words in test set that don't appear in train set, use index of <unknown>
train_body, train_claim = word_vectorizer(train_data, word2index, max_body_len=body_size, max_claim_len=claim_size)
val_body, val_claim = word_vectorizer(val_data, word2index, max_body_len=body_size, max_claim_len=claim_size)


# perform random under/over sampling to prevent class imbalance
train_body, train_claim, train_labels = random_sampler(train_body, train_claim, train_labels, type='over')


print("Training set shape: bodies", train_body.shape)
print("Training set shape: claims", train_claim.shape)
print("Training set shape: labels", train_labels.shape)

# data size params
n_train = train_body.shape[0]
n_val = val_body.shape[0]

print("Training Size", n_train)
print("Validation Size", n_val)


# initialize placeholders
input_body = Input((body_size,))
input_claim = Input((claim_size,))

# encoders
# embed the input sequence into a sequence of vectors
body_encoder_m = Sequential()
body_encoder_m.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))
body_encoder_m.add(Dropout(0.3))  # output: (samples, body_size, embedding_dim)

# embed the input into a sequence of vectors of size claim_size
body_encoder_c = Sequential()
body_encoder_c.add(Embedding(input_dim=vocab_size, output_dim=claim_size))
body_encoder_c.add(Dropout(0.3))  # output: (samples, body_size, claim_size)

# embed the claim into a sequence of vectors
claim_encoder = Sequential()
claim_encoder.add(Embedding(input_dim=vocab_size,
                            output_dim=embedding_dim,
                            input_length=claim_size))
claim_encoder.add(Dropout(0.3))  # output: (samples, claim_size, embedding_dim)

# encode article bodies and claims (which are indices) as sequences of dense vectors
body_encoded_m = body_encoder_m(input_body)
body_encoded_c = body_encoder_c(input_body)
claim_encoded = claim_encoder(input_claim)

# compute a 'match' between the body vector sequence and the claim vector sequence
match = dot([body_encoded_m, claim_encoded], axes=(2, 2))
match = Activation('softmax')(match)  # shape: (samples, body_size, claim_size)

# add the match matrix with the second input vector sequence
response = add([match, body_encoded_c])  # (samples, body_size, claim_size)
response = Permute((2, 1))(response)  # (samples, claim_size, body_size)

# concatenate the match matrix with the claim vector sequence
stance = concatenate([response, claim_encoded])

# the original paper uses a matrix multiplication for this reduction step.
# we choose to use a RNN instead.
stance = LSTM(32)(stance)  # shape: (samples, 32)

# one regularization layer -- more would probably be needed.
stance = Dropout(0.3)(stance)
stance = Dense(output_size)(stance)  # (samples, output_size)
# we output a probability distribution over the four stances
preds = Activation('softmax')(stance)

# build the model
model = Model([input_body, input_claim], preds)
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# train
model.fit([train_body, train_claim], train_labels,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=([val_body, val_claim], val_labels))
          
# model.fit([train_body, train_claim], train_labels,
#           batch_size=batch_size,
#           epochs=epochs)
#
# model.predict([val_body, val_claim], batch_size=batch_size)

# print model summary
print(model.summary())