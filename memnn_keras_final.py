from __future__ import print_function

from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout
from keras.layers import add, dot, concatenate
from keras.layers import LSTM, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.initializers import Constant
import numpy as np

from dataproc_utils import *
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

batch_size = 64
epochs = 25
random_state = 42
body_size = 20
claim_size = 12
embedding_dim = 25
output_size = 4  # size of the output vector

# open saved wordvecs from file and make id dicts
w2v = load_wordvecs('twitter_glo_vecs\\wordvecs25d.txt')
print(len(w2v), 'pretrained embeddings')

# load data and labels
data = load_proc_data('train_bodies.txt', 'train_claims.txt', split_pars=False)
labels = [label for body, claim, label in data]
y = np.array(labels)

print('First input tuple (body, claim, stance):\n', data[0])

# train/validation split
train_data, val_data, train_labels, val_labels = train_test_split(data, y, test_size=.2, random_state=random_state)


# create a vocabulary dict from train data
word2freq = make_word_freq_V(train_data)
word2index = word2idx(word2freq, pretrained=w2v)

vocab_size = len(word2index)
print('Vocab size:', vocab_size, 'unique words in the train set')

# vectorize input words (turn each word into its index from the word2index dict)
# for new words in test set that don't appear in train set, use index of <unknown>
train_body, train_claim = word_vectorizer(train_data, word2index, max_body_len=body_size, max_claim_len=claim_size)
val_body, val_claim = word_vectorizer(val_data, word2index, max_body_len=body_size, max_claim_len=claim_size)


# prepare embedding matrix
embedding_matrix = np.zeros((vocab_size + 1, embedding_dim))
for w, i in word2index.items():
    #print(i, w2v[w])
    embedding_matrix[i] = w2v[w]


# initialize tfidf tokenizer
# fit on concatenated list of bodies and claims
# transform bodies and claims separately
# measure cosine similarity btw tfidf representations of bodies & claims to compute p_tfidf (claim-evidence sim vector)
# output shape (len(texts), nb_words)


# load pre-trained word vectors into embedding layers
# we set trainable to false to keep the embeddings fixed
embedding_body = Embedding(vocab_size + 1,
                            embedding_dim,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=body_size,
                            trainable=False)

embedding_claim = Embedding(vocab_size + 1,
                            embedding_dim,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=claim_size,
                            trainable=False)

# initialize placeholders and embed pre-trained word vectors
input_body = Input(shape=(body_size,), dtype='int32')
input_claim = Input(shape=(claim_size,), dtype='int32')
embedded_body = embedding_body(input_body)
embedded_claim = embedding_claim(input_claim)

# train two 1D convnets with maxpooling (should be maxout (??))
cnn_body = Conv1D(100, 5, padding='same', activation='relu')(embedded_body)
cnn_body = MaxPooling1D(5, padding='same')(cnn_body)

cnn_claim = Conv1D(100, 5, padding='same', activation='relu')(embedded_claim)
cnn_claim = MaxPooling1D(5, padding='same')(cnn_claim)

# train two lstms
lstm_body = LSTM(100)(embedded_body)
lstm_claim = LSTM(100)(embedded_claim)

### lstm_body = lstm_body * p_tfidf (multiply??)

p_lstm = dot([lstm_body, lstm_claim], axes=(2, 2))
p_lstm = Activation('softmax')(p_lstm)  # shape: (samples, body_size, claim_size)

### cnn_body = cnn_body * p_lstm (multiply??)

p_cnn = dot([lstm_body, lstm_claim], axes=(2, 2))
p_cnn = Activation('softmax')(p_cnn)  # shape: (samples, body_size, claim_size)

output = ###

output = Dense(128, activation='relu')(output)
preds = Dense(output_size, activation='softmax')(output)


# build the model
model = Model([input_body, input_claim], preds)
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# train
model.fit(train_body, train_labels,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(val_body, val_labels))

# print model summary
print(model.summary())