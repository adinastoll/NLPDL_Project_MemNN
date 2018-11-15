from __future__ import print_function

from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Reshape, Dropout
from keras.layers import add, dot, multiply, concatenate
from keras.layers import LSTM, Conv1D, TimeDistributed, Lambda
from keras.initializers import Constant
from keras import backend as K
import numpy as np

from dataproc_utils import load_wordvecs, load_proc_data, make_word_freq_V, word2idx, vocab_vectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf



batch_size = 64
epochs = 25
random_state = 42
n_pars = 9 # max number of paragraphs from each document
par_size = 15  # max paragraph length (num of words in each paragraph)
claim_size = 15  # max num of words in each claim
embedding_dim = 25  # should be 100
output_size = 4  # size of the output vector, corresponds to the number of classes

# open saved wordvecs from file
w2v = load_wordvecs('twitter_glo_vecs\\wordvecs25d.txt')
print(len(w2v), 'pretrained embeddings')

# load data and labels
data = load_proc_data('processed_data\\train_bodies.txt', 'processed_data\\train_claims.txt', split_pars=True)
labels = [label for body, claim, label in data]
y = np.array(labels)


# initialize tfidf tokenizer
# fit on concatenated list of bodies and claims
# and transform bodies/pars and claims at the same time
# measure cosine similarity btw tfidf representations of bodies & claims to compute p_tfidf (claim-evidence sim vector)
# tfidf_body shape (len(claims), n_pars, vocab size)
# tfidf_claim shape (len(claims), vocab size)
# p_tfidf output shape: (len(claims), n_pars)

# load pre-computed p_tfidf similarity matrix for train data
p_tfidf = np.loadtxt('processed_data\\p_tfidf_train.txt', dtype=np.float32)
print('Shape of similarity matrix train p_tfidf:', p_tfidf.shape)

# train/validation split
train_data, val_data, train_p_tfidf, val_p_tfidf, train_labels, val_labels = train_test_split(data, p_tfidf, y,
                                                                                              test_size=.2,
                                                                                              random_state=random_state)
#print('First input tuple (body, claim, stance):\n', train_data[0])
#print('First input p_tfidf:\n', train_p_tfidf[0])


# create a vocabulary dict from train data (we exclude rare words, which appear only once)
word2freq = make_word_freq_V(train_data, fmin=2)
word2index = word2idx(word2freq, pretrained=w2v)
vocab_size = len(word2index)
print('Vocab size:', vocab_size, 'unique words in the train set which have glove embeddings')

# vectorize input words (turn each word into its index from the word2index dict)
# for new words in test set that don't appear in train set, use index of <unknown>
train_body, train_claim = vocab_vectorizer(train_data, word2index, max_par_len=par_size, max_claim_len=claim_size)
val_body, val_claim = vocab_vectorizer(val_data, word2index, max_par_len=par_size, max_claim_len=claim_size)


# prepare embedding matrix
embedding_matrix = np.zeros((vocab_size + 1, embedding_dim))
for w, i in word2index.items():
    embedding_matrix[i] = w2v[w]

# load pre-trained word vectors into embedding layers
# we set trainable to false to keep the embeddings fixed
embedding_body = Embedding(vocab_size + 1,
                            embedding_dim,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=(n_pars, par_size,),
                            trainable=False)

embedding_claim = Embedding(vocab_size + 1,
                            embedding_dim,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=claim_size,
                            trainable=False)

# initialize input placeholders and embed pre-trained word vectors
input_body = Input(shape=(n_pars, par_size,), dtype='int32')
input_claim = Input(shape=(claim_size,), dtype='int32')
input_p_tfidf = Input(shape=(n_pars,), dtype='float32')

print('input body', input_body.shape)     # (?, 9, 15)
print('input claim', input_claim.shape)    # (?, 15)
print('input p_tfidf', input_p_tfidf.shape)  # (?, 9)

embedded_body = embedding_body(input_body)
embedded_claim = embedding_claim(input_claim)

print('embedded body', embedded_body.shape)   # (?, 9, 15, 25)
print('embedded claim', embedded_claim.shape)  # (?, 15, 25)

# train two 1D convnets (should be time distributed with maxout layer)
cnn_body = TimeDistributed(Conv1D(100, 5, padding='valid', activation='relu'))(embedded_body)
cnn_body = Lambda(lambda x: K.max(x, axis=-1, keepdims=False))(cnn_body)  # this should be maxout
#cnn_body = Lambda(lambda x: tf.contrib.layers.maxout(x, num_units=1))(cnn_body) ## does not work for some reason!!?

cnn_claim = Conv1D(100, 5, padding='valid', activation='relu')(embedded_claim)
cnn_claim = Lambda(lambda x: K.max(x, axis=-1, keepdims=False))(cnn_claim)  # this should be maxout
#cnn_claim = Lambda(lambda x: tf.contrib.layers.maxout(x, num_units=1))(cnn_claim) ## does not work

# maxout eliminates the last dimension from the cnn representations:
# converts cnn_body with shape (?, 9, 11, 100) to (?, 9, 11)
# and cnn_claim with shape (?, 11, 100) to (?, 11)
print('cnn_body shape', cnn_body.shape)  # (?, 9, 11)
print('cnn_claim shape', cnn_claim.shape)  # (?, 11)


# train two lstms
lstm_body = TimeDistributed(LSTM(100))(embedded_body)
lstm_claim = (LSTM(100))(embedded_claim)

print('lstm body', lstm_body.shape) # (?, 9, 100)
print('lstm claim', lstm_claim.shape) # (?, 100)

# reshape tfidf sim matrix layer from (?, 9) into (?, 9, 1)
reshaped_p_tfidf = Reshape((n_pars, 1))(input_p_tfidf)
lstm_body = multiply([lstm_body, reshaped_p_tfidf])
### tensor shapes: (samples, n_pars, 100) * (samples, n_pars, 1) => (?, 9, 100)
print('lstm_body * p_tfidf', lstm_body.shape)  # (?, 9, 100)
print('lstm_claim', lstm_claim.shape)  # (?, 100)

## p_lstm = lstm_claim.T x M x lstm_body[j]  a.k.a. wtf is M?
## if normalize=True, then the output of the dot product is the cosine similarity between the two samples
p_lstm = dot([lstm_body, lstm_claim], axes=(2, 1), normalize=True)
print('p_lstm', p_lstm.shape)  # (samples, 9)

### cnn_body = cnn_body * p_lstm
# reshape sim matrix layer from (?, 9) into (?, 9, 1)
p_lstm = Reshape((n_pars, 1))(p_lstm)
cnn_body = multiply([cnn_body, p_lstm])
print('cnn_body * p_lstm', cnn_body.shape) # (?, 9, 11)
print('cnn_claim', cnn_claim.shape)        # (?, 11)

## p_cnn = cnn_claim.T x M' x cnn_body[j]  a.k.a. wtf is M'?
## if normalize=True, then the output of the dot product is the cosine similarity between the two samples
p_cnn = dot([cnn_body, cnn_claim], axes=(2, 1), normalize=True)
print('p_cnn', p_cnn.shape)  # (?, 9)


# no clue whats going from here onward
## o = [mean(cnn_body); [max(p_cnn); mean(p_cnn)]; [max(p_lstm); mean(p_lstm)]; [max(p_tfidf); mean(p_tfidf)]]
mean_cnn_body = Lambda(lambda x: K.mean(x, axis=2))(cnn_body)
print('mean cnn body', mean_cnn_body.shape)  # (?, 9)

# taking mean and max similarities
max_p_cnn = Lambda(lambda x: K.max(x, axis=1))(p_cnn)
mean_p_cnn = Lambda(lambda x: K.mean(x, axis=1))(p_cnn)
max_p_lstm = Lambda(lambda x: K.max(x, axis=1))(p_lstm)
mean_p_lstm = Lambda(lambda x: K.mean(x, axis=1))(p_lstm)
max_p_tfidf = Lambda(lambda x: K.max(x, axis=1))(reshaped_p_tfidf)
mean_p_tfidf = Lambda(lambda x: K.mean(x, axis=1))(reshaped_p_tfidf)

# reshape some layers to make their dimensions compatible
max_p_cnn = Reshape((1,))(max_p_cnn)
mean_p_cnn = Reshape((1,))(mean_p_cnn)

output = concatenate([mean_cnn_body,
                      max_p_cnn, mean_p_cnn,
                      max_p_lstm, mean_p_lstm,
                      max_p_tfidf, mean_p_tfidf])

print('output', output.shape)  # (?, 15)

response = concatenate([output, lstm_claim, cnn_claim])
print('response layer:', response.shape)   # (?, 126)

# home stretch
stance = Dense(300, activation='relu')(response)
stance = Dropout(0.3)(stance)
preds = Dense(output_size, activation='softmax')(stance)


# build the model
model = Model([input_body, input_claim, input_p_tfidf], preds)
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# train
model.fit([train_body, train_claim, train_p_tfidf], train_labels,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=([val_body, val_claim, val_p_tfidf], val_labels))

# print model summary
print(model.summary())

