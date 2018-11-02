import numpy as np
from dataproc_utils import *


# open saved wordvecs from file and make id dicts
# w2v = open_wordvecs('wordvecs25.txt')
# w2i, i2w, i2v = make_id_dicts(w2v)


data = load_proc_data('train_bodies.txt', 'train_claims.txt')
labels = [label for body, claim, label in data]



word2freq = make_word_freq_V(data)
word2index = word2idx(word2freq)
d, s = vocab_vectorizer(data, word2index)
y = label2onehot(labels)


print(d[0])
print(s[0])
print(y[0])