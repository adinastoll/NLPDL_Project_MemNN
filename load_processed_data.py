from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np
from dataproc_utils import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from memn2n import MemN2N

#np.set_printoptions(suppress=True, threshold=np.nan)
tf.app.flags.DEFINE_string('f', '', 'kernel')

tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate for SGD.")
tf.app.flags.DEFINE_float("anneal_rate", 25, "Number of epochs between halving the learnign rate.")
tf.app.flags.DEFINE_float("anneal_stop_epoch", 100, "Epoch number to end annealed lr schedule.")
tf.app.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("evaluation_interval", 5, "Evaluate and print results every x epochs")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.app.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.app.flags.DEFINE_integer("epochs", 50, "Number of epochs to train for.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Embedding size for embedding matrices.")
tf.app.flags.DEFINE_integer("memory_size", 50, "Maximum size of memory.")
tf.app.flags.DEFINE_integer("task_id", 1, "bAbI task id, 1 <= id <= 20")
tf.app.flags.DEFINE_integer("random_state", 42, "Random state.")
tf.app.flags.DEFINE_string("data_dir", "data/tasks_1-20_v1-2/en/", "Directory containing bAbI tasks")
FLAGS = tf.app.flags.FLAGS

memory_size = 9
sentence_size = 20
vocab_size = 4  # size of the output vector


# open saved wordvecs from file and make id dicts
# w2v = open_wordvecs('wordvecs25.txt')
# w2i, i2w, i2v = make_id_dicts(w2v)

# load data and labels
data = load_proc_data('train_bodies.txt', 'train_claims.txt')
labels = [label for body, claim, label in data]
y = label2onehot(labels)
#print(data[0])

# train/validation/test split
train_data, test_data, train_labels, test_labels = train_test_split(data, y, test_size=.2, random_state=FLAGS.random_state)
test_data, val_data, test_labels, val_labels = train_test_split(test_data, test_labels, test_size=.5, random_state=FLAGS.random_state)

# create a vocabulary dict from train data
word2freq = make_word_freq_V(train_data)
word2index = word2idx(word2freq)

# vectorize input words (turn each word into its index from the word2index dict)
# for new words in test set that don't appear in train set, use index of <unknown>
trainS, trainQ = vocab_vectorizer(train_data, word2index, max_par_len=sentence_size, max_claim_len=sentence_size)
valS, valQ = vocab_vectorizer(val_data, word2index, max_par_len=sentence_size, max_claim_len=sentence_size)
testS, testQ = vocab_vectorizer(test_data, word2index, max_par_len=sentence_size, max_claim_len=sentence_size)


print("Training set shape: bodies", trainS.shape)
print("Training set shape: claims", trainQ.shape)
print("Training set shape: labels", train_labels.shape)

# data size params
n_train = trainS.shape[0]
n_val = valS.shape[0]
n_test = testS.shape[0]

print("Training Size", n_train)
print("Validation Size", n_val)
print("Testing Size", n_test)

# the computed predictions are distinct integer values, not one-hot vectors
train_true = np.argmax(train_labels, axis=1)
val_true = np.argmax(val_labels, axis=1)
test_true = np.argmax(test_labels, axis=1)


tf.set_random_seed(FLAGS.random_state)
batch_size = FLAGS.batch_size

batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
batches = [(start, end) for start, end in batches]

with tf.Session() as sess:
    model = MemN2N(batch_size, vocab_size, sentence_size, memory_size,
                   FLAGS.embedding_size,
                   session=sess,
                   hops=FLAGS.hops,
                   max_grad_norm=FLAGS.max_grad_norm)
    for t in range(1, FLAGS.epochs+1):
        # Stepped learning rate
        if t - 1 <= FLAGS.anneal_stop_epoch:
            anneal = 2.0 ** ((t - 1) // FLAGS.anneal_rate)
        else:
            anneal = 2.0 ** (FLAGS.anneal_stop_epoch // FLAGS.anneal_rate)
        lr = FLAGS.learning_rate / anneal

        np.random.shuffle(batches)
        total_cost = 0.0
        for start, end in batches:
            s = trainS[start:end]
            q = trainQ[start:end]
            a = train_labels[start:end]
            cost_t = model.batch_fit(s, q, a, lr)
            total_cost += cost_t

        if t % FLAGS.evaluation_interval == 0:
            train_preds = []
            for start in range(0, n_train, batch_size):
                end = start + batch_size
                s = trainS[start:end]
                q = trainQ[start:end]
                pred = model.predict(s, q)
                train_preds += list(pred)

            val_preds = model.predict(valS, valQ)
            train_acc = accuracy_score(np.array(train_preds), train_true)
            val_acc = accuracy_score(val_preds, val_true)

            print('-----------------------')
            print('Epoch', t)
            print('Total Cost:', total_cost)
            print('Training Accuracy:', train_acc)
            print('Validation Accuracy:', val_acc)
            print('-----------------------')

    test_preds = model.predict(testS, testQ)
    test_acc = accuracy_score(test_preds, test_true)
    print("Testing Accuracy:", test_acc)
