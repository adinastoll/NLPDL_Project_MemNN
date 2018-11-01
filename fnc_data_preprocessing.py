import re
import string
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from dataproc_utils import *
#pd.set_option('max_colwidth', 400)


# load and split document bodies into paragraphs
bodies = pd.read_csv('data/train_bodies.csv')   
pars = split_paraphs(bodies)

# load and convert stances into a list of claims
stances = pd.read_csv('data/train_stances.csv')
stances = stance_toint(stances)
claims, labels = get_claims_labels(stances)


# remove all non-ascii characters
proc_pars = remove_nonascii(pars)
proc_claims = remove_nonascii(claims)


# replace all links, twitter handles, hashtags, and remove punctuation
old_patterns = [r'(https?:)?//\S+\b|www\.(\w+\.)+\S*',
                r'pic.twitter\S*',
                r'@\w+',
                r'#\w+',
                '[%s]' % re.escape(',.;:?!"#$%&()*+/<=>@[\]^_`{|}~')]
                
new_patterns = ['urlzz', 'urlzz', 'userzz', 'hashtagzz', '']


for i in range(len(old_patterns)):
    op = old_patterns[i]
    np = new_patterns[i]  
    proc_pars = replace_pattern(op, np, proc_pars)
    proc_claims = replace_pattern(op, np, proc_claims)


# convert to lowercase and tokenize
proc_pars = tokenize_lines(proc_pars)
proc_claims = tokenize_lines(proc_claims)

# convert all numbers to number token
number = r'\d+[,\.]?\d*,?'
proc_pars = replace_pattern_tokenized(number, r'<number>', proc_pars)
proc_claims = replace_pattern_tokenized(number, r'<number>', proc_claims)

# remove hyphens and single quotes, but leave contraction apostrophes
quot_mark = r"'\s|'(?!(m\b|t\b|ll\b|d\b|s\b|re\b))|-+$"
proc_pars = replace_pattern_tokenized(quot_mark, '', proc_pars)
proc_claims = replace_pattern_tokenized(quot_mark, '', proc_claims)

# trim bodies, keep at most 9 paragraphs from each body
trimmed_bodies = trim_bodies(proc_pars)

# uncomment to save preprocessed data to files
# save_proc_bodies('train_bodies.txt', trimmed_bodies)
# save_proc_claims('train_claims.txt', proc_claims, labels)


# construct vocabulary
V = make_V(trimmed_bodies, proc_claims)
print(len(V))

# create a new dictionary where all words with frequency 1
# are converted to <unknown> token
V_freq = remove_rare(V, fmin=2)
print(len(V_freq))
        
        
# remove old placeholder keys and insert new ones that have a corresponding glove vec        
new = ['<url>', '<user>', '<hashtag>']   
old = ['urlzz', 'userzz', 'hashtagzz']

V_freq = remove_placeholder_keys(V_freq, old, new)
        
# extract glove vecs that correspond to the words in our vocabulary
w2v = extract_wordvecs('glove_twitter/glove.twitter.27B.25d.txt', V_freq)

# uncomment to save the extracted word vecs to a file
# write_wordvecs_tofile('wordvecs25.txt', w2v)
