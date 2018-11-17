import numpy as np
import pandas as pd
import nltk
import re
import string
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def split_paraphs(df):
    paragraphs = []
    
    for col, row in df.iterrows():
        bid = row['Body ID']
        body = row['articleBody']
        paras = body.split('\n\n')
        for p in paras:
            paragraphs.append((bid, p))
            
    return paragraphs

    
def stance_toint(df):
    df.loc[df['Stance'] == 'unrelated', 'Stance'] = 0
    df.loc[df['Stance'] == 'discuss', 'Stance'] = 1
    df.loc[df['Stance'] == 'agree', 'Stance'] = 2
    df.loc[df['Stance'] == 'disagree', 'Stance'] = 3
    return df

    
def get_claims_labels(df):
    claims = []
    labels = []
    
    for col, row in df.iterrows():
        bid = row['Body ID']
        line = row['Headline']
        claims.append((bid, line))  

        stance = row['Stance']
        labels.append(stance)      
        
    return claims, labels
     

def remove_nonascii(lines):
    ascii_chars = set(string.printable)
    processed = [[bid, ''.join(filter(lambda x: x in ascii_chars, s))] for bid, s in lines]
    return processed
    
    
def replace_pattern(old_pat, new_pat, lines):
    compiled_old = re.compile(old_pat)
    processed = [[bid, re.sub(compiled_old, new_pat, s)] for bid, s in lines]
    
    return processed
   

def tokenize_lines(lines):
    tokenized = [[bid, nltk.word_tokenize(s.lower())] for bid, s in lines]
    return tokenized

    
def replace_pattern_tokenized(old_pat, new_pat, tokens):
    compiled_old = re.compile(old_pat)
    processed = [[bid, [re.sub(compiled_old, new_pat, tok) for tok in line]] for bid, line in tokens]
    return processed
       

def trim_bodies(body_pars, keep_count=9, keep_length=None):
    paragraph_dict = {}
    count_dict = {}
    
    for bid, par in body_pars:
        if keep_length:
            par = par[1][:keep_length]
        
        if bid not in count_dict:
            count_dict[bid] = 0
            paragraph_dict[bid] = []
        
        if count_dict[bid] < keep_count:
            paragraph_dict[bid].append((bid, par))
            count_dict[bid] = count_dict[bid] + 1
                
    trimmed_pars = []  
    for k, v in paragraph_dict.items():
        trimmed_pars += v
        
    return trimmed_pars


def trim_claims(claims, keep_length=12):
    trimmed_claims = []
    
    for c in claims:
        new_claim = c[1][:keep_length]
        trimmed_claims.append([c[0], new_claim])
    return trimmed_claims


def save_proc_bodies(filename, bodies):
    all_pars = [str(bid) + ' ' + ' '.join(p) for bid, p in bodies if len(p) > 0]
    txt_file = '\n'.join(all_pars)
    
    with open(filename, 'w') as f:
        f.write(txt_file)


def save_proc_claims(filename, claims, labels):
    all_claims = []

    for i in range(len(labels)):
        bid = claims[i][0]
        claim = claims[i][1]
        label = labels[i]
        all_claims.append(str(bid) + ' ' + ' '.join(claim) + ' ' + str(label))
        
    txt_file = '\n'.join(all_claims)
    
    with open(filename, 'w') as f:
        f.write(txt_file)
        
        
def parse_proc_bodies(all_bodies):
    bodies = []

    for line in all_bodies:
        line = line.strip().split()
        bid = int(line[0])
        par = line[1:]
        bodies.append((bid, par))
    return bodies
    
 
def parse_proc_bodies_dict(all_bodies, split_pars=True, tokenize=True):
    bid2pars = {}

    for line in all_bodies:
        line = line.strip().split()
        bid = int(line[0])
        par = line[1:]

        if tokenize is False:
            par = ' '.join(par)

        if bid in bid2pars:
            if split_pars:
                bid2pars[bid].append(par)
            else:
                bid2pars[bid].extend(par)
        else:
            if split_pars:
                bid2pars[bid] = [par]
            else:
                bid2pars[bid] = par
    return bid2pars

    
def parse_proc_claims(all_claims):
    claims = []
    labels = []

    for line in all_claims:
        line = line.strip().split()
        bid = int(line[0])
        claim = line[1:-1]
        label = int(line[-1])
        claims.append((bid, claim))
        labels.append(label)
            
    return claims, labels


def load_file(filename):
    with open(filename) as f:
        lines = f.readlines()

    return lines


def load_proc_data(bodies_filename, claims_filename, split_pars=True):
    all_bodies = load_file(bodies_filename)
    all_claims = load_file(claims_filename)

    b2p = parse_proc_bodies_dict(all_bodies, split_pars=split_pars)

    data = []
    for line in all_claims:
        line = line.strip().split()
        bid = int(line[0])
        claim = line[1:-1]
        label = int(line[-1])

        if bid in b2p:
            data.append((b2p[bid], claim, label))

    return data


def make_word_freq_V(data, fmin=None):
    V = {'<unknown>': 0}

    for b, c, _ in data:
        for par in b:
            for word in par:
                V[word] = V.get(word, 0) + 1

        for word in c:
            V[word] = V.get(word, 0) + 1

    if fmin is not None:
        most_freq_V = {'<unknown>': 0}

        for word, count in V.items():
            if count < fmin:
                most_freq_V['<unknown>'] += 1
            else:
                most_freq_V[word] = count

        V = most_freq_V

    return V


def word2idx(vocab, pretrained=None):
    if pretrained is None:
        word_idx = {w: i+1 for i, w in enumerate(vocab)}
    else:
        word_idx = {w: i+1 for i, w in enumerate(vocab.keys() & pretrained.keys())}

    return word_idx


def make_V(body_pars, claims):
    V = {}
    for par in body_pars:
        for word in par[1]:
            V[word] = V.get(word, 0) + 1
            
    for line in claims:
        for word in line[1]:
            V[word] = V.get(word, 0) + 1
            
    return V

    
def remove_rare(V_dict, fmin=2):
    most_freq_V = {'<unknown>': 0}

    for k, v in V_dict.items():
        if v < fmin:
            most_freq_V['<unknown>'] += 1
        else:
            most_freq_V[k] = v
    return most_freq_V

    
def remove_placeholder_keys(V_dict, old_keys, new_keys):
    for i in range(len(new_keys)):
        old_k = old_keys[i]
        if old_k in V_dict:
            V_dict[new_keys[i]] = V_dict[old_k]
            del V_dict[old_k]
    return V_dict

    
def extract_wordvecs(filename, V_dict):
    vec_dict = {}
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            word = line[0]
            if word in V_dict:
                vec_dict[word] = line[1:]
    return vec_dict
    
    
def write_wordvecs_tofile(filename, vec_dict):
    txt_file = '\n'.join([k + ' ' + ' '.join(v) for k, v in vec_dict.items()])
    
    with open(filename, 'w') as f:
        f.write(txt_file)

        
def load_wordvecs(filename):
    w2v = {}
    with open(filename) as f:
        for line in f:
            line = line.strip().split()
            w2v[line[0]] = [float(x) for x in line[1:]]
    return w2v

        
def make_id_dicts(k2v_dict):
    k2i, i2k, i2v, i = {}, {}, {}, 0
    
    for k, v in k2v_dict.items():
        k2i[k] = i
        i2k[i] = k
        i2v[i] = v
        i += 1
    return k2i, i2k, i2v


def vocab_vectorizer(data, w2i, max_par_num=9, max_par_len=30, max_claim_len=30):
    nclaims = len(data)

    d = np.zeros((nclaims, max_par_num, max_par_len), dtype=np.int32)
    s = np.zeros((nclaims, max_claim_len), dtype=np.int32)

    for i in range(nclaims):
        max_npars = max_par_num
        max_claim_length = max_claim_len

        body, claim, _ = data[i]
        npars = len(body)

        if npars < max_npars:
            npars, max_npars = max_npars, npars

        for j in range(max_npars):
            max_par_length = max_par_len
            par = body[j]
            par_len = len(par)

            if par_len < max_par_length:
                par_len, max_par_length = max_par_length, par_len

            for k in range(max_par_length):
                pword = par[k]

                if pword in w2i:
                    d[i, j, k] = w2i[pword]
                else:
                    d[i, j, k] = w2i['<unknown>']

        claim_len = len(claim)
        if claim_len < max_claim_length:
            claim_len, max_claim_length = max_claim_length, claim_len

        for m in range(max_claim_length):
            cword = claim[m]

            if cword in w2i:
                s[i, m] = w2i[cword]
            else:
                s[i, m] = w2i['<unknown>']

    return d, s


def word_vectorizer(data, w2i, max_body_len=30, max_claim_len=12):
    nclaims = len(data)

    d = np.zeros((nclaims, max_body_len), dtype=np.int32)
    s = np.zeros((nclaims, max_claim_len), dtype=np.int32)

    for i in range(nclaims):
        body_len = max_body_len
        claim_len = max_claim_len

        body, claim, _ = data[i]
        nwords_body = len(body)

        if nwords_body < body_len:
            nwords_body, body_len = body_len, nwords_body

        for j in range(body_len):
            pword = body[j]

            if pword in w2i:
                d[i, j] = w2i[pword]
            else:
                d[i, j] = w2i['<unknown>']

        nwords_claim = len(claim)
        if nwords_claim < claim_len:
            nwords_claim, claim_len = claim_len, nwords_claim

        for k in range(claim_len):
            cword = claim[k]

            if cword in w2i:
                s[i, k] = w2i[cword]
            else:
                s[i, k] = w2i['<unknown>']

    return d, s



def label2onehot(labels):
    n = len(labels)
    onehot_labels = np.zeros((n, 4), dtype=np.int32)

    for i in range(n):
        label = labels[i]
        onehot_labels[i, label] = 1

    return onehot_labels


def random_sampler(X_body, X_claim, X_p_tfidf, y, type='under', random_state=42):

    if type == 'under':
        rs = RandomUnderSampler(random_state=random_state)
    elif type == 'over':
        rs = RandomOverSampler(random_state=random_state)
    else:
        raise ValueError('Incorrect sampler type.')

    body_shape = X_body.shape
    if len(body_shape) > 2:
        n, m, s = body_shape
        X_body = X_body.reshape((n, -1))

        X = np.hstack((X_body, X_claim, X_p_tfidf))
        X_resampled, y_resampled = rs.fit_resample(X, y)

        X_body_resampled = X_resampled[:, :(m * s)].reshape((-1, m, s))
        X_claim_resampled = X_resampled[:, (m * s): -m]
        X_p_tfidf_resampled = X_resampled[:, -m:]

    else:
        n, m = body_shape

        X = np.hstack((X_body, X_claim, X_p_tfidf))
        X_resampled, y_resampled = rs.fit_resample(X, y)

        X_body_resampled = X_resampled[:, :m]
        X_claim_resampled = X_resampled[:, m:-m]
        X_p_tfidf_resampled = X_resampled[:, -m:]

    return X_body_resampled, X_claim_resampled, X_p_tfidf_resampled, y_resampled

