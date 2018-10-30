import numpy as np
import pandas as pd
import nltk
import re
import string


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


def trim_claims(claims, keep_length=10):
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
        
        
def open_proc_bodies(filename):
    bodies = []
    with open(filename) as f:
        for line in f:
            line = line.strip().split()
            bid = int(line[0])
            par = line[1:]
            bodies.append((bid, par))
    return bodies

    
def open_proc_claims(filename):
    claims = []
    labels = []
    
    with open(filename) as f:
        for line in f:
            line = line.strip().split()
            bid = int(line[0])
            claim = line[1:-1]
            label = int(line[-1])
            claims.append((bid, claim))
            labels.append(label)
            
    return claims, labels
    
    
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

        
def open_wordvecs(filename):
    w2v = {}
    with open(filename) as f:
        for line in f:
            line = line.strip().split()
            w2v[line[0]] = line[1:]
    return w2v

        
def make_id_dicts(word2vec_dict):
    w2i, i2w, i2v, i = {}, {}, {}, 0
    
    for k, v in word2vec_dict.items():
        w2i[k] = i
        i2w[i] = k
        i2v[i] = v
        i += 1
    return w2i, i2w, i2v
    