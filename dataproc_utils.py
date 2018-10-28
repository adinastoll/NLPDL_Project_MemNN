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
            paragraphs.append([bid, p])
            
    return paragraphs
    

def df_tolist(df):
    claims = []
    for col, row in df.iterrows():
        bid = row['Body ID']
        line = row['Headline']
        claims.append([bid, line])
            
    return claims

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
    

def trim_paraphs(body_pars, keep_length=9):
    trimmed_pars = []
    
    for par in body_pars:
        new_par = par[1][:keep_length]
        trimmed_pars.append([par[0], new_par])
    return trimmed_pars


def trim_claims(claims, keep_length=10):
    trimmed_claims = []
    
    for c in claims:
        new_claim = c[1][:keep_length]
        trimmed_claims.append([c[0], new_claim])
    return trimmed_claims

    
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
    