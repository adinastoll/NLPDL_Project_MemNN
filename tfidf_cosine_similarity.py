import numpy as np
from dataproc_utils import load_file, parse_proc_bodies_dict, parse_proc_claims
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# joins all paragraphs and claims to enable applying the tfidf vectorizer on the whole dataset
def join_pars_claims(bodies_file, claims_file, max_par_num=9):
    bodies = load_file(bodies_file)
    claims = load_file(claims_file)
    b2p = parse_proc_bodies_dict(bodies, tokenize=False)
    bid_claim, _ = parse_proc_claims(claims)
    n_bodies = len(b2p)

    # create body id to body index dict
    b2i = {bid: i for i, bid in enumerate(b2p.keys())}
    all_bodies = [['' for i in range(max_par_num)] for j in range(n_bodies)]

    for bid, pars in b2p.items():
        for i in range(len(pars)):
            all_bodies[b2i[bid]][i] = pars[i]

    all_pars = [par for body in all_bodies for par in body]

    # create claim index to body id dict
    ci2bid = {ci: bid[0] for ci, bid in enumerate(bid_claim)}
    all_claims = [' '.join(c) for b, c in bid_claim]
    all_pars_claims = all_pars + all_claims

    return all_pars_claims, b2i, ci2bid


# computes cosine sim btw each claim and all 9 paragraphs from the corresponding article body
# and stores them in p_tfidf similarity matrix of size (len(claims), max_par_num)
# its best to save the matrix to file to avoid computation which takes about 5 min
def tfidf_cosine_sim(all_tfidf_vecs, b2i, ci2bid, max_par_num=9, save_sim_matrix=False, p_tfidf_filename=None):
    n_bodies = len(b2i)
    n_claims = len(ci2bid)

    tfidf_pars = all_tfidf_vecs[:(n_bodies * max_par_num)]
    tfidf_claims = all_tfidf_vecs[(n_bodies * max_par_num):]

    p_tfidf = np.zeros((n_claims, max_par_num), dtype=np.float32)

    for i in range(n_claims):
        tfidf_c = tfidf_claims[i]
        b_id = ci2bid[i]
        b_ind = b2i[b_id]

        for j in range(max_par_num):
            tfidf_b = tfidf_pars[b_ind + j]
            p_tfidf[i, j] = cosine_similarity(tfidf_c, tfidf_b)

    if save_sim_matrix:
        np.savetxt(p_tfidf_filename, p_tfidf, delimiter=' ')

    return p_tfidf



train_pars_claims, train_b2i, train_ci2bid = join_pars_claims('processed_data\\train_bodies.txt',
                                                              'processed_data\\train_claims.txt')

# initialize tfidf vectorizer
tf_vec = TfidfVectorizer(ngram_range=(1, 1),
                         stop_words=None,
                         strip_accents=None,
                         tokenizer=str.split,
                         lowercase=False)

# fit-transform the train data and save the computed similarities to txt file
train_tfidf_vecs = tf_vec.fit_transform(train_pars_claims)

# it takes about 5 minutes to compute all similarities for the training data
train_p_tfidf = tfidf_cosine_sim(train_tfidf_vecs, train_b2i, train_ci2bid,
                                 save_sim_matrix=True, p_tfidf_filename='processed_data\\p_tfidf_train.txt')


# to do later when we have processed the test data!
test_pars_claims, test_b2i, test_ci2bid = join_pars_claims('processed_data\\test_bodes.txt',
                                                           'processed_data\\test_claims.txt')

# transform the test data and save the computed similarities to txt file
test_tfidf_vecs = tf_vec.transform(test_pars_claims)

test_p_tfidf = tfidf_cosine_sim(test_tfidf_vecs, test_b2i, test_ci2bid,
                                save_sim_matrix=True, p_tfidf_filename='processed_data\\p_tfidf_test.txt')