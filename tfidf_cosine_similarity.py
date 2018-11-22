import numpy as np
from dataproc_utils import load_file, parse_proc_claims
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# joins all paragraphs and claims to enable applying the tfidf vectorizer on the whole dataset
def join_pars_claims(data, max_par_num=9):
    n = len(data)
    bodies = [body for body, _, _ in data]
    claims = [claim for _, claim, _ in data]

    padded_bodies = [[[''] for i in range(max_par_num)] for j in range(n)]

    for i, body in enumerate(bodies):
        for j, par in enumerate(body):
            padded_bodies[i][j] = par

    all_pars = [par for body in padded_bodies for par in body]
    all_pars_claims = all_pars + claims

    return all_pars_claims


# computes cosine sim btw each claim and all 9 paragraphs from the corresponding article body
# and stores them in p_tfidf similarity matrix of size (len(claims), max_par_num)
# its best to save the matrix to file to avoid computation which takes about 5 min
def tfidf_cosine_sim(all_tfidf_vecs, max_par_num=9, save_sim_matrix=False, out_filename=None):
    n = all_tfidf_vecs.shape[0] // 10  # number of claim-body pairs

    tfidf_pars = all_tfidf_vecs[:(n * max_par_num)]
    tfidf_claims = all_tfidf_vecs[(n * max_par_num):]

    p_tfidf = np.zeros((n, max_par_num), dtype=np.float32)

    for i in range(n):
        # indicator that it's working
        if i % 1000 == 0:
            print("\rComputing similarities: {0:.2f} % done.".format((i/n) * 100), end='')

        tfidf_c = tfidf_claims[i]

        for j in range(max_par_num):
            tfidf_b = tfidf_pars[i * max_par_num + j]
            p_tfidf[i, j] = cosine_similarity(tfidf_c, tfidf_b)

    if save_sim_matrix:
        np.savetxt(out_filename, p_tfidf, delimiter=' ')

    return p_tfidf



def tfidf_fit_transform(train_data, val_data, test_data):

    # prepare the training data
    train_pars_claims = join_pars_claims(train_data)

    # prepare the validation data
    val_pars_claims = join_pars_claims(val_data)

    # prepare the test data
    test_pars_claims = join_pars_claims(test_data)

    # initialize tfidf vectorizer
    tf_vec = TfidfVectorizer(ngram_range=(1, 1),
                             stop_words=None,
                             strip_accents=None,
                             tokenizer=lambda x: x,
                             lowercase=False)

    # fit-transform the train data and save the computed similarities to txt file
    train_tfidf_vecs = tf_vec.fit_transform(train_pars_claims)

    # it takes 3-4 minutes to compute all similarities for the training data
    train_p_tfidf = tfidf_cosine_sim(train_tfidf_vecs,
                                     save_sim_matrix=True,
                                     out_filename='processed_data\\p_tfidf_train.txt')

    # transform the val data and save the computed similarities to txt file
    val_tfidf_vecs = tf_vec.transform(val_pars_claims)

    val_p_tfidf = tfidf_cosine_sim(val_tfidf_vecs,
                                   save_sim_matrix=True,
                                   out_filename='processed_data\\p_tfidf_val.txt')

    # transform the test data and save the computed similarities to txt file
    test_tfidf_vecs = tf_vec.transform(test_pars_claims)

    test_p_tfidf = tfidf_cosine_sim(test_tfidf_vecs,
                                    save_sim_matrix=True,
                                    out_filename='processed_data\\p_tfidf_test.txt')

    return train_p_tfidf, val_p_tfidf, test_p_tfidf