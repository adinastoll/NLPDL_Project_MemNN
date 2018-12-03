## Memory Network Architecture

**Input:**  
1. a document (article body) segmented into paragraphs (potential pieces of evidence)  
2. a textual statement containing a claim (article headline)

**Output:**  
1. the stance of a document with respect to the corresponding claim
- *agree*, *disagree*, *discuss*, *unrelated*

**Inference Outputs:**  
1. *k* most similar paragraphs with their similarity scores  
2. *k* most similar snippets with their similarity scores

-----------------------------------------

#### 1. Input Encoding / Vectorization

**Dense Representation:** word embeddings pre-trained on Twitter data (GloVe)
```
dense body  (n_samples, n_paragraphs=9, max_paragraph_len=15, embedding_dim=100)
dense claim (n_samples, max_claim_len=15, embedding_dim=100)
```

**Sparse Representation:** term frequency–inverse document frequency
```
sparse body  (n_samples, n_paragraphs=9, vocab_size)
sparse claim (n_samples, vocab_size)
```

#### 2. Memory Representation
```
dense body ---> TimeDistributed (LSTM, 100 units) -----------> lstm body (n_samples, 9, 100)
dense body ---> TimeDistributed (CNN, 100 filters, size 5) --> cnn body  (n_samples, 9, 11, 100)
cnn body -----> MaxOut --------------------------------------> cnn body  (n_samples, 9, 11)

dense claim --> TimeDistributed (LSTM, 100 units) -----------> lstm claim (n_samples, 100)
dense claim --> TimeDistributed (CNN, 100 filters, size 5) --> cnn claim  (n_samples, 11, 100)
cnn claim ----> MaxOut --------------------------------------> cnn claim  (n_samples, 11)
```

#### 3. Inference and Generalization
```
sparse body x sparse claim ---> p tfidf (n_samples, 9)  # similarity matrix
lstm body * p tfidf ----------> lstm body               # memory update
lstm body x lstm claim -------> p lstm (n_samples, 9)   # similarity matrix
cnn body * p lstm ------------> cnn body                # memory update
cnn body x cnn claim ---------> p cnn (n_samples, 9)    # similarity matrix
```

#### 4. Output Memory Representation
```
concatenate [ mean(cnn body),
	      max(p cnn), mean(p cnn),
	      max(p lstm), mean(p lstm),
	      max(p tfidf), mean(p tfidf) ] --> output
```

#### 5. Final Response (Class Prediction)
```
concatenate [ output, lstm claim, cnn claim ] --> response
response ---> MLP (300 units, relu) ------------> response
response ---> DropOut (0.5) --------------------> response
response ---> MLP (4 units, softmax) -----------> prediction
```

#### 6. Inference Outputs
- a set of evidences (paragraphs) with similarity scores
- a set of snippets from the most similar paragraph with similarity scores
