import numpy as np
from collections import Counter


def compute_tf(text):
    word_count = Counter(text.split())
    tf = {}
    for word, count in word_count.items():
        tf[word] = count / len(word_count)
    return tf


def compute_idf(documents):
    word_count = Counter()
    for document in documents:
        word_count.update(set(document.split()))
    idf = {}
    for word, count in word_count.items():
        idf[word] = np.log(len(documents) / count)
    return idf


def compute_tfidf(text, idf):
    tf = compute_tf(text)
    tfidf = {}
    for word, val in tf.items():
        if word in idf:
            tfidf[word] = val * idf[word]
        else:
            tfidf[word] = 0
    return tfidf


def compute_corpus_tfidf(corpus):
    idf = compute_idf(corpus)
    tfidf_corpus = [compute_tfidf(text, idf) for text in corpus]
    return tfidf_corpus


print("IDF: ", compute_corpus_tfidf("I love you nirmala"))
