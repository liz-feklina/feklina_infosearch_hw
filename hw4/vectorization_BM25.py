import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def compute_index(corpus):  # функция вычисляет BM25
    k = 2
    b = 0.75
    len_c = len(corpus)

    count_vectorizer = CountVectorizer()
    tf = count_vectorizer.fit_transform(corpus)

    len_d = tf.sum(axis=1)
    avdl = len_d.mean()
    idf = [np.log(len_c / (tf[:, j].count_nonzero())) for j in range(tf.shape[1])]
    den = (k * (1 - b + b * len_d / avdl))  # часть знаменателя

    index = tf.copy()
    for i, j in zip(*tf.nonzero()):
        new = (tf[i, j] * idf[j] * (k + 1)) / (tf[i, j] + den[i])
        index[i, j] = new

    return [index, count_vectorizer]
