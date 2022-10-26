import pickle
from preprocessing import lemmatize
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def create_lem_corp(data):
    lem_data = []
    for line in data:
        lem_data.append(lemmatize(line))
    return lem_data


def index_tfidf(corpus):
    len_c = len(corpus)

    count_vectorizer = CountVectorizer()
    tf = count_vectorizer.fit_transform(corpus)
    idf = [np.log(len_c / (tf[:, j].count_nonzero())) for j in range(tf.shape[1])]

    index = tf.copy()
    for i, j in zip(*tf.nonzero()):
        new = tf[i, j] * idf[j]
        index[i, j] = new

    return [index, count_vectorizer]


def index_bm25(corpus):
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


def index_bert():
    pass


with open('.\\data\\answers.pk', 'rb') as f:
    answers = pickle.load(f)
with open('.\\data\\questions.pk', 'rb') as f:
    questions = pickle.load(f)

'''
import torch
import os

# lem_answers = create_lem_corp(answers)
# with open('.\\data\\lem_answers.pk', 'wb') as fin:
    # pickle.dump(lem_answers, fin)

with open('.\\data\\lem_answers.pk', 'rb') as f:
    lem_answers = pickle.load(f)

# new_lem_answers = []
# for answer in lem_answers:
    # new = ' '.join(answer)
    # new_lem_answers.append(new)
# with open('.\\data\\lem_answers.pk', 'wb') as fin:
    # pickle.dump(new_lem_answers, fin)

tfidf_index, tfidf_vectorizer = index_tfidf(lem_answers)
bm25_index, bm25_vectorizer = index_bm25(lem_answers)
all_data = [tfidf_index, tfidf_vectorizer, bm25_index, bm25_vectorizer]
names = ['tfidf_index', 'tfidf_vectorizer', 'bm25_index', 'bm25_vectorizer']
for a in range(len(all_data)):
    print(names[a], type(all_data[a]))
    with open(f'.\\data\\{names[a]}.pk', 'wb') as fin:
        pickle.dump(all_data[a], fin)


def unite_data(directory):
    for root, dirs, files in os.walk('.\\data\\embs\\' + directory):
        for name in files:
            print(name)
            with open(f'.\\data\\embs\\{directory}\\{name}', 'rb') as file:
                new_piece = pickle.load(file)
                try:
                    emb = torch.cat((emb, new_piece))
                except NameError:
                    emb = new_piece
    return emb


answers_embeddings = unite_data('ans')
with open('.\\data\\answers_embeddings.pk', 'wb') as fin:
    pickle.dump(answers_embeddings, fin)
'''
