from load_data import load
import pickle
import torch
import numpy as np
from preprocessing_BM25 import create_data
from vectorization_BM25 import compute_index


# Берт
with open('question_embeddings.pk', 'rb') as f:
    question_embeddings = pickle.load(f)
raw_corpus, sentence_embeddings, corp_questions = load()
scores = torch.matmul(sentence_embeddings, torch.transpose(question_embeddings, 0, 1))
s = 0
for i in range(len(scores)):
    sorted_scores_indx = np.argsort(scores[i].numpy(), axis=0)[::-1][:5]
    if i in sorted_scores_indx.ravel():
        s += 1
print(s/len(sentence_embeddings), '- значение метрики для Берта')


# BM25
raw_corpus, questions_names, corpus, questions = create_data('data.jsonl', 10000)
index, vectorizer = compute_index(corpus)
ques_index = vectorizer.transform(questions).transpose()
scores = (index * ques_index).toarray()
s = 0
for i in range(len(scores)):
    sorted_scores_indx = np.argsort(scores[i], axis=0)[::-1][:5]
    if i in sorted_scores_indx.ravel():
        s += 1
print(s/len(scores), '- значение метрики для BM25')
