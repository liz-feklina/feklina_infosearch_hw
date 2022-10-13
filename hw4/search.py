from vectorization import compute_index
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# функция вычисляет вектор запроса, перемножает с индексом и возвращает отсортированные результаты
def search_req(request, sentence_embeddings, raw_corpus):
    query_vec = compute_index([request])[0]
    scores = cosine_similarity(sentence_embeddings, query_vec)
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    fin_res = np.array(raw_corpus)[sorted_scores_indx.ravel()]
    return fin_res
