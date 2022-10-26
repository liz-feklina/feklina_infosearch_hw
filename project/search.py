from preprocessing import lemmatize, compute_index
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def search(request, data, method='tf-idf'):
    answers = data['answers']
    questions = data['questions']
    if method == 'tf-idf':
        index = data['tfidf_index']
        vectorizer = data['tfidf_vectorizer']
        fin_req = ' '.join(lemmatize(request))
        query = vectorizer.transform([fin_req])
    elif method == 'bm-25':
        index = data['bm25_index']
        vectorizer = data['bm25_vectorizer']
        fin_req = ' '.join(lemmatize(request))
        query = vectorizer.transform([fin_req])
    elif method == 'bert':
        index = data['bert_index']
        tokenizer = data['bert_tokenizer']
        model = data['bert_model']
        query = compute_index(request, tokenizer, model)[0]
    else:
        return 0
    results = compute_result(query, index, questions, answers)
    return results


def compute_result(query, index, questions, answers):
    scores = cosine_similarity(index, query)
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    res_answers = np.array(answers)[sorted_scores_indx.ravel()]
    res_questions = np.array(questions)[sorted_scores_indx.ravel()]
    results = [(res_questions[i], res_answers[i]) for i in range(len(res_answers))]
    return results[:50]
