from sklearn.metrics.pairwise import cosine_similarity


# функция вычисляет вектор запроса, косинусную близость и возвращает отсортированные номера серий
def search_req(lem_req, index, vectorizer, series_list):
    ans = vectorizer.transform([lem_req]).toarray()
    cos_sim = cosine_similarity(ans, index)
    all_results = [x for _, x in sorted(zip(cos_sim[0], series_list))]
    all_results.reverse()
    return all_results
