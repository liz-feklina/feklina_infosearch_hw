# функция вычисляет вектор запроса, перемножает с индексом и возвращает отсортированные результаты
def search_req(lem_req, index, vectorizer):
    query = vectorizer.transform([lem_req]).transpose()
    results = index * query
    values = [(results[i, 0], i) for i in results.nonzero()[0]]
    fin_res = sorted(values, reverse=True)
    return fin_res
