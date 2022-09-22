from sklearn.feature_extraction.text import TfidfVectorizer


def compute_index(corpus):  # функция вычисляет матрицу tf-idf
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus)
    return [matrix, vectorizer]
