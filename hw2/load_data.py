from preprocessing import create_data
from vectorization import compute_index
import numpy as np
import pickle


def load():   # функция пытается открыть сохранённые данные, и если их нет, то вычисляет их и сохраняет
    try:
        index = np.load('index.npy')
        with open('vectorizer.pk', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('series_list.pk', 'rb') as f:
            series_list = pickle.load(f)
    except FileNotFoundError:
        filepath = input('Введите путь к файлам: ')
        corpus, series_list = create_data(filepath)
        with open('corpus.pk', 'wb') as fin:
            pickle.dump(corpus, fin)
        with open('series_list.pk', 'wb') as fin:
            pickle.dump(series_list, fin)
        index, vectorizer = compute_index(corpus)
        np.save('index.npy', index.toarray())
        with open('vectorizer.pk', 'wb') as fin:
            pickle.dump(vectorizer, fin)
    return [index, vectorizer, series_list]
