from preprocessing import create_data
from vectorization import compute_index
import pickle


def load():   # функция пытается открыть сохранённые данные, и если их нет, то вычисляет их и сохраняет
    try:
        with open('index.pk', 'rb') as f:
            index = pickle.load(f)
        with open('vectorizer.pk', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('corpus.pk', 'rb') as f:
            corpus = pickle.load(f)
        with open('raw_corpus.pk', 'rb') as f:
            raw_corpus = pickle.load(f)

    except FileNotFoundError:
        filepath = input('Введите путь к файлам: ')
        raw_corpus, corpus = create_data(filepath, 50000)
        with open('corpus.pk', 'wb') as fin:
            pickle.dump(corpus, fin)
        with open('raw_corpus.pk', 'wb') as fin:
            pickle.dump(raw_corpus, fin)
        index, vectorizer = compute_index(corpus)
        with open('index.pk', 'wb') as fin:
            pickle.dump(index, fin)
        with open('vectorizer.pk', 'wb') as fin:
            pickle.dump(vectorizer, fin)

    return [raw_corpus, corpus, index, vectorizer]
