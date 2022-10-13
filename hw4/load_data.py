from preprocessing import create_data
from vectorization import compute_index
import pickle


def load():   # функция пытается открыть сохранённые данные, и если их нет, то вычисляет их и сохраняет
    try:
        with open('raw_corpus.pk', 'rb') as f:
            raw_corpus = pickle.load(f)
        with open('corp_questions.pk', 'rb') as f:
            corp_questions = pickle.load(f)
        with open('sentence_embeddings.pk', 'rb') as f:
            sentence_embeddings = pickle.load(f)

    except FileNotFoundError:
        filepath = input('Введите путь к файлам: ')
        raw_corpus, corp_questions = create_data(filepath, 50000)
        with open('raw_corpus.pk', 'wb') as fin:
            pickle.dump(raw_corpus, fin)
        with open('corp_questions.pk', 'wb') as fin:
            pickle.dump(corp_questions, fin)
        sentence_embeddings, tokenizer, model = compute_index(raw_corpus)
        with open('sentence_embeddings.pk', 'wb') as fin:
            pickle.dump(sentence_embeddings, fin)

    return [raw_corpus, sentence_embeddings, corp_questions]
