from string import punctuation
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
import os

PUNCT = punctuation + '-'
morph = MorphAnalyzer()
SW = stopwords.words('russian')


def lemmatize(my_line):  # делает из строки список лемм без стоп-слов
    my_line = my_line.translate(str.maketrans('', '', PUNCT))
    words = my_line.split()
    lem_words = [morph.parse(w)[0].normal_form for w in words]
    filtered = [w for w in lem_words if w not in SW]
    return filtered


def create_data(filepath):  # обрабатываем корпус, проходя по файлам и применяя lemmatize
    corpus = []
    series_list = []
    for root, dirs, files in os.walk(filepath):
        for name in files:
            series_list.append(name[10:14])
            clean_text = ''
            with open(os.path.join(root, name), 'r', encoding='utf-8') as f:
                text = f.read()[1:].splitlines()
                for line in text:
                    clean_text += ' '
                    clean_text += ' '.join(lemmatize(line))
                corpus.append(clean_text[1:])
    return [corpus, series_list]
    # в corpus лежат обработанные документы
    # в series_list номера серий, мы будем пользоваться ими для выдачи поиска
