from pymorphy2 import MorphAnalyzer
from string import punctuation
from nltk.corpus import stopwords
import os
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from collections import defaultdict
from config import filepath

morph = MorphAnalyzer()
punctuation += '-'
sw = stopwords.words('russian')


def lemmatize(my_line):
    my_line = my_line.translate(str.maketrans('', '', punctuation))
    words = my_line.split()
    lem_words = [morph.parse(w)[0].normal_form for w in words]
    filtered = [w for w in lem_words if w not in sw]
    return filtered


def main():
    corpus = []
    series_list = []

    # проходим по файлам и к каждому построчно применяем функцию, обрабатывающую текст
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
    # в corpus лежат обработанные документы
    # в series_list номера серий, мы будем пользоваться ими позже для наглядности

    # для начала создадим обратный индекс в формате матрицы
    vectorizer = CountVectorizer(analyzer='word')
    matr = vectorizer.fit_transform(corpus)
    wordlist = vectorizer.get_feature_names()
    matrix_freq = np.asarray(matr.sum(axis=0)).ravel()

    # находим самые частые и самые редкие слова
    max_result = np.where(matrix_freq == np.amax(matrix_freq))
    freq_words = []
    for i in max_result[0]:
        freq_words.append(wordlist[i])
    print('Самые частотные слова:\n', *freq_words)
    print('Частотность:\n', matrix_freq[max_result[0][0]])
    print('\n')

    min_result = np.where(matrix_freq == np.amin(matrix_freq))
    unfreq_words = []
    for i in min_result[0]:
        unfreq_words.append(wordlist[i])
    print('Самые редкие слова:\n', *unfreq_words[:10], '...', *unfreq_words[-10:])
    print('Таких слов:\n', len(unfreq_words))
    print('Частотность:\n', matrix_freq[min_result[0][0]])
    print('\n')

    # транспонируем, чтобы было легче искать слова, которые есть во всех документах
    matr_t = matr.toarray().transpose()
    uni_words = []
    for i in range(np.shape(matr_t)[0]):
        if 0 not in matr_t[i]:
            uni_words.append(wordlist[i])
    print('Слова, которые есть во всех документах:', *uni_words, '\n')

    # определяем частотность имён
    print('Моника:', matrix_freq[vectorizer.vocabulary_.get('мона')] +
          matrix_freq[vectorizer.vocabulary_.get('моника')])
    print('Рэйчел:', matrix_freq[vectorizer.vocabulary_.get('рейчел')] +
          matrix_freq[vectorizer.vocabulary_.get('рэйчел')])
    print('Чендлер:', matrix_freq[vectorizer.vocabulary_.get('чендлер')] +
          matrix_freq[vectorizer.vocabulary_.get('чендлера')])
    print('Росс:', matrix_freq[vectorizer.vocabulary_.get('росс')])
    print('Фиби:', matrix_freq[vectorizer.vocabulary_.get('фиби')] +
          matrix_freq[vectorizer.vocabulary_.get('фибс')])
    print('Джоуи:', matrix_freq[vectorizer.vocabulary_.get('джой')] +
          matrix_freq[vectorizer.vocabulary_.get('джо')] +
          matrix_freq[vectorizer.vocabulary_.get('джоуя')] +
          matrix_freq[vectorizer.vocabulary_.get('джоо')])
    print('\n\n')

    # теперь создадим обратный индекс-словарь
    # ключ - слово, значение - список документов (с повторами, чтобы можно было считать частотность)
    i_dict = defaultdict(list)
    for i in range(len(corpus)):
        for word in corpus[i].split():
            i_dict[word].append(series_list[i])

    freq = []
    for key in i_dict:
        freq.append((key, len(i_dict[key])))
    freq = sorted(freq, key=lambda x: x[1], reverse=True)
    print('Список слов по убыванию частотности:', *freq[:10], '...', *freq[-10:])

    uni_words_2 = []
    for key in i_dict:
        if len(set(i_dict[key])) == 162:
            uni_words_2.append((key, len(set(i_dict[key]))))
    print('Слова, которые есть во всех документах:', *uni_words_2, '\n')

    # определяем частотность имён
    print('Моника:', len((i_dict['мона'])) + len((i_dict['моника'])))
    print('Рэйчел:', len((i_dict['рейчел'])) + len((i_dict['рэйчел'])))
    print('Чендлер:', len((i_dict['чендлер'])) + len((i_dict['чендлера'])))
    print('Росс:', len((i_dict['росс'])))
    print('Фиби:', len((i_dict['фиби'])) + len((i_dict['фибс'])))
    print('Джоуи:', len((i_dict['джой'])) + len((i_dict['джо'])) + len((i_dict['джоуя'])) + len((i_dict['джоо'])))


if __name__ == '__main__':
    main()
