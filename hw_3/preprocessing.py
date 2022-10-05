from string import punctuation
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
import json

PUNCT = punctuation + '-'
morph = MorphAnalyzer()
SW = stopwords.words('russian')


def lemmatize(my_line):  # делает из строки список лемм без стоп-слов
    my_line = my_line.translate(str.maketrans('', '', PUNCT))
    words = my_line.split()
    lem_words = [morph.parse(w)[0].normal_form for w in words]
    filtered = [w for w in lem_words if w not in SW]
    return filtered


def create_data(filepath, len_c):  # сохраняем корпус и обрабатываем, проходя по ответам, находя лучший и лемматизируя
    with open(filepath, 'r', encoding='utf-8') as f:
        data = list(f)[:len_c]

    corpus = []
    raw_corpus = []
    for i in range(len(data)):
        answers = json.loads(data[i])['answers']
        if len(answers) != 0:
            try:
                rating = sorted(answers, key=lambda answer: int(answer['author_rating']['value']), reverse=True)
                best_a = rating[0]['text']
                raw_corpus.append(best_a)
                clean_text = ' '.join(lemmatize(best_a))
                corpus.append(clean_text)
            except ValueError:
                pass  # там есть ровно один странный пример, где вместо value пустая строка

    return [raw_corpus, corpus]
    # в corpus лежат обработанные документы, в raw_corpus ответы без обработки, они нужны для итоговой выдачи
