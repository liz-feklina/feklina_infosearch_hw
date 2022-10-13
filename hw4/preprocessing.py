import json


def create_data(filepath, len_c):  # сохраняем корпус и обрабатываем, проходя по ответам, находя лучший
    with open(filepath, 'r', encoding='utf-8') as f:
        data = list(f)[:len_c]

    raw_corpus = []
    corp_questions = []
    for i in range(len(data)):
        corp_data = json.loads(data[i])
        answers = corp_data['answers']
        if len(answers) != 0:
            try:
                rating = sorted(answers, key=lambda answer: int(answer['author_rating']['value']), reverse=True)
                best_a = rating[0]['text']
                raw_corpus.append(best_a)
                corp_questions.append(corp_data['question'])
            except ValueError:
                pass  # там есть ровно один странный пример, где вместо value пустая строка

    return raw_corpus, corp_questions
    # в raw_corpus лежат ответы без обработки, в corp_questions соответствущие вопросы
