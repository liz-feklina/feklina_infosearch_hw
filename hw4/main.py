from load_data import load
from search import search_req


def main():
    raw_corpus, sentence_embeddings, corp_questions = load()
    request = input('Напишите запрос или "все!", если хотите закончить: ')
    while request != 'все!':
        answers = search_req(request, sentence_embeddings, raw_corpus)
        if len(answers) == 0:
            print('Нет результатов!')
        else:
            for i in range(5):
                print(f'Результат №{i+1}: "{answers[i][:50]}..."')
        request = input('Напишите запрос или "все!", если хотите закончить: ')


if __name__ == '__main__':
    main()
