from preprocessing import lemmatize
from load_data import load
from search import search_req


def main():
    raw_corpus, corpus, index, vectorizer = load()
    request = input('Напишите запрос или "все!", если хотите закончить: ')
    while request != 'все!':
        lem_req = ' '.join(lemmatize(request))
        answers = search_req(lem_req, index, vectorizer)
        if len(answers) == 0:
            print('Нет результатов!')
        else:
            for ans in answers[:5]:
                print(f'"{raw_corpus[ans[1]][:50]}..."        (результат: {ans[0]})')
        request = input('Напишите запрос или "все!", если хотите закончить: ')


if __name__ == '__main__':
    main()
