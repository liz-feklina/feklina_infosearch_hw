from preprocessing import lemmatize
from load_data import load
from search import search_req


def main():
    index, vectorizer, series_list = load()
    # wordlist = vectorizer.get_feature_names()
    request = input('Напишите запрос или "все!", если хотите закончить: ')
    while request != 'все!':
        lem_req = ' '.join(lemmatize(request))
        answer = search_req(lem_req, index, vectorizer, series_list)
        print(answer[:5])
        request = input('Напишите запрос или "все!", если хотите закончить: ')


if __name__ == '__main__':
    main()
