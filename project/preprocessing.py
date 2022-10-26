from string import punctuation
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
import pickle
from transformers import AutoTokenizer, AutoModel
import torch

PUNCT = punctuation + '-'
morph = MorphAnalyzer()
SW = stopwords.words('russian')


def lemmatize(my_line):  # делает из строки список лемм без стоп-слов
    my_line = my_line.translate(str.maketrans('', '', PUNCT))
    words = my_line.split()
    lem_words = [morph.parse(w)[0].normal_form for w in words]
    filtered = [w for w in lem_words if w not in SW]
    return filtered


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def compute_index(query, tokenizer, model):
    # Load AutoModel from huggingface model repository

    # Tokenize sentences
    encoded_input = tokenizer(query, padding=True, truncation=True, max_length=24, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, mean pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    return [sentence_embeddings, tokenizer, model]


def load_data():
    data = dict()
    names = ['answers', 'questions', 'tfidf_index', 'tfidf_vectorizer', 'bm25_index', 'bm25_vectorizer', 'bert_index']
    for a in range(len(names)):
        with open(f'.\\data\\{names[a]}.pk', 'rb') as f:
            data[names[a]] = pickle.load(f)
    data['bert_model'] = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    data['bert_tokenizer'] = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    return data
