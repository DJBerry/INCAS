import pandas as pd
import os
import numpy as np
import re
from collections import Counter
import spacy
from tqdm import tqdm
from string import digits
import nltk
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
import gensim
from gensim import corpora, models

nltk.download('stopwords')
nlp = spacy.load('en_core_web_md')
stp = stopwords.words('english')

concerns = [
    "ukraine",
    "russia",
    "NATO",
    "refugees",
    "defense",
    "economy",
    "economic sanctions",
    "energy"
]

languages = ["en", "fr"]

# preprocess corpus
# def lemmatize(sent):
#     s = [token.lemma_ for token in nlp(sent)]
#     s = ' '.join(s)
#     return s
#
# def lemmatize2(text):
#     token = WordNetLemmatizer().lemmatize(text, pos='v')
#     return token
#
# def lemmatize_stemming(text):
#     token = WordNetLemmatizer().lemmatize(text, pos='v')
#     return SnowballStemmer("english").stem(token)
#
# irrelevant_chars="~?!./\:;+=&^%$#@(,)[]_*"
# emoji_pattern = re.compile("["
#         u"\U0001F600-\U0001F64F"  # emoticons
#         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#         u"\U0001F680-\U0001F6FF"  # transport & map symbols
#         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#                            "]+", flags=re.UNICODE)
#
# def deep_clean(x):
#     x = x.lower()
#     x = re.sub(r'http\S+', '', x)
#     remove_digits = str.maketrans('', '', digits)
#     remove_chars = str.maketrans('', '', irrelevant_chars)
#     x = x.translate(remove_digits)
#     x = x.translate(remove_chars)
#     x = emoji_pattern.sub(r'', x)
#     x = x.replace('!', '')
#     x = x.replace('?', '')
#     x = x.replace('@', '')
#     x = x.replace('&', '')
#     x = x.replace('$', '')
#     x = x.replace('``', '')
#     x = x.replace("'s", '')
#     x = x.replace("''", '')
#     x = [t for t in x.split() if len(t)>3]
#     x = ' '.join(x)
#     return x

def lemmatize_stemming(text):
    token = WordNetLemmatizer().lemmatize(text, pos='v')
    return token

def preprocess_lem_stem(text):
    result = []
    for token in simple_preprocess(text):
        if token not in STOPWORDS:
            result.append(lemmatize_stemming(token))
    return result

def preprocess(text):
    result = []
    for token in simple_preprocess(text):
        if token not in STOPWORDS:
            result.append(token)
    return result

top_freq_concern_dict = {}

for concern in concerns:
    print(concern)
    filespath = 'Wiki_Content/{}/en'.format(concern)

# filespath = 'Wiki_Content/ukraine/en'
    files = os.listdir(filespath)
    text_list = []
    for file in files:
        with open(os.path.join(filespath, file), 'r', encoding="utf-8") as myfile:
            text_list.append(myfile.read())

    # split_text_list = [text.split('\n') for text in text_list]
    # processed_text_list = [lemmatize2(b) for b in text_list]
    # processed_text_list = [b.strip() for b in processed_text_list]
    # processed_text_list = [deep_clean(b) for b in processed_text_list]

    processed_text_list = [preprocess_lem_stem(text) for text in text_list]
    processed_text_list = [' '.join(text) for text in processed_text_list]

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_text_list)

    scores = zip(vectorizer.get_feature_names(), np.asarray(X.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:500]
    # print(type(sorted_scores))
    # print(sorted_scores)
    # print(len(sorted_scores))

    vocab = [word_score[0] for word_score in sorted_scores]
    # print(vocab)
    top_freq_concern_dict[concern] = vocab

en_concern_voc = pd.DataFrame.from_dict(top_freq_concern_dict)
en_concern_voc.to_csv("top_tfidf_concern_vocabulary.csv")


# en_concern_voc = pd.DataFrame.from_dict(top_freq_concern_dict)
# en_concern_voc.to_csv("top_freq_concern_vocabulary.csv")

# for item in sorted_scores:
# print "{0:50} Score: {1}".format(item[0], item[1])


# summarize
# print(vectorizer.vocabulary_)
# vocabulary_dict = vectorizer.vocabulary_
# # print(type(vocabulary_dict))
# sort_vocabulary_dict = {k: v for k, v in sorted(vocabulary_dict.items(), key=lambda item: item[1])}
# print(sort_vocabulary_dict)
# print(vectorizer.idf_)


# dictionary = corpora.Dictionary(processed_text_list)
# corpus = [dictionary.doc2bow(processed_text) for processed_text in processed_text_list]
# tfidf = models.TfidfModel(corpus)
# tfidf_corpus = tfidf[corpus]
# # Get statistics of important word
# d = {}
# for doc in tfidf_corpus:
#     for id, value in doc:
#         word = dictionary.get(id)
#         d[word] = value
#
# highest_word_list = sorted(d, key=d.get, reverse=True)[:500]
# print(highest_word_list)
# for word in highest_word_list:
#     print("{} : {}".format(word, d[word]))

# category_words = []
# for b in corpus_arr:
#     b_arr = b.split()
#     b_arr = [b for b in b_arr if b not in stp]
#     b_arr = [b for b in b_arr if b not in STOPWORDS]
#     category_words.extend(b_arr)
# category_count = Counter(category_words)
# sort_dict = {k: v for k, v in sorted(category_count.items(), key=lambda item: item[1], reverse=True)}
# # print(sort_dict)
# vocab = [word for word, count in Counter(category_count).most_common(500)]
# # print(vocab)
# top_freq_concern_dict[concern] = vocab

# en_concern_voc = pd.DataFrame.from_dict(top_freq_concern_dict)
# en_concern_voc.to_csv("top_freq_concern_vocabulary.csv")
