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
def lemmatize(sent):
    s = [token.lemma_ for token in nlp(sent)]
    s = ' '.join(s)
    return s

def lemmatize2(text):
    token = WordNetLemmatizer().lemmatize(text, pos='v')
    return token

def lemmatize_stemming(text):
    token = WordNetLemmatizer().lemmatize(text, pos='v')
    return SnowballStemmer("english").stem(token)

irrelevant_chars="~?!./\:;+=&^%$#@(,)[]_*"
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

def deep_clean(x):
    x = x.lower()
    x = re.sub(r'http\S+', '', x)
    remove_digits = str.maketrans('', '', digits)
    remove_chars = str.maketrans('', '', irrelevant_chars)
    x = x.translate(remove_digits)
    x = x.translate(remove_chars)
    x = emoji_pattern.sub(r'', x)
    x = x.replace('!', '')
    x = x.replace('?', '')
    x = x.replace('@', '')
    x = x.replace('&', '')
    x = x.replace('$', '')
    x = x.replace('``', '')
    x = x.replace("'s", '')
    x = x.replace("''", '')
    x = [t for t in x.split() if len(t)>3]
    x = ' '.join(x)
    return x

top_freq_concern_dict = {}

for concern in concerns:
    print(concern)
    filespath = 'Wiki_Content/{}/en'.format(concern)
    files = os.listdir(filespath)
    corpus = ''
    for file in files:
        with open(os.path.join(filespath, file), 'r', encoding="utf-8") as myfile:
            corpus = corpus + ' ' + myfile.read()

    corpus_arr = corpus.split('\n')
    corpus_arr = [lemmatize2(b) for b in corpus_arr]
    corpus_arr = [b.strip() for b in corpus_arr]
    corpus_arr = [deep_clean(b) for b in corpus_arr]
    # print(corpus_arr)
    category_words = []
    for b in corpus_arr:
        b_arr = b.split()
        b_arr = [b for b in b_arr if b not in stp]
        b_arr = [b for b in b_arr if b not in STOPWORDS]
        category_words.extend(b_arr)
    category_count = Counter(category_words)
    sort_dict = {k: v for k, v in sorted(category_count.items(), key=lambda item: item[1], reverse=True)}
    # print(sort_dict)
    vocab = [word for word, count in Counter(category_count).most_common(500)]
    # print(vocab)
    top_freq_concern_dict[concern] = vocab

en_concern_voc = pd.DataFrame.from_dict(top_freq_concern_dict)
en_concern_voc.to_csv("top_freq_concern_vocabulary.csv")
