import pandas as pd
import string
import os
import numpy as np
import re
from collections import Counter
import sage
import spacy
from tqdm import tqdm

baseline=''
filespath='./Baseline'
files=os.listdir(filespath)
for file in files:
    with open(os.path.join(filespath,file),'r', encoding="utf-8") as myfile:
        baseline=baseline+' '+myfile.read()

nlp = spacy.load('en_core_web_md')

def lemmatize(sent):
    s=[token.lemma_ for token in nlp(sent)]
    s=' '.join(s)
    return s

baseline_arr=baseline.split('\n')
#baseline_arr=[lemmatize(b) for b in tqdm(baseline_arr)]
baseline_arr=[b.strip() for b in baseline_arr]

irrelevant_chars="~?!./\:;+=&^%$#@(,)[]_*"
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

from string import digits
def deep_clean(x):
    x=x.lower()
    x=re.sub(r'http\S+', '', x)
    remove_digits = str.maketrans('', '', digits)
    remove_chars = str.maketrans('', '', irrelevant_chars)
    x = x.translate(remove_digits)
    x = x.translate(remove_chars)
    x = emoji_pattern.sub(r'', x)
    x=x.replace('!','')
    x=x.replace('?','')
    x=x.replace('@','')
    x=x.replace('&','')
    x=x.replace('$','')
    x=x.replace('``','')
    x=x.replace("'s",'')
    x=x.replace("''",'')
    x=[t for t in x.split() if len(t)>3]
    x=' '.join(x)
    return x

baseline_arr=[deep_clean(b) for b in baseline_arr]

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stp=stopwords.words('english')

base_words=[]
for b in tqdm(baseline_arr):
    b_arr=b.split()
    b_arr=[b for b in b_arr if b not in stp]
    base_words.extend(b_arr)
base_count=Counter(base_words)

def ret_scores(eta,K=100):
    scores=eta[(-eta).argsort()[:K]]
    return scores

# filespaths = ['education', 'Coronavirus Origins', 'Lockdowns and safety measures', 'Masking', 'Therapeutics', 'Vaccines and Vaccine Hesitancy']
filespaths = ['Lockdowns and safety measures']
for filespath in filespaths:
    files = os.listdir(filespath)
    words_dict = {}
    for file in tqdm(files):
        category_arr = open(os.path.join(filespath, file), 'r', encoding="utf-8").readlines()
        # category_arr=[lemmatize(t) for t in category]
        category_arr = [deep_clean(t.strip()) for t in category_arr]

        category_words = []
        for b in category_arr:
            b_arr = b.split()
            b_arr = [b for b in b_arr if b not in stp]
            category_words.extend(b_arr)
        category_count = Counter(category_words)

        vocab = [word for word, count in Counter(category_count).most_common(5000)]
        x_terr = np.array([category_count[word] for word in vocab])
        x_base = np.array([base_count[word] for word in vocab]) + 1.

        mu = np.log(x_base) - np.log(x_base.sum())
        eta = sage.estimate(x_terr, mu)

        category = sage.topK(eta, vocab, K=200)
        scores = ret_scores(eta, 200)
        category_dict = {}
        for i in range(len(category)):
            category_dict[category[i]] = scores[i]
        words_dict[file] = category_dict

    result = words_dict[files[1]]
    output_df = pd.DataFrame(list(result.items()),columns=['words', 'score'])
    output_df.to_csv("unigram/{}.csv".format(filespath))

# baseline=''
# filespath='./Baseline'
# files=os.listdir(filespath)
# for file in files:
#     with open(os.path.join(filespath,file),'r', encoding="utf-8") as myfile:
#         baseline=baseline+' '+myfile.read()
#
# baseline_arr=baseline.split('\n')
# #baseline_arr=[lemmatize(b) for b in tqdm(baseline_arr)]
# baseline_arr=[b.strip() for b in baseline_arr]
#
# irrelevant_chars="~?!./\:;+=&^%$#@(,)-[]-_*"
# emoji_pattern = re.compile("["
#         u"\U0001F600-\U0001F64F"  # emoticons
#         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#         u"\U0001F680-\U0001F6FF"  # transport & map symbols
#         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#                            "]+", flags=re.UNICODE)
#
# from string import digits
# def deep_clean(x):
#     x=x.lower()
#     x=re.sub(r'http\S+', '', x)
#     remove_digits = str.maketrans('', '', digits)
#     remove_chars = str.maketrans('', '', irrelevant_chars)
#     x = x.translate(remove_digits)
#     x = x.translate(remove_chars)
#     x = emoji_pattern.sub(r'', x)
#     x=x.replace('!','')
#     x=x.replace('?','')
#     x=x.replace('@','')
#     x=x.replace('&','')
#     x=x.replace('$','')
#     x=x.replace('``','')
#     x=x.replace("'s",'')
#     x=x.replace("''",'')
#     x=[t for t in x.split() if len(t)>3]
#     x=' '.join(x)
#     return x
#
# baseline_arr=[deep_clean(b) for b in baseline_arr]
#
# from nltk.corpus import stopwords
# stp=stopwords.words('english')
#
# import nltk
# nltk.download('punkt')
#
# from nltk import word_tokenize
# from nltk import ngrams
#
# base_words = []
# for b in tqdm(baseline_arr):
#     nltk_tokens = word_tokenize(b)
#
#     b_arr = list(ngrams(nltk_tokens, 2))
#     # b_arr=b.split()
#
#     # b_arr=[b for b in b_arr if b not in stp]
#     base_words.extend(b_arr)
# base_count = Counter(base_words)
#
# def ret_scores(eta,K=100):
#     scores=eta[(-eta).argsort()[:K]]
#     return scores
#
# # filespaths = ['education', 'Coronavirus Origins', 'Lockdowns and safety measures', 'Masking', 'Therapeutics', 'Vaccines and Vaccine Hesitancy']
# filespaths = ['Lockdowns and safety measures']
# for filespath in filespaths:
#     files = os.listdir(filespath)
#     words_dict = {}
#     for file in tqdm(files):
#         category_arr = open(os.path.join(filespath, file), 'r', encoding="utf-8").readlines()
#         # category_arr=[lemmatize(t) for t in category]
#         category_arr = [deep_clean(t.strip()) for t in category_arr]
#
#         category_words = []
#         for b in category_arr:
#             nltk_tokens = word_tokenize(b)
#             b_arr = list(ngrams(nltk_tokens, 2))
#             # b_arr=b.split()
#             # b_arr=[b for b in b_arr if b not in stp]
#             category_words.extend(b_arr)
#         category_count = Counter(category_words)
#
#         vocab = [word for word, count in Counter(category_count).most_common(5000)]
#         x_terr = np.array([category_count[word] for word in vocab])
#         x_base = np.array([base_count[word] for word in vocab]) + 1.
#
#         mu = np.log(x_base) - np.log(x_base.sum())
#
#         eta = sage.estimate(x_terr, mu)
#
#         category = sage.topK(eta, vocab, K=200)
#         scores = ret_scores(eta, 200)
#         category_dict = {}
#         for i in range(len(category)):
#             category_dict[category[i]] = scores[i]
#         words_dict[file] = category_dict
#
#     result = words_dict[files[-1]]
#     output_df = pd.DataFrame(list(result.items()),columns=['words', 'score'])
#     output_df.to_csv("bigrams/{}.csv".format(filespath))

# baseline=''
# filespath='./Baseline'
# files=os.listdir(filespath)
# for file in files:
#     with open(os.path.join(filespath,file),'r', encoding="utf-8") as myfile:
#         baseline=baseline+' '+myfile.read()
#
# baseline_arr=baseline.split('\n')
# #baseline_arr=[lemmatize(b) for b in tqdm(baseline_arr)]
# baseline_arr=[b.strip() for b in baseline_arr]
#
# irrelevant_chars="~?!./\:;+=&^%$#@(,)-[]-_*"
# emoji_pattern = re.compile("["
#         u"\U0001F600-\U0001F64F"  # emoticons
#         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#         u"\U0001F680-\U0001F6FF"  # transport & map symbols
#         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#                            "]+", flags=re.UNICODE)
#
# from string import digits
# def deep_clean(x):
#     x=x.lower()
#     x=re.sub(r'http\S+', '', x)
#     remove_digits = str.maketrans('', '', digits)
#     remove_chars = str.maketrans('', '', irrelevant_chars)
#     x = x.translate(remove_digits)
#     x = x.translate(remove_chars)
#     x = emoji_pattern.sub(r'', x)
#     x=x.replace('!','')
#     x=x.replace('?','')
#     x=x.replace('@','')
#     x=x.replace('&','')
#     x=x.replace('$','')
#     x=x.replace('``','')
#     x=x.replace("'s",'')
#     x=x.replace("''",'')
#     x=[t for t in x.split() if len(t)>3]
#     x=' '.join(x)
#     return x
#
# baseline_arr=[deep_clean(b) for b in baseline_arr]
#
# from nltk.corpus import stopwords
# stp=stopwords.words('english')
#
# from nltk import word_tokenize
# from nltk import ngrams
#
# base_words = []
# for b in tqdm(baseline_arr):
#     nltk_tokens = word_tokenize(b)
#
#     b_arr = list(ngrams(nltk_tokens, 3))
#     # b_arr=b.split()
#
#     # b_arr=[b for b in b_arr if b not in stp]
#     base_words.extend(b_arr)
# base_count = Counter(base_words)
#
# def ret_scores(eta,K=100):
#     scores=eta[(-eta).argsort()[:K]]
#     return scores
#
# # filespaths = ['education', 'Coronavirus Origins', 'Lockdowns and safety measures', 'Masking', 'Therapeutics', 'Vaccines and Vaccine Hesitancy']
# filespaths = ['Lockdowns and safety measures']
# for filespath in filespaths:
#     files = os.listdir(filespath)
#     words_dict = {}
#     for file in tqdm(files):
#         category_arr = open(os.path.join(filespath, file), 'r', encoding="utf-8").readlines()
#         # category_arr=[lemmatize(t) for t in category]
#         category_arr = [deep_clean(t.strip()) for t in category_arr]
#
#         category_words = []
#         for b in category_arr:
#             nltk_tokens = word_tokenize(b)
#             b_arr = list(ngrams(nltk_tokens, 3))
#             # b_arr=b.split()
#             # b_arr=[b for b in b_arr if b not in stp]
#             category_words.extend(b_arr)
#         category_count = Counter(category_words)
#
#         vocab = [word for word, count in Counter(category_count).most_common(5000)]
#         x_terr = np.array([category_count[word] for word in vocab])
#         x_base = np.array([base_count[word] for word in vocab]) + 1.
#
#         mu = np.log(x_base) - np.log(x_base.sum())
#
#         eta = sage.estimate(x_terr, mu)
#
#         category = sage.topK(eta, vocab, K=200)
#         scores = ret_scores(eta, 200)
#         category_dict = {}
#         for i in range(len(category)):
#             category_dict[category[i]] = scores[i]
#         words_dict[file] = category_dict
#
#     result = words_dict[files[-1]]
#     output_df = pd.DataFrame(list(result.items()),columns=['words', 'score'])
#     output_df.to_csv("trigrams/{}.csv".format(filespath))
















