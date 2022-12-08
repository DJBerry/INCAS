import pandas as pd
import string
import os
import numpy as np
import re
from collections import Counter
import sage
import spacy
from tqdm import tqdm
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
# stp=stopwords.words('english')
stp=stopwords.words('french')

# nlp = spacy.load('en_core_web_md')
nlp = spacy.load('fr_core_news_sm')

CAT_DIR = ["ukraine", "russia", "refugees", "NATO", "energy", "economy", "economic sanctions", "defense"]
LANGUAGE = 'fr'

baseline_files = []
for cat in CAT_DIR:
    # filespath = './Wiki_Content/{}/{}'.format(cat, LANGUAGE)
    filespath = './Wiki_Content/{}/ts_fr'.format(cat)
    files = os.listdir(filespath)
    for file in files:
        baseline_files.append(filespath+"/"+file)

for cat in CAT_DIR:
    filespath = './Wiki_Content/{}/su_fr'.format(cat)
    files = os.listdir(filespath)
    for file in files:
        baseline_files.append(filespath + "/" + file)

# print(baseline_files)

baseline = ''
for file in baseline_files:
    with open(file,'r', encoding="utf-8") as myfile:
        baseline = baseline+' '+myfile.read()

def lemmatize(sent):
    s=[token.lemma_ for token in nlp(sent)]
    s=' '.join(s)
    return s

baseline_arr=baseline.split('\n')
baseline_arr=[lemmatize(b) for b in tqdm(baseline_arr)]
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

base_words=[]
for b in tqdm(baseline_arr):
    b_arr=b.split()
    b_arr=[b for b in b_arr if b not in stp]
    base_words.extend(b_arr)
base_count=Counter(base_words)

def ret_scores(eta,K=100):
    scores=eta[(-eta).argsort()[:K]]
    return scores

# CAT_DIR = ["ukraine", "russia", "refugees", "NATO", "energy", "economy", "economic sanctions", "defense"]
for target_concern in tqdm(CAT_DIR):
    # target_concern = "energy"
    print(target_concern)
    target_files = []
    filespath = './Wiki_Content/{}/ts_fr'.format(target_concern)
    files = os.listdir(filespath)
    for file in files:
        target_files.append(filespath+"/"+file)

    filespath2 = './Wiki_Content/{}/su_fr'.format(target_concern)
    files2 = os.listdir(filespath2)
    for file in files2:
        target_files.append(filespath2 + "/" + file)
    # print(target_files)
    # break

    category_arr= []
    for file in tqdm(target_files):
        lines_text = open(file, 'r', encoding="utf-8").readlines()
        for line_text in lines_text:
            category_arr.append(line_text)

    words_dict = {}

    category_arr=[lemmatize(t) for t in category_arr]
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
    words_dict = category_dict
    print(words_dict.keys())
    # for k, v in words_dict.items():
    #     print("{}: {}".format(k, v))

    # result = words_dict[files[1]]
    output_df = pd.DataFrame(list(words_dict.items()),columns=['words', 'score'])
    output_df.to_csv("lexicon/fr_all/{}_{}_sage.csv".format(target_concern, LANGUAGE), encoding="utf-8")




