import numpy as np
import pandas as pd
import metachange
from datetime import datetime
from tqdm import tqdm
import time

from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None)

SAMPLE_FILE = "phase_1b_sample.p"
NUM_FEATURE = 1000
MIN_RANGE = 86400
MAX_DEPTH = 3

print(SAMPLE_FILE)
print(NUM_FEATURE)
print(MIN_RANGE)
print(MAX_DEPTH)

# df = pd.read_pickle("phase_1b_twitter_dataset.p")
# df = df[["title", "contentText", "timePublished", "datetime", "date"]]
# date_list = df["date"].tolist()
# uni_date_list = list(set(date_list))
# uni_date_list.sort()
#
# sample_df_list = []
# date_num_dict = {}
# for date in uni_date_list:
#     target_df = df[df["date"] == date]
#     date_num_dict[date] = len(target_df)
#     if len(target_df) > 3000:
#         sample_df_list.append(target_df.sample(3000))
#     else:
#         sample_df_list.append(target_df)
# sample_df = pd.concat(sample_df_list, ignore_index=True)
sample_df = pd.read_pickle(SAMPLE_FILE)

import spacy
import nltk
import re
nltk.download('stopwords')

from nltk.corpus import stopwords
# stp=stopwords.words('english')
stp=stopwords.words('french')

# nlp = spacy.load('en_core_web_md')
nlp = spacy.load('fr_core_news_sm')

def lemmatize(sent):
    s=[token.lemma_ for token in nlp(sent)]
    s=' '.join(s)
    return s

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

start_time = time.time()

structural_features_raw = []
for row in tqdm(sample_df.itertuples(), total=len(sample_df)):
    contentText = row[2]
    lem_text = lemmatize(contentText)
    deep_clean_text = deep_clean(lem_text.strip())
    text_arr = deep_clean_text.split()
    clean_text_arr = [word for word in text_arr if word not in stp]
    clean_text = ' '.join(clean_text_arr)
    structural_features_raw.append({
        "clean_text": clean_text
    })
structural_features_df = pd.DataFrame(structural_features_raw, index=sample_df.index)
input_df = pd.concat([sample_df, structural_features_df], axis=1)

preprocess_time = time.time()

from sklearn.feature_extraction.text import TfidfVectorizer
processed_text_str_list = structural_features_df['clean_text'].tolist()
vectorizer = TfidfVectorizer(max_features=NUM_FEATURE)
vectorizer.fit(processed_text_str_list)
features_raw = []
for row in tqdm(input_df.itertuples(), total=len(input_df)):
    text_str = row[-1]
    text_str_list = [text_str]
    vector = vectorizer.transform(text_str_list)
    vector = vector.toarray()
    features_raw.append({
        "tf_idf_vec": vector[0],
    })
features_df = pd.DataFrame(features_raw, index=input_df.index)
tfidf_feature_df = pd.concat([input_df, features_df], axis=1)

tfidf_feature_df['encoded_timestamp'] = [time/1000 for time in tfidf_feature_df['timePublished'].values]

vec_list = tfidf_feature_df['tf_idf_vec'].tolist()
date_list = tfidf_feature_df['encoded_timestamp'].tolist()
vec_array = np.array(vec_list)
date_array = np.array(date_list)

vectorizer_time = time.time()

clf_rf = RandomForestClassifier(max_depth=32, criterion="entropy", random_state=0)
res_multi, res_multi_result = metachange.change_point_tree(vec_array, date_array, clf_rf, min_range=MIN_RANGE, max_d=MAX_DEPTH)

end_time = time.time()

print("preoprocess time: {}".format(preprocess_time - start_time))
print("vectorizer time: {}".format(vectorizer_time - preprocess_time))
print("changepoint detection time: {}".format(end_time - vectorizer_time))

def make_node_text(data):
    t_left = data["t_left"]
    t_right = data["t_right"]

    if "t0" in data:
        header = f't_0 = {datetime.fromtimestamp(data["t0"]).date()}\n alpha = {data["alpha"]:.4f}'
    else:
        header = "Leaf"
    return f"{header}\nRange:{datetime.fromtimestamp(t_left).date()}-{datetime.fromtimestamp(t_right).date()}"


tree = metachange.show_tree(res_multi, make_node_text, fname="standard_test_1.pdf")
# print(res_multi_result)

t0_mean_list = []
alpha_mean_list = []
for changePoint_dict in res_multi_result:
    t0_mean_list.append(changePoint_dict["t0_mean"])
    alpha_mean_list.append(changePoint_dict["alpha_mean"])
change_point_df = pd.DataFrame(list(zip(t0_mean_list, alpha_mean_list)), columns =['time', 'alpha'])
sort_change_point_df = change_point_df.sort_values("time")

start_timestamp = date_list[0]
end_timestamp = date_list[1]

changepoint_timestamp_list = sort_change_point_df["time"].tolist()
changepoint_alpha_list = sort_change_point_df["alpha"].tolist()

changepoint_result_list = []
prefix_list = []
prefix_list.append(start_timestamp)
for timestamp in changepoint_timestamp_list:
    prefix_list.append(timestamp)

suffix_list = [timestamp for timestamp in changepoint_timestamp_list]
suffix_list.append(end_timestamp)

for i in range(len(changepoint_alpha_list)):
    change_point_result = (datetime.fromtimestamp(prefix_list[i]).date(), datetime.fromtimestamp(suffix_list[i]).date(), changepoint_alpha_list[i])
    changepoint_result_list.append(change_point_result)

# print(changepoint_result_list)

for changepoint in changepoint_result_list:
    print("({}, {}, {:.2f})".format(changepoint[0], changepoint[1], changepoint[2]))
