import statistics
import pandas as pd
import numpy as np
import igraph as ig
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import random
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

pd.set_option('display.max_columns', None)

PRO_DATA_DIR = "compare/master_timeslice_two_remainder_actor_filtered_protagonist/"
# PRO_DATA = "emotion_master_timeslice_two_remainder_actor_filtered_protagonist.p"
PRO_DATA = "emotion_master_timeslice_two_remainder_actor_filtered_protagonist.p"

USC_DATA_DIR = "compare/master_timeslice_two_remainder_actor_filtered_usc_ta1/"
USC_DATA = "emotion_master_timeslice_two_remainder_actor_filtered_usc_ta1.p"

protagonist_df = pd.read_pickle("{}{}".format(PRO_DATA_DIR, PRO_DATA))
usc_df = pd.read_pickle("{}{}".format(USC_DATA_DIR, USC_DATA))
usc_df = usc_df[["type", "text", "confidence", "date"]]

print(len(protagonist_df))
print(protagonist_df.sample(5))

print(len(usc_df))
print(usc_df.sample(5))
print("==========================")

# type_list = protagonist_df["type"].tolist()
text_list = protagonist_df["text"].tolist()
# type_set = set(type_list)
text_set = set(text_list)
# print(type_set)
print(text_set)

# type_list = protagonist_df["type"].tolist()
text_list = usc_df["text"].tolist()
# type_set = set(type_list)
text_set = set(text_list)
# print(type_set)
print(text_set)

pro_emotion_type_list = ['Pride, including National Pride', 'Joy, Happiness', 'Amusement', 'Admiration, Love', 'Optimism, Hope',
                     'Fear, Pessimism', 'Anger, Hate, Contempt, Disgust', 'Embarrassment, Guilt, Shame, Sadness',
                     'Positive-other', 'Negative-other', 'emotion present', 'emotion not present']

usc_emotion_type_list = ['Pride, including National Pride', 'Joy, Happiness', 'Amusement', 'Admiration, Love', 'Optimism, Hope',
                     'Fear, Pessimism', 'Anger, Hate, Contempt, Disgust', 'Embarrassment, Guilt, Shame, Sadness',
                     'Positive-other', 'Negative-other']

# pro_agenda_type_list = ['Vote for ENTITY', 'Vote',  'Vote against ENTITY',
#                         'Belief that you/GROUP are at risk', 'Belief that ENTITY or GROUP is moral/ethical/honest/beneficial',
#                         'Belief that ENTITY or GROUP is immoral/unethical/dishonest/harmful',
#                         'Belief that your actions can lead to a good outcome/hope for GROUP',
#                         'Belief that the election process is flawed and/or manipulated', 'Share information and opinions Actor endorses',
#                         'Take action: protest/demonstrate/attend rally/volunteer/campaign', 'agenda not present', 'cta present', 'cta not present']
#
# usc_agenda_type_list = ['vote for', 'vote against', 'Believe that ENTITY or GROUP is moral/ethical/honest/beneficial',
#                         'Believe that ENTITY or GROUP is immoral/unethical/dishonest/harmful']

# pro_concern_type_list = ['degradation', 'degradation of entity', 'betrayal (outgroup)', 'loyalty (ingroup)', 'authority', 'righteous cause',
#                          'support of entity', 'non-narrative', 'nationalism/patriotism',
#                          'victimization/oppression', 'subversion', 'NM',
#                          'harm', 'resistance', 'order, govt legitimacy, & govt failure', 'purity', 'cheating',
#                          'care', 'fairness', 'concern present', 'concern not present']
#
# usc_concern_type_list = ['Economy', 'Relationship with Russia', 'National Identity and National Pride', 'Immigration and Refugees',
#                          'Religion', 'Democracy', 'Environment and Climate Change', 'Terrorism and Counterterrorism',
#                          'Fake News/Misinformation', 'International Alliance Organizations']


pro_usc_df = pd.DataFrame()
for i in range(len(usc_emotion_type_list)):
    df = usc_df[usc_df["text"] == usc_emotion_type_list[i]]

    date_set = set(df['date'].tolist())
    confidence_times_dict = {k: 0 for k in date_set}
    confidence_value_dict = {k: 0.0 for k in date_set}

    for row in tqdm(df.itertuples(), total=len(df)):
        date = row[-1]
        confidence = row[-2]
        # confidence = row[1]
        confidence_times_dict[date] += 1
        confidence_value_dict[date] += confidence

    confidence_date_dict = {k: confidence_value_dict[k] / confidence_times_dict[k] for k in date_set}
    confidence_date_df = pd.DataFrame(confidence_date_dict.items(), columns=['date', 'confidence'])
    confidence_date_df = confidence_date_df.sort_values(by='date')

    confidence_list = confidence_date_df["confidence"].tolist()
    pro_usc_df[usc_emotion_type_list[i]] = confidence_list
    # print(confidence_date_df)
    # break

for i in range(len(pro_emotion_type_list)):
    df = protagonist_df[protagonist_df["text"] == pro_emotion_type_list[i]]

    date_set = set(df['date'].tolist())
    confidence_times_dict = {k: 0 for k in date_set}
    confidence_value_dict = {k: 0.0 for k in date_set}

    for row in tqdm(df.itertuples(), total=len(df)):
        date = row[-1]
        confidence = row[-2]
        # confidence = row[1]
        confidence_times_dict[date] += 1
        confidence_value_dict[date] += confidence

    confidence_date_dict = {k: confidence_value_dict[k] / confidence_times_dict[k] for k in date_set}
    confidence_date_df = pd.DataFrame(confidence_date_dict.items(), columns=['date', 'confidence'])
    confidence_date_df = confidence_date_df.sort_values(by='date')

    confidence_list = confidence_date_df["confidence"].tolist()
    pro_usc_df[pro_emotion_type_list[i]+" (p)"] = confidence_list

print(pro_usc_df)
pro_usc_corr_df = pro_usc_df.corr()

pro_usc_corr_df = pro_usc_corr_df[['Pride, including National Pride', 'Joy, Happiness', 'Amusement', 'Admiration, Love', 'Optimism, Hope',
                     'Fear, Pessimism', 'Anger, Hate, Contempt, Disgust', 'Embarrassment, Guilt, Shame, Sadness',
                     'Positive-other', 'Negative-other']].loc[['Pride, including National Pride (p)', 'Joy, Happiness (p)', 'Amusement (p)', 'Admiration, Love (p)', 'Optimism, Hope (p)',
                     'Fear, Pessimism (p)', 'Anger, Hate, Contempt, Disgust (p)', 'Embarrassment, Guilt, Shame, Sadness (p)',
                     'Positive-other (p)', 'Negative-other (p)', 'emotion present (p)', 'emotion not present (p)']]

# pro_usc_corr_df.to_pickle("pro_usc_emotion_corr_matrix.p")

# pro_usc_corr_df = pd.read_pickle("pro_usc_emotion_corr_matrix.p")
pro_usc_corr_df.to_json("pro_usc_emotion_corr_matrix2.json")

# fig, ax = plt.subplots(figsize=[12, 9])
#
# sns.heatmap(pro_usc_corr_df, annot=True, cmap="YlGnBu")
# plt.xticks(rotation=15, ha='right')
# # plt.yticks(np.arange(0.55, 0.75, 0.02))
# plt.title("Correlation Matrix (Emotion)")
# ax.set_xlabel('Protagonist')
# ax.set_ylabel('USC')
# # plt.legend()
# plt.savefig("corr_matrix_emotion")
# plt.show()












