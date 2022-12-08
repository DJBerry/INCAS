import statistics
import pandas as pd
import numpy as np
import igraph as ig
import pickle
from tqdm import tqdm
from datetime import datetime
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

# # match ids
# ID_DATASET = "1a_annotations_released_20220524.csv"
# INFO_DATASET = "usc-isi_eval_data_set_2022-04-20 23_43_35_merged_dedup.json"
#
# id_df = pd.read_csv(ID_DATASET)
# # print(len(id_df))
# # print(id_df.sample(5))
#
# info_df = pd.read_json(INFO_DATASET, lines=True)
# # print(len(info_df))
# # print(info_df.sample(5))
#
# structural_features_raw1 = []
# structural_features_raw2 = []
# for row in tqdm(id_df.itertuples(), total=len(id_df)):
#        id = row[1]
#        target_df = info_df[info_df["id"] == id]
#
#        if target_df.empty:
#               continue
#
#        structural_features_raw1.append(target_df)
#
#        target_df2 = id_df[id_df["ID"] == id]
#        target_dict = target_df2.to_dict('records')
#        structural_features_raw2.append(target_dict)
#
# target_info_df = pd.concat(structural_features_raw1, ignore_index=True)
# # target_id_df = pd.concat(structural_features_raw2, ignore_index=True)
# # target_info_df = target_info_df.drop("id", axis=1)
#
# # result_df = pd.concat([target_id_df, target_info_df], axis=1, join='inner')
# target_info_df["ground_truth"] = structural_features_raw2
# result_df = target_info_df
# print(len(result_df))
# print(result_df.sample(5))
#
# result_df.to_pickle("4.5k_truth_data.p")



agenda_list = ['agenda-1.1', 'agenda-1.2', 'agenda-1.3', 'agenda-1.4',
       'agenda-1.5', 'agenda-1.6', 'agenda-2.1', 'agenda-2.2.1',
       'agenda-2.2.2', 'agenda-2.2.3', 'agenda-2.2.4', 'agenda-2.3']

agenda_translate = [
    "Believe that the election process is flawed and/or manipulated by ENTITY (including potential foreign interference)(usc)",
    "Believe that the election process is fair and has not been manipulated(usc)",
    "Believe that ENTITY or GROUP is immoral/unethical/dishonest/harmful(usc)",
    "Believe that ENTITY or GROUP is moral/ethical/honest/beneficial(usc)",
    "Believe that you/GROUP are at risk(usc)",
    "Believe that a good outcome/hope for GROUP is possible(usc)",
    "Share information and opinions author endorses(usc)",
    "Vote for ENTITY(usc)",
    "Vote against ENTITY(usc)",
    "Vote(usc)",
    "Don’t vote(usc)",
    "Take action: protest/demonstrate/attend rally/volunteer/campaign(usc)"
]

concern_list = ['concern-3.1', 'concern-3.2', 'concern-3.3', 'concern-3.4',
       'concern-3.5', 'concern-3.6', 'concern-3.7', 'concern-3.8',
       'concern-3.9', 'concern-3.10', 'concern-3.11']

concern_translate = [
    "Economy(usc)",
    "Terrorism and counterterrorism(usc)",
    "Religion(usc)",
    "Immigration and refugees(usc)",
    "International alliance organizations(usc)",
    "Relationship with Russia(usc)",
    "National Identity and national pride(usc)",
    "Environment and climate change(usc)",
    "Fake news/misinformation(usc)",
    "Character of ENTITY (candidates, other key figures)(usc)",
    "Democracy(usc)"
]

emotion_list = ['emotion-4.1',
       'emotion-4.2', 'emotion-4.3', 'emotion-4.4', 'emotion-4.5',
       'emotion-4.6', 'emotion-4.7', 'emotion-4.8', 'emotion-4.9',
       'emotion-4.10']

emotion_translate = [
    "Anger, hate, contempt, disgust(usc)",
    "Embarrassment, guilt, shame, sadness(usc)",
    "Admiration, love(usc)",
    "Optimism, hope(usc)",
    "Joy, happiness(usc)",
    "Pride, incl. national pride(usc)",
    "Fear, pessimism(usc)",
    "Amusement(usc)",
    "Positive-other(usc)",
    "Negative-other(usc)"
]

agenda_dict = {}
concern_dict = {}
emotion_dict = {}

for i in range(len(agenda_list)):
    agenda_dict[agenda_list[i]] = agenda_translate[i]

for i in range(len(concern_list)):
    concern_dict[concern_list[i]] = concern_translate[i]

for i in range(len(emotion_list)):
    emotion_dict[emotion_list[i]] = emotion_translate[i]


TRUTH_DATASET = "4.5k_truth_data.p"

df = pd.read_pickle(TRUTH_DATASET)
print(len(df))
# print(df.sample(5))

structural_features_raw = []
for row in tqdm(df.itertuples(), total=len(df)):
    twitter_data = row[6]
    twitter_features_dict = twitter_data['twitterData']
    structural_features_raw.append(twitter_features_dict)

structural_features_df = pd.DataFrame(structural_features_raw, index=df.index)
result_df = pd.concat([df, structural_features_df], axis=1)

result_df['timePublished'] = [datetime.fromtimestamp(time/1000) for time in result_df['timePublished'].values]
dates = [pd.to_datetime(dt).date() for dt in result_df['timePublished'].values]
result_df['date'] = dates

result_df = result_df.sort_values(by='date')
twitter_data = result_df.drop(["mediaType", "embeddedUrls", "imageUrls", "dataTags", "geolocation"], axis=1)
print(twitter_data.sample(5))

date_set = set(twitter_data['date'].tolist())

# confidence_times_dict = {k: 0 for k in date_set}
# agendas_confidence_value_dict = {k: 0.0 for k in date_set}
# concern_confidence_value_dict = {k: 0.0 for k in date_set}
# emotion_confidence_value_dict = {k: 0.0 for k in date_set}

# for row in tqdm(twitter_data.itertuples(), total=len(twitter_data)):
#     date = row[-1]
#     annotations = (row[9])[0]
#     # usc_annotations = row[8]
#     # print(annotations)
#     # print(usc_annotations)
#     # break
#
#     confidence_times_dict[date] += 1
#
#     for agenda in agenda_list:
#         agendas_confidence_value_dict[date] += annotations[agenda]
#
#     for emotion in emotion_list:
#         emotion_confidence_value_dict[date] += annotations[emotion]
#
#     for concern in concern_list:
#         concern_confidence_value_dict[date] += annotations[concern]
#
# agendas_date_dict = {k: agendas_confidence_value_dict[k] / confidence_times_dict[k] if confidence_times_dict[k]>0 else 0.0 for k in date_set}
# concern_date_dict = {k: concern_confidence_value_dict[k] / confidence_times_dict[k] if confidence_times_dict[k]>0 else 0.0 for k in date_set}
# emotion_date_dict = {k: emotion_confidence_value_dict[k] / confidence_times_dict[k] if confidence_times_dict[k]>0 else 0.0 for k in date_set}
#
# # print(confidence_times_dict)
# # print(agendas_date_dict)
# # print(concern_date_dict)
# # print(emotion_date_dict)
#
# confidence_date_df = pd.DataFrame(confidence_times_dict.items(), columns=['date', 'times'])
# agendas_date_df = pd.DataFrame(agendas_date_dict.items(), columns=['date', 'agenda'])
# concern_date_df = pd.DataFrame(concern_date_dict.items(), columns=['date', 'concern'])
# emotion_date_df = pd.DataFrame(emotion_date_dict.items(), columns=['date', 'emotion'])
#
# confidence_date_df = confidence_date_df.sort_values(by='date')
# agendas_date_df = agendas_date_df.sort_values(by='date')
# concern_date_df = concern_date_df.sort_values(by='date')
# emotion_date_df = emotion_date_df.sort_values(by='date')
#
# final_df = confidence_date_df.join(agendas_date_df['agenda'])
# final_df = final_df.join(concern_date_df['concern'])
# final_df = final_df.join(emotion_date_df['emotion'])
#
# # print(final_df)
#
# final_df.to_pickle("ground_truth_ace.p")

filter_agenda_df_list = []
filter_concern_df_list = []
filter_emotion_df_list = []

for row in tqdm(twitter_data.itertuples(), total=len(twitter_data)):
    date = row[-1]
    annotations = (row[8])
    # print(annotations)

    filter_agenda_dict = {k: 0.0 for k in agenda_list}
    filter_concern_dict = {k: 0.0 for k in concern_list}
    filter_emotion_dict = {k: 0.0 for k in emotion_list}

    for annotation in annotations:
        annotation_type = annotation["type"]
        if 'agenda' in annotation_type:
            filter_agenda_dict[annotation_type] = annotation["confidence"]
        elif 'concern' in annotation_type:
            filter_concern_dict[annotation_type] = annotation["confidence"]
        elif 'emotion' in annotation_type:
            filter_emotion_dict[annotation_type] = annotation["confidence"]

    # print(filter_agenda_dict)
    # print(filter_concern_dict)
    # print(filter_emotion_dict)

    new_agenda_dict = {}
    new_concern_dict = {}
    new_emotion_dict = {}

    for agenda in agenda_list:
        k = agenda_dict[agenda]
        new_agenda_dict[k] = filter_agenda_dict[agenda]

    for concern in concern_list:
        k = concern_dict[concern]
        new_concern_dict[k] = filter_concern_dict[concern]

    for emotion in emotion_list:
        k = emotion_dict[emotion]
        new_emotion_dict[k] = filter_emotion_dict[emotion]

    # print(new_agenda_dict)
    # print(new_concern_dict)
    # print(new_emotion_dict)
    #
    # break
    #
    filter_agenda_df = pd.DataFrame(new_agenda_dict, index=[0])
    filter_concern_df = pd.DataFrame(new_concern_dict, index=[0])
    filter_emotion_df = pd.DataFrame(new_emotion_dict, index=[0])

    filter_agenda_df_list.append(filter_agenda_df)
    filter_concern_df_list.append(filter_concern_df)
    filter_emotion_df_list.append(filter_emotion_df)

final_agenda_df = pd.concat(filter_agenda_df_list, ignore_index=True)
final_concern_df = pd.concat(filter_concern_df_list, ignore_index=True)
final_emotion_df = pd.concat(filter_emotion_df_list, ignore_index=True)

print(len(final_agenda_df))
print(len(final_concern_df))
print(len(final_emotion_df))

# print(final_agenda_df.sample(3))
# print(final_concern_df.sample(3))
# print(final_emotion_df.sample(20))

final_agenda_df.to_pickle("usc_agenda_5k.p")
final_concern_df.to_pickle("usc_concern_5k.p")
final_emotion_df.to_pickle("usc_emotion_5k.p")

