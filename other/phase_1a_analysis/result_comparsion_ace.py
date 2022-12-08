import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

pd.set_option('display.max_columns', None)

DATA_DIR = 'dataset/'
JSON_FILE = 'master_timeslice_two_remainder_actor_filtered_protagonist'

df = pd.read_json('{}/{}.json'.format(DATA_DIR, JSON_FILE), lines=True)
# df = pd.read_pickle("sample_1k_tweet_master_timeslice_two_remainder_actor_filtered_usc_ta1.json.p")
# print(df)

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

twitter_data = result_df.drop(["mediaTypeAttributes", "mediaType", "embeddedUrls", "imageUrls", "dataTags"], axis=1)
# print(twitter_data)

agendas_list = []
concern_list = []
emotion_list = []

for row in tqdm(twitter_data.itertuples(), total=len(twitter_data)):
    date = row[-1]
    annotations = row[7]
    annotation_df = pd.DataFrame(annotations)
    # print(annotation_df)

    agendas_df = annotation_df[annotation_df["type"].str.contains('agenda')]
    concern_df = annotation_df[annotation_df["type"].str.contains('concern')]
    emotion_df = annotation_df[annotation_df["type"].str.contains('emotion')]

    agendas_df['date'] = date
    concern_df['date'] = date
    emotion_df['date'] = date

    agendas_list.append(agendas_df)
    concern_list.append(concern_df)
    emotion_list.append(emotion_df)

agendas_all_df = pd.concat(agendas_list, ignore_index=True)
concern_all_df = pd.concat(concern_list, ignore_index=True)
emotion_all_df = pd.concat(emotion_list, ignore_index=True)

agendas_all_df = agendas_all_df.sort_values(by='date')
concern_all_df = concern_all_df.sort_values(by='date')
emotion_all_df = emotion_all_df.sort_values(by='date')

agendas_all_df = agendas_all_df.drop(["providerName", "id", "name"], axis=1)
concern_all_df = concern_all_df.drop(["providerName", "id", "name"], axis=1)
emotion_all_df = emotion_all_df.drop(["providerName", "id", "name"], axis=1)

agendas_all_df.to_pickle("compare/{}/agenda_{}.p".format(JSON_FILE, JSON_FILE))
concern_all_df.to_pickle("compare/{}/concern_{}.p".format(JSON_FILE, JSON_FILE))
emotion_all_df.to_pickle("compare/{}/emotion_{}.p".format(JSON_FILE, JSON_FILE))

