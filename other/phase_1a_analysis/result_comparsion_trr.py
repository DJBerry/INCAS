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

# tweet_df = twitter_data[twitter_data["engagementType"] == "tweet"]
# retweet_df = twitter_data[twitter_data["engagementType"] == "retweet"]
# reply_df = twitter_data[twitter_data["engagementType"] == "reply"]

iter_type = ["tweet", "retweet", "reply"]
for type in iter_type:
    print("parse {} ...".format(type))
    target_df = twitter_data[twitter_data["engagementType"] == type]
    date_set = set(target_df['date'].tolist())

    confidence_times_dict = {k: 0 for k in date_set}
    confidence_value_dict = {k: 0.0 for k in date_set}

    agendas_confidence_times_dict = {k: 0 for k in date_set}
    agendas_confidence_value_dict = {k: 0.0 for k in date_set}

    concern_confidence_times_dict = {k: 0 for k in date_set}
    concern_confidence_value_dict = {k: 0.0 for k in date_set}

    emotion_confidence_times_dict = {k: 0 for k in date_set}
    emotion_confidence_value_dict = {k: 0.0 for k in date_set}

    agendas_df_list = []
    concern_df_list = []
    emotion_df_list = []

    for row in tqdm(target_df.itertuples(), total=len(target_df)):
        date = row[-1]
        annotations = row[7]
        annotation_df = pd.DataFrame(annotations)
        # print(annotation_df)

        # all
        confidence_times = len(annotation_df)
        confidence_times_dict[date] += confidence_times
        confidence_list = annotation_df['confidence'].tolist()
        confidence_value_dict[date] += sum(confidence_list)

        # agendas, concern, emotion
        agendas_df = annotation_df[annotation_df["type"].str.contains('agenda')]
        concern_df = annotation_df[annotation_df["type"].str.contains('concern')]
        emotion_df = annotation_df[annotation_df["type"].str.contains('emotion')]

        agendas_times = len(agendas_df)
        concern_times = len(concern_df)
        emotion_times = len(emotion_df)

        agendas_confidence_times_dict[date] += agendas_times
        concern_confidence_times_dict[date] += concern_times
        emotion_confidence_times_dict[date] += emotion_times

        agendas_list = agendas_df['confidence'].tolist()
        concern_list = concern_df['confidence'].tolist()
        emotion_list = emotion_df['confidence'].tolist()

        agendas_confidence_value_dict[date] += sum(agendas_list)
        concern_confidence_value_dict[date] += sum(concern_list)
        emotion_confidence_value_dict[date] += sum(emotion_list)

        # extract ace
        agendas_df = annotation_df[annotation_df["type"].str.contains('agenda')]
        concern_df = annotation_df[annotation_df["type"].str.contains('concern')]
        emotion_df = annotation_df[annotation_df["type"].str.contains('emotion')]

        agendas_df['date'] = date
        concern_df['date'] = date
        emotion_df['date'] = date

        agendas_df_list.append(agendas_df)
        concern_df_list.append(concern_df)
        emotion_df_list.append(emotion_df)

    confidence_date_dict = {k: confidence_value_dict[k] / confidence_times_dict[k] for k in date_set}

    agendas_date_dict = {k: agendas_confidence_value_dict[k] / agendas_confidence_times_dict[k] if agendas_confidence_times_dict[k]>0 else 0.0 for k in date_set}
    concern_date_dict = {k: concern_confidence_value_dict[k] / concern_confidence_times_dict[k] if concern_confidence_times_dict[k]>0 else 0.0 for k in date_set}
    emotion_date_dict = {k: emotion_confidence_value_dict[k] / emotion_confidence_times_dict[k] if emotion_confidence_times_dict[k]>0 else 0.0 for k in date_set}

    confidence_date_df = pd.DataFrame(confidence_date_dict.items(), columns=['date', 'confidence_all'])
    agendas_date_df = pd.DataFrame(agendas_date_dict.items(), columns=['date', 'agenda'])
    concern_date_df = pd.DataFrame(concern_date_dict.items(), columns=['date', 'concern'])
    emotion_date_df = pd.DataFrame(emotion_date_dict.items(), columns=['date', 'emotion'])

    confidence_date_df = confidence_date_df.sort_values(by='date')
    agendas_date_df = agendas_date_df.sort_values(by='date')
    concern_date_df = concern_date_df.sort_values(by='date')
    emotion_date_df = emotion_date_df.sort_values(by='date')

    final_df = confidence_date_df.join(agendas_date_df['agenda'])
    final_df = final_df.join(concern_date_df['concern'])
    final_df = final_df.join(emotion_date_df['emotion'])

    final_df.to_pickle("compare/{}/{}_all_ace_{}.p".format(JSON_FILE, type, JSON_FILE))

    agendas_all_df = pd.concat(agendas_df_list, ignore_index=True)
    concern_all_df = pd.concat(concern_df_list, ignore_index=True)
    emotion_all_df = pd.concat(emotion_df_list, ignore_index=True)

    agendas_all_df = agendas_all_df.sort_values(by='date')
    concern_all_df = concern_all_df.sort_values(by='date')
    emotion_all_df = emotion_all_df.sort_values(by='date')

    agendas_all_df = agendas_all_df.drop(["providerName", "id", "name"], axis=1)
    concern_all_df = concern_all_df.drop(["providerName", "id", "name"], axis=1)
    emotion_all_df = emotion_all_df.drop(["providerName", "id", "name"], axis=1)

    # print(agendas_all_df)

    agendas_all_df.to_pickle("compare/{}/{}_agenda_{}.p".format(JSON_FILE, type, JSON_FILE))
    concern_all_df.to_pickle("compare/{}/{}_concern_{}.p".format(JSON_FILE, type, JSON_FILE))
    emotion_all_df.to_pickle("compare/{}/{}_emotion_{}.p".format(JSON_FILE, type, JSON_FILE))


