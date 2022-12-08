import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

pd.set_option('display.max_columns', None)

DATA_DIR = 'dataset/'
JSON_FILE = 'master_both_timeslice_one_remainder_actor_filterd_usc_ta1'

# df = pd.read_json('{}/{}.json'.format(DATA_DIR, JSON_FILE), lines=True)
df = pd.read_pickle("sample_tweet_master_both_timeslice_one_remainder_actor_filterd_usc_ta1.json.p")
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
print(twitter_data)
# twitter_data = twitter_data.drop([1175375])

date_set = set(twitter_data['date'].tolist())

confidence_times_dict = {k: 0 for k in date_set}
confidence_value_dict = {k: 0.0 for k in date_set}

agendas_confidence_times_dict = {k: 0 for k in date_set}
agendas_confidence_value_dict = {k: 0.0 for k in date_set}

concern_confidence_times_dict = {k: 0 for k in date_set}
concern_confidence_value_dict = {k: 0.0 for k in date_set}

emotion_confidence_times_dict = {k: 0 for k in date_set}
emotion_confidence_value_dict = {k: 0.0 for k in date_set}

for row in tqdm(twitter_data.itertuples(), total=len(twitter_data)):
    date = row[-1]
    annotations = row[7]
    # print(annotations)
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

    # tweet, retweet, reply
    #
    #
    # break

confidence_date_dict = {k: confidence_value_dict[k] / confidence_times_dict[k] for k in date_set}

agendas_date_dict = {k: agendas_confidence_value_dict[k] / agendas_confidence_times_dict[k] if agendas_confidence_times_dict[k]>0 else 0.0 for k in date_set}
concern_date_dict = {k: concern_confidence_value_dict[k] / concern_confidence_times_dict[k] if concern_confidence_times_dict[k]>0 else 0.0 for k in date_set}
emotion_date_dict = {k: emotion_confidence_value_dict[k] / emotion_confidence_times_dict[k] if emotion_confidence_times_dict[k]>0 else 0.0 for k in date_set}

# print(confidence_date_dict)
# print(agendas_date_dict)
# print(concern_date_dict)
# print(emotion_date_dict)
#
confidence_date_df = pd.DataFrame(confidence_date_dict.items(), columns=['date', 'confidence_all'])
agendas_date_df = pd.DataFrame(agendas_date_dict.items(), columns=['date', 'agenda'])
concern_date_df = pd.DataFrame(concern_date_dict.items(), columns=['date', 'concern'])
emotion_date_df = pd.DataFrame(emotion_date_dict.items(), columns=['date', 'emotion'])

confidence_date_df = confidence_date_df.sort_values(by='date')
agendas_date_df = agendas_date_df.sort_values(by='date')
concern_date_df = concern_date_df.sort_values(by='date')
emotion_date_df = emotion_date_df.sort_values(by='date')

# print(confidence_date_df)
# print(agendas_date_df)

final_df = confidence_date_df.join(agendas_date_df['agenda'])
final_df = final_df.join(concern_date_df['concern'])
final_df = final_df.join(emotion_date_df['emotion'])
# print(final_df)

final_df.to_pickle("compare/{}/all_ace_{}.p".format(JSON_FILE, JSON_FILE))



# import pandas as pd
# import numpy as np
# from datetime import datetime
# from tqdm import tqdm
#
# pd.set_option('display.max_columns', None)
#
# DATA_DIR = 'dataset/'
# JSON_FILE = 'master_both_timeslice_one_remainder_actor_filterd_usc_ta1'
#
# df = pd.read_json('{}/{}.json'.format(DATA_DIR, JSON_FILE), lines=True)
# # df = pd.read_pickle("sample_tweet_master_both_timeslice_one_remainder_actor_filterd_usc_ta1.json.p")
# # df = pd.read_pickle("bug_data_master_timeslice_two_remainder_actor_filtered_usc_ta1.p")
# # print(df)
#
# structural_features_raw = []
# for row in tqdm(df.itertuples(), total=len(df)):
#     twitter_data = row[6]
#     twitter_features_dict = twitter_data['twitterData']
#     structural_features_raw.append(twitter_features_dict)
#
# structural_features_df = pd.DataFrame(structural_features_raw, index=df.index)
# result_df = pd.concat([df, structural_features_df], axis=1)
#
# result_df['timePublished'] = [datetime.fromtimestamp(time/1000) for time in result_df['timePublished'].values]
# dates = [pd.to_datetime(dt).date() for dt in result_df['timePublished'].values]
# result_df['date'] = dates
#
# result_df = result_df.sort_values(by='date')
#
# twitter_data = result_df.drop(["mediaTypeAttributes", "mediaType", "embeddedUrls", "imageUrls", "dataTags"], axis=1)
# # print(twitter_data)
# # twitter_data = twitter_data.drop([1175375])
#
# date_set = set(twitter_data['date'].tolist())
#
# confidence_times_dict = {k: 0 for k in date_set}
# confidence_value_dict = {k: 0.0 for k in date_set}
#
# agendas_confidence_times_dict = {k: 0 for k in date_set}
# agendas_confidence_value_dict = {k: 0.0 for k in date_set}
#
# concern_confidence_times_dict = {k: 0 for k in date_set}
# concern_confidence_value_dict = {k: 0.0 for k in date_set}
#
# emotion_confidence_times_dict = {k: 0 for k in date_set}
# emotion_confidence_value_dict = {k: 0.0 for k in date_set}
#
# for row in tqdm(twitter_data.itertuples(), total=len(twitter_data)):
#     date = row[-1]
#     annotations = row[7]
#     # print(annotations)
#     # annotation_df = pd.DataFrame(annotations)
#     # print(annotation_df)
#
#     # all
#     try:
#         confidence_times = len(annotations)
#     except TypeError:
#         print(row)
#         print("===================")
#         print(annotations)
#         exit()
#     else:
#         confidence_times_dict[date] += confidence_times
#         confidence_list = [annotation['confidence'] for annotation in annotations]
#         confidence_value_dict[date] += sum(confidence_list)
#
#         # agendas, concern, emotion
#
#         agendas = []
#         concern = []
#         emotion = []
#
#         for annotation in annotations:
#             if 'agenda' in annotation['type']:
#                 agendas.append(annotation)
#             elif 'concern' in annotation['type']:
#                 concern.append(annotation)
#             elif 'emotion' in annotation['type']:
#                 emotion.append(annotation)
#
#         agendas_df = pd.DataFrame(agendas)
#         concern_df = pd.DataFrame(concern)
#         emotion_df = pd.DataFrame(emotion)
#
#         agendas_times = len(agendas_df)
#         concern_times = len(concern_df)
#         emotion_times = len(emotion_df)
#
#         agendas_confidence_times_dict[date] += agendas_times
#         concern_confidence_times_dict[date] += concern_times
#         emotion_confidence_times_dict[date] += emotion_times
#
#         agendas_list = agendas_df['confidence'].tolist()
#         concern_list = concern_df['confidence'].tolist()
#         emotion_list = emotion_df['confidence'].tolist()
#
#         agendas_confidence_value_dict[date] += sum(agendas_list)
#         concern_confidence_value_dict[date] += sum(concern_list)
#         emotion_confidence_value_dict[date] += sum(emotion_list)
#
#     # tweet, retweet, reply
#     #
#     #
#     # break
#
# confidence_date_dict = {k: confidence_value_dict[k] / confidence_times_dict[k] for k in date_set}
#
# agendas_date_dict = {k: agendas_confidence_value_dict[k] / agendas_confidence_times_dict[k] if agendas_confidence_times_dict[k]>0 else 0.0 for k in date_set}
# concern_date_dict = {k: concern_confidence_value_dict[k] / concern_confidence_times_dict[k] if concern_confidence_times_dict[k]>0 else 0.0 for k in date_set}
# emotion_date_dict = {k: emotion_confidence_value_dict[k] / emotion_confidence_times_dict[k] if emotion_confidence_times_dict[k]>0 else 0.0 for k in date_set}
#
# # print(confidence_date_dict)
# # print(agendas_date_dict)
# # print(concern_date_dict)
# # print(emotion_date_dict)
# #
# confidence_date_df = pd.DataFrame(confidence_date_dict.items(), columns=['date', 'confidence_all'])
# agendas_date_df = pd.DataFrame(agendas_date_dict.items(), columns=['date', 'agenda'])
# concern_date_df = pd.DataFrame(concern_date_dict.items(), columns=['date', 'concern'])
# emotion_date_df = pd.DataFrame(emotion_date_dict.items(), columns=['date', 'emotion'])
#
# confidence_date_df = confidence_date_df.sort_values(by='date')
# agendas_date_df = agendas_date_df.sort_values(by='date')
# concern_date_df = concern_date_df.sort_values(by='date')
# emotion_date_df = emotion_date_df.sort_values(by='date')
#
# # print(confidence_date_df)
# # print(agendas_date_df)
#
# final_df = confidence_date_df.join(agendas_date_df['agenda'])
# final_df = final_df.join(concern_date_df['concern'])
# final_df = final_df.join(emotion_date_df['emotion'])
# # print(final_df)
#
# # final_df.to_pickle("compare/{}/all_ace_{}.p".format(JSON_FILE, JSON_FILE))


