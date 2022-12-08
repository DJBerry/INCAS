import statistics
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed
import re
from collections import Counter
import random

pd.set_option('display.max_columns', None)
MAX_PROCESSES = 10


DATA_DIR = 'dataset/'
JSON_FILE_1 = '{}master_timeslice_one_remainder_actor_filtered_protagonist.json'.format(DATA_DIR)
JSON_FILE_2 = '{}master_timeslice_two_remainder_actor_filtered_protagonist.json'.format(DATA_DIR)

# df_1 = pd.read_json(JSON_FILE_1, lines=True)
df = pd.read_json(JSON_FILE_1, lines=True)
# df_2 = pd.read_json(JSON_FILE_2, lines=True)
# df = pd.concat([df_1, df_2], ignore_index=True)

# df = pd.read_pickle("sample_1k_tweet_master_timeslice_two_remainder_actor_filtered_protagonist.p")
# print(df.sample(5))

# target_df = pd.read_csv("1a_annotations_released_20220524.csv")
# print(len(target_df))
# print(target_df.sample(5))

target_list = ["3f26bcb7789db7e4e38c8a01587a22c78efd53b1", "d6b16efb6e3a73fc289bcb6ea0ac05d091b88fcf",
               "e923bb0ae6b69fd93d12cbc7a150de3555c4dd46", "368550dc868c09cd63b4e9e80acecfbdead5860f",
               "cc2376b9864ecf67e50a9383c23ef7937642d533"]

is_find = 0
structural_features_raw = []
for target_id in target_list:
    match_df = df[df["id"] == target_id]

    # if match_df.empty:
    #     continue
    #
    # is_find += 1
    # print("find {}".format(is_find))
    structural_features_raw.append(match_df)

match_dfs = pd.concat(structural_features_raw)
match_dfs.to_pickle("match_5_df.p")

# test_list = ["62528b1e3f74f6ebd569fddc98d2d540371f5b88", "dc194569a422b4273edb502a75e2b4ec4c54f539",
#              "ce124ec50253ca52828c698d06b2596447f94e49", "95598a3e739685ca0e4122bbc449b481331bc8fc",
#              "38c24ba844a1c7e451eac43e3ee4a85bb63b440d"]
#
# for test_id in test_list:
#     test_df = df[df["id"] == test_id]
#     if test_df.empty:
#         continue
#     print("got 1 test id")

# is_find = 0
# structural_features_raw = []
# for row in tqdm(target_df.itertuples(), total=len(target_df)):
#     id = row[1]
#     match_df = df[df["id"] == id]
#
#     if match_df.empty:
#         continue
#
#     is_find += 1
#     print("find {}".format(is_find))
#     structural_features_raw.append(match_df)
#
# match_dfs = pd.concat(structural_features_raw)
# match_dfs.to_pickle("match_5_df.p")


#     title = match_df["title"]
#     contentText = match_df["contentText"]
#
#     timestamp = match_df["timePublished"]
#     language = match_df["language"]
#     mediaTypeAttributes = match_df["mediaTypeAttributes"]
#
#     structural_features_raw.append({
#         "title": title,
#         "contentText": contentText,
#         "timestamp": timestamp,
#         "language": language,
#         "mediaTypeAttributes": mediaTypeAttributes
#     })
#
# structural_features_df = pd.DataFrame(structural_features_raw, index=target_df.index)
# result_df = pd.concat([target_df, structural_features_df], axis=1)
#
# result_df = result_df[['ID', 'title', "contentText", 'agenda-1.1', 'agenda-1.2', 'agenda-1.3', 'agenda-1.4',
#                        'agenda-1.5', 'agenda-1.6', 'agenda-2.1', 'agenda-2.2.1', 'agenda-2.2.2', 'agenda-2.2.3',
#                        'agenda-2.2.4', 'agenda-2.3', 'concern-3.1', 'concern-3.2', 'concern-3.3', 'concern-3.4',
#                        'concern-3.5', 'concern-3.6', 'concern-3.7', 'concern-3.8', 'concern-3.9', 'concern-3.10',
#                        'concern-3.11', 'emotion-4.1', 'emotion-4.2', 'emotion-4.3', 'emotion-4.4', 'emotion-4.5',
#                        'emotion-4.6', 'emotion-4.7', 'emotion-4.8', 'emotion-4.9', 'emotion-4.10', 'timestamp',
#                        'language', 'mediaTypeAttributes']]
#
# result_df.to_pickle("matched_1k_data")

df2 = pd.read_pickle("match_5_df.p")
print(df2)


