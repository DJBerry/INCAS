import pandas as pd
import numpy as np
from tqdm import tqdm

pd.set_option('display.max_columns', None)

# DATA_DIR = 'dataset/'
# # JSON_FILE = 'master_timeslice_two_remainder_actor_filtered_usc_ta1'
# JSON_FILE = 'master_timeslice_two_remainder_actor_filtered_protagonist'
#
# json_file = pd.read_json('{}/{}.json'.format(DATA_DIR, JSON_FILE), lines=True)
# json_file = json_file.sample(1000)
# # print(json_file)
# #
# json_file.to_pickle("sample_1k_tweet_{}.p".format(JSON_FILE))
#
# # df = pd.read_pickle("sample_tweet_master_both_timeslice_one_remainder_actor_filterd_usc_ta1.json.p")
# # # print(df)
#
# new_df = json_file.iloc[1175370:1175380]
# new_df.to_pickle("bug_data_{}.p".format(JSON_FILE))

# df = pd.read_pickle("bug_data_master_timeslice_two_remainder_actor_filtered_usc_ta1.p")
# # print(df)
# # df = df.iloc[1175373:1175377]
#
# for row in tqdm(df.itertuples(), total=len(df)):
#     annotation = row[-1]
#     # print(annotation)
#     annotation_df = pd.DataFrame(annotation)
#     print(annotation_df)


df = pd.read_pickle("sample_1k_tweet_master_timeslice_two_remainder_actor_filtered_protagonist.p")
print(df.sample(5))

for row in tqdm(df.itertuples(), total=len(df)):
    annotation = row[-1]
    print(annotation)

    annotation_df = pd.DataFrame(annotation)
    print(annotation_df)

    break



