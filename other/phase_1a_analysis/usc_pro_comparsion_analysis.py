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
MAX_PROCESSES = 10

DATASET_DIR = "compare"
USC_DIR = "master_timeslice_two_remainder_actor_filtered_usc_ta1"
PROTAGONIST_DIR = "master_timeslice_two_remainder_actor_filtered_protagonist"

tweet_type = "reply"
usc_file = "{}/{}/{}_all_ace_{}.p".format(DATASET_DIR, USC_DIR,tweet_type, USC_DIR)
protagonist_file = "{}/{}/{}_all_ace_{}.p".format(DATASET_DIR, PROTAGONIST_DIR, tweet_type, PROTAGONIST_DIR)

usc_df = pd.read_pickle(usc_file)
# protagonist_df = pd.read_pickle(protagonist_file)
print(usc_df)
# print(protagonist_df)

# iter_list = ["confidence_all", "agenda", "concern", "emotion"]
# for target in iter_list:
#     fig, ax = plt.subplots(figsize=[8, 6])
#
#     ax.plot(usc_df['date'],
#              usc_df[target],
#              color='lightcoral', linewidth=2, label='usc')
#
#     ax.plot(protagonist_df['date'],
#              protagonist_df[target],
#              color='wheat', linewidth=2, label='protagonist')
#
#     plt.xticks(rotation=30, ha='right')
#     # plt.yticks(np.arange(0.55, 0.75, 0.02))
#     plt.title("Result Comparison ({})".format(target))
#     # ax.set_xlabel('')
#     # ax.set_ylabel('result')
#     plt.legend()
#     plt.savefig("usc_pro_{}_{}.png".format(tweet_type,target))
#     plt.show()

# # print(usc_df)
# print(protagonist_df.sample(10))
#
# # type_list = protagonist_df["type"].tolist()
# text_list = protagonist_df["text"].tolist()
# # type_set = set(type_list)
# text_set = set(text_list)
# # print(type_set)
# print(text_set)
# #
# agenda_type_list = ['Pride, including National Pride', 'Positive-other', 'Fear, Pessimism', 'Admiration, Love', 'Optimism, Hope', 'Amusement', 'Anger, Hate, Contempt, Disgust', 'emotion not present', 'Negative-other', 'Embarrassment, Guilt, Shame, Sadness', 'emotion present', 'Joy, Happiness']
#
#
# # agenda
# # {'agenda-2.2.2', 'agenda-1.4', 'agenda-2.2.1', 'agenda-1.3'}
# # agenda_type_list = ['agenda-2.2.1', 'agenda-2.2.2', 'agenda-1.4', 'agenda-1.3']
# # agenda_type_exp = ['Vote for', 'Vote against', 'Believe that ENTITY or GROUP is moral\/ethical\/honest\/beneficial', 'Believe that ENTITY or GROUP is immoral\/unethical\/dishonest\/harmful']
# color = ['lightcoral', 'wheat', 'lightskyblue', 'seagreen', 'darkgray', 'brown', 'lightsalmon', 'lightgreen',
#          'dodgerblue', 'royalblue', 'blueviolet', 'magenta', 'pink', "blue", "orange", "green", "red", "purple", "olive",
#          "cyan", "yellow"]
# fig, ax = plt.subplots(figsize=[12, 6])
#
# for i in range(len(agenda_type_list)):
#     df = protagonist_df[protagonist_df["text"] == agenda_type_list[i]]
#
#     date_set = set(df['date'].tolist())
#     confidence_times_dict = {k: 0 for k in date_set}
#     confidence_value_dict = {k: 0.0 for k in date_set}
#
#     for row in tqdm(df.itertuples(), total=len(df)):
#         date = row[-1]
#         confidence = row[-2]
#         # confidence = row[1]
#         confidence_times_dict[date] += 1
#         confidence_value_dict[date] += confidence
#
#     confidence_date_dict = {k: confidence_value_dict[k] / confidence_times_dict[k] for k in date_set}
#     confidence_date_df = pd.DataFrame(confidence_date_dict.items(), columns=['date', 'confidence'])
#     confidence_date_df = confidence_date_df.sort_values(by='date')
#
#     ax.plot(confidence_date_df['date'],
#              confidence_date_df['confidence'],
#              color=color[i], linewidth=2, label=agenda_type_list[i])
#
#     # break
#
# plt.xticks(rotation=30, ha='right')
# # plt.yticks(np.arange(0.55, 0.75, 0.02))
# plt.title("Result Comparison (emotion)")
# # ax.set_xlabel('')
# # ax.set_ylabel('result')
# # plt.legend()
# # Shrink current axis by 20%
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#
# # Put a legend to the right of the current axis
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#
# plt.savefig("pro_emotion.png")
# plt.show()


# fig, ax = plt.subplots(figsize=[8, 6])
#
# ax.plot(usc_df['date'],
#          usc_df['emotion'],
#          color='lightcoral', linewidth=2, label='usc')
#
# ax.plot(protagonist_df['date'],
#          protagonist_df['emotion'],
#          color='wheat', linewidth=2, label='protagonist')
#
# plt.xticks(rotation=30, ha='right')
# # plt.yticks(np.arange(0.55, 0.75, 0.02))
# plt.title("Result Comparison (emotion)")
# # ax.set_xlabel('')
# # ax.set_ylabel('result')
# plt.legend()
# # plt.savefig("usc_pro_emotion.png")
# plt.show()


fig, ax = plt.subplots(figsize=[8, 6])

ax.plot(usc_df['date'],
         usc_df['confidence_all'],
         color='seagreen', linewidth=2, label='all')
ax.plot(usc_df['date'],
         usc_df['agenda'],
         color='wheat', linewidth=2, label='agenda')
ax.plot(usc_df['date'],
         usc_df['concern'],
         color='lightskyblue', linewidth=2, label='concern')
ax.plot(usc_df['date'],
         usc_df['emotion'],
         color='lightcoral', linewidth=2, label='emotion')

plt.xticks(rotation=30, ha='right')
# plt.yticks(np.arange(0.55, 0.75, 0.02))
plt.title("Result from USC-TA1")
# ax.set_xlabel('')
# ax.set_ylabel('result')
plt.legend()
plt.savefig("usc_{}.png".format(tweet_type))
plt.show()




