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


# df = pd.read_pickle("ground_truth_ace.p")
# print(len(df))
# print(df)
#
#
# fig, ax = plt.subplots(figsize=[8, 6])
#
#
# # plot_1 = ax.plot(df['date'],
# #          df['agenda'],
# #          color='lightcoral', linewidth=2, label='agenda')
# #
# # plot_3 = ax.plot(df['date'],
# #          df['concern'],
# #          color='wheat', linewidth=2, label='concern')
# #
# # plot_4 = ax.plot(df['date'],
# #          df['emotion'],
# #          color='cornflowerblue', linewidth=2, label='emotion')
#
# # ax2 = ax.twinx()
# #
# plot_2 = ax.plot(df['date'],
#          df['times'],
#          color='black', linewidth=2, label='num_tweets')
#
# # lns = plot_1 + plot_2
# # labels = [l.get_label() for l in lns]
# # ax.xticks(rotation=30, ha='right')
# # ax2.xticks(rotation=30, ha='right')
# # plt.legend(lns, labels, loc=0)
#
#
# plt.xticks(rotation=30, ha='right')
# plt.title("Number of tweet")
# plt.legend()
# plt.savefig("times_5k")
# plt.show()

agenda = [
    "Believe that the election process is flawed and/or manipulated by ENTITY (including potential foreign interference)",
    "Believe that the election process is fair and has not been manipulated",
    "Believe that ENTITY or GROUP is immoral/unethical/dishonest/harmful",
    "Believe that ENTITY or GROUP is moral/ethical/honest/beneficial",
    "Believe that you/GROUP are at risk",
    "Believe that a good outcome/hope for GROUP is possible",
    "Share information and opinions author endorses",
    "Vote for ENTITY",
    "Vote against ENTITY",
    "Vote",
    "Don’t vote",
    "Take action: protest/demonstrate/attend rally/volunteer/campaign"
]

concern = [
    "Economy",
    "Terrorism and counterterrorism",
    "Religion",
    "Immigration and refugees",
    "International alliance organizations",
    "Relationship with Russia",
    "National Identity and national pride",
    "Environment and climate change",
    "Fake news/misinformation",
    "Character of ENTITY (candidates, other key figures)",
    "Democracy"
]

emotion = [
    "Anger, hate, contempt, disgust",
    "Embarrassment, guilt, shame, sadness",
    "Admiration, love",
    "Optimism, hope",
    "Joy, happiness",
    "Pride, incl. national pride",
    "Fear, pessimism",
    "Amusement",
    "Positive-other",
    "Negative-other"
]

agenda_usc = [
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

concern_usc = [
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

emotion_usc = [
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

agenda_df = pd.read_pickle("agenda_5k.p")
concern_df = pd.read_pickle("concern_5k.p")
emotion_df = pd.read_pickle("emotion_5k.p")

usc_agenda_df = pd.read_pickle("usc_agenda_5k.p")
usc_concern_df = pd.read_pickle("usc_concern_5k.p")
usc_emotion_df = pd.read_pickle("usc_emotion_5k.p")


corr = pd.concat([concern_df, usc_concern_df], axis=1).corr().round(2)

corr_matrix = corr[concern].loc[concern_usc]

print(corr_matrix)

corr_matrix.to_json("usc_true_concern_corr_matrix_5k")


# mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
# # print(corr_matrix)
#
# fig, ax = plt.subplots(figsize=[8, 6])
#
# sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", mask=mask)
# plt.xticks(rotation=15, ha='right')
# # plt.yticks(np.arange(0.55, 0.75, 0.02))
# plt.title("Correlation Matrix (agenda)")
# # ax.set_xlabel('Protagonist')
# # ax.set_ylabel('USC')
# # plt.legend()
# # plt.savefig("corr_matrix_emotion")
# plt.show()









