import statistics
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

pd.set_option('display.max_columns', None)

df = pd.read_json('dataset/1_mil_final_incas_format.jsonl', lines=True)
df = df[df["mediaType"] == "Twitter"]
df['datetime'] = [datetime.fromtimestamp(time/1000) for time in df['timePublished'].values]
dates = [pd.to_datetime(dt).date() for dt in df['datetime'].values]
df['date'] = dates
df = df.sort_values(by='date')
df = df[df["date"] != np.datetime64('2017-02-28')]

structural_features_raw = []
for row in tqdm(df.itertuples(), total=len(df)):
    twitter_data = row[11]
    twitter_features_dict = twitter_data['twitterData']
    # print(twitter_features_dict)
    structural_features_raw.append(twitter_features_dict)
    # break

structural_features_df = pd.DataFrame(structural_features_raw, index=df.index)
df = pd.concat([df, structural_features_df], axis=1)

finial_df = df.drop(["name", "mentionedUsers", ""], axis=1)

