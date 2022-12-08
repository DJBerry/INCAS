import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

pd.set_option('display.max_columns', None)

DATA_DIR = 'dataset/'
JSON_FILE = 'master_both_timeslice_one_remainder_actor_filterd_usc_ta1'

df = pd.read_json('{}/{}.json'.format(DATA_DIR, JSON_FILE), lines=True)
# df = pd.read_pickle("others/sample_tweet_master_both_timeslice_one_remainder_actor_filterd_usc_ta1.json.p")
# print(df.sample(5))

structural_features_raw = []
for row in tqdm(df.itertuples(), total=len(df)):
    twitter_data = row[6]
    twitter_features_dict = twitter_data['twitterData']
    structural_features_raw.append(twitter_features_dict)

structural_features_df = pd.DataFrame(structural_features_raw, index=df.index)
result_df = pd.concat([df, structural_features_df], axis=1)

result_df['datetime'] = [datetime.fromtimestamp(time/1000) for time in result_df['timePublished'].values]
dates = [pd.to_datetime(dt).date() for dt in result_df['datetime'].values]
result_df['date'] = dates

result_df = result_df.sort_values(by='date')
# print(result_df.sample(5))

twitter_data = result_df[["id", "timePublished", "language", "engagementType", "date"]]
# twitter_data = result_df.drop(["mediaTypeAttributes", "mediaType", "embeddedUrls", "imageUrls", "dataTags"], axis=1)
# print(twitter_data.sample(5))

twitter_data.to_pickle("twitter_date_{}.p".format(JSON_FILE))

