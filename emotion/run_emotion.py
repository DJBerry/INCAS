import numpy as np
import pandas as pd
from tqdm import tqdm
from model import annotate

pd.set_option('display.max_columns', None)

# DATA_DIR = 'Phase_1B_data/'
# JSON_FILE = 'english_fr_politics_twt_sample_merged_2022-07-13 02_40_01.json'
#
# df = pd.read_json('{}{}'.format(DATA_DIR, JSON_FILE), lines=True)
# # print(len(df))
# # print(df.head(5))
#
# structural_features_raw = []
# for row in tqdm(df.itertuples(), total=len(df)):
#     id = row[2]
#     text = row[4]
#     input = [{"id": id, "contentText": text}]
#     output = annotate(input)
#
#     structural_features_raw.append({
#         "annotate_emotion": output
#     })
# structural_features_df = pd.DataFrame(structural_features_raw, index=df.index)
# result_df = pd.concat([df, structural_features_df], axis=1)
#
# print(result_df.sampel(5))
#
# result_df.to_pickle("annotate_emotion_phase_1b_sample_dataset.p")

# df = pd.read_csv('GoEmotion.tsv', sep='\t', names=["text", "unknown", "id"], header=None)
# df = pd.read_csv('SemEval2018.txt', sep='\t')
df = pd.read_pickle("get_emotions.pkl")

error = 0
emotion_features_raw = []
for row in tqdm(df.itertuples(), total=len(df)):

    text = row[2]
    input_model = [{"id": 0, "contentText": text}]
    try:
        output_model = annotate(input_model)
        emotion = {}
        for output_dict in output_model:
            emotion_type = output_dict["type"]
            confidence = output_dict["confidence"]
            emotion[emotion_type] = confidence
    except TypeError:
        emotion = 0
        error += 1

    emotion_features_raw.append({
        "emotion": emotion
    })

print("error: {}".format(error))

emotion_features_df = pd.DataFrame(emotion_features_raw, index=df.index)
result_df = pd.concat([df, emotion_features_df], axis=1)
# result_df.to_pickle("annotated_dataset/annotated_emotion_{}_discussion.p".format(SUBREDDIT))
result_df.to_pickle("annotated_get_emotion.plk")




