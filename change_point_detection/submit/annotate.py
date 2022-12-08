import numpy as np
import pandas as pd
import metachange
from model import ChangePointIndicator
import json
from incas_python.api.model.annotation import Annotation
from incas_python.api.model import Message, ListOfAnnotation


def annotate(sampling_size=1000, feature_size=1000, min_range=86400, max_depth=3):
    """
        Args:
            messages: list of dictionaries [{'id':xxx,'contentText':xxx,...},...]
            sampling_size: sampling size of each date, default: 1000
            feature_size: tfidf feature size, default: 1000
            min_range: min range of detection, default: 86400
            max_depth: max depth of change point tree, default: 3

        Returns:
            A dict whose keys are message IDs and values are lists of annotations.
    """
    # only use twitter dataset
    print("loading json file ...")

    # message_file_1 = open("sorted_t1_final.jsonl")
    # messages = [json.loads(line) for line in message_file_1]
    # message_file_2 = open("sorted_t2_final.jsonl")
    # messages.extend([json.loads(line) for line in message_file_2])
    message_file = open("formal_eval.jsonl")
    messages = [json.loads(line) for line in message_file]

    contentText_list = []
    timePublished_list = []
    for message in messages:

        # Skip empty contentText
        if not message["contentText"] or message["contentText"] == "":
            continue

        if not message["timePublished"]:
            continue

        # contentText_list.append(message["contentText"])
        # timePublished_list.append(int(message["timePublished"]))

        if message["mediaType"] == "Twitter" or message["mediaTypeAttributes"]:
            if message["mediaTypeAttributes"]["twitterData"] != "null":
                contentText_list.append(message["contentText"])
                timePublished_list.append(int(message["timePublished"]))
            else:
                continue
        else:
            continue


    # for message in messages:
    #     contentText_list.append(message["contentText"])
    #     timePublished_list.append(int(message["timePublished"]))

    twitter_df = pd.DataFrame(list(zip(contentText_list, timePublished_list)),
                              columns=['contentText', 'timePublished'])
    # print(twitter_df)
    changepoint_indicator = ChangePointIndicator(sampling_size, feature_size, min_range, max_depth)
    changepoint_indicator.run(twitter_df)

    print("annotating messages ...")
    annotations = {}
    for message in messages:
        id = message["id"]
        time = int(message["timePublished"])
        changepoint_results = changepoint_indicator.annotate(time)
        results = []
        for changepoint_result in changepoint_results:
            results.append(
                Annotation(
                    id=id,
                    type="additional-change_point",
                    text=json.dumps({
                        "datetime": changepoint_result[0].isoformat(),
                        "depth": changepoint_result[2]
                    }),
                    confidence=changepoint_result[1],
                    offsets=[],
                    providerName="ta1-usc-isi"
                ).__dict__
            )
        annotations[id] = results

    return annotations

annotations = annotate(sampling_size=1000, feature_size=500, min_range=86400, max_depth=3)
with open("annotated_change_point_10k_dataset.jsonl", 'w') as f:
    f.write(json.dumps(annotations))
