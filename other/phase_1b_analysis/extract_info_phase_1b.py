import os
import json
from glob import glob
# from tqdm import tqdm
from itertools import islice

project_dir = '/project/lerman_316/INCAS_Phase1B_Eval/final_combine'
# message_dir = os.path.join(project_dir,'sorted_t1_final.jsonl')
annot_dir = os.path.join(project_dir,'TA1-USCISI-Slice1-20221108.jsonl')
output_dir = "annotated_mace_file_1.jsonl"
# files = glob(os.path.join(annot_dir,'*.jsonl'))

# messages = [json.loads(line) for line in open(message_dir, 'r')]
#print('total # messages:', len(messages))

annotations_list = ["framing-1.1", "framing-1.2", "framing-1.3", "framing-1.4", "framing-1.5", "framing-1.6",
                    "concern-3.1", "concern-3.2", "concern-3.3", "concern-3.4", "concern-3.5", "concern-3.6",
                    "concern-3.7", "concern-3.8", "concern-3.9",
                    "emotion-4.2", "emotion-4.1", "emotion-4.3", "emotion-4.4", "emotion-4.5", "emotion-4.6",
                    "emotion-4.7"]

batch_size = 3000

f = open(annot_dir, 'r')

while True:

    messages = list(islice(f, batch_size))
    messages = [json.loads(line) for line in messages]

    if not messages:
        break

    extract_infos = []
    for message in messages:
        timestamp = message["timePublished"]
        mediaType = message["mediaType"]
        if mediaType == "Twitter" or message["mediaTypeAttributes"]:
            if message["mediaTypeAttributes"]["twitterData"] != "null":
                twitterData = message["mediaTypeAttributes"]["twitterData"]
                twitterType = twitterData["engagementType"]

                annotations = message["annotations"]
                extract_annotations = {}
                for annotation in annotations:
                    annotate_type = annotation["type"]
                    if annotate_type in annotations_list:
                        confidence = annotation["confidence"]
                        extract_annotations[annotate_type] = confidence
                    else:
                        continue
                extract_info = {}
                extract_info["timePublished"] = timestamp
                extract_info["twitterType"] = twitterType
                extract_info["annotations"] = extract_annotations
                extract_infos.append(extract_info)
            else:
                continue
        else:
            continue

    with open(output_dir, 'a') as fo:
        for info in extract_infos:
            fo.write(json.dumps(info))
            fo.write('\n')
