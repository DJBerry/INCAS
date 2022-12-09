import os
import json
from glob import glob
from tqdm import tqdm
from itertools import islice

project_dir = "./"
message_dir = os.path.join(project_dir,'sorted_t1_final.jsonl')
annot_dir = os.path.join(project_dir,'annotated_change_point_dataset')
# output_dir = os.path.join(project_dir,'sorted_t1_final_with_change_point_annot')
file = "annotated_change_point_dataset.jsonl"

# messages = [json.loads(line) for line in open(message_dir, 'r')]
#print('total # messages:', len(messages))

batch_size = 30000

f = open(message_dir, 'r')
cnt = 0
while True:
    messages = list(islice(f,batch_size))
    messages = [json.loads(line) for line in messages]

    if not messages:
        break

    # print('processing annotations from: ',file)
    with open(file,'r') as fa:
        try:
            annotations = json.load(fa)
            # print('total # annotations:', len(annotations))
        except Exception as e:
            # print(e)
            annotations = fa.readlines()
            annotations=[json.loads(a) for a in annotations]
    for m in messages:
        id = m['id']
        try:
            annotation = annotations[id]
            m['annotations'].extend(annotation)
        except:
            #m['annotations'].extend(['No SRL'])
            pass

    with open("sorted_t1_final_with_change_point_annot_2.jsonl",'a') as fo:
        for m in messages:
            fo.write(json.dumps(m))
            fo.write('\n')
    #
    # cnt += 1
