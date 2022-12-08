import os
import json
from tqdm import tqdm
filespath='/project/lerman_316/INCAS_Phase1B_Eval/sorted_t2_final_with_srl_annot'
savepath='/project/lerman_316/INCAS_Phase1B_Eval/final_combine/srl_slice2.jsonl'
files=os.listdir(filespath)
files.sort()
all_data=[]
for file in tqdm(files):
    with open(os.path.join(filespath,file),'r') as f:
        data=f.readlines()
        data=[d for d in data if "\"confidence\":" in d]
        data=[json.loads(d) for d in data]
        all_data.extend(data)
print(len(all_data))
with open(os.path.join(savepath),'w') as f:
    for m in all_data:
        f.write(json.dumps(m))
        f.write('\n')
