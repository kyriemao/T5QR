from IPython import embed
import json
from tqdm import tqdm

# with open("/data1/kelong_mao/datasets/cast20/preprocessed/has_response_denpendency_turns.txt") as f:
#     data = f.readlines()

# d = set()
# for line in data:
#     line = line.strip()
#     d.add(line)

# with open("./results/case_study_cast20.json", "r") as f:
#     data = json.load(f)

# tmp = []
# for record in data:
#     sid = record['sample_id']
#     if sid not in d:
#         continue
#     if record['which_is_better'] == "oracle_rewrite":
#         tmp.append(record)

# with open("oracle_is_better_has_repsonse_dependency.json", "w") as f:
#     f.write(json.dumps(tmp, indent=4))


d = {}
def load_collection():
    with open("/data1/kelong_mao/collections/cast/cast_collection/raw.tsv", "r") as f:
        for line in tqdm(f):
            try:
                doc_id, doc = line = line.strip().split('\t')
                doc_id = int(doc_id)
                d[doc_id] = doc
            except:
                continue

load_collection()
embed()
input()

