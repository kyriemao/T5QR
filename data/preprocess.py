from IPython import embed
import json
from sklearn.model_selection import train_test_split

def trans_train_qrecc():
    with open("train_with_in_batch_neg.json", "r") as f:
        data = f.readlines()

    with open("new_train_qrecc.json", "w") as f:
        for line in data:
            line = json.loads(line)
            record = {}
            record["sample_id"] = line["sample_id"]
            record["cur_utt_text"] = line["cur_utt_text"]
            record["oracle_utt_text"] = line["oracle_utt_text"]
            context = line["ctx_utts_text"]
            # context = [context[i] for i in range(0, len(context), 2)]
            record["ctx_utts_text"] = context
            
            f.write(json.dumps(record))
            f.write('\n')


def trans():
    with open("test_qrecc.json", "r") as f:
        data = f.readlines()
    with open("new_test_qrecc.json", "w") as f:
         for line in data:
            line = json.loads(line)
            record = {}
            record["sample_id"] = line["sample_id"]
            record["cur_utt_text"] = line["cur_utt_text"]
            record["oracle_utt_text"] = line["oracle_utt_text"]
            context = line["ctx_utts_text"]
            ctx_utts_text = [context[i] for i in range(0, len(context), 2)]
            ctx_resps_text = [context[i] for i in range(1, len(context), 2)]
            record["ctx_utts_text"] = ctx_utts_text
            record["ctx_resps_text"] = ctx_resps_text
            
            f.write(json.dumps(record))
            f.write('\n')


def train_dev_split_qrecc():
    with open("train_qrecc.json", "r") as f:
        data = f.readlines()

    train_data, dev_data = train_test_split(data, test_size=0.1, random_state=42)
    with open("new_train_qrecc.json", "w") as ft, open("new_dev_qrecc.json", "w") as fd:
        for x in train_data:
            ft.write(x)
        for x in dev_data:
            fd.write(x)

  
def trans_cast20():
    with open("cast20.json", "r") as f:
        data = f.readlines()

    with open("test_cast20.json", "w") as f:
        for line in data:
            line = json.loads(line)
            record = {}
            record["sample_id"] = line["sample_id"]
            record["cur_utt_text"] = line["cur_utt_text"]
            record["oracle_utt_text"] = line["oracle_utt_text"]
            context = line["ctx_utts_text"]
            last_response_text = line['last_response_text']
            record["ctx_utts_text"] = context
            record["ctx_resps_text"] = [last_response_text] 
                 
            f.write(json.dumps(record))
            f.write('\n')

def trans_cast19():
    with open("train_cast19.json", "r") as f:
        data = f.readlines()

    with open("test_cast19.json", "w") as f:
        for line in data:
            line = json.loads(line)
            record = {}
            record["sample_id"] = line["sample_id"]
            record["cur_utt_text"] = line["cur_utt_text"]
            record["oracle_utt_text"] = line["oracle_utt_text"]
            record["ctx_utts_text"] = line["ctx_utts_text"]
            record["ctx_resps_text"] = [] 
                 
            f.write(json.dumps(record))
            f.write('\n')

if __name__ == "__main__":
    trans_cast19()