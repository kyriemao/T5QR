from IPython import embed
import sys
sys.path.append('..')
sys.path.append('.')
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import json
import random
from tqdm import tqdm
from torch.utils.data import Dataset


class T5RewriterDataset(Dataset):
    def __init__(self, args, filename):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()
        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for deugging
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)

        for line in tqdm(data):
            record = json.loads(line)

            cur_utt_text = record['cur_utt_text']
            ctx_utts_text = record['ctx_utts_text']
            
            src_seq = ctx_utts_text + [cur_utt_text]
            src_seq.reverse()
            src_seq = " [SEP] ".join(src_seq)
            
            if "oracle_utt_text" in record: 
                tgt_seq = record["oracle_utt_text"] # oracle query
            else:
                tgt_seq = ""
            
            self.examples.append((record["sample_id"], 
                                  src_seq, 
                                  tgt_seq,
                                  cur_utt_text,
                                  ctx_utts_text))


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch):
            bt_sample_ids, bt_src_seq, bt_tgt_seq, bt_cur_utt_text, bt_ctx_utts_text = list(zip(*batch)) # unzip
            bt_src_encoding = args.tokenizer(bt_src_seq, 
                                            padding="longest", 
                                            max_length=args.max_seq_length, 
                                            truncation=True, 
                                            return_tensors="pt")
            bt_input_ids, bt_attention_mask = bt_src_encoding.input_ids, bt_src_encoding.attention_mask
            if args.collate_fn_type == "train":
                bt_tgt_encoding = args.tokenizer(bt_tgt_seq, 
                                                padding="longest", 
                                                max_length=args.max_query_length, 
                                                truncation=True, 
                                                return_tensors="pt")

                bt_labels = bt_tgt_encoding.input_ids
                # replace padding token id's of the labels by -100 so it's ignored by the loss
                bt_labels[bt_labels == args.tokenizer.pad_token_id] = -100
            else:
                bt_labels = None

            return {"bt_sample_ids": bt_sample_ids, 
                    "bt_input_ids":bt_input_ids, 
                    "bt_attention_mask":bt_attention_mask, 
                    "bt_labels": bt_labels,
                    "bt_cur_utt_text": bt_cur_utt_text,
                    "bt_ctx_utts_text": bt_ctx_utts_text,
                    "bt_oracle_utt_text": bt_tgt_seq}

        return collate_fn
