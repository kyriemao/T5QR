
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

class Rewriter:
    def __init__(self, rewriter_path):
        self.tokenizer = T5Tokenizer.from_pretrained(rewriter_path)
        self.t5 = T5ForConditionalGeneration.from_pretrained(rewriter_path)
        assert self.tokenizer.sep_token == "[SEP]"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.t5.to(self.device)
        self.max_query_length = 32
        self.max_response_length = 100
        self.max_seq_length = 256

    def __call__(self, cur_utt_text, ctx_utts_text, ctx_resps_text):
        ctx_resps_text = ctx_resps_text[-3:]     
        for i in range(len(ctx_resps_text)):
            ctx_resps_text[i] = " ".join(ctx_resps_text[i].split()[:self.max_response_length]) 
        ctx_utts_text.reverse()
        ctx_resps_text.reverse()

        src_seq = []
        src_seq.append(cur_utt_text)
        for i in range(len(ctx_utts_text)):
            if i < len(ctx_resps_text):
                src_seq.append(ctx_resps_text[i])
            src_seq.append(ctx_utts_text[i])                
        src_seq = " [SEP] ".join(src_seq)

        encoding = self.tokenizer(src_seq,
                                  padding="longest", 
                                  max_length=self.max_query_length, 
                                  truncation=True, 
                                  return_tensors="pt")

        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        outputs = self.t5.generate(input_ids=input_ids, 
                                   attention_mask=attention_mask, 
                                   do_sample=False,
                                   max_length=self.max_query_length)

        rewrite_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return rewrite_text


if __name__ == "__main__":
    rewriter_path = "./outputs/t5qr_qrecc/checkpoints/epoch-5"
    rewriter = Rewriter(rewriter_path)
    
    # An example
    cur_utt_text = "How about China"
    ctx_utts_text = ["What is the population of USA", "What is its GDP"]
    ctx_resps_text = []
    rewrite = rewriter(cur_utt_text, ctx_utts_text, ctx_resps_text)
    print("Rewrite: {}".format(rewrite))