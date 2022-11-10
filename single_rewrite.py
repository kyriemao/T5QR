
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
        self.max_seq_length = 128

    def __call__(self, query, context):
        src_seq = context + [query]
        src_seq.reverse()
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
    rewriter_path = "./outputs/t5qr_cast19/checkpoints/epoch-4"
    rewriter = Rewriter(rewriter_path)
    
    # An example
    # query = "What are its symptoms?"
    # context = ["What is throat cancer?", "Is it treatable?", "Tell me about lung cancer."]  
    # # expected rewrite: What are lung cancer's symptoms?
    # rewrite = rewriter(query, context)

    query = "What of China"
    context = ["What is the population of USA", "What is its GDP"]
    rewrite = rewriter(query, context)
    print("Rewrite: {}".format(rewrite))