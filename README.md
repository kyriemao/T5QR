# An Implementation for T5-based Conversational Query Rewriter

## Environment
```
conda create -n t5qr python=3.8
source activate t5qr
pip install -r requirements.txt
```

## Data
Currently, we only consider the previous queries as the context but not consider the preivous agent response since the response dependency is much harder to deal with.

Suppose that the current query is $q_4$ and the context is $[q_1, q_2, q_3]$, the input text sequence for T5 is $q_4$ [SEP] $q_3$ [SEP] $q_2$ [SEP] $q_1$. Note that the T5Tokenizer will additionally add an eos_token (i.e., <\/s>) to the end of the input text sequence.
The target sequence is the oralce query.

We provide examples of training file and test file in the `data` folder. 

```json
// trainig file:
// "cur_utt_text" is the current query. "oracle_utt_text" is the oracle query. "ctx_utts_text" is the context (i.e., [q_1, q_2, ...]).

{"sample_id": "31_3", "cur_utt_text": "Tell me about lung cancer.", "ctx_utts_text": ["What is throat cancer?", "Is it treatable?"], "oracle_utt_text": "Tell me about lung cancer."}

// test file:
// The test file format is the same as that of training file but does not need the "oracle_utt_text" field.
{"sample_id": "31_3", "cur_utt_text": "Tell me about lung cancer.", "ctx_utts_text": ["What is throat cancer?", "Is it treatable?"]}

```


## Training
```bash
bash run_train.sh
```


## Inference
```bash
bash run_inference.sh
```
We support using DDP for inference on multi GPUs. The corresponding rewrite is in the `"t5_rewrite"` field of the output file.


## Quick Single Rewriting
We provide a "quick" script (i.e., `single_rewrite.py`) to perform a single rewriting.
You need to set the `rewriter path`, `query`, and `context` in the script, and then run:
```python
python single_rewrite.py
```





