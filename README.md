# An Implementation for T5-based Conversational Query Rewriter

## Environment
```
conda create -n t5qr python=3.8
source activate t5qr
pip install -r requirements.txt
```

## Data

Suppose that the current query is $q_3$ and the context is $[q_1, r_1, q_2, r_2]$, the input text sequence for T5 is $q_3$ [SEP] $r_2$ [SEP] $q_2$ [SEP] $r_1$ [SEP] $q_1$. 
**We only include the last three responses at most**.
Note that the T5Tokenizer will additionally add an eos_token (i.e., <\/s>) to the end of the input text sequence.
The target sequence is the oralce query.

We provide examples of training/dev file and test file in the `data` folder. 

```json
// trainig file:
// "cur_utt_text" is the current query. "oracle_utt_text" is the oracle query. "ctx_utts_text" is the previous user query context (i.e., [q_1, q_2, ...]) and "ctx_resps_text" is the previous agent responses context (i.e., [r_1, r_2, ...])

{"sample_id": "QReCC-Train_7625_2", "cur_utt_text": "When was the movie released", "oracle_utt_text": "When was the moview Amadeus released", "ctx_utts_text": ["how does mozart die in the movie amadeus"], "ctx_resps_text": ["Mozart suddenly came down with fever and was wracked with pain.In the following days his health significantly deteriorated. He died on December 5 after lapsing into a coma."]}


// test file:
// The test file format is the same as that of training file but does not need the "oracle_utt_text" field.
```


## Training
```bash
bash run_train.sh
```
You can set the maximum training epochs to `num_train_epochs`. But the early stopping will be triggered when the dev loss does not decresae for two consecutive periods of model saving. Therefore, you should use the third-to-last saved model as the final model.

## Inference
```bash
bash run_inference.sh
```
We support using DDP for inference on multi GPUs. The corresponding rewrite is in the `"t5_rewrite"` field of the output file.



## Quick Single Rewriting
We provide a "quick" script (i.e., `single_rewrite.py`) to perform a single rewriting.
You need to set the `rewriter path`, `cur_utt_text` (i.e., the current user query), `ctx_utts_text` (i.e., the previous user queries), and `ctx_resps_text`(i.e., the previous agent responses) in the script, and then run:
```python
python single_rewrite.py
```


## Training and Inference
We randomly split the training set of [QReCC](https://github.com/apple/ml-qrecc) to new training (90%) and dev(10%) sets and use them to train a T5 rewriter model. The trained model checkpoint can be downloaded [here]().

We evaluate the model on QReCC test set, CAsT-19, and CAsT-20 test sets. 

Note that:
for QReCC:
```python
max_response_length = 100
max_seq_length = 384
```
While for CAsT-19 and 20:
```python
max_response_length=128 # As the **last** automatic canonical response in CAsT-20 is a longer passage compared with the responses in QReCC which are shorter text span.
max_seq_length=256  # CAsT-20 only include one response and CAsT-19 does not include response, so its maximum sequence length can be shorter than that of QReCC.
```



## Evaluation

After conversational query rewriting, we can evaluate the rewrites by directly comparing it with the manual oracle rewrites or on downstream ranking tasks. We provide the evaluation instruction in `evaluation` folder.





<!-- ## Checkpoint and Human Evaluation
We provide a T5QR checkpoint [here](https://drive.google.com/file/d/1V531-kafArfr8AuJYwvNOOqpeoKconwB/view?usp=share_link),  which was trained on the training data of [QReCC](https://github.com/apple/ml-qrecc) with 5 epochs. Note that only previous queries were used as the context.

We randomly select 170 turns from CAsT-19 and CAsT-20 and perform a human evaluation on this checkpoint.
Specifically, we set three levels (0, 1 ,2). "0" means incorrect rewriting, "1" means moderatelly correct rewriting (only very few turns belog to this category.), and "2" means (totally) correct rewriting.
Results are shown below:
```
#level-0: #level-1: #level-2
CAsT-19 (105 turns): [24, 6, 75]. 71.4% belongs to level-2.
CAsT-19+20 (170 turns): [68, 11, 91]. 53.5% belongs to level-2.
```
Since many turns in CAsT-20 have response dependency, so the performance of this checkpoint on CAsT-20 is significantly worse than that on CAsT-19, which only has the dependency on previous queries.

The judgement file is also provided in the above checkpoint zip file. Note that we have ignored the turn whose `cur_utt_text` is the same as the `oracle_utt_text`.
 -->


