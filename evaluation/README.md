# Evaluation Instruction

## Evaluation with the oracle rewrite
Using BLEU metric to evaluate the rewrites by comparing with the oracle counterparts.
```python
python eval_bleu.py --eval_file_path="../outputs/t5_rewrite_cast19.json" --eval_field_name="t5_rewrite"
```


## Evaluation on downstream ranking tasks
We support sparse retrieval (bm25) and dense retrieval [(ANCE)](https://github.com/microsoft/ANCE).

For BM25, run:
```bash
bash run_eval_bm25_retrieval.sh
```

For dense retrievalm run:
```bash
bash run_eval_dense_retrieval.sh
```


## Case Study
Besides the overall performance, it is also important to investigate the performance of each concrete conversational turn, from which one can compare the retrieval effectiveness of two different rewriters intuitively.
To perform such a case study, one can refer to `case_study.py`.
```python
python case_study.py
```