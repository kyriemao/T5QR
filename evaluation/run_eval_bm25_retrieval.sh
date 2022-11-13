eval_file_path="../outputs/t5_rewrite_cast20.json"
index_path="/data1/kelong_mao/indexes/cast/bm25"
qrel_file_path="/data1/kelong_mao/datasets/cast20/preprocessed/cast20_qrel.tsv"
retrieval_output_path="./results/cast20_bm25" 


python eval_bm25_retrieval.py --eval_file_path=$eval_file_path \
--eval_field_name="oracle_rewrite" \
--index_path=$index_path \
--qrel_file_path=$qrel_file_path \
--top_n=1000 \
--bm25_k1=0.82 \
--bm25_b=0.68 \
--rel_threshold=2 \
--retrieval_output_path=$retrieval_output_path \
--force_emptying_dir