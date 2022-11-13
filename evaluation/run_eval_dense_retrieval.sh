eval_file_path="../outputs/t5_rewrite_cast19.json"
index_path="/data1/kelong_mao/indexes/cast/ance/doc_embeddings"
qrel_file_path="/data1/kelong_mao/datasets/cast19/preprocessed/cast19_qrel.tsv"
retrieval_output_path="./results/cast19_ance_oracle_rewrite"



export CUDA_VISIBLE_DEVICES=1
python eval_dense_retrieval.py --eval_file_path=$eval_file_path \
--eval_field_name="oracle_rewrite" \
--qrel_file_path=$qrel_file_path \
--index_path=$index_path \
--retriever_path="/data1/kelong_mao/pretrained_models/ance-msmarco" \
--use_gpu_in_faiss \
--n_gpu_for_faiss=1 \
--top_n=1000 \
--rel_threshold=1 \
--retrieval_output_path=$retrieval_output_path \
--force_emptying_dir \