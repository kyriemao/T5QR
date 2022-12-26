eval_file_path="../../GRFD/outputs/dense_rewrite_cast20_session.json"
index_path="/data1/kelong_mao/indexes/cast/ance/doc_embeddings"
qrel_file_path="/data1/kelong_mao/datasets/cast20/preprocessed/cast20_qrel.tsv"
retrieval_output_path="../../GRFD/results/cast20_ance_dense_rewrite"



export CUDA_VISIBLE_DEVICES=7
python eval_dense_retrieval.py --eval_file_path=$eval_file_path \
--eval_field_name="dense_rewrite" \
--qrel_file_path=$qrel_file_path \
--index_path=$index_path \
--retriever_path="/data1/kelong_mao/pretrained_models/ance-msmarco" \
--use_gpu_in_faiss \
--n_gpu_for_faiss=1 \
--top_n=1000 \
--rel_threshold=2 \
--retrieval_output_path=$retrieval_output_path \
--force_emptying_dir \
