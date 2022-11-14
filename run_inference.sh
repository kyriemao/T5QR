port=28555
test_file_path="./data/test_cast20.json"
output_file_path="./outputs/t5_rewrite_cast20.json"

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
--nproc_per_node=1 \
--master_port $port \
inference_t5qr.py \
--n_gpu=1 \
--model_checkpoint_path="./outputs/t5qr_qrecc/checkpoints/epoch-5" \
--test_file_path=$test_file_path \
--output_file_path="$output_file_path" \
--use_data_percent=1.0 \
--per_gpu_eval_batch_size=64 \
--max_response_length=128 \
--max_seq_length=256 

