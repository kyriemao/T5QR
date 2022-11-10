port=28555
test_file_path="./data/test.json"
output_file_path="./outputs/t5_rewrite.json"

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
--nproc_per_node=1 \
--master_port $port \
inference_t5qr.py \
--n_gpu=1 \
--model_checkpoint_path="./outputs/t5qr_cast19/checkpoints/epoch-4" \
--test_file_path=$test_file_path \
--output_file_path="$output_file_path" \
--use_data_percent=1.0 \
--per_gpu_eval_batch_size=12 \
--max_seq_length=128 \
--collate_fn_type="test" \

