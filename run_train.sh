model_tag="t5qr_qrecc"
output_dir_path="./outputs/$model_tag"
train_file_path="./data/train_qrecc.json"

export CUDA_VISIBLE_DEVICES=1
python train_t5qr.py --model_path="t5-base" \
--train_file_path=$train_file_path \
--output_dir_path=$output_dir_path \
--log_path="$output_dir_path/log" \
--output_checkpoint_path="$output_dir_path/checkpoints" \
--log_print_steps=0.1 \
--model_save_steps=1.0 \
--use_data_percent=1.0 \
--num_train_epochs=5 \
--train_batch_size=64 \
--max_seq_length=128 \
--collate_fn_type="train" \
--need_output \
--force_emptying_dir \