model_tag="t5qr_qrecc"
output_dir_path="./outputs/$model_tag"
train_file_path="./data/qrecc/train_qrecc.json"
dev_file_path="./data/qrecc/dev_qrecc.json"

export CUDA_VISIBLE_DEVICES=1
python train_t5qr.py --model_path="t5-base" \
--train_file_path=$train_file_path \
--dev_file_path=$dev_file_path \
--output_dir_path=$output_dir_path \
--log_path="$output_dir_path/log" \
--output_checkpoint_path="$output_dir_path/checkpoints" \
--log_print_steps=0.1 \
--model_save_steps=1.0 \
--use_data_percent=1.0 \
--num_train_epochs=20 \
--train_batch_size=48 \
--dev_batch_size=48 \
--max_response_length=100 \
--max_seq_length=384 \
--need_output