WANDB_PROJECT=pretrain-mistral-7b python3 train.py \
--model_name_or_path huseinzol05/dummy-mistral-7b \
--share_directory "/home/ubuntu/share" \
--torch_dtype "bfloat16" \
--train_file "/home/ubuntu/share/mosaic-combine-all" \
--block_size 4096 \
--num_workers 80