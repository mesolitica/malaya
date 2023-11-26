WANDB_PROJECT=pretrain-mistral-3b python3 train.py \
--model_name_or_path huseinzol05/dummy-mistral-3b \
--share_directory "/home/ubuntu/share" \
--torch_dtype "bfloat16" \
--train_file "/home/ubuntu/share/combine-all" \
--block_size 4096 \
--num_workers 20