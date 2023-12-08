WANDB_PROJECT=pretrain-mamba-1.4b python3 train.py \
--model_name_or_path huseinzol05/dummy-mamba-1.4b \
--share_directory "/home/ubuntu/share" \
--train_file "/home/ubuntu/share/mosaic-combine-all" \
--block_size 4096 \
--num_workers 20 \
--checkpoint_steps 100