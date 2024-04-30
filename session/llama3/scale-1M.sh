PATH=$PATH:~/.local/bin
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1024' 

~/.local/bin/accelerate launch \
--config_file accelerate_configs/single_node.yaml \
train_checkpoint.py \
--batch-size 1 \
--gradient-accumulate-every 4 \
--output-dir ./output/7B_32K_bs_1M_rope_1M_step_1000_lr_2e-5 \
--wandb EasyContext \
--max-train-steps 1000  \
--learning-rate 2e-5  \
--dataset malaysia-ai/malaysian-dataset-llama3-128k \
--model mesolitica/llama-3-8b-8192-hf  \
--seq-length 32768 \
--rope-theta 1000000 \
--parallel_mode data_parallel

~/.local/bin/accelerate launch \
--config_file  accelerate_configs/single_node.yaml \
train.py \
--batch-size 1 \
--gradient-accumulate-every 2 \
--output-dir ./output/7B_64K_bs_1M_rope_5M_step_1000_lr_2e-5 \
--seed 2022 \
--wandb EasyContext \
--max-train-steps 1000  \
--learning-rate 2e-5  \
--dataset malaysia-ai/malaysian-dataset-llama3-128k \
--model output/7B_32K_bs_1M_rope_1M_step_1000_lr_2e-5  \
--seq-length 65536 \
--rope-theta 5000000 \
--parallel_mode data_parallel

~/.local/bin/accelerate launch \
--config_file  accelerate_configs/single_node.yaml \
train.py \
--batch-size 1 \
--gradient-accumulate-every 4  \
--output-dir  ./output/7B_0.256M_bs_1M_rope_10M_step_500_lr_2e-5 \
--seed 2023 \
--wandb EasyContext \
--max-train-steps 500  \
--learning-rate 2e-5  \
--dataset malaysia-ai/malaysian-dataset-llama3-1M \
--model output/7B_64K_bs_1M_rope_5M_step_1000_lr_2e-5  \
--seq-length 256000 \
--rope-theta 10000000 \
--parallel_mode zigzag_ring_attn

~/.local/bin/accelerate launch \
--config_file accelerate_configs/single_node.yaml \
train.py \
--batch-size 1 \
--gradient-accumulate-every 4  \
--output-dir  ./output/7B_0.256M_bs_1M_rope_25M_step_500_lr_2e-5 \
--seed 2024 \
--wandb EasyContext \
--max-train-steps 500  \
--learning-rate 2e-5  \
--dataset malaysia-ai/malaysian-dataset-llama3-1M \
--model output/7B_0.256M_bs_1M_rope_10M_step_500_lr_2e-5  \
--seq-length 256000 \
--rope-theta 25000000 \
--parallel_mode zigzag_ring_attn

~/.local/bin/accelerate launch \
--config_file accelerate_configs/single_node.yaml \
train.py \
--batch-size 1 \
--gradient-accumulate-every 2  \
--output-dir  ./output/7B_0.5M_bs_1M_rope_100M_step_300_lr_2e-5 \
--seed 2026 \
--wandb EasyContext \
--max-train-steps 300  \
--learning-rate 2e-5  \
--dataset malaysia-ai/malaysian-dataset-llama3-1M \
--model output/7B_0.256M_bs_1M_rope_50M_step_150_lr_2e-5  \
--seq-length 512000 \
--rope-theta 100000000 \
--parallel_mode zigzag_ring_attn

~/.local/bin/accelerate launch \
--config_file accelerate_configs/single_node.yaml \
train.py \
--batch-size 1 \
--gradient-accumulate-every 2  \
--output-dir  ./output/7B_0.5M_bs_1M_rope_250M_step_90_lr_2e-5 \
--seed 2027 \
--wandb EasyContext \
--max-train-steps 90  \
--learning-rate 1e-5  \
--dataset malaysia-ai/malaysian-dataset-llama3-1M \
--model output/7B_0.5M_bs_1M_rope_100M_step_300_lr_2e-5  \
--seq-length 512000 \
--rope-theta 250000000 \
--parallel_mode zigzag_ring_attn