PATH=$PATH:~/.local/bin
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1024' 

~/.local/bin/accelerate launch \
--config_file accelerate_configs/single_node.yaml \
train_checkpoint.py \
--batch-size 1 \
--gradient-accumulate-every 32 \
--output-dir ./output/7B_32768 \
--wandb EasyContext-32768 \
--max-train-steps 1000  \
--learning-rate 2e-5  \
--dataset malaysia-ai/malaysian-dataset-llama3-128k \
--model mesolitica/llama-3-8b-8192-hf  \
--seq-length 32768 \
--rope-theta 15300000 \
--parallel_mode zigzag_ring_attn

~/.local/bin/accelerate launch \
--config_file accelerate_configs/single_node.yaml \
train_checkpoint.py \
--batch-size 1 \
--gradient-accumulate-every 32 \
--output-dir ./output/7B_65536 \
--wandb EasyContext-65536 \
--max-train-steps 1000  \
--learning-rate 2e-5  \
--dataset malaysia-ai/malaysian-dataset-llama3-128k \
--model mesolitica/llama-3-8b-8192-hf  \
--seq-length 65536 \
--rope-theta 15300000 \
--parallel_mode zigzag_ring_attn

~/.local/bin/accelerate launch \
--config_file accelerate_configs/single_node.yaml \
train_checkpoint.py \
--batch-size 1 \
--gradient-accumulate-every 16 \
--output-dir ./output/7B_262144 \
--wandb EasyContext-262144 \
--max-train-steps 1000  \
--learning-rate 2e-5  \
--dataset malaysia-ai/malaysian-dataset-llama3-1M \
--model ./output/7B_65536  \
--seq-length 262144 \
--rope-theta 207100000 \
--parallel_mode zigzag_ring_attn

~/.local/bin/accelerate launch \
--config_file accelerate_configs/single_node.yaml \
train_checkpoint.py \
--batch-size 1 \
--gradient-accumulate-every 16 \
--output-dir ./output/7B_524288 \
--wandb EasyContext-524288 \
--max-train-steps 1000  \
--learning-rate 2e-5  \
--dataset malaysia-ai/malaysian-dataset-llama3-1M \
--model ./output/7B_524288  \
--seq-length 524288 \
--rope-theta 1060000000 \
--parallel_mode zigzag_ring_attn