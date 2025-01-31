TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 3 \
-m train_grpo \
--model_name huseinzol05/Llama-3.2-3B-Malaysian