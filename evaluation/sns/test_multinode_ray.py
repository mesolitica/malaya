import os
import time
import ray
import torch
import torch.distributed as dist
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer, get_device

class RayConnection:
    def __init__(self, address, **kwargs):
        ray.init(address=address, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, typ, value, traceback):
        ray.shutdown()

def train_func():

    device = get_device()
    dist.barrier()

    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])

    tensor_size_mb = 512
    tensor = torch.ones(tensor_size_mb * 250_000, dtype=torch.float32).to(device)

    num_iters = 10
    torch.cuda.synchronize()
    times = []
    for _ in range(num_iters):
        dist.barrier()
        start = time.time()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        dist.barrier()
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    data_mb = tensor.element_size() * tensor.numel() / 1e9
    if rank == 0:
        print(f"Average AllReduce time over {num_iters} runs: {avg_time:.4f} s | Avg Bandwidth: {data_mb/avg_time:.2f} GB/s")
    
    return

def main():
    with RayConnection("ray://localhost:10001"):
        scaling_config = ScalingConfig(
            num_workers=16,
            use_gpu=True,
        )
        ray_trainer = TorchTrainer(
            train_func,
            scaling_config=scaling_config,
        )
        result = ray_trainer.fit()

if __name__ == "__main__":
    main()
