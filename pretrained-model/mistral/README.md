# Pretrain Mistral

## prerequisites 

1. Install libraries,

```bash
pip3 install -r requirements.txt
```

### Flash Attention 2

1. Install dependencies,

```bash
pip3 install flash-attn --no-build-isolation -U
pip3 install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary
```

## Full Parameter Finetuning

Dataset prepared at https://github.com/malaysia-ai/text-dataset-dedup#pretrain

### VM Spec

We use Azure Kubernetes Standard_NC96ads_A100_v4 for each FPF.

1. 96 vCPU
2. 880 GB RAM
3. 4x A100 80GB with topology,

```bash
nvidia-smi topo -m
```

```text
        GPU0    GPU1    GPU2    GPU3    CPU Affinity    NUMA Affinity
GPU0     X      NV12    SYS     SYS     0-23    0
GPU1    NV12     X      SYS     SYS     24-47   1
GPU2    SYS     SYS      X      NV12    48-71   2
GPU3    SYS     SYS     NV12     X      72-95   3
```

- When you use Kubernetes, `/dev/shm` default to 64 MB, and this caused bus error for NCCL. this issue happened on AWS p5 and Azure NC A100 v4-series, so to solve it, scale it up, eg, https://github.com/malaysia-ai/jupyter-gpu/blob/main/aks/jupyter4.yaml#L99
- Always perform NCCL test, https://github.com/NVIDIA/nccl-tests,

```bash
git clone https://github.com/NVIDIA/nccl-tests && cd nccl-tests
make
./build/all_gather_perf -g 4
```

```text
# nThread 1 nGpus 4 minBytes 33554432 maxBytes 33554432 step: 1048576(bytes) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid    622 on jupyter4-0 device  0 [0x00] NVIDIA A100 80GB PCIe
#  Rank  1 Group  0 Pid    622 on jupyter4-0 device  1 [0x00] NVIDIA A100 80GB PCIe
#  Rank  2 Group  0 Pid    622 on jupyter4-0 device  2 [0x00] NVIDIA A100 80GB PCIe
#  Rank  3 Group  0 Pid    622 on jupyter4-0 device  3 [0x00] NVIDIA A100 80GB PCIe
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
    33554432       2097152     float    none      -1    973.2   34.48   25.86      0    969.7   34.60   25.95      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 25.905
```

Good to go!

### 191M, 4096 Context length

```bash
bash run-191M.sh
```

https://wandb.ai/mesolitica/mistral-158M?workspace=user-husein-mesolitica