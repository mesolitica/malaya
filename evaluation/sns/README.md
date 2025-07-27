# SNS

Evaluate SNS server using Qwen2.5 models.

## LLM Engine

### vLLM Requirements

```bash
python3 -m venv --without-pip vllm
wget https://bootstrap.pypa.io/get-pip.py
vllm/bin/python3 get-pip.py
vllm/bin/pip3 install vllm==0.8.5.post1 aiohttp "ray[default]" "ray[train]"
```

#### Multi-nodes using Ray

```bash
# Elect any head node
NCCL_SOCKET_IFNAME=eno8303 GLOO_SOCKET_IFNAME=eno8303 ~/vllm/bin/ray start --block --head --port=6379 --dashboard-host=0.0.0.0
# other nodes
NCCL_SOCKET_IFNAME=eno8303 GLOO_SOCKET_IFNAME=eno8303 ~/vllm/bin/ray start --block --address=10.10.100.104:6379
```

To verify multi-nodes is working, you can run simple reduce scatter using Ray,

```bash
python3 test_multinode_ray.py
```

```
(RayTrainWorker pid=2754037, ip=10.10.100.107) Average AllReduce time over 10 runs: 0.0059 s | Avg Bandwidth: 88.79 GB/s
```


### TensorRT-LLM Requirements

```bash
python3 -m venv --without-pip tensorrt
wget https://bootstrap.pypa.io/get-pip.py
tensorrt/bin/python3 get-pip.py
tensorrt/bin/pip3 install tensorrt_llm==0.18.2 aiohttp flash_attn
tensorrt/bin/pip3 install -U "nvidia-modelopt[all]"
```

### 14B vLLM

#### FP16

```bash
./vllm/bin/vllm serve "mesolitica/Malaysian-Qwen2.5-14B-Instruct" \
--served-model-name "model" \
--tensor_parallel_size "8"
```

Stress test,

```bash
python3 benchmark.py \
--model "model" \
--save "Malaysian-Qwen2.5-14B-Instruct-warmup" \
--rps-list "5"

python3 benchmark.py \
--model "model" \
--save "Malaysian-Qwen2.5-14B-Instruct"
```

#### FP8

```bash
./vllm/bin/vllm serve "mesolitica/Malaysian-Qwen2.5-14B-Instruct-FP8" \
--served-model-name "model" \
--tensor_parallel_size "8"
```

Stress test,

```bash
python3 benchmark.py \
--model "model" \
--save "Malaysian-Qwen2.5-14B-Instruct-FP8-warmup" \
--rps-list "5"

python3 benchmark.py \
--model "model" \
--save "Malaysian-Qwen2.5-14B-Instruct-FP8"
```

### 14B TensorRT-LLM

#### FP16

```bash
./tensorrt/bin/trtllm-serve "mesolitica/Malaysian-Qwen2.5-14B-Instruct" \
--tp_size "8" --host "0.0.0.0" --backend pytorch
```

Stress test,

```bash
python3 benchmark.py \
--model "mesolitica/Malaysian-Qwen2.5-14B-Instruct" \
--save "Malaysian-Qwen2.5-14B-Instruct-warmup-tensorrt" \
--rps-list "5"

python3 benchmark.py \
--model "mesolitica/Malaysian-Qwen2.5-14B-Instruct" \
--save "Malaysian-Qwen2.5-14B-Instruct-warmup-tensorrt" \
--rps-list "5"

python3 benchmark.py \
--model "mesolitica/Malaysian-Qwen2.5-14B-Instruct" \
--save "Malaysian-Qwen2.5-14B-Instruct-warmup-tensorrt" \
--rps-list "5"

python3 benchmark.py \
--model "Malaysian-Qwen2.5-14B-Instruct" \
--save "Malaysian-Qwen2.5-14B-Instruct-tensorrt"
```

### 72B vLLM

#### FP16

```bash
./vllm/bin/vllm serve "mesolitica/Malaysian-Qwen2.5-72B-Instruct" \
--served-model-name "model" \
--tensor_parallel_size "8"
```

Stress test,

```bash
python3 benchmark.py \
--model "model" \
--save "Malaysian-Qwen2.5-72B-Instruct-warmup" \
--rps-list "5"

python3 benchmark.py \
--model "model" \
--save "Malaysian-Qwen2.5-72B-Instruct"
```

#### FP8

```bash
./vllm/bin/vllm serve "mesolitica/Malaysian-Qwen2.5-72B-Instruct-FP8" \
--served-model-name "model" \
--tensor_parallel_size "8"
```

Stress test,

```bash
python3 benchmark.py \
--model "model" \
--save "Malaysian-Qwen2.5-72B-Instruct-FP8-warmup" \
--rps-list "5"

python3 benchmark.py \
--model "model" \
--save "Malaysian-Qwen2.5-72B-Instruct-FP8"
```

#### FP16 2 Nodes

```bash
NCCL_SOCKET_IFNAME=eno8303 GLOO_SOCKET_IFNAME=eno8303 \
~/vllm/bin/vllm serve "mesolitica/Malaysian-Qwen2.5-72B-Instruct" \
--served-model-name "model" \
--tensor-parallel-size 8 \
--pipeline-parallel-size 2
```

Stress test,

```bash
python3 benchmark.py \
--model "model" \
--save "Malaysian-Qwen2.5-72B-Instruct-warmup-2nodes" \
--rps-list "5"

python3 benchmark.py \
--model "model" \
--save "Malaysian-Qwen2.5-72B-Instruct-2nodes"
```


#### FP8 2 Nodes

```bash
NCCL_SOCKET_IFNAME=eno8303 GLOO_SOCKET_IFNAME=eno8303 \
~/vllm/bin/vllm serve "mesolitica/Malaysian-Qwen2.5-72B-Instruct-FP8" \
--served-model-name "model" \
--tensor-parallel-size 8 \
--pipeline-parallel-size 2
```

Stress test,

```bash
python3 benchmark.py \
--model "model" \
--save "Malaysian-Qwen2.5-72B-Instruct-FP8-warmup-2nodes" \
--rps-list "5"

python3 benchmark.py \
--model "model" \
--save "Malaysian-Qwen2.5-72B-Instruct-FP8-2nodes"
```

### 72B TensorRT-LLM

#### FP16

```bash
./tensorrt/bin/trtllm-serve "mesolitica/Malaysian-Qwen2.5-72B-Instruct" \
--tp_size "8" --host "0.0.0.0" --backend pytorch
```

Stress test,

```bash
python3 benchmark.py \
--model "mesolitica/Malaysian-Qwen2.5-72B-Instruct" \
--save "Malaysian-Qwen2.5-72B-Instruct-warmup-tensorrt" \
--rps-list "5"

python3 benchmark.py \
--model "mesolitica/Malaysian-Qwen2.5-72B-Instruct" \
--save "Malaysian-Qwen2.5-72B-Instruct-warmup-tensorrt" \
--rps-list "5"

python3 benchmark.py \
--model "mesolitica/Malaysian-Qwen2.5-72B-Instruct" \
--save "Malaysian-Qwen2.5-72B-Instruct-warmup-tensorrt" \
--rps-list "5"

python3 benchmark.py \
--model "Malaysian-Qwen2.5-72B-Instruct" \
--save "Malaysian-Qwen2.5-72B-Instruct-tensorrt"
```