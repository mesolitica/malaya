# Installation TF 1.15.5 on RTX / A100

If you compiled Tensorflow 1.15.X on RTX / A100 with CUDA 11.X, or even if you added CUDA 10.0 in your system, variables initialization is very slow, lucky we have NVIDIA fork to make it RTX / A100 compatile with 1.15.X, read more at https://developer.nvidia.com/blog/accelerating-tensorflow-on-a100-gpus/

## how-to

1. Initialize virtual env,

```bash
python3 -m venv tf-nvidia
```

2. Install NVIDIA Tensorflow,

```bash
~/tf-nvidia/bin/pip3 install nvidia-pyindex
~/tf-nvidia/bin/pip3 install nvidia-tensorflow
```

3. Run python3 code to test NVIDIA Tensorflow,

```bash
~/tf-nvidia/bin/python3 test-gpu.py
```

4. Add in jupyter notebook kernel,

```bash
~/tf-nvidia/bin/pip3 install ipykernel
~/tf-nvidia/bin/python3 -m ipykernel install --user --name=tf1
```

5. Install tensor2tensor,

```bash
~/tf-nvidia/bin/pip3 install tensor2tensor --no-deps
~/tf-nvidia/bin/pip3 install requests \
tensorflow-probability==0.7.0 \
scipy sympy tqdm gym==0.17.1 \
Pillow tensorflow-datasets==3.2.1 \
pypng tensorflow-gan mesh-tensorflow==0.1.13 \
tensorflow-estimator==1.15.2
```