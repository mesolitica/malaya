#!/bin/bash

apt update
apt install unzip ffmpeg -y
apt update && apt install -y locales
locale-gen en_US.UTF-8
cd /workspace
wget https://www.7-zip.org/a/7z2301-linux-x64.tar.xz
tar -xf 7z2301-linux-x64.tar.xz
pip3 install huggingface-hub wandb multiprocess

cmd1="
export LC_ALL=en_US.UTF-8; export LANG=en_US.UTF-8;
cd /workspace
python3 -c \"
from huggingface_hub import snapshot_download
snapshot_download(repo_id='mesolitica/Malaysian-audio-QA-pre-training', repo_type='dataset', 
                  allow_patterns = 'slice-audio.z*', local_dir = './')
\"
/workspace/7zz x slice-audio.zip -y -mmt40
rm filtered-24k_processed.z*
"

cmd2="
export LC_ALL=en_US.UTF-8; export LANG=en_US.UTF-8;
cd /workspace
python3 -c \"
from huggingface_hub import snapshot_download
snapshot_download(repo_id='mesolitica/malaysian-youtube-audio-instructions', repo_type='dataset', 
                  allow_patterns = 'filter-audio.7z.*', local_dir = './')
\"
/workspace/7zz x filter-audio.7z.001 -y -mmt40
rm filter-audio.7z.*
"

bash -c "$cmd1" &
pid1=$!

bash -c "$cmd2" &
pid2=$!

wait $pid1 $pid2

wget https://huggingface.co/datasets/huseinzol05/malaysian-audio-qa-pretraining/resolve/main/audio-qa-pretrained.jsonl