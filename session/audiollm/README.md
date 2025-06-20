# AudioLLM

Combine Whisper Encoder with any LLM to become Audio LLM.

## Prepare dataset

1. Prepare all dataset, can check each notebook for preparation in [preparation](preparation).
2. Prepare multipacking for audio understanding or stage 1, [multipack-audio-understanding.ipynb](multipack-audio-understanding.ipynb).
3. Prepare multipacking for speech understanding or stage 2. In Stage 2, we combine both speech and text dataset,

- First multipack text instructions, [packing-text-instructions.ipynb](packing-text-instructions.ipynb).
- Finally, multipack speech instructions combine with multipack text instructions, [multipack-speech-instructions.ipynb](multipack-speech-instructions.ipynb).

## Train

1. Stage 1, [qwen2.5-7b-lora-64.sh](qwen2.5-7b-lora-64.sh).
2. Stage 2, [qwen2.5-7b-lora-64-stage2.sh](qwen2.5-7b-lora-64-stage2.sh).