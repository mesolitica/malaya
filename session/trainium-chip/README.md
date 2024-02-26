```bash
MALLOC_ARENA_MAX=64 neuron_parallel_compile torchrun --nproc_per_node=32 run_clm.py \
 --model_id mesolitica/tinyllama-1.1b-4096-fpf \
 --dataset_path /home/ubuntu/mosaic-chat-instructions-rag \
 --bf16 True \
 --learning_rate 2e-5 \
 --output_dir litellama-runs \
 --overwrite_output_dir True \
 --skip_cache_push True \
 --per_device_train_batch_size 10 \
 --gradient_checkpointing True \
 --tensor_parallel_size 4 \
 --max_steps 10 \
 --logging_steps 1 \
 --gradient_accumulation_steps 8
```

```
2024-02-26 16:02:55.000243:  97  INFO ||NEURON_PARALLEL_COMPILE||: Running trial run (add option to terminate trial run early; also ignore trial run's generated outputs, i.e. loss, checkpoints)
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
> initializing tensor model parallel with size 4
> initializing pipeline model parallel with size 1
> initializing data parallel with size 8
2024-02-26 16:03:05.000270:  175  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000270:  179  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000270:  187  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000270:  183  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000270:  172  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000270:  178  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000270:  169  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000270:  167  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000271:  195  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000271:  171  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000271:  188  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000271:  185  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000271:  168  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000271:  189  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000271:  165  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000271:  192  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000271:  174  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000271:  173  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000271:  191  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000271:  186  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000271:  193  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000271:  177  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000271:  184  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000272:  181  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000272:  180  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000272:  196  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000272:  194  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000272:  170  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000289:  166  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000289:  176  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000290:  190  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:05.000293:  182  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
***** Running training *****
  Num examples = 346,940
  Num Epochs = 1
  Instantaneous batch size per device = 10
  Total train batch size (w. parallel, distributed & accumulation) = 640
  Gradient Accumulation steps = 8
  Total optimization steps = 10
  Number of trainable parameters = 275,081,216
  0%|          | 0/10 [00:00<?, ?it/s]You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
2024-02-26 16:03:32.000509:  20277  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:32.000510:  20277  INFO ||NEURON_CC_WRAPPER||: Extracting graphs (/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_5274156739761925271+d41d8cd9/model.hlo.pb) for ahead-of-time parallel compilation. No compilation was done.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
2024-02-26 16:03:35.000414:  25554  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:35.000416:  25554  INFO ||NEURON_CC_WRAPPER||: Extracting graphs (/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_2599822411245392803+d41d8cd9/model.hlo.pb) for ahead-of-time parallel compilation. No compilation was done.
2024-02-26 16:03:35.000449:  25556  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:35.000451:  25556  INFO ||NEURON_CC_WRAPPER||: Extracting graphs (/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_15521117223395126027+d41d8cd9/model.hlo.pb) for ahead-of-time parallel compilation. No compilation was done.
2024-02-26 16:03:35.000501:  25559  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:35.000501:  25560  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:35.000503:  25559  INFO ||NEURON_CC_WRAPPER||: Extracting graphs (/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_18226885169323101566+d41d8cd9/model.hlo.pb) for ahead-of-time parallel compilation. No compilation was done.
2024-02-26 16:03:35.000504:  25560  INFO ||NEURON_CC_WRAPPER||: Extracting graphs (/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_16718126519652317380+d41d8cd9/model.hlo.pb) for ahead-of-time parallel compilation. No compilation was done.
2024-02-26 16:03:38.000445:  25803  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:38.000447:  25803  INFO ||NEURON_CC_WRAPPER||: Extracting graphs (/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_1897587528045977639+d41d8cd9/model.hlo.pb) for ahead-of-time parallel compilation. No compilation was done.
2024-02-26 16:03:38.000549:  25805  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:38.000551:  25805  INFO ||NEURON_CC_WRAPPER||: Extracting graphs (/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_15254528341555394658+d41d8cd9/model.hlo.pb) for ahead-of-time parallel compilation. No compilation was done.
2024-02-26 16:03:38.000785:  25807  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:38.000787:  25807  INFO ||NEURON_CC_WRAPPER||: Extracting graphs (/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_10789312904300651302+d41d8cd9/model.hlo.pb) for ahead-of-time parallel compilation. No compilation was done.
2024-02-26 16:03:38.000815:  25809  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:38.000817:  25809  INFO ||NEURON_CC_WRAPPER||: Extracting graphs (/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_16926392219237896526+d41d8cd9/model.hlo.pb) for ahead-of-time parallel compilation. No compilation was done.
 10%|█         | 1/10 [00:08<01:17,  8.64s/it]2024-02-26 16:03:44.000204:  29410  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:44.000207:  29410  INFO ||NEURON_CC_WRAPPER||: Extracting graphs (/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_11532192028300124032+d41d8cd9/model.hlo.pb) for ahead-of-time parallel compilation. No compilation was done.
2024-02-26 16:03:44.000341:  29412  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:44.000344:  29412  INFO ||NEURON_CC_WRAPPER||: Extracting graphs (/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_9205396726650978396+d41d8cd9/model.hlo.pb) for ahead-of-time parallel compilation. No compilation was done.
2024-02-26 16:03:44.000566:  29414  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:44.000568:  29414  INFO ||NEURON_CC_WRAPPER||: Extracting graphs (/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_10773640740194008060+d41d8cd9/model.hlo.pb) for ahead-of-time parallel compilation. No compilation was done.
2024-02-26 16:03:44.000826:  29416  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:44.000828:  29416  INFO ||NEURON_CC_WRAPPER||: Extracting graphs (/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_13381711796159894734+d41d8cd9/model.hlo.pb) for ahead-of-time parallel compilation. No compilation was done.
2024-02-26 16:03:47.000456:  29545  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:47.000457:  29545  INFO ||NEURON_CC_WRAPPER||: Extracting graphs (/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_5731444939583576096+d41d8cd9/model.hlo.pb) for ahead-of-time parallel compilation. No compilation was done.
{'loss': 0.0, 'learning_rate': 1.8e-05, 'epoch': 0.0}
 10%|█         | 1/10 [00:15<01:17,  8.64s/it]2024-02-26 16:03:47.000679:  29848  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:47.000681:  29848  INFO ||NEURON_CC_WRAPPER||: Extracting graphs (/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_931450859643720011+d41d8cd9/model.hlo.pb) for ahead-of-time parallel compilation. No compilation was done.
 20%|██        | 2/10 [00:16<01:07,  8.43s/it]2024-02-26 16:03:52.000449:  34577  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:52.000451:  34577  INFO ||NEURON_CC_WRAPPER||: Extracting graphs (/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_640892455732324657+d41d8cd9/model.hlo.pb) for ahead-of-time parallel compilation. No compilation was done.
2024-02-26 16:03:52.000702:  34579  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:52.000704:  34579  INFO ||NEURON_CC_WRAPPER||: Extracting graphs (/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_16105616828272085633+d41d8cd9/model.hlo.pb) for ahead-of-time parallel compilation. No compilation was done.
2024-02-26 16:03:52.000744:  34581  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:52.000746:  34581  INFO ||NEURON_CC_WRAPPER||: Extracting graphs (/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_15067706583171545378+d41d8cd9/model.hlo.pb) for ahead-of-time parallel compilation. No compilation was done.
2024-02-26 16:03:52.000750:  34583  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:03:52.000752:  34583  INFO ||NEURON_CC_WRAPPER||: Extracting graphs (/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_6875444972028399005+d41d8cd9/model.hlo.pb) for ahead-of-time parallel compilation. No compilation was done.
{'loss': 0.0, 'learning_rate': 1.6000000000000003e-05, 'epoch': 0.0}
{'loss': 0.0, 'learning_rate': 1.4e-05, 'epoch': 0.0}
{'loss': 0.0, 'learning_rate': 1.2e-05, 'epoch': 0.0}
{'loss': 0.0, 'learning_rate': 1e-05, 'epoch': 0.0}
{'loss': nan, 'learning_rate': 8.000000000000001e-06, 'epoch': 0.0}
{'loss': nan, 'learning_rate': 6e-06, 'epoch': 0.0}7/10 [00:31<00:08,  2.71s/it]
{'loss': 338649581355008.0, 'learning_rate': 4.000000000000001e-06, 'epoch': 0.0}
{'loss': 0.0, 'learning_rate': 2.0000000000000003e-06, 'epoch': 0.0}
{'loss': 0.0, 'learning_rate': 0.0, 'epoch': 0.0}
100%|██████████| 10/10 [00:36<00:00,  1.94s/it]

Training completed. Do not forget to share your model on huggingface.co/models =)


{'train_runtime': 36.3738, 'train_samples_per_second': 175.951, 'train_steps_per_second': 0.275, 'train_loss': nan, 'epoch': 0.0}
100%|██████████| 10/10 [00:36<00:00,  3.64s/it]
2024-02-26 16:04:13.000129:  97  INFO ||NEURON_PARALLEL_COMPILE||: New graph list from script: /var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_5274156739761925271+d41d8cd9/model.hlo.pb
	/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_2599822411245392803+d41d8cd9/model.hlo.pb
	/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_15521117223395126027+d41d8cd9/model.hlo.pb
	/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_18226885169323101566+d41d8cd9/model.hlo.pb
	/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_16718126519652317380+d41d8cd9/model.hlo.pb
	/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_1897587528045977639+d41d8cd9/model.hlo.pb
	/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_15254528341555394658+d41d8cd9/model.hlo.pb
	/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_10789312904300651302+d41d8cd9/model.hlo.pb
	/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_16926392219237896526+d41d8cd9/model.hlo.pb
	/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_11532192028300124032+d41d8cd9/model.hlo.pb
	/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_9205396726650978396+d41d8cd9/model.hlo.pb
	/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_10773640740194008060+d41d8cd9/model.hlo.pb
	/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_13381711796159894734+d41d8cd9/model.hlo.pb
	/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_5731444939583576096+d41d8cd9/model.hlo.pb
	/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_931450859643720011+d41d8cd9/model.hlo.pb
	/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_640892455732324657+d41d8cd9/model.hlo.pb
	/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_16105616828272085633+d41d8cd9/model.hlo.pb
	/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_15067706583171545378+d41d8cd9/model.hlo.pb
	/var/tmp/neuron-compile-cache/neuronxcc-2.12.68.0+4480452af/MODULE_6875444972028399005+d41d8cd9/model.hlo.pb
2024-02-26 16:04:13.000130:  97  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
2024-02-26 16:04:13.000142:  87302  INFO ||NEURON_CACHE||: Current remaining items are 19, locked are 0, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000143:  87303  INFO ||NEURON_CACHE||: Current remaining items are 19, locked are 0, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000145:  87304  INFO ||NEURON_CACHE||: Current remaining items are 19, locked are 0, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000146:  87305  INFO ||NEURON_CACHE||: Current remaining items are 19, locked are 0, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000148:  87306  INFO ||NEURON_CACHE||: Current remaining items are 19, locked are 0, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000150:  87307  INFO ||NEURON_CACHE||: Current remaining items are 19, locked are 0, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000151:  87309  INFO ||NEURON_CACHE||: Current remaining items are 19, locked are 0, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000153:  87308  INFO ||NEURON_CACHE||: Current remaining items are 19, locked are 0, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000153:  87303  INFO ||NEURON_CACHE||: Current remaining items are 18, locked are 1, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000154:  87304  INFO ||NEURON_CACHE||: Current remaining items are 18, locked are 1, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000154:  87305  INFO ||NEURON_CACHE||: Current remaining items are 18, locked are 1, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000154:  87306  INFO ||NEURON_CACHE||: Current remaining items are 18, locked are 1, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000155:  87307  INFO ||NEURON_CACHE||: Current remaining items are 18, locked are 1, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000155:  87309  INFO ||NEURON_CACHE||: Current remaining items are 18, locked are 1, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000156:  87308  INFO ||NEURON_CACHE||: Current remaining items are 18, locked are 1, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000157:  87304  INFO ||NEURON_CACHE||: Current remaining items are 17, locked are 2, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000157:  87305  INFO ||NEURON_CACHE||: Current remaining items are 17, locked are 2, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000158:  87306  INFO ||NEURON_CACHE||: Current remaining items are 17, locked are 2, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000158:  87307  INFO ||NEURON_CACHE||: Current remaining items are 17, locked are 2, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000158:  87309  INFO ||NEURON_CACHE||: Current remaining items are 17, locked are 2, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000159:  87308  INFO ||NEURON_CACHE||: Current remaining items are 17, locked are 2, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000159:  87302  INFO ||NEURON_CC_WRAPPER||: Call compiler with cmd: ['neuronx-cc', '--target=trn1', 'compile', '--framework', 'XLA', '/tmp/no-user/neuroncc_compile_workdir/45b92cda-f162-44e0-b24a-d4db519d0388/model.MODULE_18226885169323101566+d41d8cd9.hlo.pb', '--output', '/tmp/no-user/neuroncc_compile_workdir/45b92cda-f162-44e0-b24a-d4db519d0388/model.MODULE_18226885169323101566+d41d8cd9.neff', '--verbose=35']
2024-02-26 16:04:13.000160:  87305  INFO ||NEURON_CACHE||: Current remaining items are 16, locked are 3, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000161:  87306  INFO ||NEURON_CACHE||: Current remaining items are 16, locked are 3, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000161:  87307  INFO ||NEURON_CACHE||: Current remaining items are 16, locked are 3, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000162:  87309  INFO ||NEURON_CACHE||: Current remaining items are 16, locked are 3, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000162:  87308  INFO ||NEURON_CACHE||: Current remaining items are 16, locked are 3, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000162:  87303  INFO ||NEURON_CC_WRAPPER||: Call compiler with cmd: ['neuronx-cc', '--target=trn1', 'compile', '--framework', 'XLA', '/tmp/no-user/neuroncc_compile_workdir/ffc35e06-4b8a-48af-98fe-445aa04a73bb/model.MODULE_10773640740194008060+d41d8cd9.hlo.pb', '--output', '/tmp/no-user/neuroncc_compile_workdir/ffc35e06-4b8a-48af-98fe-445aa04a73bb/model.MODULE_10773640740194008060+d41d8cd9.neff', '--verbose=35']
2024-02-26 16:04:13.000163:  87306  INFO ||NEURON_CACHE||: Current remaining items are 15, locked are 4, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000164:  87307  INFO ||NEURON_CACHE||: Current remaining items are 15, locked are 4, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000164:  87309  INFO ||NEURON_CACHE||: Current remaining items are 15, locked are 4, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000164:  87308  INFO ||NEURON_CACHE||: Current remaining items are 15, locked are 4, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000165:  87304  INFO ||NEURON_CC_WRAPPER||: Call compiler with cmd: ['neuronx-cc', '--target=trn1', 'compile', '--framework', 'XLA', '/tmp/no-user/neuroncc_compile_workdir/bf0eabb3-6737-4f6e-bf3e-1082577edcab/model.MODULE_15521117223395126027+d41d8cd9.hlo.pb', '--output', '/tmp/no-user/neuroncc_compile_workdir/bf0eabb3-6737-4f6e-bf3e-1082577edcab/model.MODULE_15521117223395126027+d41d8cd9.neff', '--verbose=35']
2024-02-26 16:04:13.000166:  87307  INFO ||NEURON_CACHE||: Current remaining items are 14, locked are 5, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000166:  87309  INFO ||NEURON_CACHE||: Current remaining items are 14, locked are 5, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000167:  87308  INFO ||NEURON_CACHE||: Current remaining items are 14, locked are 5, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000167:  87305  INFO ||NEURON_CC_WRAPPER||: Call compiler with cmd: ['neuronx-cc', '--target=trn1', 'compile', '--framework', 'XLA', '/tmp/no-user/neuroncc_compile_workdir/b619b3f7-c636-487d-a515-f696d3505531/model.MODULE_13381711796159894734+d41d8cd9.hlo.pb', '--output', '/tmp/no-user/neuroncc_compile_workdir/b619b3f7-c636-487d-a515-f696d3505531/model.MODULE_13381711796159894734+d41d8cd9.neff', '--verbose=35']
2024-02-26 16:04:13.000167:  87309  INFO ||NEURON_CACHE||: Current remaining items are 13, locked are 6, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000168:  87308  INFO ||NEURON_CACHE||: Current remaining items are 13, locked are 6, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000168:  87306  INFO ||NEURON_CC_WRAPPER||: Call compiler with cmd: ['neuronx-cc', '--target=trn1', 'compile', '--framework', 'XLA', '/tmp/no-user/neuroncc_compile_workdir/214c0bf3-d155-4bb1-a744-06bf7e69a1ac/model.MODULE_931450859643720011+d41d8cd9.hlo.pb', '--output', '/tmp/no-user/neuroncc_compile_workdir/214c0bf3-d155-4bb1-a744-06bf7e69a1ac/model.MODULE_931450859643720011+d41d8cd9.neff', '--verbose=35']
2024-02-26 16:04:13.000169:  87308  INFO ||NEURON_CACHE||: Current remaining items are 12, locked are 7, failed are 0, done are 0, total is 19
2024-02-26 16:04:13.000169:  87307  INFO ||NEURON_CC_WRAPPER||: Call compiler with cmd: ['neuronx-cc', '--target=trn1', 'compile', '--framework', 'XLA', '/tmp/no-user/neuroncc_compile_workdir/3a95058a-3891-4ece-ac2c-ba955859a8d4/model.MODULE_16926392219237896526+d41d8cd9.hlo.pb', '--output', '/tmp/no-user/neuroncc_compile_workdir/3a95058a-3891-4ece-ac2c-ba955859a8d4/model.MODULE_16926392219237896526+d41d8cd9.neff', '--verbose=35']
2024-02-26 16:04:13.000170:  87309  INFO ||NEURON_CC_WRAPPER||: Call compiler with cmd: ['neuronx-cc', '--target=trn1', 'compile', '--framework', 'XLA', '/tmp/no-user/neuroncc_compile_workdir/17fe8a37-8710-4099-a8be-8209c5a5663a/model.MODULE_15254528341555394658+d41d8cd9.hlo.pb', '--output', '/tmp/no-user/neuroncc_compile_workdir/17fe8a37-8710-4099-a8be-8209c5a5663a/model.MODULE_15254528341555394658+d41d8cd9.neff', '--verbose=35']
2024-02-26 16:04:13.000171:  87308  INFO ||NEURON_CC_WRAPPER||: Call compiler with cmd: ['neuronx-cc', '--target=trn1', 'compile', '--framework', 'XLA', '/tmp/no-user/neuroncc_compile_workdir/31fad675-ac08-4fdf-8c61-3625b91c0988/model.MODULE_16718126519652317380+d41d8cd9.hlo.pb', '--output', '/tmp/no-user/neuroncc_compile_workdir/31fad675-ac08-4fdf-8c61-3625b91c0988/model.MODULE_16718126519652317380+d41d8cd9.neff', '--verbose=35']
........
Compiler status PASS
2024-02-26 16:04:16.000243:  87306  INFO ||NEURON_CACHE||: Current remaining items are 11, locked are 7, failed are 0, done are 1, total is 19
2024-02-26 16:04:16.000245:  87306  INFO ||NEURON_CC_WRAPPER||: Call compiler with cmd: ['neuronx-cc', '--target=trn1', 'compile', '--framework', 'XLA', '/tmp/no-user/neuroncc_compile_workdir/899636b5-ce0f-40ab-b434-06764971cbc5/model.MODULE_6875444972028399005+d41d8cd9.hlo.pb', '--output', '/tmp/no-user/neuroncc_compile_workdir/899636b5-ce0f-40ab-b434-06764971cbc5/model.MODULE_6875444972028399005+d41d8cd9.neff', '--verbose=35']
.................................................................................................................................................................................................................................................................................................................................................................................................................................................
```

```bash
MALLOC_ARENA_MAX=64 torchrun --nproc_per_node=32 run_clm.py \
 --model_id mesolitica/tinyllama-1.1b-4096-fpf \
 --dataset_path /home/ubuntu/mosaic-chat-instructions-rag \
 --bf16 True \
 --learning_rate 2e-5 \
 --output_dir litellama-runs \
 --overwrite_output_dir True \
 --skip_cache_push True \
 --per_device_train_batch_size 10 \
 --gradient_checkpointing True \
 --tensor_parallel_size 4 \
 --max_steps 10 \
 --logging_steps 1 \
 --gradient_accumulation_steps 8
```

```
train steps per second -> 0.257
1 steps = 4096 tokens
0.257 * 4096 = X
2xl = X tokens
```