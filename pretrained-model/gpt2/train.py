import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

batch_size = 32
sample_length = 1023
epoch = 10
num_train_steps = 300000
num_warmup_steps = 10
checkpoint = 1000
length = 1024
learning_rate = 1e-4
use_memory_saving_gradients = True
only_train_transformer_layers = False
accumulate_gradients = 1

import json
from gpt_2.src import model, encoder
import custom_optimization

with open('bahasa-vocab.json', 'r') as f:
    en = json.load(f)
with open('bahasa-merges.txt', 'r', encoding = 'utf-8') as f:
    bpe_data = f.read()

bpe_merges = [
    tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]
]
enc_malay = encoder.Encoder(encoder = en, bpe_merges = bpe_merges)

import gpt_2_simple
import pickle

from gpt_2_simple.src import model, sample, encoder, memory_saving_gradients
from gpt_2_simple.src.load_dataset import load_dataset, Sampler
from gpt_2_simple.src.accumulate import AccumulatingOptimizer
from gpt_2_simple import get_available_gpus
import tensorflow as tf

sess = gpt_2_simple.start_tf_sess()

hparams = model.default_hparams()
with open('models/345M/hparams.json') as f:
    hparams.override_from_dict(json.load(f))

hparams.__dict__

context_none = tf.compat.v1.placeholder(tf.int32, [None, None])
gpus = []
gpus = get_available_gpus()

output = model.model(hparams = hparams, X = context_none, gpus = gpus)
loss = tf.reduce_mean(
    input_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels = context_none[:, 1:], logits = output['logits'][:, :-1]
    )
)

context = tf.compat.v1.placeholder(tf.int32, [1, None])

tf_sample = sample.sample_sequence(
    hparams = hparams,
    length = sample_length,
    context = context,
    batch_size = 1,
    temperature = 1.0,
    top_k = 40,
)

all_vars = [v for v in tf.compat.v1.trainable_variables() if 'model' in v.name]
train_vars = (
    [v for v in all_vars if '/h' in v.name]
    if only_train_transformer_layers
    else all_vars
)
opt = tf.compat.v1.train.AdamOptimizer(learning_rate = learning_rate)

if accumulate_gradients > 1:
    if use_memory_saving_gradients:
        exit(
            'Memory saving gradients are not implemented for gradient accumulation yet.'
        )
    opt = AccumulatingOptimizer(opt = opt, var_list = train_vars)
    opt_reset = opt.reset()
    opt_compute = opt.compute_gradients(loss)
    opt_apply = opt.apply_gradients()
    summary_loss = tf.compat.v1.summary.scalar('loss', opt_apply)

else:
    if use_memory_saving_gradients:
        opt_grads = memory_saving_gradients.gradients(loss, train_vars)
    else:
        opt_grads = tf.gradients(ys = loss, xs = train_vars)
    opt_grads = list(zip(opt_grads, train_vars))
    opt_apply = opt.apply_gradients(opt_grads)
    summary_loss = tf.compat.v1.summary.scalar('loss', loss)

# opt = custom_optimization.create_optimizer(
#     loss, learning_rate, num_train_steps, num_warmup_steps
# )
# summary_loss = tf.compat.v1.summary.scalar('loss', loss)


def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass


checkpoint_dir = 'checkpoint'
run_name = 'run1'
checkpoint_path = os.path.join(checkpoint_dir, run_name)
maketree(checkpoint_path)
checkpoint_path

summary_log = tf.compat.v1.summary.FileWriter(checkpoint_path)
saver = tf.compat.v1.train.Saver(var_list = all_vars, max_to_keep = 5)
sess.run(tf.compat.v1.global_variables_initializer())

ckpt = tf.train.latest_checkpoint(checkpoint_path)
if ckpt is None:
    ckpt = tf.train.latest_checkpoint('models/345M/')

print('Loading checkpoint', ckpt)
saver.restore(sess, ckpt)

import pickle

with open('dataset.pkl', 'rb') as fopen:
    dataset = pickle.load(fopen)[:1]

step = 0
for e in range(epoch):
    for l in range(len(dataset)):
        s = []
        for i in range(0, len(dataset[l]) - length, length):
            s.append((i, i + length))

        for i in range(0, len(s), batch_size):
            index = min(i + batch_size, len(s))
            batch_x = []
            for k in s[i:index]:
                batch_x.append(dataset[l][k[0] : k[1]])

            (_, v_loss, v_summary) = sess.run(
                (opt_apply, loss, summary_loss),
                feed_dict = {context_none: batch_x},
            )

            summary_log.add_summary(v_summary, step)
            print(step, v_loss)

            if step % checkpoint == 0:
                print(step, 'saving checkpoint and generating sample')
                saver.save(
                    sess,
                    os.path.join(checkpoint_path, 'model'),
                    global_step = step,
                )
                out = sess.run(tf_sample, feed_dict = {context: batch_x[:1]})
                print(enc_malay.decode(out[0]))

            step += 1

saver.save(sess, os.path.join(checkpoint_path, 'model'), global_step = step)
