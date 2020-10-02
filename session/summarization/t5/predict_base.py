import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import json
import tensorflow as tf
import tensorflow_datasets as tfds
import t5

import tensorflow_text

tf.compat.v1.reset_default_graph()
sess = tf.InteractiveSession()
meta_graph_def = tf.compat.v1.saved_model.load(sess, ['serve'], 'model')
signature_def = meta_graph_def.signature_def['serving_default']
pred = lambda x: sess.run(
    fetches = signature_def.outputs['outputs'].name,
    feed_dict = {signature_def.inputs['input'].name: x},
)

# https://malaya-dataset.s3-ap-southeast-1.amazonaws.com/summary/test-set-cnn.json
with open('test-set-cnn.json') as fopen:
    data = json.load(fopen)['X']

print(len(data))

from tqdm import tqdm

results = []
batch = 4

try:
    for i in tqdm(range(0, (len(data) // batch) * batch, batch)):
        x = data[i : i + batch]
        r = pred(x).tolist()
        r = [k.decode() for k in r]
        results.extend(r)
except:
    pass

with open('output-base.json', 'w') as fopen:
    json.dump(results, fopen)
