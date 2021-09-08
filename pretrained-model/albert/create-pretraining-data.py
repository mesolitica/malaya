import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/husein/t5/prepare/mesolitica-tpu.json'

import subprocess
import os
import itertools
from glob import glob
from multiprocessing import Pool
from google.cloud import storage
import tokenization
import tensorflow as tf
import random
from create_pretraining_data import create_training_instances, write_instance_to_example_files

import sys

files = glob('/home/husein/pure-text/splitted/*.txt*')
files.extend(random.sample(glob('/home/husein/pure-text/the-pile/*.txt'), 10))

directory = sys.argv[1] or 'tfrecord'
os.system(f'mkdir {directory}')
global_count = 0

tf.logging.set_verbosity(tf.logging.INFO)


def loop(files):
    client = storage.Client()
    bucket = client.bucket('mesolitica-tpu-general')
    files, index = files
    output_files = f'{directory}/albert-{index}-{global_count}.tfrecord'
    print(f'Output filename: {output_files}')
    files = ','.join(files)
    tokenizer = tokenization.FullTokenizer(
        vocab_file='sp10m.cased.albert.vocab',
        do_lower_case=False,
        spm_model_file='sp10m.cased.albert.model',
    )
    dupe_factor = 5
    command = f"""
    python3 create_pretraining_data.py \
    --input_file={files} \
    --output_file={output_files} \
    --vocab_file=sp10m.cased.albert.vocab \
    --spm_model_file=sp10m.cased.albert.model \
    --do_lower_case=False --dupe_factor={dupe_factor}
    """
    subprocess.call(command, shell=True,)

    blob = bucket.blob(f'albert-data/{output_files}')
    blob.upload_from_filename(output_files)
    os.system(f'rm {output_files}')


def chunks(l, n):
    count = 0
    for i in range(0, len(l), n):
        yield l[i: i + n], count
        count += 1


def multiprocessing(strings, function, cores=16):
    df_split = chunks(strings, len(strings) // cores)
    pool = Pool(cores)
    pooled = pool.map(function, df_split)
    pool.close()
    pool.join()


batch_size = 12
max_core = 6
for i in range(0, len(files), batch_size):
    b = files[i: i + batch_size]
    multiprocessing(b, loop, min(len(b), max_core))
    global_count += 1
