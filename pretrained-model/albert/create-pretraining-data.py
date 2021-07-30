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
files.extend(glob('/home/husein/pure-text/the-pile/*.txt'))

directory = sys.argv[1] or 'tfrecord'
os.system(f'mkdir {directory}')


def loop(files):
    client = storage.Client()
    bucket = client.bucket('mesolitica-tpu-general')
    files, index, postfix = files
    output_files = f'{directory}/albert-{index}-{postfix}.tfrecord'
    print(f'Output filename: {output_files}')
    files = ','.join(files)
    tokenizer = tokenization.FullTokenizer(
        vocab_file='sp10m.cased.albert.vocab',
        do_lower_case=False,
        spm_model_file='sp10m.cased.albert.model',
    )

    input_files = []
    for input_pattern in files.split(','):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info('*** Reading from input files ***')
    for input_file in input_files:
        tf.logging.info('  %s', input_file)

    rng = random.Random(random.randint(1, 999999))
    instances = create_training_instances(
        input_files,
        tokenizer,
        max_seq_length=128,
        dupe_factor=2,
        short_seq_prob=0.1,
        masked_lm_prob=0.15,
        max_predictions_per_seq=20,
        rng=rng,
    )

    tf.logging.info('number of instances: %i', len(instances))

    write_instance_to_example_files(
        instances,
        tokenizer,
        max_seq_length=128,
        max_predictions_per_seq=20,
        output_files=output_files.split(','),
    )

    blob = bucket.blob(f'albert-data/{output_files}')
    blob.upload_from_filename(output_files)
    os.system(f'rm {output_files}')


def chunks(l, n, postfix):
    count = 0
    for i in range(0, len(l), n):
        yield l[i: i + n], count, postfix
        count += 1


def multiprocessing(strings, function, postfix, cores=16):
    df_split = chunks(strings, len(strings) // cores, postfix)
    pool = Pool(cores)
    pooled = pool.map(function, df_split)
    pool.close()
    pool.join()


batch_size = 10
for i in range(0, len(files), batch_size):
    multiprocessing(files[i: i + batch_size], loop, i, cores=4)
