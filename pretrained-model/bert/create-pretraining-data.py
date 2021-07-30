import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/husein/t5/prepare/mesolitica-tpu.json'

import tensorflow as tf
from multiprocessing import Pool
from glob import glob
import json
import re
import random
import tokenization
import os
import itertools
import sys
from google.cloud import storage
from create_pretraining_data import create_training_instances, write_instance_to_example_files

files = glob('/home/husein/pure-text/splitted/*.txt*')
files.extend(glob('/home/husein/pure-text/the-pile/*.txt'))
random.shuffle(files)

tokenizer = tokenization.FullTokenizer(
    vocab_file='BERT.wordpiece', do_lower_case=False
)
directory = sys.argv[1]
os.system(f'mkdir {directory}')


def loop(files):
    client = storage.Client()
    bucket = client.bucket('mesolitica-tpu-general')
    input_files, index = files
    output_file = f'{directory}/bert-{index}.tfrecord'

    print('*** Reading from input files ***')
    for input_file in input_files:
        print(input_file)

    max_seq_length = 128
    dupe_factor = 5
    max_predictions_per_seq = 20
    masked_lm_prob = 0.15
    short_seq_prob = 0.1
    rng = random.Random(12345)
    instances = create_training_instances(
        input_files,
        tokenizer,
        max_seq_length,
        dupe_factor,
        short_seq_prob,
        masked_lm_prob,
        max_predictions_per_seq,
        rng,
    )

    print('*** Writing to output files ***')

    write_instance_to_example_files(
        instances,
        tokenizer,
        max_seq_length,
        max_predictions_per_seq,
        [output_file],
    )

    blob = bucket.blob(f'bert-data/{output_file}')
    blob.upload_from_filename(output_file)
    os.system(f'rm {output_file}')


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


multiprocessing(files, loop)
