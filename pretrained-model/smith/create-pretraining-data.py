import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ[
    'GOOGLE_APPLICATION_CREDENTIALS'
] = '/home/husein/t5/prepare/mesolitica-tpu.json'

from create_pretraining_data import process_documents
from glob import glob
import tokenization
from multiprocessing import Pool
import itertools
import os
from google.cloud import storage

tokenizer = tokenization.FullTokenizer(
    vocab_file = 'BERT.wordpiece', do_lower_case = False
)

files = glob('/home/husein/pure-text/splitted/*.txt')

os.system('mkdir tfrecord')


def loop(files):
    client = storage.Client()
    bucket = client.bucket('mesolitica-tpu-general')
    for file in files:
        output_files = f'tfrecord/smith-{os.path.split(file)[1]}.tfrecord'
        process_documents(file, output_files, tokenizer)
        blob = bucket.blob(f'smith-data/{output_files}')
        blob.upload_from_filename(output_files)
        os.system(f'rm {output_files}')


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def multiprocessing(strings, function, cores = 10):
    df_split = chunks(strings, len(strings) // cores)
    pool = Pool(cores)
    pooled = pool.map(function, df_split)
    pool.close()
    pool.join()


multiprocessing(files, loop)
