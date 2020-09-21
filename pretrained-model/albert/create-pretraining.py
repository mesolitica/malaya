import subprocess
from glob import glob
import os
from multiprocessing import Pool
import itertools

files = glob('../pure-text/splitted/*.txt*')


def loop(files):
    for file in files:
        print('Reading from input file', file)
        output_files = f'albert-{os.path.split(file)[1]}.tfrecord'
        if 'common-crawl' in file:
            dupe_factor = 2
        else:
            dupe_factor = 10

        print(f'Output filename: {output_files}, dupe factor: {dupe_factor}')

        subprocess.call(
            f'python3 create_pretraining_data.py --input_file={file} --output_file={output_files} --vocab_file=sp10m.cased.v10.vocab --spm_model_file=sp10m.cased.v10.model --do_lower_case=False --dupe_factor={dupe_factor}',
            shell = True,
        )


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def multiprocessing(strings, function, cores = 16):
    df_split = chunks(strings, len(strings) // cores)
    pool = Pool(cores)
    pooled = pool.map(function, df_split)
    pool.close()
    pool.join()


multiprocessing(files, loop)
