import subprocess
from glob import glob

files = glob('dumping-*.txt')

for no, file in enumerate(files):
    print('Reading from input files', file)

    output_files = f'albert-{no}.tfrecord'
    print('Output filename', output_files)
    subprocess.call(
        f'python3 create_pretraining_data.py --input_file={file} --output_file={output_files} --vocab_file=sp10m.cased.v10.vocab --spm_model_file=sp10m.cased.v10.model --do_lower_case=False --dupe_factor=5',
        shell = True,
    )
