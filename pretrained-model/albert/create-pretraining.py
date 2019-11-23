import subprocess

files = '../parliament-text.txt,../wiki-text.txt,../dumping-twitter.txt,../news-text.txt,../dumping-instagram.txt'.split(
    ','
)

for no, file in enumerate(files):
    print('Reading from input files', file)

    output_files = 'albert-%d.tfrecord' % (no)
    print('Output filename', output_files)
    subprocess.call(
        'python3 create_pretraining_data.py --input_file=%s --output_file=%s --vocab_file=sp10m.cased.v8.vocab --spm_model_file=sp10m.cased.v8.model --do_lower_case=False --dupe_factor=5'
        % (file, output_files),
        shell = True,
    )
