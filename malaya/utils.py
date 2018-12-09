from tqdm import tqdm
import tensorflow as tf
import requests


def download_file(url, filename):
    r = requests.get(
        'http://s3-ap-southeast-1.amazonaws.com/huseinhouse-storage/' + url,
        stream = True,
    )
    total_size = int(r.headers['content-length'])
    with open(filename, 'wb') as f:
        for data in tqdm(
            iterable = r.iter_content(chunk_size = 1048576),
            total = total_size / 1048576,
            unit = 'MB',
            unit_scale = True,
        ):
            f.write(data)


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph
