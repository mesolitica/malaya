from tqdm import tqdm
import tensorflow as tf
import requests
import numpy as np

def download_file(url, filename):
    r = requests.get('http://s3-ap-southeast-1.amazonaws.com/huseinhouse-storage/'+url, stream = True)
    total_size = int(r.headers['content-length'])
    with open(filename, 'wb') as f:
        for data in tqdm(iterable = r.iter_content(chunk_size = 1048576), total = total_size/1048576, unit = 'MB'):
            f.write(data)

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph

def str_idx(corpus, dic, UNK=0):
    maxlen = max([len(i) for i in corpus])
    X = np.zeros((len(corpus),maxlen))
    for i in range(len(corpus)):
        for no, k in enumerate(corpus[i][:maxlen][::-1]):
            try:
                X[i,-1 - no]=dic[k]
            except Exception as e:
                X[i,-1 - no]=UNK
    return X

def generate_char_seq(batch,idx2word,char2idx):
    x = [[len(idx2word[i]) for i in k] for k in batch]
    maxlen = max([j for i in x for j in i])
    temp = np.zeros((batch.shape[0],batch.shape[1],maxlen),dtype=np.int32)
    for i in range(batch.shape[0]):
        for k in range(batch.shape[1]):
            for no, c in enumerate(idx2word[batch[i,k]].lower()):
                temp[i,k,-1-no] = char2idx[c]
    return temp
