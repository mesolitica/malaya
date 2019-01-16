
# coding: utf-8

# In[1]:


import word2vec
import numpy as np
import tensorflow as tf
import json
import os
import re
from unidecode import unidecode
os.environ['CUDA_VISIBLE_DEVICES'] = ''


# In[2]:


with open('wiki-ms.txt') as fopen:
    sentences = fopen.read()


def cleaning(string):
    string = re.sub(
        'http\S+|www.\S+',
        '',
        ' '.join(
            [i for i in string.split() if i.find('#') < 0 and i.find('@') < 0]
        ),
    )
    string = unidecode(string).replace('.', '. ').replace(',', ', ')
    string = re.sub('[^A-Za-z ]+', ' ', string)
    string = re.sub(r'[ ]+', ' ', string).strip()
    string = ' '.join(
            [
                i
                for i in re.findall('[\\w\']+|[;:\-\(\)&.,!?"]', string)
                if len(i)
            ]
        )
    return string.lower()
# In[3]:

sentences = cleaning(sentences).split()

word_array, dictionary, rev_dictionary, num_lines, num_words = word2vec.build_word_array(sentences,vocab_size=1000000)


# In[4]:


len(dictionary)


# In[5]:


X, Y = word2vec.build_training_set(word_array)
graph_params = {'batch_size': 32,
                'vocab_size': np.max(X)+1,
                'embed_size': 256,
                'hid_size': 256,
                'neg_samples': 128,
                'learn_rate': 0.01,
                'momentum': 0.9,
                'embed_noise': 0.1,
                'hid_noise': 0.3,
                'epoch':10,
                'optimizer': 'Momentum'}


# In[6]:


split = round(X.shape[0]*0.9)
train_X, train_Y = X[:split, :], Y[:split, :]
test_X, test_Y = X[split:, :], Y[split:, :]


# In[7]:


model = word2vec.Model(graph_params)
print('model built, vocab size %d, document length %d'%(np.max(X)+1, len(word_array)))


# In[ ]:


embed_weights, nce_weights = model.train(train_X, train_Y, test_X, test_Y,
                                         graph_params['epoch'],
                                         graph_params['batch_size'])


# In[ ]:


import pickle
with open('word2vec-wiki-256.p', 'wb') as fopen:
    pickle.dump({'dictionary':dictionary,'rev_dictionary':rev_dictionary,
                 'embed_weights':embed_weights,'nce_weights':nce_weights}, fopen)

