import requests
import json
import argparse
import pickle
import os
import time
import random
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', help='filename')
args = parser.parse_args()
filename = args.filename
retry = 3

class Pointer:
    def __init__(self, filename):
        self.filename = filename
        self.index = 0

    def _save(self):
        with open(self.filename, 'wb') as fopen:
            pickle.dump(self.index, fopen)

    def increment(self):
        self.index += 1
        self._save()

    def load(self):
        if not os.path.exists(self.filename):
            return
        with open(self.filename, 'rb') as fopen:
            self.index = pickle.load(fopen)


pointer = Pointer(f'{filename}.pickle')
pointer.load()

file = open(f'{filename}.ms-requested', 'a')

with open(filename) as fopen:
    for i, l in tqdm(enumerate(fopen)):
        if i >= pointer.index:
            data = json.loads(l)
            if 'r' in data and 'result' in data['r']:
                text = data['r']['result']
                k = 0
                while True:
                    try:
                        r = requests.post('http://localhost:8999/api', timeout=5, json={
                            'text': text,
                            'from': 'en',
                            'to': 'ms',
                            'lite': True,
                        })
                        r = r.json()
                        if 'error' in r:
                            t = r['message']
                            print(k, f'{t}, sleep for 2.0')      
                            
                        else:
                            break
                    except Exception as e:
                        print(k, e)
                    
                    time.sleep(2.0)
                    k += 1
                    
                data = {'src': data, 'r': r}

                d = json.dumps(data)
                file.write(f'{d}\n')
                file.flush()

            pointer.index = i
            pointer._save()
            
            time.sleep(random.uniform(0.5, 1.0))

file.close()
