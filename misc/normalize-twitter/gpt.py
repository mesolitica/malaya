import os

os.environ['OPENAI_API_KEY'] = ''

import json
import argparse
import pickle
from tqdm import tqdm
import ast
import requests

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


pointer = Pointer(f'{filename}.gpt.pickle')
pointer.load()

file = open(f'{filename}.gpt.requested', 'a')

with open(filename) as fopen:
    for i, l in tqdm(enumerate(fopen)):
        if i >= pointer.index:
            data = json.loads(l)
            text = data['original']
            r = None
            for k in range(retry):
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + os.getenv('OPENAI_API_KEY', ''),
                }
                end = 'return JSON structure ({"english", "malay"})'
                text = f"text, `{text}`, translate text to standard english and standard malay, {end}"
                json_data = {
                    'model': 'gpt-3.5-turbo',
                    'messages': [
                        {'role': 'user', 'content': text}
                    ],
                    'temperature': 0.1,
                    'max_tokens': 384,
                }
                try:

                    response = requests.post(
                        'https://api.openai.com/v1/chat/completions',
                        headers=headers,
                        json=json_data,
                        timeout=30.0)

                    c = response.json()['choices'][0]['message']['content'].strip()
                    try:
                        r = json.loads(c)
                        break
                    except BaseException:
                        pass

                    try:
                        r = ast.literal_eval(c)
                        break
                    except BaseException:
                        pass

                except Exception as e:
                    print(k, e)
            data = {'src': data, 'r': r}

            d = json.dumps(data)
            file.write(f'{d}\n')
            file.flush()

            pointer.index = i
            pointer._save()

file.close()
