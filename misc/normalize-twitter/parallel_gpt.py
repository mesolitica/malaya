# https://raw.githubusercontent.com/teelinsan/camoscio/main/data/alpaca_data.json

import os

os.environ['OPENAI_API_KEY'] = ''

import openai
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    stop_after_delay,
)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', help='filename')
args = parser.parse_args()
filename = args.filename

# read eviroment variable
openai.api_key = os.environ.get('OPENAI_API_KEY')


@retry(wait=wait_random_exponential(min=1, max=60),
       stop=(stop_after_delay(60) | stop_after_attempt(1)))
def translate_text(value):
    end = 'return JSON structure ({"english", "malay"})'
    text = f"text, `{value}`, translate text to standard english and standard malay, {end}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": text},
        ],
        max_tokens=1024,
        temperature=0.1,
    )
    return response.choices[0]["message"]["content"].strip()


def translate_item(value):
    try:
        return (value, translate_text(value))
    except BaseException:
        return None


MAX_PARALLEL_REQUESTS = 50

data = []
with open(filename) as fopen:
    for i, l in tqdm(enumerate(fopen)):
        try:
            data.append(json.loads(l)['original'])
        except BaseException:
            pass

CHUNK_SIZE = 1000
start = 0
end = len(data)
print(start, end)
# Translate the data in chunks of 1000 items
for i in range(start, end, CHUNK_SIZE):
    start = i
    end = i + CHUNK_SIZE

    new_filename = f'{filename}_{start}_to_{end}.json'
    print(new_filename)
    if os.path.exists(new_filename):
        continue

    translated_data = []
    data_new = data[start:end]

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_REQUESTS) as executor:
        futures = {executor.submit(translate_item, item): item for item in data_new}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Translating"):
            translated_data.append(future.result())

    with open(new_filename, 'w') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=4)

    print(
        f"Translation complete. The translated data is saved in '{new_filename}'")
