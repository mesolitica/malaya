from tqdm import tqdm
import requests
chunk_size = 1024

def download_file(url, filename):
    r = requests.get(url, stream = True)
    total_size = int(r.headers['content-length'])
    with open(filename, 'wb') as f:
        for data in tqdm(iterable = r.iter_content(chunk_size = chunk_size), total = total_size/chunk_size, unit = 'KB'):
            f.write(data)
