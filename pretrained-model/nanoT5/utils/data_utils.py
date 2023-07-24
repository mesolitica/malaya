import torch
import json
import linecache


class DatasetFixed(torch.utils.data.Dataset):
    def __init__(self, filename, tokenizer, before_mask_input_length):
        self.filename = filename
        self.tokenizer = tokenizer
        self.before_mask_input_length = before_mask_input_length
        self.len = 0
        with open(self.filename) as fopen:
            for l in tqdm(fopen):
                self.len += 1

    def __getitem__(self, idx):
        line = json.loads(linecache.getline(self.filename, idx + 1))
        return line

    def __len__(self):
        return self.len


class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, filename, tokenizer, before_mask_input_length):
        self.filename = filename
        self.f = open(self.filename)
        self.tokenizer = tokenizer
        self.before_mask_input_length = before_mask_input_length

    def __iter__(self):
        t = []
        while True:

            while len(t) < self.before_mask_input_length:
                line = self.f.readline()
                if not line:
                    self.f.close()
                    self.f = open(self.filename)
                    line = self.f.readline()
                try:
                    line = json.loads(line)
                    t_ = self.tokenizer(line)['input_ids']
                    t.extend(t_)
                except BaseException:
                    pass

            t_ = t[: self.before_mask_input_length]
            t = t[self.before_mask_input_length:]
            yield {'input_ids': t_}
