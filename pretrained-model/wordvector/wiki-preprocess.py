import json

with open('dumping-wiki-20-july-2019.json') as fopen:
    wiki = json.load(fopen)

with open('wiki-ms.txt', 'w') as fopen:
    fopen.write(' '.join(wiki))
