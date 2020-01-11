# How-to-Crawl

**_Last update 11-November-2019, still usable, as long not use GCP ip addresses._**

1. Install dependencies

Malaya must installed first.

For ubuntu / debian based
```bash
pip3 install bs4 newspaper3k fake_useragent unidecode
apt-get install libxml2-dev libxslt-dev libjpeg-dev zlib1g-dev libpng12-dev -y
curl https://raw.githubusercontent.com/codelucas/newspaper/master/download_corpora.py | python3
```

For Mac OS
```bash
brew install libxml2 libxslt
brew install libtiff libjpeg webp little-cms2
pip3 install bs4 newspaper3k fake_useragent unidecode
curl https://raw.githubusercontent.com/codelucas/newspaper/master/download_corpora.py | python3
```

2. Run main.py

```bash
python3 main.py -i "isu mahathir" -s 2009 -e 2019 -l 10
```

You can read more about crawler in [Malaya Wiki](https://github.com/DevconX/Malaya/wiki).

## Issues crawled

You can get download some crawled data from https://github.com/huseinzol05/Malaya-Dataset#news-crawled