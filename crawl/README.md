# How-to-Crawl

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

You can get download some crawled data from [here](https://s3-ap-southeast-1.amazonaws.com/huseinhouse-storage/crawler-data.zip)

Last updated (15th August 2018), crawled until the end of google results.

1. isu agama
2. isu air
3. isu artis
4. isu astro
5. isu bahasa melayu
6. isu barisan nasional
7. isu dunia
8. isu ekonomi
9. isu harga
10. isu kerajaan
11. isu kesihatan
12. isu lgbt
13. isu mahathir
14. isu malaysia
15. isu minyak
16. isu najib razak
17. isu pelakon
18. isu pembangkang
19. isu politik
20. isu rosmah
21. isu sekolah
22. isu sosial media
23. isu sosial
24. isu sultan melayu
25. isu teknologi
26. isu tm
