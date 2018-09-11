# How-to-Crawl

1. Install dependencies
```bash
sudo apt update
sudo apt install python3-pip -y
export LC_ALL=C
sudo pip3 install bs4 newspaper3k fake_useragent unidecode
sudo apt-get install libxml2-dev libxslt-dev libjpeg-dev zlib1g-dev libpng12-dev -y
curl https://raw.githubusercontent.com/codelucas/newspaper/master/download_corpora.py | python3
```
2. Edit main.py
```python
# can be anything
issue ='isu sekolah'
# edit parameters as you want
google_news_run(issue, limit=1000, year_start=2010, year_end=2019, debug=False, sleep_time_every_ten_articles=10)
3. Run main.py
```python3 main.py
```

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
