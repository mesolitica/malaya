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
