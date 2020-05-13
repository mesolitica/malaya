from __future__ import print_function
import errno
import logging
import os
import pickle
import random
import re
import time
import requests
import threading
import json
import urllib.request
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from newspaper import Article
from datetime import datetime, timedelta
from dateutil import parser
from queue import Queue
from urllib.parse import quote
from unidecode import unidecode

NUMBER_OF_CALLS_TO_GOOGLE_NEWS_ENDPOINT = 0

GOOGLE_NEWS_URL = 'https://www.google.com.my/search?q={}&source=lnt&tbs=cdr%3A1%2Ccd_min%3A{}%2Ccd_max%3A{}&tbm=nws&start={}'

logging.basicConfig(
    level = logging.DEBUG, format = '%(asctime)s - %(levelname)s - %(message)s'
)


def get_date(load):
    try:
        date = re.findall(
            '[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', load
        )
        return '%s-%s-%s' % (date[2], date[0], date[1])
    except Exce:
        return False


def run_parallel_in_threads(target, args_list):
    globalparas = []
    result = Queue()

    def task_wrapper(*args):
        result.put(target(*args))

    threads = [
        threading.Thread(target = task_wrapper, args = args)
        for args in args_list
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    while not result.empty():
        globalparas.append(result.get())
    globalparas = list(filter(None, globalparas))
    return globalparas


def forge_url(q, start, year_start, year_end):
    global NUMBER_OF_CALLS_TO_GOOGLE_NEWS_ENDPOINT
    NUMBER_OF_CALLS_TO_GOOGLE_NEWS_ENDPOINT += 1
    return GOOGLE_NEWS_URL.format(
        q.replace(' ', '+'), str(year_start), str(year_end), start
    )


def extract_links(content):
    soup = BeautifulSoup(content, 'html.parser')
    # return soup
    today = datetime.now().strftime('%m/%d/%Y')
    links_list = [
        v.attrs['href']
        for v in soup.find_all('a', {'class': 'l lLrAF'.split()})
    ]
    dates_list = [
        v.text for v in soup.find_all('div', {'class': 'dhIWPd'.split()})
    ]
    output = []
    for (link, date) in zip(links_list, dates_list):
        try:
            date = date.split('-')
            if (
                date[1].find('hour') >= 0
                or date[1].find('minute') >= 0
                or date[1].find('分鐘') >= 0
                or date[1].find('小時') >= 0
            ):
                date[1] = today
            elif date[1].find('day') >= 0 or date[1].find('日') >= 0:
                count = date[1].split(' ')[0]
            else:
                try:
                    date[1] = parser.parse(date[1]).strftime('%m-%d-%Y')
                except:
                    date[1] = 'null'
            output.append((link, date[0].strip(), date[1]))
        except Exception as e:
            print(e)
            continue
    return output


def get_article(link, news, date):
    article = Article(link)
    article.download()
    article.parse()
    article.nlp()
    lang = 'ENGLISH'
    if len(article.title) < 5 or len(article.text) < 5:
        lang = 'INDONESIA'
        print('found BM/ID article')
        article = Article(link, language = 'id')
        article.download()
        article.parse()
        article.nlp()
    return {
        'title': article.title,
        'url': link,
        'authors': article.authors,
        'top-image': article.top_image,
        'text': article.text,
        'keyword': article.keywords,
        'summary': article.summary,
        'news': news,
        'date': date,
        'language': lang,
    }


def google_news_run(
    keyword,
    limit = 10,
    year_start = 2010,
    year_end = 2011,
    debug = True,
    sleep_time_every_ten_articles = 0,
):
    num_articles_index = 0
    ua = UserAgent()
    results = []
    while num_articles_index < limit:
        url = forge_url(keyword, num_articles_index, year_start, year_end)
        if debug:
            logging.debug('For Google -> {}'.format(url))
            logging.debug(
                'Total number of calls to Google = {}'.format(
                    NUMBER_OF_CALLS_TO_GOOGLE_NEWS_ENDPOINT
                )
            )
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36'
        }
        success = False
        try:
            response = requests.get(url, headers = headers, timeout = 60)
            if (
                str(response.content).find(
                    'In the meantime, solving the above CAPTCHA will let you continue to use our services'
                )
                >= 0
            ):
                print('whops, blocked')
                return results
            links = extract_links(response.content)
            nb_links = len(links)
            if nb_links == 0 and num_articles_index == 0:
                print(
                    'No results fetched. Either the keyword is wrong or you have been banned from Google. Retry tomorrow or change of IP Address.'
                )
                return results
            if nb_links == 0:
                print('No more news to read for keyword {}.'.format(keyword))
                return results
            for link in links:
                try:
                    results.append(get_article(*link))
                except:
                    pass
            success = True
        except requests.exceptions.Timeout:
            logging.debug(
                'Google news Timeout. Maybe the connection is too slow. Skipping.'
            )
            continue
        num_articles_index += 10
        if debug and sleep_time_every_ten_articles != 0:
            logging.debug(
                'Program is going to sleep for {} seconds.'.format(
                    sleep_time_every_ten_articles
                )
            )
        time.sleep(sleep_time_every_ten_articles)
    return results
