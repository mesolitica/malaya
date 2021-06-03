from core import google_news_run
import json
import os
import logging

topics = ['mimpi', 'angan-angan']

for topic in topics:
    topic = topic.lower()
    # topic = 'isu ' + topic
    file = topic + '.json'
    if file in os.listdir(os.getcwd()):
        print('passed: ', file)
        continue

    print('crawling', topic)
    results = google_news_run(
        topic,
        limit=100000,
        year_start=2000,
        year_end=2021,
        debug=False,
        sleep_time_every_ten_articles=10
    )

    with open(file, 'w') as fopen:
        fopen.write(json.dumps(results))
