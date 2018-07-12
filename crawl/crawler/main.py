from core import *
import json

issue ='isu sekolah'
results = google_news_run(issue, limit=1000, year_start=2010, year_end=2019, debug=False, sleep_time_every_ten_articles=10)
with open(issue+'.json','w') as fopen:
    fopen.write(json.dumps(results))
