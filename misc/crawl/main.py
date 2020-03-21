import sys
import argparse


def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            '%s is an invalid positive int value' % value
        )
    return ivalue


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--issue', required = True, help = 'issue to search')
ap.add_argument(
    '-s',
    '--start',
    type = check_positive,
    required = True,
    help = 'year start to crawl',
)
ap.add_argument(
    '-e',
    '--end',
    type = check_positive,
    required = True,
    help = 'year end to crawl',
)
ap.add_argument(
    '-l',
    '--limit',
    type = check_positive,
    required = True,
    help = 'limit of articles to crawl',
)
ap.add_argument(
    '-p',
    '--sleep',
    type = check_positive,
    default = 10,
    help = 'seconds to sleep for every 10 articles',
)
ap.add_argument(
    '-m', '--malaya', default = False, help = 'boolean to use Malaya'
)
args = vars(ap.parse_args())

from core import google_news_run
import json

xgb_model = None

if args['malaya']:
    import malaya

    xgb_model = malaya.xgb_detect_languages()


results = google_news_run(
    args['issue'],
    limit = args['limit'],
    year_start = args['start'],
    year_end = args['end'],
    debug = False,
    sleep_time_every_ten_articles = args['sleep'],
    xgb_model = xgb_model,
)

with open(args['issue'] + '.json', 'w') as fopen:
    fopen.write(json.dumps(results))
