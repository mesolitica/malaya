## How-to-Translate

**_Last update 30-September-2019, still usable._**

Some of us want to translate a dataset, like Malaya itself, to find a bahasa based dataset is very hard, so one way to do it, we use translator to translate from a specific language dataset to bahasa based dataset. Everyone agreed that, Google Translate is the best online translator in this world, and to use it massively, we need to pay the service, the API. It is very expensive, really really expensive.

So work around, I code selenium using PhantomJS as the backbone headless browser, plus multithreading!

1. Install dependencies

For ubuntu or debian based,
```bash
wget https://bitbucket.org/ariya/phantomjs/downloads/phantomjs-2.1.1-linux-x86_64.tar.bz2
tar xvjf phantomjs-2.1.1-linux-x86_64.tar.bz2
apt install libfontconfig
cp phantomjs-2.1.1-linux-x86_64/bin/phantomjs /usr/bin/
```

For mac,
```bash
brew install phantomjs
```

Install Python packages,
```bash
pip install selenium tqdm
```

2. Open example from [translator.ipynb](translator.ipynb) for your own kickstart!

## Disclaimer

1. Anytime Google will change the Xpath for the result, so some of time this will not able to work, submit an issue, I will update it.
2. I am not aware of any data protection, use it for your own solely purpose!
