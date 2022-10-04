from malaya.dictionary.english.words import words as _english_words
from malaya.dictionary.bahasa.words import words as _malay_words
from malaya.dictionary.bahasa.cambridge_words import words as _cambridge_malay_words
from malaya.dictionary.bahasa.kamus_dewan import words as _kamus_dewan_words
from malaya.dictionary.bahasa.dbp import words as _dbp_words
from malaya.dictionary.bahasa.negeri import negeri
from malaya.dictionary.bahasa.city import city
from malaya.dictionary.bahasa.country import country
from malaya.dictionary.bahasa.places import places
from malaya.dictionary.bahasa.daerah import daerah
from malaya.dictionary.bahasa.parlimen import parlimen
from malaya.dictionary.bahasa.adun import adun
from malaya.text.rules import rules_normalizer
from typing import List

ENGLISH_WORDS = {i for i in _english_words if len(i) > 2} | {'me', 'on', 'of'}
MALAY_WORDS = _malay_words
CAMBRIDGE_MALAY_WORDS = _cambridge_malay_words
KAMUS_DEWAN_WORDS = _kamus_dewan_words
DBP_WORDS = _dbp_words

available_requests = True
try:
    import requests
    from requests.structures import CaseInsensitiveDict

    headers_corpus = CaseInsensitiveDict()
    headers_corpus["Accept"] = "*/*"
    headers_corpus["Accept-Language"] = "en-MY,en;q=0.9,en-US;q=0.8,ms;q=0.7"
    headers_corpus["Cache-Control"] = "no-cache"
    headers_corpus["Connection"] = "keep-alive"
    headers_corpus["Content-Type"] = "application/x-www-form-urlencoded; charset=UTF-8"
    headers_corpus["Cookie"] = "__ssds=3; __ssuzjsr3=a9be0cd8e; __uzmaj3=f3203f85-ceb6-46a0-a558-b1e11a637ecd; __uzmbj3=1661748827; ASP.NET_SessionId=vgtl1vyxxybw4m3rpkpvatmg; _gid=GA1.3.1764105897.1661921894; _ga=GA1.1.1010277242.1661748827; __uzmcj3=1743010683960; __uzmdj3=1661934381; _ga_50X5PQW96W=GS1.1.1661934348.9.1.1661934385.0.0.0"
    headers_corpus["Origin"] = "http://sbmb.dbp.gov.my"
    headers_corpus["Pragma"] = "no-cache"
    headers_corpus["Referer"] = "http://sbmb.dbp.gov.my/korpusdbp/Search2.aspx"
    headers_corpus["User-Agent"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36"
    headers_corpus["X-MicrosoftAjax"] = "Delta=true"
    headers_corpus["X-Requested-With"] = "XMLHttpRequest"

except Exception as e:
    available_requests = False

available_bs4 = True
try:
    from bs4 import BeautifulSoup
except Exception as e:
    available_bs4 = False

parser = None


def check_requests():
    if not available_requests:
        raise ModuleNotFoundError(
            'requests not installed. Please install it by `pip3 install requests` and try again.'
        )


def check_bs4():
    if not available_bs4:
        raise ModuleNotFoundError(
            'beautifulsoup4 not installed. Please install it by `pip3 install beautifulsoup4` and try again.'
        )


def keyword_wiktionary(
    word: str,
    acceptable_lang: List[str] = ['brunei malay', 'malay'],
):
    """
    crawl https://en.wiktionary.org/wiki/ to check a word is a malay word.

    Parameters
    ----------
    word: str
    acceptable_lang: List[str], optional (default=['brunei malay', 'malay'])
        acceptable languages in wiktionary section.

    Returns
    -------
    result: Dict
    """
    global parser

    try:
        from wiktionaryparser import WiktionaryParser
    except BaseException:
        raise ModuleNotFoundError(
            'wiktionaryparser not installed. Please install it by `pip3 install wiktionaryparser` and try again.'
        )

    if parser is None:
        parser = WiktionaryParser()

    response = parser.session.get(parser.url.format(word), params={'oldid': None})
    parser.soup = BeautifulSoup(response.text.replace('>\n<', '><'), 'html.parser')
    parser.current_word = word
    parser.clean_html()
    results = {}
    for lang in acceptable_lang:
        results[lang] = parser.get_word_data(lang)
    return results


def keyword_dbp(word: str, parse: bool = False):
    """
    crawl https://prpm.dbp.gov.my/cari1?keyword= to check a word is a malay word.

    Parameters
    ----------
    word: str
    parse: bool, optional (default=False)
        if True, will parse using BeautifulSoup.

    Returns
    -------
    result: Dict
    """
    check_requests()

    url = f'https://prpm.dbp.gov.my/Cari1?keyword={word}'
    r = requests.get(url)
    if parse:
        check_bs4()

        soup = BeautifulSoup(r._content)
        definitions = soup.select("div[class*=tab-pane]")
        definitions = [d.text for d in definitions]
        spans = soup.find_all('span', id='MainContent_SearchInfoTesaurus_lblTesaurus')
        if len(spans):
            tesaurus = spans[0]
            tds = tesaurus.findAll('td')
            selected_td = None
            for td in tds:
                if 'javascript:showModalDialog' in str(td):
                    selected_td = td
            if selected_td:
                tesaurus = [a.text for a in selected_td.find_all('a')]
            else:
                tesaurus = None
        else:
            tesaurus = None
        if len(definitions):
            return {'definisi': definitions, 'tesaurus': tesaurus}
        else:
            return False
    else:
        if b'Definisi :' in r._content:
            return True
        else:
            return False


def corpus_dbp(word):
    """
    crawl http://sbmb.dbp.gov.my/korpusdbp/Search2.aspx to search corpus based on a word.

    Parameters
    ----------
    word: str

    Returns
    -------
    result: pandas.core.frame.DataFrame
    """
    import pandas as pd

    check_requests()
    check_bs4()

    url = 'http://sbmb.dbp.gov.my/korpusdbp/Search2.aspx'
    data = f"ctl00%24ContentPlaceHolder1%24txtKata={word}&ctl00%24RadScriptManager1=ctl00%24ContentPlaceHolder1%24ctl00%24ContentPlaceHolder1%24pnlContentPanel%7Cctl00%24ContentPlaceHolder1%24cmdSearch&RadScriptManager1_TSM=%3B%3BTelerik.Web.UI%2C%20Version%3D2014.3.1024.45%2C%20Culture%3Dneutral%2C%20PublicKeyToken%3D121fae78165ba3d4%3Aen-US%3A45fa33f7-195b-4d8e-a1d1-d5955cf24e2c%3A16e4e7cd%3Aed16cbdc%3A365331c3%3A24ee1bba%3A92fe8ea0%3Af46195d3%3Afa31b949%3A874f8ea2%3Ac128760b%3A19620875%3A490a9d4e%3A88144a7a&__EVENTTARGET=ctl00%24ContentPlaceHolder1%24cmdSearch&__EVENTARGUMENT=&__VIEWSTATE=%2FwEPDwUKLTMwMzk1OTI4MA9kFgJmD2QWAgIDD2QWAgIDD2QWBgIDD2QWBAIBD2QWAgICDw8WAh4RVXNlU3VibWl0QmVoYXZpb3JoZGQCAw8PFgIeB1Zpc2libGVoZBYEAgEPFCsAAhQrAAJkEBYCZgIBFgIUKwACZGQUKwACZGQPFgJmZhYBBW9UZWxlcmlrLldlYi5VSS5SYWRUYWIsIFRlbGVyaWsuV2ViLlVJLCBWZXJzaW9uPTIwMTQuMy4xMDI0LjQ1LCBDdWx0dXJlPW5ldXRyYWwsIFB1YmxpY0tleVRva2VuPTEyMWZhZTc4MTY1YmEzZDRkZAIDDxQrAAJkFQIKS29ua29yZGFucwhLb2xva2FzaRYEZg9kFgQCAQ8PFgIfAGhkZAIDDzwrAA4CABQrAAIPFgIeElJlc29sdmVkUmVuZGVyTW9kZQspc1RlbGVyaWsuV2ViLlVJLlJlbmRlck1vZGUsIFRlbGVyaWsuV2ViLlVJLCBWZXJzaW9uPTIwMTQuMy4xMDI0LjQ1LCBDdWx0dXJlPW5ldXRyYWwsIFB1YmxpY0tleVRva2VuPTEyMWZhZTc4MTY1YmEzZDQBZBcAARYCFgsPAgUUKwAFZDwrAAUBBAULS29udGVrc0tpcmlkPCsABQEEBQxLb250ZWtzS2FuYW48KwAFAQQFDU1ha2x1bWF0QmFoYW5kZRQrAAALKXpUZWxlcmlrLldlYi5VSS5HcmlkQ2hpbGRMb2FkTW9kZSwgVGVsZXJpay5XZWIuVUksIFZlcnNpb249MjAxNC4zLjEwMjQuNDUsIEN1bHR1cmU9bmV1dHJhbCwgUHVibGljS2V5VG9rZW49MTIxZmFlNzgxNjViYTNkNAE8KwAHAAspdVRlbGVyaWsuV2ViLlVJLkdyaWRFZGl0TW9kZSwgVGVsZXJpay5XZWIuVUksIFZlcnNpb249MjAxNC4zLjEwMjQuNDUsIEN1bHR1cmU9bmV1dHJhbCwgUHVibGljS2V5VG9rZW49MTIxZmFlNzgxNjViYTNkNAFkZGRkZmQCAQ9kFgYCAQ8UKwACZBAWAmYCARYCFCsAAmRkFCsAAmRkDxYCZmYWAQV5VGVsZXJpay5XZWIuVUkuUmFkVG9vbEJhckJ1dHRvbiwgVGVsZXJpay5XZWIuVUksIFZlcnNpb249MjAxNC4zLjEwMjQuNDUsIEN1bHR1cmU9bmV1dHJhbCwgUHVibGljS2V5VG9rZW49MTIxZmFlNzgxNjViYTNkNGQCAw9kFgoCAQ8PZBYCHgVzdHlsZQURdGV4dC1hbGlnbjpyaWdodDtkAgcPEGQQFQgHUGVydGFtYQRLZS0yBEtlLTMES2UtNARLZS01BEtlLTYES2UtNwRLZS04FQgBMQEyATMBNAE1ATYBNwE4FCsDCGdnZ2dnZ2dnFgFmZAIJDxBkEBUIB1BlcnRhbWEES2UtMgRLZS0zBEtlLTQES2UtNQRLZS02BEtlLTcES2UtOBUIATEBMgEzATQBNQE2ATcBOBQrAwhnZ2dnZ2dnZxYBZmQCCw8PFgIfAGhkZAIPD2QWAmYPZBYCZg9kFgICAQ88KwAOAgAUKwACDxYCHwILKwQBZBcAARYCFgsPAgUUKwAFZDwrAAUBBAULS29udGVrc0tpcmlkPCsABQEEBQxLb250ZWtzS2FuYW48KwAFAQQFDU1ha2x1bWF0QmFoYW5kZRQrAAALKwUBPCsABwALKwYBZGRkZGZkAgUPZBYCAgEPPCsADgIAFCsAAg8WAh8CCysEAWQXAAEWAhYLZGRlFCsAAAsrBQE8KwAHAAsrBgFkZGRkZmQCBQ8UKwADDxYIHhdFbmFibGVBamF4U2tpblJlbmRlcmluZ2geHEVuYWJsZUVtYmVkZGVkQmFzZVN0eWxlc2hlZXRnHwILKwQBHhVFbmFibGVFbWJlZGRlZFNjcmlwdHNnZGRkZAIHDw8WAh8EaGRkGAEFHl9fQ29udHJvbHNSZXF1aXJlUG9zdEJhY2tLZXlfXxYBBS5jdGwwMCRDb250ZW50UGxhY2VIb2xkZXIxJHJ3QXN5bmNQcm9jZXNzV2luZG93li5%2BC%2BdmzAEF6P6astcpGGrBT5A%3D&__VIEWSTATEGENERATOR=93ABCA44&__EVENTVALIDATION=%2FwEdAAORbmMsdI9RgJFn3DgB0hnlDMG9drBKQHQbvep7utWD7%2F3VBRkrGvjMj%2Bf9fGNUz2vF5aj38Uz5QuvKWes0idadGtr%2BCg%3D%3D&ctl00_ContentPlaceHolder1_rwAsyncProcessWindow_ClientState=&__ASYNCPOST=true&RadAJAXControlID=ctl00_ContentPlaceHolder1_RadAjaxManager1"
    r = requests.post(url, headers=headers_corpus, data=data)
    soup = BeautifulSoup(r._content)

    table = soup.find_all('table', id='ctl00_ContentPlaceHolder1_rgKonkordan_ctl00')
    if not len(table):
        return False

    df = pd.read_html(str(table[0]))
    if not len(df):
        return False

    df = df[0].iloc[1:-3]
    df.columns = ['No', 'Konteks Kiri', 'Kata', 'Konteks Kanan', 'Maklumat Artikel/Bahan']
    return df


def is_malay(word, stemmer=None):
    """
    Check a word is a malay word.

    Parameters
    ----------
    word: str
    stemmer: Callable, optional (default=None)
        a Callable object, must have `stem_word` method.

    Returns
    -------
    result: bool
    """
    if word.lower() in rules_normalizer:
        return False

    if stemmer is not None:
        if not hasattr(stemmer, 'stem_word'):
            raise ValueError('stemmer must have `stem_word` method')

        word = stemmer.stem_word(word)
        if word.lower() in rules_normalizer:
            return False

    return word in MALAY_WORDS or word in CAMBRIDGE_MALAY_WORDS or word in KAMUS_DEWAN_WORDS or word in DBP_WORDS


def is_english(word):
    """
    Check a word is an english word.

    Parameters
    ----------
    word: str

    Returns
    -------
    result: bool
    """
    is_in = False
    if word in ENGLISH_WORDS:
        is_in = True
    elif len(word) > 1 and word[-1] in 's' and word[:-1] in ENGLISH_WORDS:
        is_in = True
    return is_in


def is_malaysia_location(string):
    string_lower = string.lower()
    title = string_lower.title()
    if string_lower in negeri or title in city or title in country or title in daerah or title in parlimen or title in adun:
        return True
    return False
