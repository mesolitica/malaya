from selenium import webdriver
import threading
from queue import Queue
from tqdm import tqdm
import warnings
import time

warnings.filterwarnings('ignore')

span = '/html/body/div[2]/div[1]/div[2]/div[1]/div[1]/div[2]/div[3]/div[1]/div[2]/div/span[1]'
base_url = 'https://translate.google.com/#'


class Translate:
    def __init__(self, from_lang, to_lang):
        self.driver = webdriver.PhantomJS()
        self.driver.set_page_load_timeout(30)
        final_url = base_url + from_lang.lower() + '/' + to_lang.lower()
        self.driver.get(final_url)

    def translate(self, string):
        assert isinstance(string, str), 'input must be a string'
        self.driver.find_element_by_id('source').clear()
        self.driver.find_element_by_id('source').send_keys(string)
        time.sleep(2)
        text = [elem.text for elem in self.driver.find_elements_by_xpath(span)]
        if len(text):
            return text[0]
        else:
            return '?'


def task_translate(translator, string):
    return translator.translate(string)


def run_parallel_in_threads(args_list, target = task_translate):
    globalparas = [None] * len(args_list)
    result = Queue()

    def task_wrapper(*args):
        result.put((target(*args), args_list.index(args)))

    threads = [
        threading.Thread(target = task_wrapper, args = args)
        for args in args_list
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    while not result.empty():
        res = result.get()
        globalparas.insert(res[1], res[0])
    globalparas = list(filter(None, globalparas))
    return globalparas


class Translate_Concurrent:
    def __init__(self, batch_size, from_lang = 'en', to_lang = 'ms'):
        self._batch_size = batch_size
        self._translators = [
            Translate(from_lang, to_lang) for _ in range(self._batch_size)
        ]

    def translate_batch(self, strings):
        assert isinstance(strings, list) and isinstance(
            strings[0], str
        ), 'input must be list of strings'
        translated = []
        for no in tqdm(range(0, len(strings), self._batch_size), ncols = 70):
            data = strings[no : no + self._batch_size]
            combined = [
                (self._translators[i], data[i]) for i in range(len(data))
            ]
            translated.extend(run_parallel_in_threads(combined))
        return translated
