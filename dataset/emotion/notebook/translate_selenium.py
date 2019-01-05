from selenium import webdriver
import threading
from queue import Queue
import warnings
import time

warnings.filterwarnings("ignore")

span = '/html/body/div[2]/div[1]/div[2]/div[1]/div[1]/div[2]/div[2]/div[1]/div[2]/div/span[1]'
base_url = 'https://translate.google.com/#'

class Translate:
    def __init__(self, from_lang, to_lang):
        self.driver = webdriver.PhantomJS()
        self.driver.set_page_load_timeout(30)
        final_url = base_url + from_lang.lower() + '/' + to_lang.lower()
        self.driver.get(final_url)
    
    def translate(self, string):
        self.driver.find_element_by_id('source').clear()
        self.driver.find_element_by_id('source').send_keys(string.lower())
        time.sleep(1)
        text = [elem.text for elem in self.driver.find_elements_by_xpath(span)]
        return text[0]
    
def task_translate(translator, string):
    return translator.translate(string)
    
def run_parallel_in_threads(args_list, target = task_translate):
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