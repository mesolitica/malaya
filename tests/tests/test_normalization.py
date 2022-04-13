import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MALAYA_USE_HUGGINGFACE'] = 'true'

import sys
import malaya
import logging

logging.basicConfig(level=logging.DEBUG)

string1 = 'krajaan patut bagi pencen awal skt kpd warga emas supaya emosi'
string2 = 'Husein ska mkn aym dkat kampng Jawa'
string3 = 'Melayu malas ni narration dia sama je macam men are trash. True to some, false to some.'
string4 = 'Tapi tak pikir ke bahaya perpetuate myths camtu. Nanti kalau ada hiring discrimination despite your good qualifications because of your race tau pulak marah. Your kids will be victims of that too.'
string5 = 'DrM cerita Melayu malas semenjak saya kat University (early 1980s) and now as i am edging towards retirement in 4-5 years time after a career of being an Engineer, Project Manager, General Manager'
string6 = 'blh bntg dlm kls nlp sy, nnti intch'
string7 = '031 313.212-2341'


def test_normalization():
    corrector = malaya.spell.probability()
    normalizer = malaya.normalize.normalizer(corrector)
    string = 'boleh dtg 8pagi esok tak atau minggu depan? 2 oktober 2019 2pm, tlong bayar rm 3.2k sekali tau'
    normalizer.normalize(string)
