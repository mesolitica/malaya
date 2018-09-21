import os, shutil
from pathlib import Path
import sys

try:
    os.mkdir(str(Path.home())+'/Malaya')
except:
    pass
    
with open(str(Path.home())+'/Malaya/version', 'w') as fopen:
    fopen.write('0.0')
import malaya

def test_make_directory():
    del sys.modules['malaya']
    shutil.rmtree(str(Path.home())+'/Malaya')
    import malaya
    with open(str(Path.home())+'/Malaya/version', 'w') as fopen:
        fopen.write('0.0')
    del sys.modules['malaya']
    import malaya

def test_directory():
    with open(str(Path.home())+'/Malaya/something', 'w') as fopen:
        fopen.write('something')
    del sys.modules['malaya']
    import malaya
    assert os.path.exists(str(Path.home())+'/Malaya')

def test_print():
    malaya.describe_pos_malaya()
    malaya.describe_pos()
    malaya.describe_entities_malaya()
    malaya.describe_entities()
