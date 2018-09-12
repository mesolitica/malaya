import malaya
import os
from pathlib import Path

def test_directory():
     assert os.path.exists(str(Path.home())+'/Malaya')

def test_print():
    malaya.describe_pos_malaya()
    malaya.describe_entities_malaya()
    malaya.describe_entities()
