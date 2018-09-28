from malaya.text_functions import malaya_textcleaning

def test_malaya_textcleaning():
    assert len(malaya_textcleaning('saya sebenarnya sukakan awak, hahahaha')) > 0

def test_malaya_textcleaning_trash():
    assert not len(malaya_textcleaning('asdsad asdsad easdcxv'))
