from malaya.text_functions import malaya_textcleaning

def test_malaya_textcleaning():
    assert len(malaya_textcleaning('saya sebenarnya sukakan awak, hahahaha')) > 0
