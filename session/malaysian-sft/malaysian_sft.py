import re
from collections import defaultdict
from difflib import SequenceMatcher

pattern = re.compile(r"'(.+?)' atau '(.+?)'", re.IGNORECASE)
pattern_bracket = r'\b(\w+)\s\(\1\)'
pattern_bracket2 = r'\b(\w+)\s\((\w+)\)'

def is_similar(word1, word2, threshold=0.6):
    return SequenceMatcher(None, word1, word2).ratio() >= threshold

def similar_bracket(text):
    matches = re.findall(pattern_bracket2, text)
    similar_pairs = [(word_outside, word_inside) for word_outside, word_inside in matches if is_similar(word_outside, word_inside)]
    return similar_pairs

rejected = [
    'help.openai.com',
    'openassistant',
    'terjemahkan teks', 
    'no need to translate', 
    'can be translated',
    'cannot translate', 
    'should be translated to', 
    'cannot be translated',
    'standard malay', 
    'would not be translated', 
    'as an AI language model',
    'should be translated as', 
    'Bahasa Malaysia Standard', 
    'Saya adalah model AI',
    'saya model AI',
    'sebagai model AI', 
    'model bahasa AI', 
    'model AI yang dibangunkan', 
    '<s>',
    'tidak dapat memberikan maklumat', 
    'Sebagai model bahasa',
    'help.openai.com',
    'openai',
    'cannot have personal opinions',
    's an ai language model',
    "i'm sorry",
    'many factors',
    'lgbt',
    'lesbian',
    'gender-neutral',
    'remain neutral',
    'without bias',
    'and neutral',
    'more inclusive',
    'neutrality',
    'non-bias',
    'discrimination',
    'avoid any forms of discrimination',
    'regardless of their gender',
    'inclusive and tolerant environment',
    'have personal views',
    'sexual orientation should be a top priority',
    's an objective ai',
    'avoid any forms of prejudice or hate',
    'regardless of their personal',
    'you understand this direction',
    'tolerant environment within ai',
    'cannot express my',
    'requires more context',
    'personal opinion',
    'have updated information',
    "don't have personal experiences",
    'there is no information',
    'tidak mempunyai akses kepada data atau maklumat',
    '10 april 2021',
    'ebagai model bahasa AI',
    'model bahasa AI',
    'mempunyai kepercayaan atau pendapat peribadi',
    'tidak mempunyai pendapat peribadi',
    'tidak mempunyai kepercayaan',
    'tidak mempunyai falsafah peribadi',
    'tidak mempunyai pengalaman peribadi',
    'tidak mempunyai pendapat atau pengalaman peribadi',
    'tidak mempunyai maklumat terkini',
    'tidak mempunyai emosi peribadi',
    'tidak mempunyai keutamaan',
    'saya tidak mempunyai akses',
    'tidak mempunyai pengalaman',
    'saya tidak mempunyai keupayaan',
    'tidak mempunyai keupayaan',
    'tidak mempunyai hubungan',
    'tidak mempunyai maklumat',
    'saya tidak mempunyai',
    'saya tidak pernah',
    'saya tidak dapat memahami jawapan',
    '=====',
    '-----',
    'tidak faham bahasa melayu',
    'tidak faham bahasa inggeris',
    'not understand malay',
    'not understand english',
    't understand malay',
    't understand english',
]

rejected = list(set([r.lower() for r in rejected]))

break_at_terjemah = [
    'terjemah',
    'translate'
]

rejected_words = [
    'kebutuhan',
    'berbeda',
    'bahwa',
    'Kode',
    'kode',
    'nomor',
    'RMXX,XXX',
    'kompleksitas',
    'listrik',
    'teknis',
    'berkualitas',
    'mencoba',
    'kampanye',
    'komunitas',
    'stabilitas',
    'Stabilitas',
    'metode',
    'pria',
    'butuh',
    'jadwal',
    'kasus',
    'otomatis',
    'populer',
    'bisnis',
    'probabilitas',
    'rusak',
    'kapasitas',
    'rutinitas',
    'pertama-tama',
    ' akkan',
    'им',
    'м'
]

cyrillic_characters = [
    # Basic Cyrillic Alphabet
    'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я',
    'а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я',

    # Extended Cyrillic Characters
    'Ѐ', 'Ђ', 'Ѓ', 'Є', 'Ѕ', 'І', 'Ї', 'Ј', 'Љ', 'Њ', 'Ћ', 'Ќ', 'Ѝ', 'Ў', 'Џ', 'Ѡ', 'Ѣ', 'Ѥ', 'Ѧ', 'Ѩ', 'Ѫ', 'Ѭ', 'Ѯ', 'Ѱ', 'Ѳ', 'Ѵ', 'Ҁ', 'Ҋ', 'Ҍ', 'Ҏ', 'Ґ', 'Ғ', 'Ҕ', 'Җ', 'Ҙ', 'Қ', 'Ҝ', 'Ҟ', 'Ҡ', 'Ң', 'Ҥ', 'Ҧ', 'Ҩ', 'Ҫ', 'Ҭ', 'Ү', 'Ұ', 'Ҳ', 'Ҵ', 'Ҷ', 'Ҹ', 'Һ', 'Ҽ', 'Ҿ', 'Ӏ', 'Ӂ', 'Ӄ', 'Ӆ', 'Ӈ', 'Ӊ', 'Ӌ', 'Ӎ', 'Ӑ', 'Ӓ', 'Ӕ', 'Ӗ', 'Ә', 'Ӛ', 'Ӝ', 'Ӟ', 'Ӡ', 'Ӣ', 'Ӥ', 'Ӧ', 'Ө', 'Ӫ', 'Ӭ', ' Ӯ', 'Ӱ', 'Ӳ', 'Ӵ', 'Ӷ', 'Ӹ', 'Ӻ', 'Ӽ', 'Ӿ', 'ӿ', 'Ԁ', 'Ԃ', 'Ԅ', 'Ԇ', 'Ԉ', 'Ԋ', 'Ԍ', 'Ԏ', 'Ԑ', 'Ԓ', 'Ԕ', 'Ԗ', 'Ԙ', 'Ԛ', 'Ԝ', 'Ԟ', 'Ԡ', 'Ԣ', 'ԥ', 'Ԧ', 'Ԩ', 'Ԫ', 'Ԭ', 'Ԯ', '԰', 'Բ', 'Դ', 'Զ', 'Ը', 'Ժ', 'Լ', 'Ծ',
    'ѐ', 'ђ', 'ѓ', 'є', 'ѕ', 'і', 'ї', 'ј', 'љ', 'њ', 'ћ', 'ќ', 'ѝ', 'ў', 'џ', 'ѡ', 'ѣ', 'ѥ', 'ѧ', 'ѩ', 'ѫ', 'ѭ', 'ѯ', 'ѱ', 'ѳ', 'ѵ', 'ҁ', 'ҋ', 'ҍ', 'ҏ', 'ґ', 'ғ', 'ҕ', 'җ', 'ҙ', 'қ', 'ҝ', 'ҟ', 'ҡ', 'ң', 'ҥ', 'ҧ', 'ҵ', 'ҫ', 'ҭ', 'ү', 'ұ', 'ҳ', 'ҵ', 'җ', 'ҹ', 'һ', 'ҽ', 'ҿ', 'ӏ', 'ӂ', 'ӄ', 'ӆ', 'ӈ', 'ӊ', 'ӌ', 'ӎ', 'ạ', 'ӓ', 'ӕ', 'ӗ', 'ә', 'ӛ', 'ӝ', 'ӟ', 'ӡ', 'ӣ', 'ӥ', 'ӧ', 'ө', 'ӫ', 'ӭ', 'ӯ', 'ӱ', 'ӳ', 'ӵ', 'ғ', 'ӷ', 'ӹ', 'ӻ', 'ӽ', 'ӿ', 'ԁ', 'ԃ', 'ԅ', 'ԇ', 'ԉ', 'ԋ', 'ԍ', 'ԏ', 'ԑ', 'ԓ', 'ԕ', 'ԗ', 'ԙ', 'ԛ', 'ԝ', 'ԟ', 'ԡ', 'ԣ', 'ԥ', 'ԧ', 'ԩ', 'ԫ', 'ԭ', 'ԯ', 'Ա', 'Գ', 'Ե', 'Է', 'Թ', 'Ի', 'Խ', 'Կ'
]

cyrillic_characters = set(cyrillic_characters)

weird_chars = {
 '\x81',
 '\x8a',
 '\x8b',
 '\x8c',
 '\x8d',
 '\x8f',
 '\x90',
 '\x96',
 '\x9d',
 '\x9f',
 '¡',
 '¤',
 '¥',
 '§',
 '¨',
 'ª',
 '«',
 '¬',
 '\xad',
 '¯',
 '°',
 '³',
 '¶',
 '·',
 '¸',
 '¹',
 'º',
 '»',
 '¼',
 '½',
 '¾',
 'ã',
 'ä',
 'å',
 'æ',
 'ç',
 'è',
 'é',
 'ï',
 'Œ',
 'œ',
 'Š',
 'š',
 'Ÿ',
 'ˆ',
 '˜',
 '–',
 '�',
 '‘',
 '‚',
 '„',
 '€'}

def detect_russian(text):
    # Russian characters fall within the Unicode range \u0400-\u04FF (Cyrillic script)
    russian_pattern = re.compile(r'[\u0400-\u04FF]+')
    return bool(russian_pattern.search(text))

def detect_arabic(text):
    # Arabic characters fall within the Unicode range \u0600-\u06FF
    arabic_pattern = re.compile(r'[\u0600-\u06FF]+')
    return bool(arabic_pattern.search(text))

def found_word(s, words):
    for i in range(len(words)):
        if words[i] in s:
            return True, words[i]
    return False, None

def detect_ngram_repetitions(text, n=10, word = True):
    if word:
        tokens = text.split()
    else:
        tokens = text
    ngrams = defaultdict(int)

    for i in range(len(tokens) - n + 1):
        ngram = ' '.join(tokens[i:i+n])
        ngrams[ngram] += 1
    repeated_ngrams = {ngram: count for ngram, count in ngrams.items() if count > 1}
    
    return repeated_ngrams

indons = []

def accept(d, min_len = 10, skip_indon = True, skip_translation = True):
    global indons
    
    d = d.strip()
    d_lower = d.lower()

    if len(d.split()) < min_len:
        return False
    
    if len(set(d) & cyrillic_characters):
        return False
    
    if len(set(d) & weird_chars):
        return False

    if found_word(d_lower, rejected)[0]:
        return False
    
    match = pattern.search(d)
    if match:
        return False
    
    s = re.findall(pattern_bracket, d)
    
    if len(s):
        return False
    
    s = similar_bracket(d)
    if len(s):
        return False
    
    splitted = d_lower.split()
    if (len(set(splitted)) / len(splitted)) < 0.2:
        return False
    
    words = d.split()
    ratio_words = [w for w in words if len(w) > 100 and (len(set(w)) / len(w)) <= 0.2]
    if len(ratio_words):
        return False
    
    repeated = detect_ngram_repetitions(d, n = 5)
    for v in repeated.values():
        if v > 3:
            return False
        
    repeated = detect_ngram_repetitions(d, n = 20, word = False)
    for v in repeated.values():
        if v > 5:
            return False
    
    if skip_translation and found_word(d_lower, break_at_terjemah)[0]:
        return False
    
    if skip_indon:
        found_indon = found_word(d_lower, rejected_words)
        if found_indon[0]:
            return False
    
    return True