from malaya.text.regex import _expressions
from malaya.text.function import split_into_sentences
import re
import html


class Tokenizer:
    def __init__(self, **kwargs):
        """
        Load Tokenizer object.
        Check supported regex pattern at
        https://github.com/huseinzol05/Malaya/blob/master/malaya/text/regex.py#L85

        Parameters
        ----------
        emojis: bool, optional (default=False)
            True to keep emojis.
        urls: bool, optional (default=True)
            True to keep urls.
        urls_improved: bool, optional (default=True)
            True to keep urls, better version.
        tags: bool, optional (default=True)
            True to keep tags: <tag>.
        emails: bool, optional (default=True)
            True to keep emails.
        users: bool, optional (default=True)
            True to keep users handles: @cbaziotis.
        hashtags: bool, optional (default=True)
            True to keep hashtags.
        phones: bool, optional (default=True)
            True to keep phones.
        percents: bool, optional (default=True)
            True to keep percents.
        money: bool, optional (default=True)
            True to keep money expressions.
        date: bool, optional (default=True)
            True to keep date expressions.
        time: bool, optional (default=True)
            True to keep time expressions.
        time_pukul: bool, optional (default=True)
            True to keep time `pukul` expressions.
        acronyms: bool, optional (default=True)
            True to keep acronyms.
        emoticons: bool, optional (default=True)
            True to keep emoticons.
        censored: bool, optional (default=True)
            True to keep censored words: f**k.
        emphasis: bool, optional (default=True)
            True to keep words with emphasis: *very* good.
        numbers: bool, optional (default=True)
            True to keep numbers.
        numbers_with_shortform: bool, optional (default=True)
            True to keep numbers with shortform.
        temperature: bool, optional (default=True)
            True to keep temperatures
        distance: bool, optional (default=True)
            True to keep distances.
        volume: bool, optional (default=True)
            True to keep volumes.
        duration: bool, optional (default=True)
            True to keep durations.
        weight: bool, optional (default=True)
            True to keep weights.
        hypen: bool, optional (default=True)
            True to keep hypens.
        ic: bool, optional (default=True)
            True to keep Malaysian IC.
        title: bool, optional (default=True)
            True to keep title with dot, Dr. ayam -> ['Dr.', 'ayam']
        parliament: bool, optional (default=True)
            True to keep P.123 / D.123
        hijri_year: bool, optional (default=True)
            True to keep pada hijri year expressions.
        hari_bulan: bool, optional (default=True)
            True to keep pada hari bulan expressions.
        pada_tarikh: bool, optional (default=True)
            True to keep pada tarikh expressions.
        word_dash: bool, optional (default=True)
            True to keep pada a word that at least 2 dashes expressions.
        """

        pipeline = []
        self.regexes = _expressions

        emojis = kwargs.get('emojis', False)
        urls = kwargs.get('urls', True)
        urls_improved = kwargs.get('urls_improved', True)
        tags = kwargs.get('tags', True)
        emails = kwargs.get('emails', True)
        users = kwargs.get('users', True)
        hashtags = kwargs.get('hashtags', True)
        cashtags = kwargs.get('cashtags', True)
        phones = kwargs.get('phones', True)
        percents = kwargs.get('percents', True)
        money = kwargs.get('money', True)
        date = kwargs.get('date', True)
        time = kwargs.get('time', True)
        time_pukul = kwargs.get('time_pukul', True)
        acronyms = kwargs.get('acronyms', True)
        emoticons = kwargs.get('emoticons', True)
        censored = kwargs.get('censored', True)
        emphasis = kwargs.get('emphasis', True)
        numbers = kwargs.get('numbers', True)
        numbers_with_shortform = kwargs.get('numbers_with_shortform', True)
        temperatures = kwargs.get('temperature', True)
        distances = kwargs.get('distance', True)
        volumes = kwargs.get('volume', True)
        durations = kwargs.get('duration', True)
        weights = kwargs.get('weight', True)
        hypens = kwargs.get('hypen', True)
        ic = kwargs.get('ic', True)
        title = kwargs.get('title', True)
        parliament = kwargs.get('parliament', True)
        hijri_year = kwargs.get('hijri_year', True)
        hari_bulan = kwargs.get('hari_bulan', True)
        pada_tarikh = kwargs.get('pada_tarikh', True)
        word_dash = kwargs.get('word_dash', True)
        passport = kwargs.get('passport', True)

        if word_dash:
            pipeline.append(self.regexes['word_dash'])

        if title:
            pipeline.append(self.regexes['title'])

        if parliament:
            pipeline.append(self.regexes['parliament'])

        if urls:
            pipeline.append(self.regexes['url'])

        if urls_improved:
            pipeline.append(self.regexes['url_v2'])
            pipeline.append(self.regexes['url_dperini'])

        if tags:
            pipeline.append(self.regexes['tag'])

        if emails:
            pipeline.append(self.wrap_non_matching(self.regexes['email']))

        if users:
            pipeline.append(self.wrap_non_matching(self.regexes['user']))

        if hashtags:
            pipeline.append(self.wrap_non_matching(self.regexes['hashtag']))

        if cashtags:
            pipeline.append(self.wrap_non_matching(self.regexes['cashtag']))

        if phones:
            pipeline.append(self.wrap_non_matching(self.regexes['phone']))

        if percents:
            pipeline.append(self.wrap_non_matching(self.regexes['percent']))

        if money:
            pipeline.append(self.wrap_non_matching(self.regexes['money']))

        if date:
            pipeline.append(self.wrap_non_matching(self.regexes['date']))

        if time:
            pipeline.append(self.wrap_non_matching(self.regexes['time']))

        if time_pukul:
            pipeline.append(self.wrap_non_matching(self.regexes['time_pukul']))

        if acronyms:
            pipeline.append(self.wrap_non_matching(self.regexes['acronym']))

        if emoticons:
            pipeline.append(self.regexes['ltr_face'])
            pipeline.append(self.regexes['rtl_face'])

        if censored:
            pipeline.append(self.wrap_non_matching(self.regexes['censored']))

        if emphasis:
            pipeline.append(self.wrap_non_matching(self.regexes['emphasis']))

        if emoticons:
            pipeline.append(
                self.wrap_non_matching(self.regexes['rest_emoticons'])
            )

        if temperatures:
            pipeline.append(self.wrap_non_matching(self.regexes['temperature']))

        if distances:
            pipeline.append(self.wrap_non_matching(self.regexes['distance']))

        if volumes:
            pipeline.append(self.wrap_non_matching(self.regexes['volume']))

        if durations:
            pipeline.append(self.wrap_non_matching(self.regexes['duration']))

        if weights:
            pipeline.append(self.wrap_non_matching(self.regexes['weight']))

        if ic:
            pipeline.append(self.wrap_non_matching(self.regexes['ic']))

        if numbers_with_shortform:
            pipeline.append(self.regexes['number_with_shortform'])

        if numbers:
            pipeline.append(self.regexes['number'])

        if emojis:
            pipeline.append(self.regexes['emoji'])

        if hypens:
            pipeline.append(self.regexes['hypen'])
        
        if hijri_year:
            pipeline.append(self.wrap_non_matching(self.regexes['hijri_year']))
        
        if hari_bulan:
            pipeline.append(self.wrap_non_matching(self.regexes['hari_bulan']))
        
        if pada_tarikh:
            pipeline.append(self.wrap_non_matching(self.regexes['pada_tarikh']))

        if passport:
            pipeline.append(self.wrap_non_matching(self.regexes['passport']))

        pipeline.append(self.regexes['apostrophe'])
        pipeline.append(self.regexes['word'])

        if emoticons:
            pipeline.append(
                self.wrap_non_matching(self.regexes['eastern_emoticons'])
            )

        # keep repeated puncts as one term
        # pipeline.append(r"")

        pipeline.append('(?:\\S)')  # CATCH ALL remaining terms

        self.tok = re.compile(r'({})'.format('|'.join(pipeline)))

    @staticmethod
    def wrap_non_matching(exp):
        return '(?:{})'.format(exp)

    def tokenize(self, string: str, lowercase: bool = False):
        """
        Tokenize string into words.

        Parameters
        ----------
        string : str
        lowercase: bool, optional (default=False)

        Returns
        -------
        result: List[str]
        """
        escaped = html.unescape(string)
        tokenized = self.tok.findall(escaped)
        tokenized = [t[0] if isinstance(t, tuple) else t for t in tokenized]
        tokenized_all = []
        for t in tokenized:
            if len(re.findall(r'\.{2,}', t)):
                splitted = [w if len(w) else '.' for w in t.split('.')]
                tokenized_all.extend(splitted)
            else:
                tokenized_all.append(t)
        tokenized = [re.sub(r'[ ]+', ' ', t).strip() for t in tokenized_all]

        if lowercase:
            tokenized = [t.lower() for t in tokenized]

        return tokenized


class SentenceTokenizer:
    def __init__(self):
        pass

    def tokenize(self, string, minimum_length=5):
        """
        Tokenize string into multiple strings.

        Parameters
        ----------
        string : str
        minimum_length: int, optional (default=5)
            minimum length to assume a string is a string, default 5 characters.

        Returns
        -------
        result: List[str]
        """
        return split_into_sentences(string, minimum_length=minimum_length)
