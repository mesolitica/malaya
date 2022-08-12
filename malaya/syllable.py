import re


class SyllableTokenizer:
    def __init__(self):
        # originally from https://github.com/fahadh4ilyas/syllable_splitter/blob/master/SyllableSplitter.py
        # https://en.wikipedia.org/wiki/Comparison_of_Indonesian_and_Standard_Malay#Syllabification
        # https://www.researchgate.net/figure/Syllable-structures-of-Malay-words_tbl1_290018725
        self.consonant = ['b', 'c', 'd', 'f', 'g', 'h', 'j',
                          'k', 'l', 'm', 'n', 'p', 'q', 'r',
                          's', 't', 'v', 'w', 'x', 'y', 'z',
                          'ng', 'ny', 'sy', 'ch', 'dh', 'gh',
                          'kh', 'ph', 'sh', 'th']

        self.double_consonant = ['ll', 'ks', 'rs', 'rt']

        self.vocal = ['a', 'e', 'i', 'o', 'u']

    def split_letters(self, string):
        letters = []
        arrange = []

        while string != '':
            letter = string[:2]

            if letter.lower() in self.double_consonant:

                if string[2:] != '' and string[2].lower() in self.vocal:
                    letters += [letter[0]]
                    arrange += ['c']
                    string = string[1:]

                else:
                    letters += [letter]
                    arrange += ['c']
                    string = string[2:]

            elif letter.lower() in self.consonant:
                letters += [letter]
                arrange += ['c']
                string = string[2:]

            elif letter.lower() in self.vocal:
                letters += [letter]
                arrange += ['v']
                string = string[2:]

            else:
                letter = string[0]

                if letter.lower() in self.consonant:
                    letters += [letter]
                    arrange += ['c']
                    string = string[1:]

                elif letter.lower() in self.vocal:
                    letters += [letter]
                    arrange += ['v']
                    string = string[1:]

                else:
                    letters += [letter]
                    arrange += ['s']
                    string = string[1:]

        return letters, ''.join(arrange)

    def split_syllables_from_letters(self, letters, arrange):
        consonant_index = re.search('vc{2,}', arrange)
        while consonant_index:
            i = consonant_index.start()+1
            letters = letters[:i+1]+['|']+letters[i+1:]
            arrange = arrange[:i+1]+'|'+arrange[i+1:]
            consonant_index = re.search('vc{2,}', arrange)

        vocal_index = re.search(r'v{2,}', arrange)
        while vocal_index:
            i = vocal_index.start()
            letters = letters[:i+1]+['|']+letters[i+1:]
            arrange = arrange[:i+1]+'|'+arrange[i+1:]
            vocal_index = re.search(r'v{2,}', arrange)

        vcv_index = re.search(r'vcv', arrange)
        while vcv_index:
            i = vcv_index.start()
            letters = letters[:i+1]+['|']+letters[i+1:]
            arrange = arrange[:i+1]+'|'+arrange[i+1:]
            vcv_index = re.search(r'vcv', arrange)

        sep_index = re.search(r'[cvs]s', arrange)
        while sep_index:
            i = sep_index.start()
            letters = letters[:i+1]+['|']+letters[i+1:]
            arrange = arrange[:i+1]+'|'+arrange[i+1:]
            sep_index = re.search(r'[cvs]s', arrange)

        sep_index = re.search(r's[cvs]', arrange)
        while sep_index:
            i = sep_index.start()
            letters = letters[:i+1]+['|']+letters[i+1:]
            arrange = arrange[:i+1]+'|'+arrange[i+1:]
            sep_index = re.search(r's[cvs]', arrange)

        return ''.join(letters).split('|')

    def tokenize(self, string):
        """
        Tokenize string into multiple strings using syllable patterns.

        Parameters
        ----------
        string : str

        Returns
        -------
        result: List[str]
        """
