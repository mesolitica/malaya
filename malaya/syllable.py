import re
import logging

logger = logging.getLogger(__name__)


class SyllableTokenizer:
    def __init__(self):
        """
        originally from https://github.com/fahadh4ilyas/syllable_splitter/blob/master/SyllableSplitter.py
        improved `cuaca` double vocal `ua` based on https://en.wikipedia.org/wiki/Comparison_of_Indonesian_and_Standard_Malay#Syllabification
        improved `rans` double consonant `ns` based on https://www.semanticscholar.org/paper/Syllabification-algorithm-based-on-syllable-rules-Musa-Kadir/a819f255f066ae0fd7a30b3534de41da37d04ea1
        """
        self.consonant = ['b', 'c', 'd', 'f', 'g', 'h', 'j',
                          'k', 'l', 'm', 'n', 'p', 'q', 'r',
                          's', 't', 'v', 'w', 'x', 'y', 'z',
                          'ng', 'ny', 'sy', 'ch', 'dh', 'gh',
                          'kh', 'ph', 'sh', 'th']

        self.double_consonant = ['ll', 'ks', 'rs', 'rt', 'ns']

        self.double_vocal = ['ua']

        self.vocal = ['a', 'e', 'i', 'o', 'u']

    def split_letters(self, string):
        letters = []
        arrange = []

        while string != '':
            letter = string[:2]
            logger.info(f'letter: {letter}, string: {string}')

            if letter.lower() in self.double_consonant:

                if string[2:] != '' and string[2].lower() in self.vocal:
                    letters += [letter[0]]
                    arrange += ['c']
                    string = string[1:]

                else:
                    letters += [letter]
                    arrange += ['c']
                    string = string[2:]

            elif letter.lower() in self.double_vocal:
                letters += [letter]
                arrange += ['v']
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
        Example from https://www.semanticscholar.org/paper/Syllabification-algorithm-based-on-syllable-rules-Musa-Kadir/a819f255f066ae0fd7a30b3534de41da37d04ea1/figure/0,
        'cuaca' -> ['cua', 'ca']
        'insurans' -> ['in', 'su', 'rans']
        'praktikal' -> ['prak', 'ti', 'kal']
        'strategi' -> ['stra', 'te', 'gi']
        'ayam' -> ['a', 'yam']
        'anda' -> ['an', 'da']
        'hantu' -> ['han', 'tu']

        Parameters
        ----------
        string : str

        Returns
        -------
        result: List[str]
        """
        letters, arrange = self.split_letters(string)
        return self.split_syllables_from_letters(letters, arrange)
