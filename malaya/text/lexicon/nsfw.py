from herpetologist import check_type
from typing import List


class Lexicon:
    def __init__(self, corpus):
        self._corpus = corpus
        self.nsfw_sex = self._corpus['nsfw_sex']
        self.nsfw_gambling = self._corpus['nsfw_gambling']
        self.gambling_rejected = self._corpus['gambling_rejected']
        self.rejected = self._corpus['rejected']

    @check_type
    def predict(self, strings: List[str]):
        """
        classify list of strings.

        Parameters
        ----------
        strings: List[str]

        Returns
        -------
        result: List[str]
        """

        labels = []
        for string in strings:
            lowered = string.lower()
            labelled = False

            for word in self.nsfw_sex:
                if (
                    word in lowered
                    and all([r not in lowered for r in self.rejected])
                    and len(lowered) < 5000
                ):
                    labels.append('sex')
                    labelled = True
                    continue

            if labelled:
                continue

            for word in self.nsfw_gambling:
                if (
                    word in lowered
                    and all([r not in lowered for r in self.rejected])
                    and all([r not in lowered for r in self.gambling_rejected])
                    and len(lowered) < 5000
                ):
                    labels.append('gambling')
                    labelled = True
                    continue

            if labelled:
                continue

            labels.append('negative')

        return labels
