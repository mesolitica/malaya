import torch


class Base:
    def compile(self):
        if getattr(self, 'use_ctranslate2', False):
            raise ValueError('`compile` method not able to use for ctranslate2 model.')
        self.model = torch.compile(self.model)

    def eval(self, **kwargs):
        if getattr(self, 'use_ctranslate2', False):
            raise ValueError('`eval` method not able to use for ctranslate2 model.')
        return self.model.eval(**kwargs)

    def cuda(self, **kwargs):
        if getattr(self, 'use_ctranslate2', False):
            raise ValueError('`cuda` method not able to use for ctranslate2 model.')
        return self.model.cuda(**kwargs)
