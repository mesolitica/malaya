import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from malaya.model.stem import Base as BaseStem
from malaya.model.syllable import Base as BaseSyllable
from malaya.model.syllable import replace_same_length
from malaya_boilerplate.torch_utils import to_numpy
from malaya.text.function import phoneme_textcleaning

SOS_token = 0


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long,
                                    device=self.device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        if target_tensor is not None:
            maxlen = target_tensor.shape[1]
        else:
            maxlen = encoder_outputs.shape[1] * 2

        for i in range(maxlen):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None  # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(RNN, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, dropout_p=dropout_p)
        self.decoder = DecoderRNN(hidden_size, input_size)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_tensor, target_tensor=None):
        input_tensor = input_tensor.to(self.device)
        if target_tensor is not None:
            target_tensor = target_tensor.to(self.device)
        encoder_out = self.encoder(input_tensor)
        return self.decoder(*encoder_out, target_tensor)

    def greedy_decoder(self, input_tensor):
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            encoder_out = self.encoder(input_tensor)
            decoder_outputs, decoder_hidden, _ = self.decoder(*encoder_out)
            return decoder_outputs


class Model:
    def __init__(self, model, hidden_size, pth):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = RNN(input_size=self.tokenizer.vocab_size, hidden_size=hidden_size)
        self.model.load_state_dict(torch.load(pth, map_location='cpu'))

    def cuda(self):
        return self.model.cuda()

    def eval(self):
        return self.model.eval()

    def forward(self, strings):
        input_ids = self.tokenizer(strings, padding=True, return_tensors='pt')['input_ids']
        argmax = to_numpy(self.model.greedy_decoder(input_ids)).argmax(axis=-1)
        results = []
        for a in argmax:
            results_ = []
            for a_ in a:
                if a_ == self.tokenizer.eos_token_id:
                    break
                else:
                    results_.append(a_)
            results.append(results_)
        return self.tokenizer.batch_decode(results)


class Stem(Model, BaseStem):
    def __init__(self, *args, **kwargs):
        Model.__init__(self, *args, **kwargs)

    def stem_word(self, word):
        """
        Stem a word, this also include lemmatization using greedy decoder.

        Parameters
        ----------
        string : str

        Returns
        -------
        result: str
        """

        return self.forward([word])[0]

    def stem(self, string: str):
        """
        Stem a string, this also include lemmatization.

        Parameters
        ----------
        string : str

        Returns
        -------
        result: str
        """

        return super().stem(string)


class Syllable(Model, BaseSyllable):
    def __init__(self, *args, **kwargs):
        Model.__init__(self, *args, **kwargs)

    def tokenize_word(self, word: str):
        """
        Tokenize a word using greedy decoder.

        Parameters
        ----------
        word: str

        Returns
        -------
        result: str
        """
        predicted = self.forward([word])[0]
        r_ = replace_same_length(word, predicted)
        if r_[0]:
            predicted = r_[1]
        return predicted.split('.')

    def tokenize(self, string):
        """
        Tokenize string into multiple strings.

        Parameters
        ----------
        string: str

        Returns
        -------
        result: List[str]
        """

        return super().tokenize(string)


class Phoneme(Model):
    def __init__(self, *args, **kwargs):
        Model.__init__(self, *args, **kwargs)

    def predict(self, strings):
        """
        Convert to target strings.

        Parameters
        ----------
        strings : List[str]

        Returns
        -------
        result: List[str]
        """
        strings = [phoneme_textcleaning(s) for s in strings]
        return self.forward(strings)
