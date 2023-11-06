import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

try:
    from transformers import TextIteratorStreamer
except BaseException:
    TextIteratorStreamer = None


class LLM:
    def __init__(
        self,
        model,
        use_ctranslate2=False,
        **kwargs,
    ):
        self.model_name = model
        self.tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
        self.use_ctranslate2 = use_ctranslate2

        if self.use_ctranslate2:
            from malaya_boilerplate.converter import ctranslate2_generator

            self.model = ctranslate2_generator(model, **kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)

    def _get_input_llama(
        self,
        query,
        keys={'role', 'content'},
        roles={'system', 'user', 'assistant'},
    ):
        """
        We follow exactly Llama2 chatbot tokens.
        https://github.com/facebookresearch/llama/blob/main/llama/generation.py
        """
        if not len(query):
            raise ValueError('`query` must length as least 2.')

        for i in range(len(query)):
            for k in keys:
                if k not in query[i]:
                    raise ValueError(f'{i} message does not have `{k}` key.')

            if query[i]['role'] not in roles:
                raise ValueError(f'Only accept `{roles}` for {i} message role.')

        if query[0]['role'] != 'system':
            raise ValueError('first message of `query` must `system` role.')
        if query[-1]['role'] != 'user':
            raise ValueError('last message of `query` must `user` role.')

        system = query[0]['content']
        user_query = query[-1]['content']

        users, assistants = [], []
        for q in query[1:-1]:
            if q['role'] == 'user':
                users.append(q['content'])
            elif q['role'] == 'assistant':
                assistants.append(q['content'])

        if len(users) != len(assistants):
            raise ValueError(
                'model only support `system, `user` and `assistant` roles, starting with `system`, then `user` and alternating (u/a/u/a/u...)')

        texts = [f'<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n']
        for u, a in zip(users, assistants):
            texts.append(f'{u.strip()} [/INST] {a.strip()} </s><s>[INST] ')
        texts.append(f'{user_query.strip()} [/INST]')
        prompt = ''.join(texts)
        logger.debug(f'prompt: {prompt}')
        if self.use_ctranslate2:
            return self.tokenizer.tokenize(prompt)
        else:
            inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
            return inputs['input_ids'].to(self.model.device)

    def _get_input_mistral(
        self,
        query,
        keys={'role', 'content'},
        roles={'user', 'assistant'},
    ):
        """
        We follow exactly Llama2 chatbot tokens.
        https://github.com/facebookresearch/llama/blob/main/llama/generation.py
        """
        if not len(query):
            raise ValueError('`query` must length as least 2.')

        for i in range(len(query)):
            for k in keys:
                if k not in query[i]:
                    raise ValueError(f'{i} message does not have `{k}` key.')

            if query[i]['role'] not in roles:
                raise ValueError(f'Only accept `{roles}` for {i} message role.')

        if query[0]['role'] != 'system':
            raise ValueError('first message of `query` must `system` role.')
        if query[-1]['role'] != 'user':
            raise ValueError('last message of `query` must `user` role.')

        system = query[0]['content']
        user_query = query[-1]['content']

        users, assistants = [], []
        for q in query[1:-1]:
            if q['role'] == 'user':
                users.append(q['content'])
            elif q['role'] == 'assistant':
                assistants.append(q['content'])

        if len(users) != len(assistants):
            raise ValueError(
                'model only support `system, `user` and `assistant` roles, starting with `system`, then `user` and alternating (u/a/u/a/u...)')

        texts = [f'<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n']
        for u, a in zip(users, assistants):
            texts.append(f'{u.strip()} [/INST] {a.strip()} </s><s>[INST] ')
        texts.append(f'{user_query.strip()} [/INST]')
        prompt = ''.join(texts)
        logger.debug(f'prompt: {prompt}')
        if self.use_ctranslate2:
            return self.tokenizer.tokenize(prompt)
        else:
            inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
            return inputs['input_ids'].to(self.model.device)

    def _get_input(self, query):
        if 'llama2' in self.model_name:
            return self._get_input_llama(query=query)
        elif 'mistral' in self.model_name:
            return self._get_input_mistral(query=query)
        else:
            raise ValueError('Currently we only support Llama2 and Mistral based.')

    def generate(
        self,
        query: List[Dict[str, str]],
        **kwargs,
    ):
        """
        Generate respond from user input.

        Parameters
        ----------
        query: List[Dict[str, str]]
            [
                {
                    'role': 'system',
                    'content': 'anda adalah AI yang berguna',
                },
                {
                    'role': 'user',
                    'content': 'makanan apa yang sedap?',
                }
            ]

        **kwargs: vector arguments pass to huggingface `generate` method.
            Read more at https://huggingface.co/docs/transformers/main_classes/text_generation

            If you are using `use_ctranslate2`, vector arguments pass to ctranslate2 `translate_batch` method.
            Read more at https://opennmt.net/CTranslate2/python/ctranslate2.Translator.html?highlight=translate_batch#ctranslate2.Translator.translate_batch

        Returns
        -------
        result: str
        """
        input_ids = self._get_input(query=query)

        if self.use_ctranslate2:
            o = model.generate_batch([input_ids], **kwargs)
            output = tokenizer.decode(tokenizer.convert_tokens_to_ids(o.sequences[0]))
        else:
            with torch.no_grad():
                generation_output = self.model.generate(
                    input_ids=input_ids,
                    return_dict_in_generate=False,
                    output_scores=False,
                    **kwargs,
                )
                s = generation_output[0]
                output = self.tokenizer.decode(s, skip_special_tokens=True)

        return output

    def generate_stream(
        self,
        query: List[Dict[str, str]],
        **kwargs,
    ):
        """
        Generate respond from user input in streaming mode.

        Parameters
        ----------
        query: str
            User input.
        **kwargs: vector arguments pass to huggingface `generate` method.
            Read more at https://huggingface.co/docs/transformers/main_classes/text_generation

        Returns
        -------
        result: str
        """

        input_ids = self._get_input(query=query)

        if self.use_ctranslate2:
            step_results = self.model.generate_tokens(
                input_ids, **kwargs
            )

            tokens_buffer = []
            text_output = ''

            for step_result in step_results:
                is_new_word = step_result.token.startswith('▁')

                if is_new_word and tokens_buffer:
                    word = tokenizer.decode(tokens_buffer)
                    if word:
                        text_output += ' ' + word
                        yield text_output
                    tokens_buffer = []

                tokens_buffer.append(step_result.token_id)

            if tokens_buffer:
                word = tokenizer.decode(tokens_buffer)
                if word:
                    text_output += ' ' + word
                    yield text_output

        else:
            if TextIteratorStreamer is None:
                raise ModuleNotFoundError(
                    '`transformers.TextIteratorStreamer` is not available, please install latest transformers library.'
                )
            streamer = TextIteratorStreamer(
                self.tokenizer,
                timeout=10.,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            generate_kwargs = dict(
                {'input_ids': input_ids},
                streamer=streamer,
                **kwargs,
            )
            t = Thread(target=self.model.generate, kwargs=generate_kwargs)
            t.start()
            outputs = []
            for text in streamer:
                outputs.append(text)
                yield ''.join(outputs)
