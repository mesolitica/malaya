import torch
from threading import Thread
from malaya.text.prompt import template_alpaca, template_malaya
from malaya_boilerplate.converter import ctranslate2_generator
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

try:
    from transformers import TextIteratorStreamer
except BaseException:
    TextIteratorStreamer = None


class LLM:
    def __init__(self, model, use_ctranslate2=False, torch_dtype=None, **kwargs):

        self.tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
        self.use_ctranslate2 = use_ctranslate2

        if self.use_ctranslate2:
            self.model = ctranslate2_generator(model, **kwargs)
        else:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                logger.warning('This model is running on CPU, this will consume a lot memory.')
                self.device = 'cpu'

            if self.device == 'cuda':
                device_map = 'auto'
                major, _ = torch.cuda.get_device_capability()
                if torch_dtype is None:
                    if major >= 8:
                        logger.info('compute capability is >= 8, able to use bloat16.')
                        torch_dtype = torch.bfloat16
                    else:
                        torch_dtype = torch.float16
            else:
                device_map = {'': self.device}
                if torch_dtype is None:
                    torch_dtype = 'auto'

            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch_dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
                **kwargs,
            )

            _ = self.model.eval()

    def _get_input(self, query):
        prompt = f'#User: {query}\n#Bot: '
        logger.debug(f'prompt: {prompt}')
        if self.use_ctranslate2:
            return self.tokenizer.tokenize(prompt)
        else:
            inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
            return inputs['input_ids'].to(self.device)

    def generate(
        self,
        query: str,
        split_by='\n#Bot:',
        **kwargs,
    ):
        """
        Generate respond from user input.

        Parameters
        ----------
        query: str
            User input.

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

        return output.split(split_by)[1].strip()

    def generate_stream(
        self,
        query: str,
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

    def gradio(self, **kwargs):
        """
        Chatbot Gradio interface.

        Parameters
        ----------
        **kwargs: keyword arguments for `demo.launch`.
        """
        try:
            import gradio as gr
        except BaseException:
            raise ModuleNotFoundError(
                'gradio not installed. Please install it by `pip install gradio` and try again.'
            )

        def clear_and_save_textbox(message):
            return '', message

        def display_input(message, history):
            history.append((message, ''))
            return history

        def delete_prev_fn(history):
            try:
                message, _ = history.pop()
            except IndexError:
                message = ''
            return history, message or ''

        def generate(
            message,
            history_with_input,
            system_prompt,
            max_new_tokens: int,
            temperature,
            top_p,
            top_k,
        ):

            history = history_with_input[:-1]
            if self.use_ctranslate2:
                generator = self.generate_stream(
                    message,
                    max_length=max_new_tokens,
                    sampling_topk=top_k,
                    sampling_topp=top_p,
                    sampling_temperature=temperature
                )
            else:
                generator = self.generate_stream(
                    message,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_beams=1,
                )
            try:
                first_response = next(generator)
                yield history + [(message, first_response)]
            except StopIteration:
                yield history + [(message, '')]
            for response in generator:
                yield history + [(message, response)]

        with gr.Blocks() as demo:

            with gr.Group():
                chatbot = gr.Chatbot(label='Chatbot')
                with gr.Row():
                    textbox = gr.Textbox(
                        container=False,
                        show_label=False,
                        placeholder='Type a message...',
                        scale=10,
                    )
                    submit_button = gr.Button('Submit',
                                              variant='primary',
                                              scale=1,
                                              min_width=0)
            with gr.Row():
                retry_button = gr.Button('🔄  Retry', variant='secondary')
                undo_button = gr.Button('↩️ Undo', variant='secondary')
                clear_button = gr.Button('🗑️  Clear', variant='secondary')

            saved_input = gr.State()

            with gr.Accordion(label='Advanced options', open=False):
                system_prompt = gr.Textbox(label='System prompt',
                                           value='',
                                           lines=6)
                max_new_tokens = gr.Slider(
                    label='Max new tokens',
                    minimum=1,
                    maximum=4096,
                    step=1,
                    value=1024,
                )
                temperature = gr.Slider(
                    label='Temperature',
                    minimum=0.1,
                    maximum=4.0,
                    step=0.1,
                    value=1.0,
                )
                top_p = gr.Slider(
                    label='Top-p (nucleus sampling)',
                    minimum=0.05,
                    maximum=1.0,
                    step=0.05,
                    value=0.95,
                )
                top_k = gr.Slider(
                    label='Top-k',
                    minimum=1,
                    maximum=1000,
                    step=1,
                    value=50,
                )

            gr.Examples(
                examples=[
                    'siapa perdana menteri malaysia',
                    'camne nak code numpy 2d array',
                ],
                inputs=textbox,
                outputs=[textbox, chatbot],
                cache_examples=False,
            )

            textbox.submit(
                fn=clear_and_save_textbox,
                inputs=textbox,
                outputs=[textbox, saved_input],
                api_name=False,
                queue=False,
            ).then(
                fn=display_input,
                inputs=[saved_input, chatbot],
                outputs=chatbot,
                api_name=False,
                queue=False,
            ).success(
                fn=generate,
                inputs=[
                    saved_input,
                    chatbot,
                    system_prompt,
                    max_new_tokens,
                    temperature,
                    top_p,
                    top_k,
                ],
                outputs=chatbot,
                api_name=False,
            )

            button_event_preprocess = submit_button.click(
                fn=clear_and_save_textbox,
                inputs=textbox,
                outputs=[textbox, saved_input],
                api_name=False,
                queue=False,
            ).then(
                fn=display_input,
                inputs=[saved_input, chatbot],
                outputs=chatbot,
                api_name=False,
                queue=False,
            ).success(
                fn=generate,
                inputs=[
                    saved_input,
                    chatbot,
                    system_prompt,
                    max_new_tokens,
                    temperature,
                    top_p,
                    top_k,
                ],
                outputs=chatbot,
                api_name=False,
            )

            retry_button.click(
                fn=delete_prev_fn,
                inputs=chatbot,
                outputs=[chatbot, saved_input],
                api_name=False,
                queue=False,
            ).then(
                fn=display_input,
                inputs=[saved_input, chatbot],
                outputs=chatbot,
                api_name=False,
                queue=False,
            ).then(
                fn=generate,
                inputs=[
                    saved_input,
                    chatbot,
                    system_prompt,
                    max_new_tokens,
                    temperature,
                    top_p,
                    top_k,
                ],
                outputs=chatbot,
                api_name=False,
            )

            undo_button.click(
                fn=delete_prev_fn,
                inputs=chatbot,
                outputs=[chatbot, saved_input],
                api_name=False,
                queue=False,
            ).then(
                fn=lambda x: x,
                inputs=[saved_input],
                outputs=textbox,
                api_name=False,
                queue=False,
            )

            clear_button.click(
                fn=lambda: ([], ''),
                outputs=[chatbot, saved_input],
                queue=False,
                api_name=False,
            )
        return demo.queue(max_size=20).launch(**kwargs)
