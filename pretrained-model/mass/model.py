import transformer


class Model:
    def __init__(self, bert_config):
        vocab_size = bert_config.vocab_size
        hidden_size = bert_config.hidden_size
        filter_size = bert_config.intermediate_size
        num_encoder_layers = bert_config.num_hidden_layers
        num_decoder_layers = bert_config.num_hidden_layers
        num_heads = bert_config.num_attention_heads
        dropout = bert_config.attention_probs_dropout_prob
        label_smoothing = 0.1
        beam_size = 3
        self.model = transformer.TransformerEncoderDecoderModel(
            vocab_size,
            hidden_size,
            filter_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            label_smoothing,
            dropout,
        )
