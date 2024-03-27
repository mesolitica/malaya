from transformers.models.deberta_v2.modeling_deberta_v2 import *

# Integrated DeBERTaV2 with EMD


class DebertaV2EmdModel(DebertaV2Model):
    def __init__(self, config):
        super().__init__(config)
        self.z_steps = 2
        assert self.embeddings.position_biased_input == False
        self.position_embeddings = nn.Embedding(
            self.config.max_position_embeddings, self.embeddings.embedding_size
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        encoded_layers = encoder_outputs[1]

        # if self.z_steps > 1:
        #     hidden_states = encoded_layers[-2]
        #     layers = [self.encoder.layer[-1] for _ in range(self.z_steps)]
        #     query_states = encoded_layers[-1]
        #     rel_embeddings = self.encoder.get_rel_embedding()
        #     attention_mask = self.encoder.get_attention_mask(attention_mask)
        #     rel_pos = self.encoder.get_rel_pos(embedding_output)
        #     for layer in layers[1:]:
        #         query_states = layer(
        #             hidden_states,
        #             attention_mask,
        #             output_attentions=False,
        #             query_states=query_states,
        #             relative_pos=rel_pos,
        #             rel_embeddings=rel_embeddings,
        #         )
        #         encoded_layers.append(query_states)

        if self.z_steps > 0 and not self.embeddings.position_biased_input:
            seq_length = input_ids.size(1)
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            z_states = self.position_embeddings(position_ids.long())

            hidden_states = encoded_layers[-2]
            layers = [self.encoder.layer[-1] for _ in range(self.z_steps)]
            query_states = hidden_states + z_states
            rel_embeddings = self.encoder.get_rel_embedding()
            attention_mask = self.encoder.get_attention_mask(attention_mask)
            rel_pos = self.encoder.get_rel_pos(embedding_output)
            for layer in layers:
                query_states = layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=False,
                    query_states=query_states,
                    relative_pos=rel_pos,
                    rel_embeddings=rel_embeddings,
                )
                encoded_layers = encoded_layers + (query_states,)

        sequence_output = encoded_layers[-1]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[(1 if output_hidden_states else 2):]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions,
        )


class DebertaV2EmdForPreTraining(DebertaV2ForMaskedLM):
    def __init__(self, config):
        super(DebertaV2ForMaskedLM, self).__init__(config)
        self.deberta = DebertaV2EmdModel(config)
        self.cls = DebertaV2OnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()
