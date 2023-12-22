
"""PyTorch b model."""


import math
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F

from transformers.file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers import BloomConfig, BloomPreTrainedModel, BloomModel, AutoConfig, PreTrainedModel, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutput, Seq2SeqLMOutput



logger = logging.get_logger(__name__)


import torch
import torch.nn as nn

class PrefixEncoder(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, config, model):
        super().__init__()
        self.model = model

        self.dropout = torch.nn.Dropout(config.hidden_dropout)



        self.config = config

        if config.pooling==True:
            self.W = nn.Linear(config.hidden_size, 1)
            self.key= torch.nn.Linear(config.pre_seq_len, config.pre_seq_len*config.num_hidden_layers * config.hidden_size)
            self.value= torch.nn.Linear(config.pre_seq_len, config.pre_seq_len*config.num_hidden_layers * config.hidden_size)
        else:
            self.W =None
            self.key= torch.nn.Linear(config.hidden_size, config.num_hidden_layers * config.hidden_size)
            self.value= torch.nn.Linear(config.hidden_size, config.num_hidden_layers * config.hidden_size)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        batch_size=None,
        output_attentions=None,
        output_hidden_states=None,
        past_key_values=None,
        return_dict=None,
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )
        #print('here', outputs[0].shape)

        #att_w = torch.nn.functional.softmax(self.W(outputs[0]).squeeze(-1), dim=-1)
        #print('att_w', att_w.shape)
        if self.config.pooling==True:
            att_w = self.W(outputs[0]).squeeze(-1)
            key = self.key(att_w)
            value = self.value(att_w)
        else:
            key = self.key(outputs[0])
            value = self.value(outputs[0])
        #print('working', key.shape)

        #print('key', key.shape, 'value', value.shape)

        # bsz, seqlen, _ = past_key_values.shape
        past_key = key.view(
            batch_size*self.config.n_head,
            self.config.pre_seq_len,
            self.config.head_dim,
            self.config.n_layer,
        )

        past_value = value.view(
            batch_size*self.config.n_head,
            self.config.pre_seq_len,
            self.config.head_dim,
            self.config.n_layer,

        )
        #print(past_key_values.shape)
        past_key = self.dropout(past_key).permute([3, 0, 2, 1])
        past_value = self.dropout(past_value).permute([3, 0, 1, 2])
        #print('past_key', past_key.shape)
        #print('past_value', past_value.shape)

        # Create a list to store the tuples
        result = []

        # Iterate over the first dimension (24 in your case)
        for i in range(past_key.shape[0]):
            # For each iteration, create a tuple with 2 elements
            tuple_element_1 = past_key[i, ...]  # Shape: (32, 3, 128)
            tuple_element_2 = past_value[i, ...]  # Shape: (32, 128, 3)
            result.append((tuple_element_1, tuple_element_2))

        # Now result will be a list of tuples, where each tuple contains two tensors of the desired shapes
        # You can convert it to a tuple if needed
        result_tuple = tuple(result)
        #past_key_values = past_key_values.permute([3, 0, 1, 2]).split(2)
        return result_tuple





class PromptEncoder(torch.nn.Module):

    def __init__(self, config, model):
        super().__init__()
        self.model = model

        #self.projection = torch.nn.Linear(config.hidden_size, config.hidden_size)

        if config.pooling==True:
            self.W = torch.nn.Linear(config.hidden_size, 1)
            self.projection = torch.nn.Linear(config.pre_seq_len, config.pre_seq_len * config.hidden_size)
        else:
            self.W =None
            self.projection = torch.nn.Linear(config.hidden_size, config.hidden_size)

        self.config = config


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        batch_size=None,
        output_attentions=None,
        output_hidden_states=None,
        past_key_values=None,
        return_dict=None,
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )
        #print('here', outputs[0].shape)

        #print('output', output.shape)
        if self.config.pooling==True:
            att_w = self.W(outputs[0]).squeeze(-1)
            projection = self.projection(att_w).reshape(-1, self.config.pre_seq_len, self.config.hidden_size)
        else:
            projection = self.projection(outputs[0])
        #projection = self.projection(outputs[0])
        #print('working', projection.shape)

        return projection

class PrefixForSequenceClassification(PreTrainedModel):
    def __init__(self, config: AutoConfig):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.transformer = AutoModel.from_pretrained(config._name_or_path)


        self.config.head_dim = self.config.hidden_size // self.config.n_head
        #self.embeddings = self.transformer.word_embeddings

        self.pre_seq_len=config.pre_seq_len

        for param in self.transformer.parameters():
            param.requires_grad = False

        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)


        self.p_encoder = PrefixEncoder(config, self.transformer )



        #self.prompt_encoder = nn.Embedding(config.pre_seq_len, config.hidden_size)
        # Initialize weights and apply final processing
        self.post_init()


    def get_prompt(self, prefix_ids, batch_size):
        #prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)

        past_key_values = self.p_encoder(input_ids=prefix_ids, batch_size=batch_size)
        return past_key_values


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        prefix_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size = input_ids.shape[0]

        if prefix_ids is None:
            raise ValueError(f"prefix_ids is required for label tuning")



        past_key_values = self.get_prompt(prefix_ids=prefix_ids, batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.transformer.device)
        if attention_mask is not None:
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)




        #print('past_key_values', type(past_key_values), type(past_key_values[0]))
        #print(past_key_values[0][0].shape, past_key_values[0][1].shape)


        transformer_outputs = self.transformer(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        logits = self.score(hidden_states)


        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )




class PromptForSequenceClassification(PreTrainedModel):
    def __init__(self, config: AutoConfig):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.transformer = AutoModel.from_pretrained(config._name_or_path)


        self.config.head_dim = self.config.hidden_size // self.config.n_head
        #self.embeddings = self.transformer.word_embeddings

        self.pre_seq_len=config.pre_seq_len

        for param in self.transformer.parameters():
            param.requires_grad = False

        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)


        self.p_encoder = PromptEncoder(config, self.transformer )



        #self.prompt_encoder = nn.Embedding(config.pre_seq_len, config.hidden_size)
        # Initialize weights and apply final processing
        self.post_init()





    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        prefix_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size = input_ids.shape[0]

        if prefix_ids is None:
            raise ValueError(f"prefix_ids is required for label tuning")



        raw_embedding = self.transformer.word_embeddings(input_ids)

        prompts = self.p_encoder(prefix_ids)
        inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)
        input_ids=None

        prompt_attention_mask = torch.ones(prompts.shape[0], prompts.shape[1]).to(self.transformer.device)

        if attention_mask is not None:
            attention_mask = torch.cat((prompt_attention_mask, attention_mask), dim=1)




        outputs = self.transformer(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = self.score(outputs.last_hidden_state)
        logits = torch.mean(logits, dim=1)


        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



