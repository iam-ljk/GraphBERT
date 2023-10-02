import math
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertAttention
from transformers.models.bert.modeling_bert import BertIntermediate
from transformers.models.bert.modeling_bert import BertOutput

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate =  BertIntermediate(config)
        self.output = BertOutput(config)
        
    def forward(
        self,
        hidden_states,
        attention_mask = None,
        head_mask = None,
        encoder_hidden_states = None,
        encoder_attention_mask = None
        ) :
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        
        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = cross_attention_outputs[1:]
        
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output, ) + outputs
        return outputs