import math
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform
from transformers.models.bert.modeling_bert import BertAttention
from transformers.configuration_utils import PretrainedConfig

class GraphBertConfig(PretrainedConfig) :
    def __init__(self,
                 residual_type = 'none',
                 x_size = 3000,
                 y_size = 7,
                 k = 5,
                 max_wl_role_index = 100,
                 max_hop_dis_index = 100,
                 max_inti_pos_index = 100,
                 hidden_size = 32,
                 num_hidden_layer = 1,
                 num_attention_heads = 1,
                 intermediate_size = 32,
                 hidden_act = 'gelu',
                 hidden_dropout_prob = 0.5,
                 attention_probs_dropout_prob = 0.3,
                 initializer_range = 0.02,
                 layer_norm_eps = 1e-12,
                 is_decoder = False,
                ):
        super().__init__()
        self.residual_type = residual_type
        self.x_size = x_size
        self.y_size = y_size
        self.k = k
        self.max_wl_role_index = max_wl_role_index
        self.max_hop_dis_index = max_hop_dis_index
        self.max_init_pos_index = max_inti_pos_index
        self.hidden_size = hidden_size
        self.num_hidden_layer = num_hidden_layer
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.is_decoder = is_decoder







        
