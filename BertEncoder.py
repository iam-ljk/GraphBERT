import torch
import torch.nn as nn
from BertLayer import BertLayer

class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layer)])
        
    def forward(self, 
                hidden_states, 
                attention_mask = None, 
                head_mask = None, 
                encoder_hidden_states = None, 
                encoder_attention_mask = None,
                residual_h = None
                ) :
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer) :
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask)
            hidden_states = layer_outputs[0]
            
            #add residual
            if residual_h is not None:
                for index in range(hidden_states.size()[1]):
                    hidden_states[:,index,:]+=residual_h
                    
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1], )
                
        #Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states , )
        
        outputs = (hidden_states , )
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states, )
        if self.output_attentions:
            outputs = outputs + (all_attentions ,)
        return outputs