import torch
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertPooler
from BertEmbedding import BertEmbedding
from BertEncoder import BertEncoder

class Bert(BertPreTrainedModel):
    def __init__(self,config) :
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbedding(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.init_weights()

    def forward(self, raw_features, wl_role_ids, init_pos_ids, hop_dis_ids, head_mask = None, residual_h = None):
        if head_mask is None:
            head_mask = [None] * self.config.num_hidden_layer

        embedding_output = self.embeddings(raw_features=raw_features, wl_role_ids=wl_role_ids, init_pos_ids=init_pos_ids, hop_dis_ids=hop_dis_ids)
        # print("Embedding size",embedding_output.shape)
        encoder_outputs = self.encoder(embedding_output, head_mask=head_mask, residual_h=residual_h)
        sequence_output = encoder_outputs[0]
        # print("Sequence output", sequence_output.shape)
        pooled_output = self.pooler(sequence_output)
        # print("Pooled output size: ",pooled_output.shape)
        outputs = (sequence_output, pooled_output,) 
        return outputs
