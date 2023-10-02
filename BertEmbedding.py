import torch
import torch.nn as nn
import numpy as np

class BertEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.raw_feature_embeddings = nn.Linear(in_features= config.x_size, out_features= config.hidden_size)
        self.wl_role_embeddings = nn.Embedding(config.max_wl_role_index, config.hidden_size)
        self.inti_pos_embeddings = nn.Embedding(config.max_init_pos_index, config.hidden_size)
        self.hop_dis_embeddings = nn.Embedding(config.max_hop_dis_index, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden_size = config.hidden_size

    def positional_encode(self, wl_role_ids, dh=6, L=5):
        num_rows = len(wl_role_ids)
        positional_encoding = np.zeros((num_rows, L, dh))

        for row_index, row in enumerate(wl_role_ids):
            for l in range(L):
                for d in range(dh):
                    if d % 2 == 0:
                        positional_encoding[row_index, l, d] = np.sin(row[l] / (10000 ** (2 * l / dh)))
                    else:
                        positional_encoding[row_index, l, d] = np.cos(row[l] / (10000 ** (2 * l / dh)))
        positional_encoding = torch.LongTensor(positional_encoding)

        return positional_encoding


    def forward(self,
                raw_features = None,
                wl_role_ids = None,
                init_pos_ids = None,
                hop_dis_ids = None
                ) :
        raw_features_embeds = self.raw_feature_embeddings(raw_features)
        # role_embeddings = self.positional_encode(wl_role_ids, self.hidden_size, wl_role_ids.size(-1))
        # position_embeddings = self.positional_encode(init_pos_ids, self.hidden_size, init_pos_ids.size(-1))
        # hop_embeddings = self.positional_encode(hop_dis_ids, self.hidden_size, hop_dis_ids.size(-1))
        role_embeddings = self.wl_role_embeddings(wl_role_ids)
        position_embeddings = self.inti_pos_embeddings(init_pos_ids)
        hop_embeddings = self.hop_dis_embeddings(hop_dis_ids)
        
        embeddings = raw_features_embeds + role_embeddings + position_embeddings + hop_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings