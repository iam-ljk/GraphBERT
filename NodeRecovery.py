import torch
import torch.optim as optim
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from Bert import Bert

import time
from sklearn.metrics import accuracy_score

class NodeRecovery(BertPreTrainedModel) :
    def __init__(self, config, bert):
        super().__init__(config=config)
        self.bert = bert
        self.learning_record_dict = []
        self.config = config
        self.init_weights()
    
    def forward(self, raw_features, wl_role_ids, init_pos_ids, hop_dis_ids, idx = None):
        output = self.bert(raw_features, wl_role_ids, init_pos_ids, hop_dis_ids)
        sequence_output = 0
        for i in range(self.config.k+1):
            sequence_output+=output[0][:,i,:]
        sequence_output/=float(self.config.k+1)
        
        x_hat = sequence_output
        x_norm = torch.norm(x_hat, p=2, dim=1)
        nume = torch.mm(x_hat, x_hat.t())
        deno = torch.ger(x_norm, x_norm)
        cosine_similarity = nume / deno
        return cosine_similarity
    
    def trainModel(self,data, lr,weight_decay,max_epoch):
        t_begin = time.time()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        for epoch in range(max_epoch):
            t_epoch_begin = time.time()

            self.train()
            optimizer.zero_grad()

            output = self.forward(data['raw_embeddings'], data['wl_embedding'], data['init_embeddings'], data['hop_embeddings'])
            row_num, col_num = output.size()
            loss_train = torch.sum((output - data['A'].to_dense()) ** 2)/(row_num*col_num)

            loss_train.backward()
            optimizer.step()
            self.learning_record_dict.append(loss_train.item())
            print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(loss_train.item()))
        
        print("Optimization Finished!")
        
    def run(self, data,lr = 0.001, weight_decay = 5e-4, max_epoch = 200):
        self.trainModel(data,lr,weight_decay,max_epoch)
        return self.learning_record_dict