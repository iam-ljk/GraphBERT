import torch
import torch.nn.functional as F
import torch.optim as optim

import time
import numpy as np
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from Bert import Bert
from sklearn.metrics import accuracy_score


class NodeClassification(BertPreTrainedModel):
    def __init__(self, config, bert):
        super().__init__(config=config)
        self.config = config
        self.res_h = torch.nn.Linear(config.x_size, config.hidden_size)
        self.res_y = torch.nn.Linear(config.x_size, config.y_size)
        self.cls_y = torch.nn.Linear(config.hidden_size, config.y_size)
        self.bert = bert
        self.init_weights()
        self.learning_record_dict = []
    
    def forward(self, raw_features, wl_role_ids, init_pos_ids, hop_dis_ids, idx = None):
        output = self.bert(raw_features[idx], wl_role_ids[idx], init_pos_ids[idx], hop_dis_ids[idx])
        sequence_output = 0
        for i in range(self.config.k+1):
            sequence_output+=output[0][:,i,:]
        sequence_output/=float(self.config.k+1)
        
        labels = self.cls_y(sequence_output)
        return F.log_softmax(labels, dim=1)
    
    def trainModel(self, data, lr, weight_decay, max_epoch, patience=7):
        t_begin = time.time()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        best_val_accuracy = 0.0
        consecutive_epochs_without_improvement = 0

        for epoch in range(max_epoch):
            t_epoch_begin = time.time()

            self.train()
            optimizer.zero_grad()

            output = self.forward(data['raw_embeddings'], data['wl_embedding'], data['init_embeddings'], data['hop_embeddings'], data['idx_train'])
            loss_train = F.cross_entropy(output, data['y'][data['idx_train']])
            accuracy_data = {'true_y': data['y'][data['idx_train']], 'pred_y': output.max(1)[1]}
            
            train_pred_labels = accuracy_data['pred_y'].cpu().numpy()
            train_true_labels = accuracy_data['true_y'].cpu().numpy()
            train_accuracy = accuracy_score(train_true_labels, train_pred_labels)
    
            loss_train.backward()
            optimizer.step()
            
            self.eval()
            output = self.forward(data['raw_embeddings'], data['wl_embedding'], data['init_embeddings'], data['hop_embeddings'], data['idx_val'])

            loss_val = F.cross_entropy(output, data['y'][data['idx_val']])
            val_data = {'true_y': data['y'][data['idx_val']],
                             'pred_y': output.max(1)[1]}
            
            val_pred_labels = val_data['pred_y'].cpu().numpy()
            val_true_labels = val_data['true_y'].cpu().numpy()
            val_accuracy = accuracy_score(val_true_labels, val_pred_labels)
            
            self.learning_record_dict.append(val_accuracy)
            
            print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'loss_val: {:.4f}'.format(loss_val.item()),
                      'training_accuracy: {:.4f}'.format(train_accuracy),
                      'val_accuracy: {:.4f}'.format(val_accuracy),
                      'time: {:.4f}s'.format(time.time() - t_epoch_begin))
            
            # Check for early stopping
            # if val_accuracy > best_val_accuracy:
            #     best_val_accuracy = val_accuracy
            #     consecutive_epochs_without_improvement = 0
            # else:
            #     consecutive_epochs_without_improvement += 1

            # if consecutive_epochs_without_improvement >= patience:
            #     print(f'Validation accuracy has not improved for {patience} consecutive epochs. Stopping training.')
            #     break
        
        print("Optimization Finished!")
        return self.learning_record_dict
        
    def run(self, data,lr = 0.001, weight_decay = 5e-4, max_epoch = 200):
        self.trainModel(data,lr,weight_decay,max_epoch)
        return self.learning_record_dict