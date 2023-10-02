import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from Bert import Bert
from transformers.models.bert.modeling_bert import BertPreTrainedModel

class NodeConstruction(BertPreTrainedModel):
    def __init__(self, config, bert):
        super().__init__(config)
        self.config = config
        self.bert = bert
        self.reconstruction_layer = nn.Linear(config.hidden_size, config.x_size)
        self.init_weights()
        self.learning_record_dict = []

    def forward(self, raw_features, wl_role_ids, init_pos_ids, hop_dis_ids):
        output = self.bert(raw_features, wl_role_ids, init_pos_ids, hop_dis_ids)
        #sum of all neighbour and own embeddings
        print(output[0].shape)
        sequence_output = 0
        for i in range(self.config.k+1):
            sequence_output+=output[0][:,i,:]
        sequence_output/=float(self.config.k+1)
        x_hat = self.reconstruction_layer(sequence_output)

        return x_hat

    def trainModel(self, data, lr, weight_decay, max_epoch, patience=5):
        t_begin = time.time()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        best_loss = float('inf')
        consecutive_epochs_without_improvement = 0

        for epoch in range(max_epoch):
            t_epoch_begin = time.time()
            self.train()
            optimizer.zero_grad()

            output = self.forward(
                data['raw_embeddings'],
                data['wl_embedding'],
                data['init_embeddings'],
                data['hop_embeddings']
            )

            loss_train = F.mse_loss(output, data['X'])
            loss_train.backward()
            optimizer.step()

            self.learning_record_dict.append(loss_train.item())
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'time: {:.4f}s'.format(time.time() - t_epoch_begin))

            # Check for early stopping
            # if loss_train.item() < best_loss:
            #     best_loss = loss_train.item()
            #     consecutive_epochs_without_improvement = 0
            # else:
            #     consecutive_epochs_without_improvement += 1

            # if consecutive_epochs_without_improvement >= patience:
            #     print(f'Loss has not improved for {patience} consecutive epochs. Stopping training.')
            #     break

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_begin))
        return self.learning_record_dict

    def run(self, data, lr = 0.001, weight_decay = 5e-4, max_epoch = 200):
        return self.trainModel(data,lr,weight_decay,max_epoch)
        