import torch

from transformers.models.bert.modeling_bert import BertPreTrainedModel
from Bert import Bert

import time
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

class NodeClustering(BertPreTrainedModel):
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
        return sequence_output
    
    def trainModel(self, data,max_epoch,cluster_number) :
        output = self.forward(data['raw_embeddings'], data['wl_embedding'], data['init_embeddings'], data['hop_embeddings'])
        print(output.shape)
        kmeans = KMeans(n_clusters=cluster_number, max_iter=max_epoch)
        clustering_result = kmeans.fit_predict(output.tolist())
        accuracy_data = {'true_y':data['y'], 'pred_y': clustering_result}
        
        train_pred_labels = accuracy_data['pred_y']
        train_true_labels = accuracy_data['true_y']
        train_accuracy = accuracy_score(train_true_labels, train_pred_labels)
        
        print('training_accuracy: {:.4f}'.format(train_accuracy))
        self.learning_record_dict.append(train_accuracy)
        return self.learning_record_dict 
    
    def run(self,data,max_epoch,cluster_number):
        self.trainModel(data,max_epoch, cluster_number)
        return self.learning_record_dict
        
        