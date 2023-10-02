from BertConfig import GraphBertConfig
from NodeClustering import NodeClustering
from preprocessing__main import pre_processing
from Bert import Bert
from NodeConstruction import NodeConstruction

dataset_name = 'cora'
num_class = 7
num_features = 1433
num_nodes = 2708

lr = 0.001
k = 7
max_epoch = 50
x_size = num_features
hidden_size = 32
intermediate_size = 32
num_attention_heads = 1
num_hidden_layers = 1
y_class = num_class
graph_size = num_nodes
residual_type = 'graph_raw'

def clustering():
    dataset_name = 'cora'
    print('WL, dataset: ' + dataset_name)



    print('************ Start ************')
    print('GrapBert, dataset: ' + dataset_name + ', Pre-training, Node Attribute Reconstruction.')
    # ---- objection initialization setction ---------------
    data = pre_processing(dataset_name, k, True)
    
    bert_config = GraphBertConfig(
        residual_type = residual_type,
        x_size = x_size,
        y_size = y_class,
        k = k,
        hidden_size = hidden_size,
        num_hidden_layer = num_hidden_layers,
        num_attention_heads = num_attention_heads,
        intermediate_size = intermediate_size
    )
    bert = Bert(bert_config)
    construction_method = NodeConstruction(bert_config, bert)
    construction_method.run(data, lr = lr, weight_decay = 5e-4, max_epoch = max_epoch)
    
    clustering_method = NodeClustering(bert_config, bert)
    clustering_method.run(data, max_epoch, num_class)
    
if __name__ == "__main__":
    clustering()