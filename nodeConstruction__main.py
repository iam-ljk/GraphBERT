from BertConfig import GraphBertConfig
from NodeConstruction import NodeConstruction
from preprocessing__main import pre_processing
from Bert import Bert
import matplotlib.pyplot as plt
import os

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



def construction():
    dataset_name = 'cora'
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
    loss_list = construction_method.run(data, lr = lr, weight_decay = 5e-4, max_epoch = max_epoch)
    return loss_list

if __name__ == "__main__":
    loss_list = construction()
    plt.plot(range(1, len(loss_list) + 1), loss_list, marker='o', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    result_folder = 'result'
    plot_filename = os.path.join(result_folder, 'loss_plot.png')
    plt.savefig(plot_filename)
    print(f'Loss plot saved to {plot_filename}')

    plt.show()

