from NodeRecovery import NodeRecovery
from BertConfig import GraphBertConfig
from NodeClassification import NodeClassification
from preprocessing__main import pre_processing
from Bert import Bert
from NodeConstruction import NodeConstruction
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
num_attention_heads = 2
num_hidden_layers = 2
y_class = num_class
graph_size = num_nodes
residual_type = 'graph_raw'

def classification():
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
    bert1 = Bert(bert_config)
    construction_method = NodeConstruction(bert_config, bert)
    construction_method.run(data, lr = lr, weight_decay = 5e-4, max_epoch = 200)

    recovery_method = NodeRecovery(bert_config, bert)
    recovery_method.run(data, lr = lr, weight_decay = 5e-4, max_epoch = 200)
    
    classification_method = NodeClassification(bert_config, bert)
    loss_list1 = classification_method.run(data, lr = lr, weight_decay = 5e-4, max_epoch = 150)
    
    classification_method1 = NodeClassification(bert_config, bert1)
    loss_list2 = classification_method1.run(data, lr = lr, weight_decay = 5e-4, max_epoch = 150)
    
    plt.plot(range(len(loss_list1)), loss_list1, label='loss_list1')
    plt.plot(range(len(loss_list2)), loss_list2, label='loss_list2')
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.legend()

    # Create a result folder if it doesn't exist
    result_folder = 'result'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Save the plot to the result folder
    plot_filename = os.path.join(result_folder, 'accuracy_plot.png')
    plt.savefig(plot_filename)
    print(f'Loss plot saved to {plot_filename}')

    # Display the plot
    plt.show()
    
if __name__ == "__main__":
    classification()
