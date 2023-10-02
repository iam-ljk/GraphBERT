import numpy as np
import torch
from DatasetPreprocessing import DatasetPreprocessing
from WeisfeilerLehmanEmbedding import WLNodeColoring
from GraphBatching import GraphBatching
import pickle
from HopDistance import HopDistance

def save(result_destination_folder_path, result_destination_file_name, data):
    f = open(result_destination_folder_path + result_destination_file_name, 'wb')
    pickle.dump(data, f)
    f.close()


def pre_processing(dataset_name, k, load_all_tag) :
    dataset_source_folder_path = './data/' + dataset_name + '/'
    wl_folder_path = './result/wl/' 
    batch_folder_path = './result/Batch/'
    hop_folder_path = './result/Hop/'

    print("Start Data Preprocessing")
    preprocessing_obj = DatasetPreprocessing(
        dataset_source_folder_path = dataset_source_folder_path,
        dataset_name = dataset_name,
        k = k,
        c = 0.15,
        data = None,
        load_all_tag = load_all_tag,
        compute_s = True,
        batch_size = None
        )
    data = preprocessing_obj.run()

    print("Start WL Preprocessing")
    #Find WL for each node => WL(vi) = N
    wl_node_obj = WLNodeColoring(
        node_list = data['idx'],
        link_list = data['edges']
        )
    colours = wl_node_obj.run()
    save(wl_folder_path, dataset_name, colours)

    max_key = max(colours, key=lambda k: colours[k])
    max_value = colours[max_key]

    print("Maximum Key:", max_key)
    print("Maximum Value:", max_value)

    print("Start Batching")
    #Divide graph into subgraph
    batching_obj = GraphBatching(
        intimacy_matrix = data['S'],
        index_id_map = data['index_id_map'],
        k = k
        )
    #will get node_id and intimacy values
    batchings = batching_obj.run()
    save(batch_folder_path, dataset_name + '_' + str(k), batchings)


    print("Stack Hop Counting")
    #Hop counting 
    hop_obj = HopDistance(
        node_list = data['idx'],
        link_list = data['edges'],
        dataset_name = dataset_name,
        k = k
        )
    #This will be n*n matries
    hopings = hop_obj.run()
    save(hop_folder_path, dataset_name + '_' + str(k), hopings)
    return data

if __name__ == "__main__":
    pre_processing('cora',7,load_all_tag = False)

