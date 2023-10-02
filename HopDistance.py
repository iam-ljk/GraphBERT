import networkx as nx
import pickle

class HopDistance():
    def __init__(self, node_list, link_list,dataset_name, k) :
        super().__init__()
        self.node_list = node_list
        self.link_list = link_list
        self.dataset_name = dataset_name
        self.k = k

    def hopDistance(self):
        G = nx.Graph()
        G.add_nodes_from(self.node_list)
        G.add_edges_from(self.link_list)

        #loading batch-subgraphs
        f = open('./result/Batch/' + self.dataset_name + '_' + str(self.k), 'rb')
        batch_dict = pickle.load(f)
        f.close()

        hop_dict = {}
        for node in batch_dict:
            if node not in hop_dict: hop_dict[node] = {}
            for neighbor, score in batch_dict[node]:
                try:
                    hop = nx.shortest_path_length(G, source=node, target=neighbor)
                except:
                    hop = 99
                hop_dict[node][neighbor] = hop
        return hop_dict

    def run(self):
        return self.hopDistance()
