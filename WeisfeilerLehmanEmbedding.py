import hashlib

class WLNodeColoring():
    def __init__(self,node_list,link_list):
        super().__init__()
        self.node_list = node_list
        self.link_list = link_list
        self.node_color_dict = {}
        self.node_neighbor_dict = {}
        self.max_iter = 2
    
    def create_neighbours(self):
        for node in self.node_list:
            self.node_color_dict[node] = 1
            self.node_neighbor_dict[node] = {}

        for pair in self.link_list:
            u1, u2 = pair
            if u1 not in self.node_neighbor_dict:
                self.node_neighbor_dict[u1] = {}
            if u2 not in self.node_neighbor_dict:
                self.node_neighbor_dict[u2] = {}
            self.node_neighbor_dict[u1][u2] = 1
            self.node_neighbor_dict[u2][u1] = 1
    
    def WL_recursion(self):
        iteration_count = 1
        while True:
            new_color_dict = {}
            for node in self.node_list:
                # print(node)
                neighbors = self.node_neighbor_dict[node]
                neighbor_color_list = [self.node_color_dict[neb] for neb in neighbors]
                color_string_list = [str(self.node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
                color_string = "_".join(color_string_list)
                hash_object = hashlib.md5(color_string.encode())
                hashing = hash_object.hexdigest()
                new_color_dict[node] = hashing
            color_index_dict = {k: v+1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
            for node in new_color_dict:
                new_color_dict[node] = color_index_dict[new_color_dict[node]]
            if self.node_color_dict == new_color_dict or iteration_count == self.max_iter:
                return
            else:
                self.node_color_dict = new_color_dict
            iteration_count += 1

    def run(self):
        self.create_neighbours()
        self.WL_recursion()
        # print(self.node_color_dict)
        return self.node_color_dict

    
