class GraphBatching():
    def __init__(self, intimacy_matrix, index_id_map, k = 5):
        super().__init__()
        self.intimacy_matrix = intimacy_matrix
        self.index_id_map = index_id_map
        self.k = k
    
    def find_top_k_intimacy(self):
        user_top_k_neighbor_intimacy_dict = {}
        for node_index in self.index_id_map:
            node_id = self.index_id_map[node_index]
            s = self.intimacy_matrix[node_index]
            s[node_index] = -1000.0
            top_k_neighbor_index = s.argsort()[-self.k:][::-1]
            user_top_k_neighbor_intimacy_dict[node_id] = []
            for neighbor_index in top_k_neighbor_index:
                neighbor_id = self.index_id_map[neighbor_index]
                user_top_k_neighbor_intimacy_dict[node_id].append((neighbor_id, s[neighbor_index]))
        return user_top_k_neighbor_intimacy_dict
    
    def run(self):
        return self.find_top_k_intimacy()