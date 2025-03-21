'''
graph.py 包含了 NeighborFinder 类，用于构建和查询图的邻居信息。

邻接列表初始化：将原始的邻接列表（adj_list）转换为多个数组（cas_l, node_idx_l, node_ts_l, edge_idx_l, off_set_l），便于后续的高效查询。
时间约束的邻居查找：find_before 方法在给定时间截断点（cut_time）之前，找到特定节点（src_idx）在给定CAS ID（cas_id）下的邻居节点，确保只考虑时间上发生在截断点之前的传播行为。
获取时序邻居：get_temporal_neighbor 方法批量处理多个查询，获取每个源节点在指定时间之前的邻居信息，并根据是否进行均匀采样（uniform）选择邻居节点。
k-hop邻居采样：find_k_hop 方法通过迭代调用 get_temporal_neighbor，逐层获取k-hop邻居，用于多层图神经网络的输入。
'''
import numpy as np
import torch

class NeighborFinder:
    def __init__(self, adj_list, uniform=False):
        """
        Params
        ------
        cas_dict: Dict{'cas_id': [start, end]}
        node_idx_l: List[int]
        node_ts_l: List[int]
        off_set_l: List[int], such that node_idx_l[off_set_l[i]:off_set_l[i + 1]] = adjacent_list[i]
        """ 
       
        cas_l, node_idx_l, node_ts_l, edge_idx_l, off_set_l = self.init_off_set(adj_list)
        self.cas_l = cas_l
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = edge_idx_l
        
        self.off_set_l = off_set_l
        
        self.uniform = uniform
        
    def init_off_set(self, adj_list):
        """
        Params
        ------
        adj_list: List[List[int]]
        
        """
        cas_l = []
        n_idx_l = []
        n_ts_l = []
        e_idx_l = []
        off_set_l = [0]
        # i is idx
        for i in range(len(adj_list)):
            curr = adj_list[i]
            #curr = sorted(curr, key=lambda x: x[2])
            cas_l.extend([x[0] for x in curr])
            n_idx_l.extend([x[1] for x in curr])
            e_idx_l.extend([x[2] for x in curr])
            n_ts_l.extend([x[3] for x in curr])
            off_set_l.append(len(n_idx_l))
        cas_l = np.array(cas_l)
        n_idx_l = np.array(n_idx_l)
        n_ts_l = np.array(n_ts_l)
        e_idx_l = np.array(e_idx_l)
        off_set_l = np.array(off_set_l)

        assert(len(n_idx_l) == len(n_ts_l))
        assert(off_set_l[-1] == len(n_ts_l))
        
        return cas_l, n_idx_l, n_ts_l, e_idx_l, off_set_l
        
    def find_before(self, cas_id, src_idx, cut_time):
        """
    
        Params
        ------
        src_idx: int
        cut_time: float
        """
        cas_l = self.cas_l
        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        edge_idx_l = self.edge_idx_l
        off_set_l = self.off_set_l
        neighbors_cas = cas_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        cas_idx = np.where(neighbors_cas == cas_id)
        neighbors_idx = node_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]][cas_idx]
        neighbors_ts = node_ts_l[off_set_l[src_idx]:off_set_l[src_idx + 1]][cas_idx]
        neighbors_e_idx = edge_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]][cas_idx]
        
        if len(neighbors_idx) == 0 or len(neighbors_ts) == 0:
            return neighbors_idx, neighbors_ts, neighbors_e_idx

        left = 0
        right = len(neighbors_idx) - 1
        
        while left + 1 < right:
            mid = (left + right) // 2
            curr_t = neighbors_ts[mid]
            if curr_t < cut_time:
                left = mid
            else:
                right = mid
                
        if neighbors_ts[right] <= cut_time:
            return neighbors_idx[:right+1], neighbors_e_idx[:right+1], neighbors_ts[:right+1]
        else:
            return neighbors_idx[:left+1], neighbors_e_idx[:left+1], neighbors_ts[:left+1]


    def get_temporal_neighbor(self, cas_l, src_idx_l, cut_time_l, num_neighbors=20):
        """
        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        assert(len(src_idx_l) == len(cut_time_l))
        out_ngh_node_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.float32)
        out_ngh_eidx_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        for i, (cas_id, src_idx, cut_time) in enumerate(zip(cas_l, src_idx_l, cut_time_l)):
            ngh_idx, ngh_eidx, ngh_ts = self.find_before(cas_id, src_idx, cut_time)
            #print(len(ngh_idx))
            if len(ngh_idx) > 0:
                if self.uniform:
                    sampled_idx = np.random.randint(0, len(ngh_idx), num_neighbors)
                    
                    out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
                    out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
                    out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx]
                    
                    # resort based on time
                    pos = out_ngh_t_batch[i, :].argsort()
                    out_ngh_node_batch[i, :] = out_ngh_node_batch[i, :][pos]
                    out_ngh_t_batch[i, :] = out_ngh_t_batch[i, :][pos]
                    out_ngh_eidx_batch[i, :] = out_ngh_eidx_batch[i, :][pos]
                else:
                    ngh_ts = ngh_ts[:num_neighbors]
                    ngh_idx = ngh_idx[:num_neighbors]
                    ngh_eidx = ngh_eidx[:num_neighbors]
                    
                    assert(len(ngh_idx) <= num_neighbors)
                    assert(len(ngh_ts) <= num_neighbors)
                    assert(len(ngh_eidx) <= num_neighbors)
                    
                    out_ngh_node_batch[i, num_neighbors - len(ngh_idx):] = ngh_idx
                    out_ngh_t_batch[i, num_neighbors - len(ngh_ts):] = ngh_ts
                    out_ngh_eidx_batch[i,  num_neighbors - len(ngh_eidx):] = ngh_eidx
                    
        return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch

    def find_k_hop(self, k, src_idx_l, cut_time_l, num_neighbors=20):
        """Sampling the k-hop sub graph
        """
        x, y, z = self.get_temporal_neighbor(src_idx_l, cut_time_l, num_neighbors)
        node_records = [x]
        eidx_records = [y]
        t_records = [z]
        for _ in range(k -1):
            ngn_node_est, ngh_t_est = node_records[-1], t_records[-1] # [N, *([num_neighbors] * (k - 1))]
            orig_shape = ngn_node_est.shape
            ngn_node_est = ngn_node_est.flatten()
            ngn_t_est = ngh_t_est.flatten()
            out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch = self.get_temporal_neighbor(ngn_node_est, ngn_t_est, num_neighbors)
            out_ngh_node_batch = out_ngh_node_batch.reshape(*orig_shape, num_neighbors) # [N, *([num_neighbors] * k)]
            out_ngh_eidx_batch = out_ngh_eidx_batch.reshape(*orig_shape, num_neighbors)
            out_ngh_t_batch = out_ngh_t_batch.reshape(*orig_shape, num_neighbors)

            node_records.append(out_ngh_node_batch)
            eidx_records.append(out_ngh_eidx_batch)
            t_records.append(out_ngh_t_batch)
        return node_records, eidx_records, t_records

            

