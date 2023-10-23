import faiss
import numpy as np

class faissIndex:
    def __init__(self, dim, n_centroids, metric):
        self.dim = dim
        self.n_centriods = n_centroids
        assert metric in ('INNER_PRODUCT', 'L2'), "Input metric not in 'INNER_PRODUCT' or 'L2'"
        self.metric = faiss.METRIC_INNER_PRODUCT if metric == 'INNER_PRODUCT' else faiss.METRIC_L2
        self._build_index()
        return
    
    def _build_index(self):
        self._quantizer = faiss.IndexFlatL2(self.dim)
        self.index = faiss.IndexIVFFlat(self._quantizer, self.dim, self.n_centriods, self.metric)
        self.is_trained = self.index.is_trained
        self.n_samples = 0 # 查询向量池中的向量个数
        self.items = np.array([]) # 向量池中向量对应的item，数量应与self.n_samples保持一致，即向量与item一一对应
        return True
    
    def reset_index(self, dim, n_centroids, metric):
        self.dim = dim
        self.n_centriods = n_centroids
        assert metric in ('INNER_PRODUCT', 'L2'), "Input metric not in 'INNER_PRODUCT' or 'L2'"
        self.metric = faiss.METRIC_INNER_PRODUCT if metric == 'INNER_PRODUCT' else faiss.METRIC_L2
        self._build_index()
        return
    
    def train(self, vectors_train):
        self.index.train(vectors_train)
        self.is_trained = self.index.is_trained
        return
    
    def add(self, vectors, items=None):
        if not items.empty: # 当有输入items时，验证之前的item和vector数量是否匹配，以及当前输入
            assert len(vectors) == len(items), "Length of vectors ({n_vectors}) and items ({n_items}) don't match, please check your input.".format(n_vectors=len(vectors), n_items=len(items))
            assert self.n_samples == len(self.items), "Amounts of added vectors and items don't match, cannot add more items."
            self.items = np.append(self.items, items.to_numpy())
        else:
            assert len(self.items) == 0, "There were items added previously, please added corresponding items in this batch."
        self.index.add(vectors)
        self.n_samples += len(vectors)
        return
    
    def search(self, query_vector, k, n_probe=1):
        assert query_vector.shape[1] == self.dim, "The dimension of query vector ({dim_vector}) doesn't match the training vector set ({dim_index})!".format(dim_vector=query_vector.shape[1], dim_index=self.dim)
        assert self.is_trained, "Faiss index is not trained, please train index first!"
        assert self.n_samples > 0, "Faiss index doesn't have any vector for query, please add vectors into index first!"
        self.index.nprobe = n_probe
        D, I = self.index.search(query_vector, k)
        return D, I
    # k = 30 # 对每条向量（每行）寻找最近k个物料
    # n_probe = 5 # 每次查询只查询最近邻n_probe个聚类
    def search_items(self, query_vector, k, n_probe=1):
        D, I = self.search(query_vector, k, n_probe)
        R = [self.items[i] for i in I]
        return R, D, I
