


class GraphGPUCache(object):
    def __init__(self, graph, rank, cache_size):
        self.rank = rank
        self.cache_size = cache_size # number of max cache nodes

        # gpu cache
        self.local_cache = {} # key: node id, value: node feature

    def cache_feats_in_gpu(self, keys, values):
        pass

    def fetch_feats_from_cpu(self, keys):
        pass
    
    def fetch_feats_from_remote(self, keys):
        pass

    def construct_mfgs(self, keys):
        pass


