from collections import Counter
import pickle
import faiss

class FaissSearch():
    def __init__(self):
        self.dataset = None
        self.labels = None
        self.classes = None
        self.index = None

    def flat_search(self, dataset, labels, classes):
        self.dataset = dataset
        self.labels = labels
        self.classes = classes
        num_vect, num_dim = self.dataset.shape

        self.index = faiss.IndexFlatL2(num_dim)
        if not self.index.is_trained:
            self.index.train(self.dataset)

        self.index.add(self.dataset)

    def lsh_search(self, dataset, labels, classes):
        self.dataset = dataset
        self.labels = labels
        self.classes = classes
        num_vect, num_dim = dataset.shape
        nbits = num_dim * 8

        self.index = faiss.IndexLSH(num_dim, nbits)
        if not self.index.is_trained:
            self.index.train(self.dataset)

        self.index.add(self.dataset)


    def hnsw_search(self, dataset, labels, classes):
        self.dataset = dataset
        self.labels = labels
        self.classes = classes
        num_vect, num_dim = dataset.shape
        M = 16
        ef_search = 32
        ef_construction = 64

        self.index = faiss.IndexHNSWFlat(num_dim, M)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search

        if not self.index.is_trained:
            self.index.train(self.dataset)

        self.index.add(self.dataset)

    def search(self, query, k):
        D, I = self.index.search(query, k)
        sel_labels = [self.labels[i] for i in I[0]]
        _c = Counter(sel_labels)
        cls_idx = _c.most_common(1)[0][0]
        cls = self.classes[cls_idx]
        return cls, D, I

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            instance = pickle.load(f)
        return instance

