__author__ = 'pva701'


class DummyWindow:

    def __init__(self, embedding_size):
        self.embedding_size = embedding_size

    def init_with_scope(self):
        pass

    def build_graph(self, embedded_vectors, n_words, dropout_keep_prob):
        return embedded_vectors

    def output_vector_size(self):
        return self.embedding_size