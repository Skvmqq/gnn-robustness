import igraph as ig
import torch

class GraphFeatures:
    def __init__(self, data, un_directed=True):
        self.data = data
        self.un_directed = un_directed

        edges = self.data.edge_index.t().cpu().tolist()

        num_nodes = self.data.num_nodes

        self.G = ig.Graph(
            n=num_nodes,
            edges=edges,
            directed=not self.un_directed
        )

    def pagerank(self):
        return self.G.pagerank()

    def closeness_centrality(self):
        return self.G.closeness()

    def eigenvector_centrality(self):
        return self.G.eigenvector_centrality()

    def betweenness(self, normalized=True, cutoff=None):
        bc = self.G.betweenness(cutoff=cutoff)

        if normalized:
            n = self.G.vcount()
            if n > 2:
                if self.un_directed:
                    scale = 2 / ((n - 1) * (n - 2))
                else:
                    scale = 1 / ((n - 1) * (n - 2))
                bc = [v * scale for v in bc]

        return bc

    def degree_centrality(self):
        n = self.G.vcount()

        if self.un_directed:
            return [d / (n - 1) for d in self.G.degree()]
        else:
            return (
                self.G.degree(mode="in"),
                self.G.degree(mode="out")
            )

    def to_tensor(self, values):
        return torch.tensor(values, dtype=torch.float).view(-1, 1)