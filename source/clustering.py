# coding=utf-8

from scipy.cluster.hierarchy import ClusterNode, to_tree, linkage, dendrogram

from distance_matrix import DistanceVectorComputer
from preprocessing import Terms
from primitives import LinkageMatrix


class HierarchicalCluster:
    def __init__(self, labels, term_frequency, linkage_matrix: LinkageMatrix):
        self.labels = labels
        self.term_frequency = term_frequency
        self.linkage_matrix = linkage_matrix

    def to_tree(self) -> ClusterNode:
        return to_tree(self.linkage_matrix)

    def plot(self, tittle="Dendrogram", **kwargs):
        from matplotlib import pyplot as plt
        plt.figure(figsize=(20, 10))
        plt.title(tittle)
        dendrogram(self.linkage_matrix, labels=self.labels, leaf_rotation=90, **kwargs)
        plt.show()


class HierarchicalClustering:
    def __init__(self, distance_metric: DistanceVectorComputer):
        self.distance_metric = distance_metric

    def cluster(self, terms: Terms, method="ward") -> HierarchicalCluster:
        distances = self.distance_metric.get_distance_vector(terms.term_occurrence())
        linkage_matrix = linkage(distances, method=method)
        return HierarchicalCluster(terms.term_labels(), terms.term_frequency(), linkage_matrix)
