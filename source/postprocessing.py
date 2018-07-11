# coding=utf-8
from typing import Dict

from scipy.cluster.hierarchy import ClusterNode

from clustering import HierarchicalCluster


class ClusterEncoder:
    @classmethod
    def to_dict(cls, cluster: HierarchicalCluster) -> Dict:
        tree = cluster.to_tree()
        frequencies = cluster.term_frequency
        names = cluster.labels

        d = {
            "words": cls.words_to_dict(tree, names, frequencies),
            "hierarchical_cluster": cls.tree_to_dict(tree)
        }

        return d

    @classmethod
    def words_to_dict(cls, tree, names, frequencies):
        words = []

        # iterate leaves
        for leaf in tree.pre_order(lambda x: x):
            words.append({
                "name": names[leaf.id],
                "weight:": frequencies[leaf.id],
                "word_id": leaf.id
            })

        return words

    @classmethod
    def tree_to_dict(cls, root: ClusterNode):
        def dictify(node: ClusterNode):
            if node.is_leaf():
                return {
                    "word_id": node.get_id(),
                    "leaf": True
                }

            else:
                return {
                    "leaf": False,
                    "left_child": dictify(node.get_left()),
                    "right_child": dictify(node.get_right()),
                    "cluster_size": node.get_count(),
                    "children_distance": node.dist
                }

        return dictify(root)




