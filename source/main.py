# coding=utf-8
import sys
from typing import Dict

import json

from clustering import HierarchicalClustering
from postprocessing import ClusterEncoder
from preprocessing import DataDecoder
from distance_matrix import DistanceVectorComputer, bitwise_similarity


def make_hierarchical_term_cluster(term_sets: Dict) -> Dict:
    terms = DataDecoder.from_term_sets(term_sets)

    distance = DistanceVectorComputer(bitwise_similarity)
    clustering = HierarchicalClustering(distance)
    clusters = clustering.cluster(terms, method="average")

    return ClusterEncoder.to_dict(clusters)


if __name__ == '__main__':
    input_file_name, output_file_name = sys.argv[1], sys.argv[2]

    input_file = open(input_file_name)
    term_sets = json.load(input_file)

    term_cluster = make_hierarchical_term_cluster(term_sets)

    output_file = open(output_file_name, "w")
    json.dump(term_cluster, output_file, indent="\t")
