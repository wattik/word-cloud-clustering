from abc import ABC

import numpy as np
from scipy.spatial.distance import pdist

from primitives import BooleanMatrix, BooleanVector, NonNegativeFloat, Vector


class DistanceVectorComputer(ABC):
    def __init__(self, metric):
        self.metric = metric

    def get_distance_vector(self, dataset: BooleanMatrix) -> Vector:
        return pdist(dataset, metric=self.metric)


def common_occurrence(a: BooleanVector, b: BooleanVector) -> NonNegativeFloat:
    return 1 - np.logical_and(a, b).mean()


def bitwise_similarity(a: BooleanVector, b: BooleanVector) -> NonNegativeFloat:
    logical_xor = np.logical_xor(a, b).sum()
    logical_or = np.logical_or(a, b).sum()
    return logical_xor / logical_or
