# coding=utf-8
from typing import NewType

import numpy as np

Vector = NewType("Vector", np.ndarray)
Matrix = NewType("Matrix", np.ndarray)

BooleanMatrix = NewType("BooleanMatrix", np.ndarray)
BooleanVector = NewType("BooleanVector", np.ndarray)

NonNegativeFloat = NewType("NonNegativeFloat", np.float)
LinkageMatrix = NewType("LinkageMatrix", np.ndarray)