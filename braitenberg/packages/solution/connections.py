from typing import Tuple

import numpy as np


def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    # TODO: write your function instead of this one
    res = np.zeros(shape=shape, dtype="float32")
    for i in range(shape[0]):
        for j in range(shape[1]):
            res[i, j] = shape[0]*shape[1] - (i+1) * (j+1)    
    return res


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    # TODO: write your function instead of this one
    res = np.zeros(shape=shape, dtype="float32")
    for i in range(shape[0]):
        for j in range(shape[1]):
            res[i, j] = shape[0]*shape[1] - (i+1) * (j+1)
    res = np.fliplr(res)
    return res
