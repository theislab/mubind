import numpy as np


def supertest(m):
    count = 0
    for i in range(m.shape[0]):
        for j in range(m.shape[0]):
            count += m[i, j]
    return np.mean(count)
