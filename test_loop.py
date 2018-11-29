import numpy as np
import timeit
from numba import jit
from numba import cuda
from numba import float64

class SmArray():
    def __init__(self, x):
        self.value = x

@jit(nopython=True)
def nan_compact(x):
    if a.value[0] > 0.2: print('true')
    out = np.empty_like(x)
    out_index = 0
    for element in x:
        if not np.isnan(element):
            out[out_index] = element
            out_index += 1
    return out[:out_index]

a = SmArray(np.random.uniform(size=1000000))
# a.value[a < 0.2] = np.nan

start = timeit.default_timer()

nan_compact(a.value)

print(timeit.default_timer()-start)
# timeit('a[~np.isnan(a)]')
# timeit('nan_compact(a)')