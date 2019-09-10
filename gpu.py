from numba import cuda
import numpy as np
from math import ceil
from decimal import *

@cuda.jit
def round_func(x, out):
    out[0] = Decimal(str(x)).quantize(Decimal('.01'), rounding=ROUND_UP)

if __name__ == '__main__':
    x = np.zeros(shape=(1), dtype=np.float32)
    out = np.zeros(shape=(1), dtype=np.int32)

    x = [5.4]

    x_d = cuda.to_device(x)
    out_d = cuda.to_device(out)

    threads = 2
    blocks = 2

    round_func[threads, blocks](x_d, out_d)

    print(out_d.copy_to_host())