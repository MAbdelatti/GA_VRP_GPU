from numba import cuda
import numpy as np

@cuda.jit
def test(a, b ,c):
    row, _col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)
    for col in range(_col, c.shape[1], stride_y):
        cuda.atomic.add(c, (row,0), b[a[row, col],a[row, col+1]])

threads_per_block = (5, 5)
blocks = (6, 6)

a = np.ones((5,5), dtype=np.int32)
b = np.ones((5,5), dtype=np.int32)
c = np.zeros((5,5), dtype=np.int32)

a_d = cuda.to_device(a)
b_d = cuda.to_device(b)
c_d = cuda.to_device(c)

test[blocks, threads_per_block](a_d, b_d, c_d)
print(c_d.copy_to_host())