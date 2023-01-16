from numba import cuda
import numpy as np
import warnings
import time
import math
warnings.filterwarnings('ignore')

@cuda.jit
def cudakernel(array, prim_array, array_len, nthreads):
    cg = cuda.grid(1)
    for t in range(cg, array_len, nthreads):
        num = array[t]
        prim = True
        for i in range(2, math.ceil(math.sqrt(num)) + 1):
            if num % i == 0 and i != num:
                prim = False
                break
        if prim:
            prim_array[t] = 1

G = 4
L = 4

array_len = 100000

nthreads = G * L

start = time.time()
array = np.arange(start=1, stop=array_len + 1, step=1)
prim_array = np.zeros(array_len)
# print('Initial array:', array)
cudakernel[G, L](array, prim_array, array_len, nthreads)
print('Number of prims:', int(np.sum(prim_array)))
print(f'Time needed: {time.time() - start}')

