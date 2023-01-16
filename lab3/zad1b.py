from numba import cuda
import numpy as np
import warnings
import math
warnings.filterwarnings('ignore')

@cuda.jit
def cudakernel(array, br, array_len, nthreads):
    cg = cuda.grid(1)
    for t in range(cg, array_len, nthreads):
        num = array[t]
        prim = True
        for i in range(2, math.ceil(math.sqrt(num)) + 1):
            if num % i == 0:
                prim = False
                break
        if prim:
            cuda.atomic.add(br, 0, 1)
        
G = 32
L = 32

array_len = 2**18

nthreads = G * L

br = np.array([0], dtype=np.int32)
array = np.arange(start=1, stop=array_len + 1, step=1)
prim_array = np.zeros(array_len)
# print('Initial array:', array)
cudakernel[G, L](array, br, array_len, nthreads)

print('Number of prims:', br[0])