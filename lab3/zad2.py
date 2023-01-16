from numba import cuda
import numpy as np
import warnings
import time
warnings.filterwarnings('ignore')

@cuda.jit
def cudakernel(array, n, nthreads):
    thread_position = cuda.grid(1)
    h = 1.0 / n
    suma = 0
    for i in range(thread_position+1, n+1, nthreads):
        x = h * (i - 0.5)
        suma += 4.0 / (1.0 + x*x)
    array[thread_position] = h * suma

G = 4
L = 4

start = time.time()
nthreads = G * L
n = 10000000

array = np.zeros(nthreads)

cudakernel[G, L](array, n, nthreads)

print('Pi:', np.sum(array))
print(f"Time: {time.time() - start}")