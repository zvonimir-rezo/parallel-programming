from numba import cuda
import numpy as np
import math
import sys
import warnings
import time
warnings.filterwarnings('ignore')
np.set_printoptions(threshold=sys.maxsize)


@cuda.jit
def jacobistep(psitmp, psi, m, n, nthreads):
    cg = cuda.grid(1)
    for i in range(cg+1, m+1, nthreads):
        for j in range(1, n+1):
            psitmp[i*(m+2)+j]=0.25*(psi[(i-1)*(m+2)+j]+psi[(i+1)*(m+2)+j]+psi[i*(m+2)+j-1]+psi[i*(m+2)+j+1])


@cuda.jit
def deltasq(psitmp, psi, m, n, dsq, nthreads):  
    cg = cuda.grid(1)
    for i in range(cg+1, m+1, nthreads):
        for j in range(1, n+1):
            tmp = psitmp[i*(m+2)+j]-psi[i*(m+2)+j]
            cuda.atomic.add(dsq, 0, tmp*tmp)


@cuda.jit
def ccpy(psitmp, psi, m, n, nthreads):
    cg = cuda.grid(1)
    for i in range(cg+1, m+1, nthreads):
        for j in range(1, n+1):
            psi[i*(m+2)+j]=psitmp[i*(m+2)+j]
    


def main():

    G = 16
    L = 16
    nthreads = G * L

    printfreq = 1000
    error, bnorm = 0, 0
    tolerance = 0
    scalefactor, numiter = 0, 0
    printfreq = 1000
    error = 0
    bnorm = 0
    tolerance = 0
    bbase = 10
    hbase = 15
    wbase = 5
    mbase = 32
    nbase = 32
    irrotational = 1
    checkerr = 0
    m, n, b, h, w = 0, 0, 0, 0, 0
    iter = 0
    i, j = 0, 0
    tstart, tstop, ttot, titer = 0, 0, 0, 0

    if tolerance > 0:
        checkerr = 1

    scalefactor, numiter = 64, 100

    if checkerr == 0:
        print(f'Scale Factor = {scalefactor}, iterations = {numiter}\n')
    else:
        print(f'Scale Factor = {scalefactor}, iterations = {numiter}, tolerance = {tolerance}\n')

    print('Irrotational flow')


    b = bbase*scalefactor
    h = hbase*scalefactor
    w = wbase*scalefactor
    m = mbase*scalefactor
    n = nbase*scalefactor

    print(f'Running CFD on {m} x {n} grid in serial\n')

    psi = np.zeros(((m+2)*(n+2)), dtype=np.float32)
    psitmp = np.zeros(((m+2)*(n+2)), dtype=np.float32)

    for i in range(b+1, b+w):
        psi[i*(m+2)+0] = i-b

    for i in range(b+w, m+1):
        psi[i*(m+2)+0] = w

    for j in range(1, h+1):
        psi[(m+1)*(m+2)+j] = w

    for j in range(h+1, h+w):
        psi[(m+1)*(m+2)+j]= w-j+h

    bnorm=0.0
    for i in range(0, m+2):
        for j in range(0, n+2):
            bnorm += psi[i*(m+2)+j]*psi[i*(m+2)+j]
    
    bnorm=math.sqrt(bnorm)

    print('\nStarting main loop...\n\n')
    tstart=time.time()

    for iter in range(1, numiter+1):
  
        jacobistep[G, L](psitmp, psi, m, n, nthreads)

        if checkerr or iter == numiter:
            dsq = np.array([0], dtype=np.float32)
            deltasq[G, L](psitmp, psi, m, n, dsq, nthreads)
            error = dsq[0]
            error=math.sqrt(error)
            error=error/bnorm

        if checkerr: 
            if error < tolerance:
                print(f'Converged on iteration {iter}\n')
                break

        ccpy[G, L](psitmp, psi, m, n, nthreads)

        if(iter%printfreq == 0):
            if not checkerr:
                print(f'Completed iteration {iter}\n')
            else:
                print(f'Completed iteration {iter}, error = {error}\n')

    if iter > numiter:
        iter=numiter

    tstop=time.time()

    ttot=tstop-tstart
    titer=ttot/iter

    print(f'\n... finished\n')
    print(f'After {iter} iterations, the error is {error}\n')
    print(f'Time for {iter} iterations was {ttot} seconds\n')
    print(f'Each iteration took {titer} seconds\n')
    print(f'... finished\n')

if __name__ == '__main__':
    main()