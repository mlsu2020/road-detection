import numpy as np

def diamond(n):
    '''Create diamond shaped kernel for  use in binary closing'''
    a = np.arange(n)
    b = np.minimum(a,a[::-1])
    return ((b[:,None]+b)>=(n-1)//2).astype(np.int)