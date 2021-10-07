from functools import partial, reduce
from itertools import combinations
from multiprocessing import Pool
import numpy.linalg as la
from tqdm import tqdm
import pickle as pkl
import numpy as np
import sys, time, gc

import scipy
import scipy.fftpack


def grf(alpha=-3.0, m=128, flag_normalize=True):
    size = int(np.sqrt(m))
    k_ind = np.mgrid[:size, :size] - int((size + 1) / 2)
    k_idx = scipy.fftpack.fftshift(k_ind)
    # Defines the amplitude as a power law 1/|k|^(alpha/2)
    amplitude = np.power(k_idx[0] ** 2 + k_idx[1] ** 2 + 1e-10, alpha / 4.0)
    amplitude[0,0] = 0
    # Draws a complex gaussian random noise with normal (circular) distribution
    noise = np.random.normal(size = (size, size)) + 1j * np.random.normal(size = (size, size))
    gfield = np.fft.ifft2(noise * amplitude).real # To real space
    return (gfield - gfield.min()) / (gfield.max() - gfield.min())

def tqit(it, verbose=False, desc=None, n=None):
    return tqdm(it, desc=desc, total=n) if verbose else it

def load(fcache):
    sys.stdout.write('[ loading %s...' % fcache)
    sys.stdout.flush()
    t0 = time.time()
    with open(fcache, 'rb') as f:
        gc.disable()
        dat = pkl.load(f)
        gc.enable()
    print(' %0.3fs' % (time.time() - t0))
    return dat

def save(dat, fcache):
    sys.stdout.write('[ saving %s...' % fcache)
    sys.stdout.flush()
    t0 = time.time()
    with open(fcache, 'wb') as f:
        gc.disable()
        pkl.dump(dat, f)
        gc.enable()
    print(' %0.3fs' % (time.time() - t0))


def insert(L, i, x):
    L[i] += [x]
    return L

def partition(f, X, n):
    return reduce(f, X, [[] for _ in range(n)])


def in_rng(c, I, open=False):
    return ((I[0] < c < I[1]) if open
        else (I[0] <= c <= I[1]))

def in_bounds(p, bounds, open=False):
    return all(map(lambda cI: in_rng(cI[0],cI[1],open), zip(p, bounds)))

def is_boundary(p, d, l):
    return not all(d < c < u - d for c,u in zip(p, l))

def to_path(vertices, nbrs):
    V = vertices.copy()
    cur = V.pop()
    path = [cur]
    while len(V):
        cur = nbrs[cur].intersection(V).pop()
        path.append(cur)
        V.remove(cur)
    return path

def diff(p):
    return p[1] - p[0]

def identity(x):
    return x

def get_delta(n, w=1, h=1):
    return 2 / (n-1) * np.sqrt(w ** 2 + h ** 2)

def lipschitz(f, P):
    return max(abs(fp - fq) / la.norm(p - q) for (fp,p),(fq,q) in tqdm(list(combinations(zip(f,P),2))))

def scale(x):
    return (x - x.min()) / (x.max() - x.min())

def stuple(s, *args, **kw):
    return tuple(sorted(s, *args, **kw))

def pmap(fun, x, *args, **kw):
    pool = Pool()
    f = partial(fun, *args, **kw)
    try:
        y = pool.map(f, x)
    except KeyboardInterrupt as e:
        print(e)
        pool.close()
        pool.join()
        sys.exit()
    pool.close()
    pool.join()
    return y

def format_float(f):
    if f.is_integer():
        return int(f)
    e = 0
    while not f.is_integer():
        f *= 10
        e -= 1
    return '%de%d' % (int(f), e)
