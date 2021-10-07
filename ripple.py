from topology.complex.simplicial import *
from topology.complex.cellular import *
from topology.complex.geometric import *

from topology.persistence.filtration import *
from topology.persistence.diagram import *

from topology.plot.util import plot_diagrams
from topology.util import lipschitz
from topology.plot.lipschitz import *
from topology.data import *

import numpy.linalg as la
import pickle as pkl
import numpy as np
import time, os
import random


def max_ext(Q, fQ, l, p):
    return min(f + l*la.norm(p - q) for q,f in zip(Q, fQ))

def min_ext(Q, fQ, l, p):
    return max(f - l*la.norm(p - q) for q,f in zip(Q, fQ))

def lipschitz_extend(D, V, P, Q, fQ, l, induce=max, verbose=True):
    it = tqdm(P, desc='[ min/max ext') if verbose else P
    es = list(zip(*[[e(Q, fQ, l, p) for e in (max_ext, min_ext)] for p in it]))
    for s in D:
        for e,k in zip(es, ('maxext', 'minext')):
            s.data[k] = V.dual(s).data[k] = induce(e[i] for i in s)
    return es


if __name__ == '__main__':
    np.random.seed(0)

    # config
    NBREAK = 10
    SAVE = True
    FIGDIR = 'figures'
    LABEL = 'alpha'
    WIDTH, HEIGHT = 1, 1
    _f, _l, _w = 2, 2, 1
    GRID_RES = 1024
    N_P = 2000
    N_Q = 1000

    # field init
    N_GRID = int(WIDTH*GRID_RES) * int(HEIGHT*GRID_RES)
    GRID_X, GRID_Y = get_grid(GRID_RES, WIDTH, HEIGHT)
    BOUNDS = np.array([[-WIDTH, WIDTH], [-HEIGHT, HEIGHT]])
    field = get_ripple(GRID_X, GRID_Y, _f, _l, _w, -3, False)
    lips = ripple_lips(0, max(WIDTH, HEIGHT), GRID_RES, _f, _l, _w) # 11.8424 #
    GRID_POINTS = np.vstack([GRID_X.flatten(), GRID_Y.flatten()]).T
    GRID_FUN = field.flatten()

    # sample init
    I_P_GRID = sorted(random.sample(range(len(GRID_POINTS)), N_P))
    P = GRID_POINTS[I_P_GRID]

    I_Q_P = sorted(random.sample(range(N_P), N_Q))
    I_Q_GRID = [I_P_GRID[i] for i in I_Q_P]
    Q = GRID_POINTS[I_Q_GRID]
    Q_FUN = GRID_FUN[I_Q_GRID]

    # complexes
    D = DelaunayComplex(P) #, verbose=True)
    V = VoronoiComplex(D) #, verbose=True)

    # run
    P_FUN_max, P_FUN_min = lipschitz_extend(D, V, P, Q, Q_FUN, lips)

    # plot
    plt.ion()
    fig, ax = plt.subplots(1,1, figsize=(6,5))
    plot_breaks(ax, D, field, BOUNDS, max(P_FUN_max), NBREAK, SAVE, FIGDIR, LABEL)
