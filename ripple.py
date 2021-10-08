from topology.complex.simplicial import *
from topology.complex.cellular import *
from topology.complex.geometric import *

from topology.persistence.filtration import *
from topology.persistence.diagram import *

from topology.util import lipschitz, dio_diagram
from topology.plot.util import plot_diagrams
from topology.plot.lipschitz import *
from topology.data import *

import numpy.linalg as la
import dionysus as dio
import pickle as pkl
import numpy as np
import time, os
import random

def induce(D, V, f, key, finduce=max):
    for s in D:
        s.data[key] = V.dual(s).data[key] = finduce(f[i] for i in s)

def max_ext(Q, fQ, l, p):
    return min(f + l*la.norm(p - q) for q,f in zip(Q, fQ))

def min_ext(Q, fQ, l, p):
    return max(f - l*la.norm(p - q) for q,f in zip(Q, fQ))

def lipschitz_extend(D, V, P, Q, fQ, l, finduce=max, verbose=True):
    it = tqdm(P, desc='[ min/max ext') if verbose else P
    es = list(zip(*[[e(Q, fQ, l, p) for e in (max_ext, min_ext)] for p in it]))
    for e,k in zip(es, ('maxext', 'minext')):
        induce(D, V, e, k, finduce)
    return es


if __name__ == '__main__':
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)

    # config
    SAVE = True
    FIGDIR = 'figures'
    LABEL = 'delaunay'
    WIDTH, HEIGHT = 1, 1
    _f, _l, _w = 2, 2, 1
    BREAKS = [-0.8, -0.6, -0.5, -0.2, 0., 0.3, 0.5, 0.7, 0.8]
    GRID_RES = 1024
    N_P = 2000 # 2000
    N_Q = 1000 # 1000

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
    P_FUN = GRID_FUN[I_P_GRID]

    I_Q_P = sorted(random.sample(range(N_P), N_Q))
    I_Q_GRID = [I_P_GRID[i] for i in I_Q_P]
    Q = GRID_POINTS[I_Q_GRID]
    Q_FUN = GRID_FUN[I_Q_GRID]

    # complexes
    DQ, D = DelaunayComplex(Q), DelaunayComplex(P)
    VQ, V = VoronoiComplex(DQ), VoronoiComplex(D)
    induce(DQ, VQ, Q_FUN, 'fun', max)
    induce(D, V, P_FUN, 'fun', max)
    P_FUN_max, P_FUN_min = lipschitz_extend(D, V, P, Q, Q_FUN, lips)

    # pers plot
    plt.ion()
    fig, ax = plt.subplots(3,3, sharex='row', sharey='row',figsize=(12,12))
    lims = (-1.5, 2)

    Fdio = dio.fill_freudenthal(field)
    Hdio = dio.homology_persistence(Fdio, progress=True)
    Ddio = dio_diagram(dio.init_diagrams(Hdio, Fdio))

    filt = {'Qfun' : Filtration(DQ, 'fun'),
            'Pfun' : Filtration(D, 'fun'),
            'maxext' : Filtration(D, 'maxext'),
            'minext' : Filtration(D, 'minext', False)}

    hom = {'Qfun' : Diagram(DQ, filt['Qfun']),
            'Pfun' : Diagram(D, filt['Pfun']),
            'maxext' : Diagram(D, filt['maxext']),
            'minext' : Diagram(D, filt['minext']),
            'image' : Diagram(D, filt['minext'], pivot=filt['maxext'])}

    plot_diagrams(ax[0,0], hom['Pfun'].diagram, lims, 'induced on P (%d pts)' % N_P)
    plot_diagrams(ax[0,1], hom['Qfun'].diagram, lims, 'induced on Q (%d pts)' % N_Q)
    plot_diagrams(ax[0,2], Ddio, lims, 'grid (%dx%d)' % (GRID_RES, GRID_RES))
    plot_diagrams(ax[1,0], hom['maxext'].diagram, lims, 'Q max extension to P')
    plot_diagrams(ax[1,1], hom['minext'].diagram, lims, 'Q min extension to P')
    plot_diagrams(ax[1,2], hom['image'].diagram, lims, 'Q image extension to P')

    for axis in ax[2]:
        axis.axis('off')
        axis.set_xlim(*BOUNDS[0])
        axis.set_ylim(*BOUNDS[1])
    if SAVE:
        if not os.path.exists(FIGDIR):
            os.mkdir(FIGDIR)
    plt.tight_layout()

    elems = []
    for i, alpha in enumerate(BREAKS):
        for x in ax[:2].flatten():
            elems += x.plot([lims[0], alpha], [alpha, alpha], c='red', ls=':', alpha=0.5, zorder=1)
            elems += x.plot([alpha, alpha], [alpha, 1.2*lims[1]], c='red', ls=':', alpha=0.5, zorder=1)
        elems += [plot_field(ax[2,2], field, BOUNDS, alpha, alpha=0.5, zorder=0)]
        for s in D:
            for axis, key, color, ord in [(ax[2,0], 'maxext', 'purple', 6), (ax[2,1], 'minext','green', 5)]:
                if s(key) <= alpha:
                    elems += plot_cell(axis, D.P, s, color, ord)
                    elems += plot_cell(ax[2,2], D.P, s, color, ord)
        plt.pause(0.1)
        if SAVE:
            fname = os.path.join(FIGDIR, '%s%da%de-1.png' % (LABEL, i, int(alpha*10)))
            print('saving %s' % fname)
            plt.savefig(fname, dpi=300)#, transparent=True)
        else:
            input('%s: %0.2f' % (LABEL, alpha))
        while len(elems):
            elems.pop().remove()
