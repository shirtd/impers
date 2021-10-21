from topology.complex.simplicial import *
from topology.complex.cellular import *
from topology.complex.geometric import *

from topology.persistence.filtration import *
from topology.persistence.diagram import *

from topology.util import lipschitz, dio_diagram, pmap
from topology.plot.util import plot_diagrams
from topology.plot.lipschitz import *
from topology.data import *

import numpy.linalg as la
import dionysus as dio
import pickle as pkl
import time, os, sys
import numpy as np
import random


def induce(D, V, f, key, finduce=max):
    for s in D:
        s.data[key] = V.dual(s).data[key] = finduce(f[i] for i in s)

def max_ext(Q, fQ, l, d, p):
    # return min(f + l*(la.norm(p - q) - d) for q,f in zip(Q, fQ))
    return min(f + l*(la.norm(p - q)) for q,f in zip(Q, fQ))

def min_ext(Q, fQ, l, d, p):
    # return max(f - l*(la.norm(p - q) + d) for q,f in zip(Q, fQ))
    # return max(f - l*(la.norm(p - q) + d) for q,f in zip(Q, fQ))
    return max(f - l*(la.norm(p - q)) for q,f in zip(Q, fQ))

def minmax_ext(Q, fQ, l, d, p):
    return [e(Q, fQ, l, d, p) for e in (max_ext, min_ext)]

def lipschitz_extend(D, V, Q, fQ, l, d=0, finduce=max, verbose=True):
    it = tqdm(D.P, desc='[ min/max ext') if verbose else D.P
    # es = list(zip(*[[e(Q, fQ, l, d, p) for e in (max_ext, min_ext)] for p in it]))
    es = list(zip(*pmap(minmax_ext, it, Q, fQ, l, d)))
    for e,k in zip(es, ('maxext', 'minext')):
        induce(D, V, e, k, finduce)
    return es

# def lipschitz_extend(D, V, Q, fQ, l, finduce=min, verbose=True):
#     it = tqdm(V.P, desc='[ min/max ext') if verbose else V.P
#     es = list(zip(*[[e(Q, fQ, l, p) for e in (max_ext, min_ext)] for p in it]))
#     for e,k in zip(es, ('maxext', 'minext')):
#         for s in V:
#             s.data[k] = V.primal(s).data[k] = finduce(e[i] for i in s)
#     return es


if __name__ == '__main__':
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)

    TEST = False
    FORCE = True # False #

    # config
    SAVE = True
    LOAD = True
    CACHE = True
    FCACHE = 'cache'
    FIGDIR = 'figures'
    LABEL = 'voronoi' # 'delaunay'
    INDUCE = max if LABEL == 'delaunay' else min
    WIDTH, HEIGHT = 1, 1
    _f, _l, _d, _w, _s = 2, 2, 100, 2, 2
    # BREAKS = [-0.8, -0.6, -0.5, -0.2, 0., 0.3, 0.5, 0.7, 0.8]
    # BREAKS = np.linspace(-0.1, 0.2, 11)[1:]
    BREAKS = np.linspace(0, 1, 11)[1:]
    GRID_RES = 512
    N_P = 2000 # 2000
    N_Q = 1000 # 1000

    CIRCLE_N = 256
    CIRCLE_R = [0.5]
    CIRCLE_C = [[0,0]]

    # field init
    N_GRID = int(WIDTH*GRID_RES) * int(HEIGHT*GRID_RES)
    GRID_X, GRID_Y = get_grid(GRID_RES, WIDTH, HEIGHT)
    BOUNDS = np.array([[-WIDTH, WIDTH], [-HEIGHT, HEIGHT]])
    # field = get_ripple(GRID_X, GRID_Y, _f, _l, _d, _w, _s, -3, False)
    field = get_circle(GRID_X, GRID_Y, CIRCLE_N, CIRCLE_R, CIRCLE_C)
    GRID_POINTS = np.vstack([GRID_X.flatten(), GRID_Y.flatten()]).T
    GRID_FUN = field.flatten()

    if TEST:
        fig, ax = plt.subplots(1,1)
        ax.imshow(field)
        plt.tight_layout()
        plt.pause(0.1)
        lips = ripple_lips(0, max(WIDTH, HEIGHT), GRID_RES, _f, _l, _d, _w, _s)
        print('lips: %f' % lips)
        sys.exit(0)

    # sample init
    I_P_GRID = sorted(random.sample(range(len(GRID_POINTS)), N_P))
    P = GRID_POINTS[I_P_GRID]
    P_FUN = GRID_FUN[I_P_GRID]

    I_Q_P = sorted(random.sample(range(N_P), N_Q))
    I_Q_GRID = [I_P_GRID[i] for i in I_Q_P]
    Q = GRID_POINTS[I_Q_GRID]
    Q_FUN = GRID_FUN[I_Q_GRID]

    lips = 1 # ripple_lips(0, max(WIDTH, HEIGHT), GRID_RES, _f, _l, _d, _w, _s)
    # complexes

    # P_FUN_max, P_FUN_min = lipschitz_extend(V, D, Q, Q_FUN, lips, min)

    # pers plot
    plt.ion()
    fig, ax = plt.subplots(3,3, figsize=(12,12))#,
                            # sharex=True, # 'row',
                            # sharey=True) # 'row')
    # lims = (-1.5, 2)
    # lims = (-0.2, 0.2)
    lims = (0, 1)

    CACHE_PATH = '%s_%s.pkl' % (LABEL, FCACHE)
    if not FORCE and LOAD and os.path.exists(CACHE_PATH):
        print('loading %s' % CACHE_PATH)
        DQ, D, VQ, V, B, delta, filt, hom, Ddio = pkl.load(open(CACHE_PATH, 'rb'))
    else:
        DQ, D = DelaunayComplex(Q), DelaunayComplex(P)
        VQ, V = VoronoiComplex(DQ), VoronoiComplex(D)
        B = D.get_boundary(BOUNDS)
        delta = max(max(la.norm(D.P[s[0]] - V.P[j]) for j in V.dual(s)) for s in D(0) if not s in B)
        induce(DQ, VQ, Q_FUN, 'fun', INDUCE)  # max) #
        induce(D, V, P_FUN, 'fun', INDUCE) # max) #
        lipschitz_extend(D, V, Q, Q_FUN, lips, delta, INDUCE) # max) #

        Fdio = dio.fill_freudenthal(field)
        Hdio = dio.homology_persistence(Fdio, progress=True)
        Ddio = dio_diagram(dio.init_diagrams(Hdio, Fdio))

        if LABEL == 'voronoi':
            filt = {'Qfun' : Filtration(VQ, 'fun'),
                    'Pfun' : Filtration(V, 'fun'),
                    'maxext' : Filtration(V, 'maxext'),
                    'minext' : Filtration(V, 'minext', False)}

            hom = {'Qfun' : Diagram(VQ, filt['Qfun']),
                    'Pfun' : Diagram(V, filt['Pfun']),
                    'maxext' : Diagram(V, filt['maxext']),
                    'minext' : Diagram(V, filt['minext']),
                    'image' : Diagram(V, filt['minext'], pivot=filt['maxext'])}

        elif LABEL == 'delaunay':
            filt = {'Qfun' : Filtration(DQ, 'fun'),
                    'Pfun' : Filtration(D, 'fun'),
                    'maxext' : Filtration(D, 'maxext'),
                    'minext' : Filtration(D, 'minext', False)}

            hom = {'Qfun' : Diagram(DQ, filt['Qfun']),
                    'Pfun' : Diagram(D, filt['Pfun']),
                    'maxext' : Diagram(D, filt['maxext']),
                    'minext' : Diagram(D, filt['minext']),
                    'image' : Diagram(D, filt['minext'], pivot=filt['maxext'])}

        if CACHE:
            print('caching %s' % CACHE_PATH)
            pkl.dump((DQ, D, VQ, V, B, delta, filt, hom, Ddio), open(CACHE_PATH, 'wb'))

    # R = {filt['minext'].index(V.dual(s)) for s in B}
    # hom[image] = Diagram(V, filt['minext'], R, pivot=filt['maxext'])

    PLOT DIAGRAMS

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
    # ax[2,1].patch.set_facecolor('gray')
    ax[2,1].add_patch(plt.Rectangle([-1,-1],2,2, color='gray', zorder=0))
    plt.tight_layout()

    # for x in ax[2]:
    #     x.scatter(Q[:,0], Q[:,1], c='red', zorder=10, marker=',', s=0.5)
    #
    for e in V(1):
        for x in ax[2]:
            x.plot(V.P[e,0], V.P[e,1], c='black', zorder=1, lw=0.5)

    elems = []
    for i, alpha in enumerate(BREAKS):
        for x in ax[:2].flatten():
            elems += x.plot([lims[0], alpha], [alpha, alpha], c='red', ls=':', alpha=0.5, zorder=1)
            elems += x.plot([alpha, alpha], [alpha, 1.2*lims[1]], c='green', ls=':', alpha=0.5, zorder=1)
        # ax[2,1].set_facecolor('gray')
        for fq,q in zip(Q_FUN, Q):
            if (alpha - fq) / lips > 0:
                elems += [ax[2,0].add_patch(plt.Circle(q, alpha - fq, color='gray', zorder=1))]
            if (fq - alpha) / lips > 0:
                elems += [ax[2,1].add_patch(plt.Circle(q, fq - alpha, color='white', zorder=1))]
        elems += [plot_field(ax[2,2], field, BOUNDS, alpha, alpha=0.5, zorder=0)]
        for s in (V if LABEL == 'voronoi' else  D): #
            for axis, key, color, ord in [(ax[2,0], 'maxext', 'purple', 6), (ax[2,1], 'minext','green', 5)]:
                if s(key) <= alpha + lips*delta * (1 if key == 'minext' else -1): # alpha:# +
                    if LABEL == 'voronoi':
                        elems += plot_voronoi(axis, V, s, color, ord)
                        elems += plot_voronoi(ax[2,2], V, s, color, ord)
                    elif LABEL == 'delaunay':
                        elems += plot_cell(axis, D, s, color, ord)
                        elems += plot_cell(ax[2,2], D, s, color, ord)
        plt.pause(0.1)
        if SAVE:
            fname = os.path.join(FIGDIR, '%s%da%de-1.png' % (LABEL, i, int(alpha*10)))
            print('saving %s' % fname)
            plt.savefig(fname, dpi=300)#, transparent=True)
        else:
            input('%s: %0.2f' % (LABEL, alpha))
        while len(elems):
            elems.pop().remove()
