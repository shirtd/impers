from topology.complex.simplicial import *
from topology.complex.cellular import *
from topology.complex.geometric import *

from topology.persistence.filtration import *
from topology.persistence.diagram import *

from topology.plot.util import plot_diagrams, plt, pqt, plot_complex

import dionysus as dio
import diode

import numpy.linalg as la
import random
import pickle as pkl
import numpy as np
import time, os

# def add_fun(K, fun, key, induce):
#     for dim, S in K.items():
#         for s in S:
#             if s in fun:
#                 s.data[key] = fun[s]
#             else:
#                 s.data[key] = induce([t(key) for t in K.faces(s)])

if __name__ == '__main__':
    np.random.seed(0)
    BOUNDS = np.array([[0, 1], [0, 1]])
    N = (20, 20)
    NOISE = 5e-2

    xs, ys = np.meshgrid(*(np.linspace(a,b,n) for (a,b),n in zip(BOUNDS, N)))
    grid_points = np.vstack((xs.flatten(), ys.flatten())).T
    random_noise = diff(BOUNDS.T) * (2*np.random.rand(*grid_points.shape) - 1) * NOISE
    P = grid_points + random_noise

    NQ = 100
    NF = 1000
    RADIUS = 0.4
    CENTER = (BOUNDS[:,1] - BOUNDS[:,0]) / 2

    Ft = np.linspace(-np.pi, np.pi, NF)
    FX =  CENTER + RADIUS * np.vstack((np.sin(Ft), np.cos(Ft))).T

    Qidx = random.sample(range(len(P)), NQ)
    Q = P[Qidx]
    FQ = np.array([min(la.norm(q - x) for x in FX) for q in Q])


    A = DelaunayComplex(P, verbose=True)
    V = VoronoiComplex(A, verbose=True)


    FVmax = [min(f + la.norm(p - q) for q,f in zip(Q, FQ)) for p in V.P]
    FVmin = [max(f - la.norm(p - q) for q,f in zip(Q, FQ)) for p in V.P]
    for s in V:
        s.data['maxext'] = min(FVmax[i] for i in s)
        s.data['minext'] = min(FVmin[i] for i in s)
        V.primal(s).data['maxext'] = s('maxext')
        V.primal(s).data['minext'] = s('minext')


    # F = Filtration(A, 'minext', False)
    # B = A.get_boundary(BOUNDS)
    # R = {F.index(s) for s in B}
    # HF = Diagram(A, F, R, verbose=True)
    #
    # G = Filtration(V, 'minext', True)
    # C = {V.dual(s) for s in B}
    # S = {G.index(s) for s in C}
    # HG = Diagram(V, G, S, coh=True, verbose=True)


    plt.ion()
    fig, ax = plt.subplots(1,1, figsize=(6,5))#, sharex=True, sharey=True)
    ax = [ax]

    MARGIN = 0.05
    MARGINS = np.array([[-MARGIN, MARGIN],[-MARGIN, MARGIN]])
    WINDOW = BOUNDS + MARGINS

    ax[0].set_xlim(*WINDOW[0]); ax[0].set_ylim(*WINDOW[1])
    ax[0].scatter(P[:,0], P[:,1], s=2, zorder=2)
    ax[0].scatter(Q[:,0], Q[:,1], s=4, c='red', zorder=10)

    for e in V.P[[s for s in V(1) if len(s) == 2]]:
        ax[0].plot(e[:,0], e[:,1], c='black', zorder=1)

    FXX = np.vstack((FX, FX[-1]))
    ax[0].plot(FXX[:,0], FXX[:,1], zorder=10, lw=1.5, c='red', ls='-')

    # plot_diagrams(ax[1], HF.diagram)
    ax[0].axis('off')
    plt.tight_layout()

    BREAKS = np.linspace(0, RADIUS*1.1, 11)[1:]

    FIGDIR = 'figures'
    if not os.path.exists(FIGDIR):
        os.mkdir(FIGDIR)

    for i, alpha in enumerate(BREAKS):
        elems = []
        for s in A:
            for key, color, ord in [('maxext', 'purple', 6), ('minext','green', 5)]:
                if s(key) <= alpha:
                    if s.dim == 0:
                        pass
                        # elems.append(ax[0].scatter(P[s,0], P[s,1], s=7, c=color, zorder=ord))
                    elif s.dim == 1:
                        elems += ax[0].plot(P[s,0], P[s,1], c=color, zorder=ord-1, alpha=0.7)
                    elif s.dim == 2:
                        elems.append(ax[0].add_patch(plt.Polygon(P[s], color=color, alpha=0.4, zorder=ord-2)))
        plt.pause(0.1)
        fname = os.path.join(FIGDIR, 'alpha%d.png' % i)
        print('saving %s' % fname)
        plt.savefig(fname, dpi=500, transparent=True)
        # input('alpha = %0.2f' % alpha)
        for e in elems:
            e.remove()
