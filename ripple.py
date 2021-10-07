from topology.complex.simplicial import *
from topology.complex.cellular import *
from topology.complex.geometric import *

from topology.persistence.filtration import *
from topology.persistence.diagram import *

from topology.plot.util import plot_diagrams, plt
from topology.util import grf, lipschitz
from scipy.spatial import KDTree

import numpy.linalg as la
import pickle as pkl
import numpy as np
import time, os
import random

# def ripple(x, y, f=1, l=1, w=1):
#     t = la.norm(np.stack((x, y), axis=2), axis=2) + 1/12
#     t[t > 1] = 1.
#     return (1 - t) - np.exp(-t / l) * np.cos(2*np.pi*f*(1-t) * np.sin(2*np.pi*w*t))

def ripple(t, f=1, l=1, w=1):
    return np.exp(-t / l) * np.cos(2*np.pi*f*t)

def ripple_grid(x, y, f=1, l=1, w=1):
    t = la.norm(np.stack((x, y), axis=2), axis=2) + 1/12
    # t[t > 1] = 1.
    # return (1 - t) - np.exp(-t / l) * np.cos(2*np.pi*f*(1-t))
    return ripple(t, f, l, w)

def ripple_lips(a=0, b=1, n=1024, f=1, l=1, w=1):
    t =  np.linspace(a, b, n)
    return lipschitz(ripple(t, f, l, w), t)

def get_grid(res=1024, width=1, height=1):
    x_rng = np.linspace(-width,width,int(width*res))
    y_rng = np.linspace(-height,height,int(height*res))
    return np.meshgrid(x_rng, y_rng)

def get_ripple(x, y, f=1, l=1, w=1, exp=-3, noise=False, scale=False):
    f = ripple_grid(x, y, f, l, w)
    if noise:
        f *= (1 + grf(exp, x.shape[0]*x.shape[1]))
    return (f - f.min()) / (f.max() - f.min()) if scale else f

def plot_cell(axis, P, s, color, ord):
    if s.dim == 0:
        return [axis.scatter(P[s,0], P[s,1], s=1, c=color, zorder=ord)]
    elif s.dim == 1:
        return axis.plot(P[s,0], P[s,1], c=color, zorder=ord-1, alpha=0.7, lw=0.5)
    elif s.dim == 2:
        return [axis.add_patch(plt.Polygon(P[s], color=color, alpha=0.4, zorder=ord-2, ec=None))]

def lipschitz_extend(D, V, P, Q, Q_FUN, LIPS):
    P_FUN_max = [min(f + LIPS*la.norm(p - q) for q,f in zip(Q, Q_FUN)) for p in tqdm(P)]
    P_FUN_min = [max(f - LIPS*la.norm(p - q) for q,f in zip(Q, Q_FUN)) for p in tqdm(P)]
    for s in D:
        s.data['maxext'] = max(P_FUN_max[i] for i in s)
        s.data['minext'] = max(P_FUN_min[i] for i in s)
        V.dual(s).data['maxext'] = s('maxext')
        V.dual(s).data['minext'] = s('minext')
    return P_FUN_max, P_FUN_min

def plot_breaks(ax, K, field, bounds, mx, nbreak=10, save=False, figdir='figures', label='alpha'):
    ax.axis('off')
    plt.tight_layout()
    ax.set_xlim(*bounds[0]); ax.set_ylim(*bounds[1])

    BREAKS = np.linspace(0, mx, nbreak+1)[1:]

    if save:
        if not os.path.exists(figdir):
            os.mkdir(figdir)

    for i, alpha in enumerate(BREAKS):
        elems = []
        _field = field.copy()
        _field[field <= alpha], _field[field > alpha]= 0., 1
        elems.append(ax.imshow(_field, 'gray', alpha=0.5, origin='lower', extent=bounds.flatten(),zorder=0))

        for s in K:
            for key, color, ord in [('maxext', 'purple', 6), ('minext','green', 5)]:
                if s(key) <= alpha:
                    elems += plot_cell(ax, K.P, s, color, ord)
        plt.pause(0.1)
        if save:
            fname = os.path.join(figdir, '%s%d.png' % (label, i))
            print('saving %s' % fname)
            plt.savefig(fname, dpi=300, transparent=True)
        else:
            input('%s: %0.2f' % label)
        for e in elems:
            e.remove()


if __name__ == '__main__':
    np.random.seed(0)

    SAVE = True
    FIGDIR = 'figures'

    WIDTH, HEIGHT = 1, 1
    BOUNDS = np.array([[-WIDTH, WIDTH], [-HEIGHT, HEIGHT]])

    _f, _l, _w = 2, 2, 1
    GRID_X, GRID_Y = get_grid(1024, WIDTH, HEIGHT)
    field = get_ripple(GRID_X, GRID_Y, _f, _l, _w, -3, False)# True)

    _LIPS = 11.8424 # ripple_lips(0, 1, 1024, _f, _l, _w)
    print(_LIPS)

    plt.ion()
    fig, ax = plt.subplots(1,1, figsize=(6,5))

    GRID_FUN = field.flatten()
    GRID_POINTS = np.vstack([GRID_X.flatten(), GRID_Y.flatten()]).T
    # print(lipschitz(GRID_FUN, GRID_POINTS)) # 5.045 #

    N_GRID = len(GRID_POINTS)
    N_P = 1000 # 2000 # N_GRID // 3
    N_Q = 500 # 1000 # N_P // 2

    I_P_GRID = sorted(random.sample(range(N_GRID), N_P))
    P = GRID_POINTS[I_P_GRID]

    I_Q_P = sorted(random.sample(range(N_P), N_Q))
    I_Q_GRID = [I_P_GRID[i] for i in I_Q_P]
    Q = GRID_POINTS[I_Q_GRID]
    Q_FUN = GRID_FUN[I_Q_GRID]

    # ax.scatter(P[:,0], P[:,1], c='black', s=1, zorder=1)
    # ax.scatter(Q[:,0], Q[:,1], c='red', s=2, zorder=2)

    D = DelaunayComplex(P, verbose=True)
    V = VoronoiComplex(D, verbose=True)

    P_FUN_max, P_FUN_min = lipschitz_extend(D, V, P, Q, Q_FUN, _LIPS)

    plot_breaks(ax, D, field, BOUNDS, max(P_FUN_max))

    # # plot_diagrams(ax[1], HF.diagram)
    # ax.axis('off')
    # plt.tight_layout()
    # ax.set_xlim(*BOUNDS[0]); ax.set_ylim(*BOUNDS[1])
    #
    # BREAKS = np.linspace(0, max(P_FUN_max), 11)[1:]
    #
    # if SAVE:
    #     if not os.path.exists(FIGDIR):
    #         os.mkdir(FIGDIR)
    #
    # for i, alpha in enumerate(BREAKS):
    #     elems = []
    #     _field = field.copy()
    #     _field[field <= alpha], _field[field > alpha]= 0., 1
    #     elems.append(ax.imshow(_field, 'gray', alpha=0.5, origin='lower', extent=BOUNDS.flatten(),zorder=0))
    #
    #     for s in D:
    #         for key, color, ord in [('maxext', 'purple', 6), ('minext','green', 5)]:
    #             if s(key) <= alpha:
    #                 elems += plot_cell(ax, P, s, color, ord)
    #     # for s in V:
    #     #     for key, color, ord in [('maxext', 'purple', 6), ('minext','green', 5)]:
    #     #         if s(key) <= alpha:
    #     #             elems.append(plot_cell(ax, V.P, s, color, ord))
    #     plt.pause(0.1)
    #     if SAVE:
    #         fname = os.path.join(FIGDIR, 'alpha%d.png' % i)
    #         print('saving %s' % fname)
    #         plt.savefig(fname, dpi=300, transparent=True)
    #     else:
    #         input('alpha = %0.2f' % alpha)
    #     for e in elems:
    #         e.remove()
