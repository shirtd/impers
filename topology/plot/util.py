import matplotlib.pyplot as plt
import pyvistaqt as pqt
import pyvista as pyv
import numpy as np
import os


COLORS = {  'edges' : '#f7f7f7', 'faces' : '#f4a582',
            'primal birth' : '#1a9641', 'primal death' : '#ca0020',
            'dual birth' : '#0571b0', 'dual death' : '#e66101'}

KWARGS = {'edges' : {'color' : COLORS['edges'], 'opacity' : 0.2},
            'faces' : {'color' : COLORS['faces'], 'opacity' : 0.03}}


def plot_complex(vfig, K, B=set(), **kwargs):
    kwargs = {**KWARGS, **kwargs}
    meshes = {'faces' : pyv.PolyData(K.P, faces=[l for s in K(2) for l in [len(s)] + list(s) if not s in B]),
                'edges' : pyv.PolyData(K.P, lines=[l for s in K(1) for l in [len(s)] + list(s) if not s in B])}
    return meshes, {k : vfig.add_mesh(v, **kwargs[k]) for k,v in meshes.items()}

def plot_cells(vfig, K, cells, dim, **kwargs):
    mesh = (pyv.PolyData(K.P, lines=[l for s in cells for l in [len(s)] + list(s)]) if dim == 1
        else pyv.PolyData(K.P, faces=[l for s in cells for l in [len(s)] + list(s)]) if dim == 2
        else pyv.PolyData(K.P[cells]))
    return mesh, vfig.add_mesh(mesh, **kwargs)

def get_lim(dgms):
    return max(d if d < np.inf else b for dgm in dgms for b,d in dgm)

def init_diagram(axis, lim):
    axis.plot([0, 1.2*lim], [0,1.2*lim], c='black', alpha=0.5, zorder=1)
    axis.plot([0, lim], [lim,lim], c='black', ls=':', alpha=0.5, zorder=1)
    axis.plot([lim, lim], [lim, 1.2*lim], c='black', ls=':', alpha=0.5, zorder=1)

def lim_dgm(dgm, lim):
    return np.array([[b, d if d < np.inf else 1.2*lim] for b,d in dgm])

def plot_diagrams(axis, dgms, lim=None, init=True):
    if lim is None:
        lim = get_lim(dgms)
    if init:
        init_diagram(axis, lim)
    elems = []
    for dim, dgm in enumerate(dgms):
        if len(dgm):
            d = lim_dgm(dgm, lim)
            elems += [axis.scatter(d[:,0], d[:,1], s=7, zorder=2, alpha=0.3, label='H%d' % dim)]
        else:
            elems += [axis.scatter([], [], s=7, zorder=2, alpha=0.3, label='H%d' % dim)]
    if init:
        axis.legend()
    return lim, elems

def save_plot(dir, prefix, name, dpi=500):
    fname = os.path.join(dir,'%s_%s.png' % (prefix, name))
    print('saving %s' % fname)
    plt.savefig(fname, dpi=dpi)

def plot_histo(axis, L, histo, name='', ymax=None, stat='count'):
    if ymax is not None:
        ax.set_ylim(0, ymax)
    for dim, h in enumerate(histo):
        axis.plot(L[1:] - 1 / len(L), h, label='H%d' % dim)
    fig.suptitle('TPers histogram %s' % name)
    ax.set_xlabel('TPers')
    ax.set_ylabel(stat)
    ax.legend()
    plt.tight_layout()
