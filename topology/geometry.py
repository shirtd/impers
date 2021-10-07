from topology.util import diff

from itertools import combinations
import numpy.linalg as la
import numpy as np


def tri_circumradius_2d(T):
    a,b,c = [la.norm(y - x) for x,y in combinations(T,2)]
    s = (a + b + c) / 2
    return (a * b * c) / (4 * np.sqrt(s*(a+b-s)*(a+c-s)*(b+c-s)))

def tri_circumcenter_2d(T):
    a, b, c = T
    ba, ca, cb = b-a, c-a, c-b
    u = cb / la.norm(cb)
    v = np.array([u[1], -u[0]])
    if np.dot(-ba,v) < 0:
        v *= -1
    R = tri_circumradius_2d(T)
    d = R * np.dot(ba, ca) / (la.norm(ba) * la.norm(ca))
    l = np.sqrt(R ** 2 - d ** 2)
    return b + u*l + v*d

def circumcenter(S):
    dim = S.shape[-2]-1
    return (tet_circumcenter(S) if dim == 3
        else tri_circumcenter(S) if dim == 2
        else edge_circumcenter(S) if dim == 1
        else S)

def tet_circumcenter(T):
    D = np.vstack(((T.T * T.T).sum(0)[None], T.T, np.ones(T.T.shape[1:])[None]))
    f = lambda ij: la.det(D[[0,ij[0],ij[1],-1]].T) * (-1) ** (ij[1]-ij[0]-1)
    c = (np.vstack(map(f, zip((2,1,1),(3,3,2)))) / (2 * la.det(D[1:].T))).T
    return c if T.ndim > 2 else c[0]

def tri_circumcenter(T):
    ca,ba = T.T[:,2] - T.T[:,0], T.T[:,1] - T.T[:,0]
    if T.shape[-1] == 3:
        baxca = np.cross(ba, ca, axis=0)
        xca = la.norm(ca, axis=0) ** 2 * np.cross(baxca, ba, axis=0)
        xba = la.norm(ba, axis=0) ** 2 * np.cross(-baxca, ca, axis=0)
        return (T.T[:,0] + (xca + xba) / (2 * la.norm(baxca, axis=0) ** 2)).T
    if T.ndim > 2:
        return np.array([tri_circumcenter_2d(t) for t in T])
    return tri_circumcenter_2d(T)

def edge_circumcenter(E):
    return E.T.sum(1).T / 2


def circumradius(S):
    dim = S.shape[-2]-1
    return (tet_circumradius(S) if dim == 3
        else tri_circumradius(S) if dim == 2
        else edge_circumradius(S) if dim == 1
        else 0.)

def tet_circumradius(T):
    if T.ndim > 2:
        return la.norm(T[:,0] - tet_circumcenter(T), axis=1)
    return la.norm(T[0] - tet_circumcenter(T))

def tri_circumradius(T):
    if T.shape[-1] == 3:
        return la.norm(T[:,0] - tri_circumcenter(T), axis=1)
    return la.norm(T[0] - tri_circumcenter(T))

def edge_circumradius(E):
    if E.ndim > 2:
        return la.norm(E.T[:,0] - E.T[:,1], axis=0) / 2
    return la.norm(diff(E)) / 2

# def torus(r1=0.5, r2=1, n=64):
#     r1, r2 = 0.5, 1
#     phi, theta = np.linspace(-np.pi, np.pi, n), np.linspace(-np.pi, np.pi, n)
#     x = (r1 * np.cos(phi) + r2) * np.cos(theta)
#     y = (r1 * np.cos(phi) + r2) * np.sin(theta)
#     z = r1 * np.sin(phi)
#     return np.vstack((x,y,z)).T
#
# def ripple(x, y, f=1, l=1, w=1):
#     t = la.norm(np.stack((x, y), axis=2), axis=2) + 1/12
#     t[t > 1] = 1.
#     return (1 - t) - np.exp(-t / l) * np.cos(2*np.pi*f*(1-t) * np.sin(2*np.pi*w*t))


# def tri_circumcenter(T):
#     ca, ba = T[2] - T[0], T[1] - T[0]
#     xca = la.norm(ca) ** 2 * np.cross(np.cross(ba, ca), ba)
#     xba = la.norm(ba) ** 2 * np.cross(np.cross(ca, ba), ca)
#     return T[0] + (xca + xba) / (2 * la.norm(np.cross(ba, ca)) ** 2)
#
# def tet_circumcenter(T):
#     alpha = np.array([[a[0], a[1], a[2], 1] for a in T])
#     Dx = np.array([[a[0]*a[0] + a[1]*a[1] + a[2]*a[2], a[1], a[2], 1] for a in T])
#     Dy = np.array([[a[0]*a[0] + a[1]*a[1] + a[2]*a[2], a[0], a[2], 1] for a in T])
#     Dz = np.array([[a[0]*a[0] + a[1]*a[1] + a[2]*a[2], a[0], a[1], 1] for a in T])
#     return np.array([la.det(Dx), -1*la.det(Dy), la.det(Dz)]) / (2 * la.det(alpha))
#
# def tri_circumradius_l(a, b, c):
#     return a*b*c / np.sqrt((a+b+c)*(b+c-a)*(c+a-b)*(a+b-c))
#
# def tri_circumradius(T):
#     return tri_circumradius_l(*map(la.norm, map(diff, combinations(T, 2))))
