from topology.util import lipschitz

import numpy.linalg as la
import pickle as pkl
import numpy as np

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

def ripple(t, f=1, l=1, d=1, w=1, s=1):
    t = w * (t + s)
    return np.exp(-t / l) * np.cos(2*np.pi*f**(1/d)*t)

def ripple_grid(x, y, f=1, l=1, d=1, w=1, s=1):
    return ripple(la.norm(np.stack((x, y), axis=2), axis=2), f, l, d, w, s)

def ripple_lips(a=0, b=1, n=1024, f=1, l=1, d=1, w=1, s=1):
    t = np.linspace(a, b, n)
    f = ripple(t, f, l, d, w, s)
    return lipschitz(f, t)

def get_grid(res=1024, width=1, height=1):
    x_rng = np.linspace(-width,width,int(width*res))
    y_rng = np.linspace(-height,height,int(height*res))
    return np.meshgrid(x_rng, y_rng)

def get_ripple(x, y, f=1, l=1, d=1, w=1, s=1, exp=-3, noise=False, scale=False):
    f = ripple_grid(x, y, f, l, d, w, s)
    if noise:
        f *= (1 + grf(exp, x.shape[0]*x.shape[1]))
    return (f - f.min()) / (f.max() - f.min()) if scale else f

def get_circle(x, y, n=128, r=[0.2, 1.], c=[[0,0],[0,0]]):
    xy = np.stack((x,y), axis=2)
    t = np.linspace(0, 2*np.pi, n)
    z = np.vstack([np.vstack((rr*np.sin(t), rr*np.cos(t))).T + np.array(cc) for rr,cc in zip(r,c)])
    return np.array([[la.norm(z - xy[i,j], axis=1).min() for j in range(xy.shape[1])] for i in range(xy.shape[0])])
