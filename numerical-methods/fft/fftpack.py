"""
Discrete Fourier Transform

@file fftpack.py
@brief Contains the routines to calculate different FFT
"""
import numpy as np

__all__ = ['fft', 'ifft']

def fft_r(x, y, n, w):
    if n == 1:
        y[0] = x[0]
    else:
        p = np.zeros(n) + np.zeros(n) * 1j
        s = np.zeros(n) + np.zeros(n) * 1j
        
        for k in range(n/2):
            p[k] = x[2*k]
            s[k] = x[2*k + 1]
        
        q = np.zeros(n/2) + np.zeros(n/2) * 1j
        t = np.zeros(n/2) + np.zeros(n/2) * 1j
        
        fft_r(p, q, n/2, w**2)
        fft_r(s, t, n/2, w**2)
        
        for k in range(n):
            y[k] = q[k % (n/2)] + (w**k) * t[k % (n/2)]
            
    return y


def fft(x):
	PI = np.pi
	n = len(x)
	y = np.zeros(n) + np.zeros(n) * 1j
	w = np.cos(2*np.pi/n) + np.sin(2*np.pi/n) * 1j
	return fft_r(x,y,n,w)

def ifft(F):
    f = np.conj( fft( np.conj(F) ) )
    return f