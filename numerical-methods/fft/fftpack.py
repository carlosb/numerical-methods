"""
Discrete Fourier Transform

Routines in this module:

fft(x)
ifft(F)
"""
import numpy as np

'''
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
Copyright (C) 4/24/17 Carlos Brito

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.*
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
'''

__all__ = ['fft', 'ifft']


def fft_r(x, y, n, w):
    """
    Recursive method for calculation of the fft.

    This method is a recursive implementation of the fft transform.

    Parameters
    ----------
    x : array_like
        This is the signal that will be transformed.
    y : array_like
        This is the transformed signal.
    n : int
        Number of entries in the signal.
    w : array_like
        Fourier divisor

    Returns
    -------
    y : array_like
        Transformed signal.

    """
    if n == 1:
        y[0] = x[0]
    else:
        p = np.zeros(n) + np.zeros(n) * 1j
        s = np.zeros(n) + np.zeros(n) * 1j

        for k in range(n / 2):
            p[k] = x[2 * k]
            s[k] = x[2 * k + 1]

        q = np.zeros(n / 2) + np.zeros(n / 2) * 1j
        t = np.zeros(n / 2) + np.zeros(n / 2) * 1j

        fft_r(p, q, n / 2, w ** 2)
        fft_r(s, t, n / 2, w ** 2)

        for k in range(n):
            y[k] = q[k % (n / 2)] + (w ** k) * t[k % (n / 2)]

    return y


def fft(x):
    """
    Calculates the fourier transform for a unidimensional
    sampled signal x.

    This method calculates the Discrete Fourier Transform of
    a signal x. It uses the Fast Fourier Transform method
    to make this computation.

    Parameters
    ----------
    x : array_like
        The signal.

    Returns
    -------
    F : array_like
        DFT of the signal x.

    """
    n = len(x)
    y = np.zeros(n) + np.zeros(n) * 1j
    w = np.cos(2 * np.pi / n) + np.sin(2 * np.pi / n) * 1j
    F = fft_r(x, y, n, w)
    return F


def ifft(F):
    """
    Calculates the inverse fourier transform for a signal
    transformed in the frequency spectrum.

    This method calculates the Inverse Fourier Transform of
    a transformed signal F.

    Parameters
    ----------
    F : array_like
        Signal F in the frequency spectrum.

    Returns
    -------
    f : array_like
        Inverse Fourier Transform of the signal F.

    """
    f = np.conj(fft(np.conj(F)))
    return f
