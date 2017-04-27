"""
Root finding methods
====================

Routines in this module:

bisection(f, a, b, eps=1e-5)
newton(f, df, eps=1e-5)
secant(f, x0, x1, eps=1e-5)
inv_cuadratic_interp(f, a, b, c, eps=1e-5)
lin_fracc_interp(f, a, b, c, eps=1e-5)
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

__all__ = ['bisection', 'newton', 'secant']


def bisection(f, a, b, eps=1e-5):
    """
    Find root of f.

    This function computes a root of the function f using the bisection method.

    Parameters
    ----------
    f : function
        Function we want to find the root of.
    a : float
        Lower bound.
    b : float
        High bound.
    eps : float
        Tolerance.

    Returns
    -------
    m : float
        Root of f.

    """
    if a > b:
        a, b = b, a

    while((b - a) > eps):
        m = a + np.float32(b - a) / 2.

        if (np.sign(f(a)) == np.sign(f(m))):
            a = m
        else:
            b = m

    return m


def newton(f, df, x0, eps=1e-5):
    """
    Find root of f.

    This method computes the root of f using Newton's method.

    Parameters
    ----------
    f : function
        Function we want to find the root of.
    df : function
        Derivative of f.
    x0 : float
        This is the starting point for the method.
    eps : float
        Tolerance.

    Returns
    -------
    root : float
        Root of f.

    """
    x_old = np.float(x0)
    x_new = x_old

    while(True):
        try:
            x_old = x_new
            x_new = x_old - f(x_old) / df(x_old)

            if(abs(x_old - x_new) <= eps):
                break

        except(ZeroDivisionError):
            return np.nan

    root = x_new
    return root


def secant(f, x0, x1, eps=1e-5):
    """
    Parameters
    ----------
    f : function
        Function we want to find the root of.
    x0 : float
        First initial value "close" to the root of f.
    x1: float
        Second initial value "close" to the root of f.
    eps : float
        Tolerance.

    Returns
    -------
    root : float
        Root of f.

    """
    x_old_0 = x0
    x_old_1 = x1

    x_new = x0 - f(x0) * (x1 - x0) / (f(x1) - f(x0))

    while True:
        x_old_0 = x_old_1
        x_old_1 = x_new
        x_new = x_old_1 - f(x_old_1) * \
            ((x_old_1 - x_old_0) / (f(x_old_1) - f(x_old_0)))

        if(abs(x_old_1 - x_new) < eps):
            break

    root = x_new
    return root


def inv_cuadratic_interp(f, a, b, c, eps=1e-5):
    """
    Find root of f.

    This method finds the root of f using the inverse cuadratic
    interpolation method.

    Parameters
    ----------
    f : function
        Function we want to find the root of.
    a : float
        First initial value.
    b : float
        Second initial value.
    c : float
        Third initial value.

    Returns
    -------
    root : float
        Root of f.

    """
    while True:
        u = f(b) / f(c)
        v = f(b) / f(a)
        w = f(a) / f(c)

        p = v * (w * (u - w) * (c - b) - (1 - u) * (b - a))
        q = (w - 1) * (u - 1) * (v - 1)

        x_new = b + p / q

        a = b
        b = c
        c = x_new

        if(abs(f(x_new)) < eps):
            break

    root = x_new
    return root


def lin_fracc_interp(f, a, b, c, eps=1e-5):
    """
    Find root of f.

    This method finds the root of f using the linear fractional
    interpolation method.

    Parameters
    ----------
    f : function
        Function we want to find the root of.
    a : float
        First initial value.
    b : float
        Second initial value.
    c : float
        Third initial value.

    Returns
    -------
    root : float
        Root of f.
    """
    while True:
        numerator = (a - c) * (b - c) * (f(a) - f(b)) * f(c)
        denominator = (a - c) * (f(c) - f(b)) * f(a) - \
            (b - c) * (f(c) - f(a)) * f(b)

        h = numerator / denominator

        x_new = c + h

        a = b
        b = c
        c = x_new

        if(abs(f(x_new)) < eps):
            break

    root = x_new
    return root
