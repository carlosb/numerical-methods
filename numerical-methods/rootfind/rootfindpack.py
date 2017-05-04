"""
Root finding methods
====================

Routines in this module:

bisection(f, a, b, eps=1e-5)
newton1(f, df, eps=1e-5)
newtonn(f, J, x0, eps=1e-5)
secant(f, x0, x1, eps=1e-5)
inv_cuadratic_interp(f, a, b, c, eps=1e-5)
lin_fracc_interp(f, a, b, c, eps=1e-5)
broyden(f, x0, B0, eps=1e-5)
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

__all__ = ['bisection', 'newton1', 'secant', 'newtonn',
           'inv_cuadratic_interp', 'lin_fracc_interp']


def bisection(f, a, b, eps=1e-5, display=False):
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
    iterations : int
        Number of iterations taken to find root.
    """
    iterations = 0

    if a > b:
        a, b = b, a

    while((b - a) > eps):
        m = a + np.float32(b - a) / 2.

        if (np.sign(f(a)) == np.sign(f(m))):
            a = m
        else:
            b = m

        if display:
            print 'iteration ', iterations
            print 'm: ', m
        iterations += 1

    return m, iterations


def newton1(f, df, x0, eps=1e-5, display=False):
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
    iterations = 0

    x_old = np.float(x0)
    x_new = x_old

    while(True):
        try:
            x_old = x_new
            x_new = x_old - f(x_old) / df(x_old)

            if display:
                print 'iteration ', iterations
                print 'x: ', x_new
            iterations += 1

            if(abs(x_old - x_new) <= eps):
                break

        except(ZeroDivisionError):
            return np.nan

    root = x_new
    return root, iterations


def secant(f, x0, x1, eps=1e-5, display=False):
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
    iterations : int
        Number of iterations taken to find root.

    """
    iterations = 0

    x_old_0 = x0
    x_old_1 = x1

    x_new = x0 - f(x0) * (x1 - x0) / (f(x1) - f(x0))

    while True:
        x_old_0 = x_old_1
        x_old_1 = x_new
        x_new = x_old_1 - f(x_old_1) * \
            ((x_old_1 - x_old_0) / (f(x_old_1) - f(x_old_0)))

        if display:
            print 'iteration ', iterations
            print 'x: ', x_new
        iterations += 1

        if(abs(x_old_1 - x_new) < eps):
            break

    root = x_new
    return root, iterations


def inv_cuadratic_interp(f, a, b, c, eps=1e-5, display=False):
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
    iterations : int
        Number of iterations taken to find root.
    """
    iterations = 0

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

        if display:
            print 'iteration ', iterations
            print 'x: ', x_new
        iterations += 1

        if(abs(f(x_new)) < eps):
            break

    root = x_new
    return root, iterations


def lin_fracc_interp(f, a, b, c, eps=1e-5, display=False):
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
    iterations : int
        Number of iterations taken to find root.
    """
    iterations = 0

    while True:
        numerator = (a - c) * (b - c) * (f(a) - f(b)) * f(c)
        denominator = (a - c) * (f(c) - f(b)) * f(a) - \
            (b - c) * (f(c) - f(a)) * f(b)

        h = numerator / denominator

        x_new = c + h

        a = b
        b = c
        c = x_new

        if display:
            print 'iteration ', iterations
            print 'x: ', x_new
        iterations += 1

        if(abs(f(x_new)) < eps):
            break

    root = x_new
    return root, iterations


def broyden(f, x0, B0, eps=1e-5, display=False):
    """
    Finds roots for functions of k-variables.

    This function utilizes Broyden's method to find roots in a
    k-dimensional function f utilizing the initial Jacobian B0
    at x0.

    Parameters
    ----------
    f : function which takes an array_like matrix and
    returns an array_like matrix
        Function we want to find the root of.
    x0 : array_like
        Initial point.
    B0 : array_like
        Jacobian of function at x0.
    eps : float
        Error tolerance.

    Returns
    -------
    root : array_like
        Root of function.
    iterations : int
        Number of iterations taken to find root.
    """
    iterations = 0

    x_new = x0
    B_new = B0
    while True:
        x_old = x_new
        B_old = B_new

        s = np.dot(np.linalg.inv(B_old), -f(x_old).T)  # solve for s
        x_new = x_old + s

        y = f(x_new) - f(x_old)

        B_new = B_old + (np.dot((y - np.dot(B_old, s)), s.T)
                         ) / (np.dot(s.T, s))

        if display:
            print 'iteration ', iterations
            print 'x:', x_new
            print 'B', B_new
        iterations += 1

        if(np.all(np.abs(x_old - x_new) <= eps)):
            break

    root = x_new
    return root, iterations


def newtonn(f, J, x0, eps=1e-5, display=False):
    """
    Finds roots for functions of k-variables.

    This function utilizes Newton's method for root finding
    to find roots in a k-dimensional function. To do this,
    it takes the Jacobian of the function and an initial
    point.

    Parameters
    ----------
    f : function which takes an array_like matrix and
    returns an array_like matrix
    J : function returning an array_like matrix
        Jacobian of function.
    x0 : array_like
        Initial point.
    eps : float
        Error tolerance.

    Returns
    -------
    root : array_like
        Root of function.
    iterations : int
        Number of iterations taken to find root.
    """
    iterations = 0

    x_old = x0
    x_new = x_old

    try:
        while True:
            x_old = x_new
            x_new = x_old - np.dot(np.linalg.inv(J(x0)), f(x_old))

            if display:
                print 'iteration ', iterations
                print 'x: ', x_new
            iterations += 1

            if(np.all(np.abs(x_old - x_new) <= eps)):
                break

    except np.linalg.LinAlgError:
        print 'Error during iteration. Matrix is probably singular'
        return None

    root = x_new
    return root, iterations
