"""
Solve methods
=============
Description not yet added.

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

__all__ = ['newton', 'broyden', 'damped_newton', 'fit']


def newton(f, J, x0, eps=1e-5, display=False):
    """
    Finds roots for functions of k-variables.
    This function utilizes Newton's method to find roots in a
    k-dimensional function f utilizing the jacobian.

    Parameters
    ----------
    f : function which takes an array_like matrix and
    returns an array_like matrix
        Function we want to find the root of.
    x0 : array_like
        Initial point.
    J : function returning array_like matrix
        Jacobian of function at x0.
    eps : float
        Error tolerance.
    Returns
    -------
    root : array_like
        Root of function.
    iterations : int
        Number of iterations taken to reach minimum.
    """
    x_new = x0
    iterations = 0
    try:
        while True:
            x_old = x_new
            x_new = x_old - np.dot(np.linalg.inv(J(x_old)), f(x_old))

            if display:
                print 'iterations: ', iterations
                print 'x: ', x_new
            iterations += 1

            if(np.all(np.abs(x_old - x_new) <= eps)):
                break

    except np.linalg.LinAlgError:
        print 'Error during iteration. Matrix is probably singular'
        return None

    root = x_new
    return root, iterations


def damped_newton(f, J, x0, eps=1e-5, display=False):
    """
    Finds roots for functions of k-variables.
    This function utilizes Newton's method to find roots in a
    k-dimensional function f utilizing the jacobian.

    Parameters
    ----------
    f : function which takes an array_like matrix and
    returns an array_like matrix
        Function we want to find the root of.
    x0 : array_like
        Initial point.
    J : function returning array_like matrix
        Jacobian of function at x0.
    eps : float
        Error tolerance.
    Returns
    -------
    root : array_like
        Root of function.
    iterations : int
        Number of iterations taken to reach minimum.
    """
    iterations = 0
    x_new = x0

    try:
        while True:
            x_old = x_new
            t = 1. / (1. + np.linalg.norm(x_new))
            x_new = x_old - t * np.dot(np.linalg.inv(J(x_old)), f(x_old))

            if display:
                print 'iterations: ', iterations
                print 'x: ', x_new
            iterations += 1

            if(np.all(np.abs(x_old - x_new) <= eps)):
                break

    except np.linalg.LinAlgError:
        print 'Error during iteration. Matrix is probably singular'
        return None

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
    """
    iterations = 0
    x_new = x0
    B_new = B0
    while True:
        x_old = x_new
        B_old = B_new

        if display:
            print 'iteration: ', iterations
            print 'x: ', x_new
        iterations += 1

        s = np.dot(np.linalg.inv(B_old), -f(x_old))  # solve for s

        x_new = x_old + s

        y = f(x_new) - f(x_old)

        B_new = B_old + (np.dot((y - np.dot(B_old, s)), s.T)
                         ) / (np.dot(s.T, s))

        if(np.all(np.abs(x_old - x_new) <= eps)):
            break

    root = x_new
    return root, iterations


def fit(f, grad_f, x0, X, y, eps=1e-5):
    old_x = np.array(x0)

    while True:

        r = y - f(X, old_x)

        jacobian = grad_f(X, old_x).T

        s = np.dot(np.linalg.pinv(jacobian), r)

        new_x = old_x + s

        if((np.abs(old_x - new_x) < eps).all()):
            break

        old_x = new_x
    return new_x
