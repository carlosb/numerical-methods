"""
Optimization Methods
====================

Routines in this module:

golden_section(a, b, f, eps=1e-5)
successive_parabolic_interp(u, v, w, f, eps=1e-5)
newton1(x0, g, h, eps=1e-5)
newtonn(x0, g, h, eps=1e-5)
nelder_mead(xs0, n, f, eps=1e-5, alpha=1., beta=2., gamma=0.5, sigma=0.5)
bfgs(x0, B0, f, gf, eps=1e-5)
gradient_descent(x0, f, g, eps=1e-5, rho=0.5, c=0.1, alpha=1e-3)

TODO
====
- Add line search to BFGS
- Add Conjugate Gradient method

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

__all__ = ['golden_section',
           'successive_parabolic_interp',
           'newton1',
           'newtonn',
           'gradient_descent',
           'bfgs']


def golden_section(a, b, f, eps=1e-5, display=False):
    """
    Finds a minimum of a function in the interval [a,b].

    This function finds a minimum in the interval [a,b] for a
    one dimensional function f using the Golden Section method.

    Parameters
    ----------
    a : float
        Lower bound of interval.
    b : float
        Higher bound of interval.
    f : function
        Function to be minimized.
    eps : float
        Error tolerance.

    Returns
    -------
    x_ : float
            A local minimum in the interval [a,b].
    iterations : int
            Number of iterations taken to reach minimum.

    """
    iterations = 0

    tau = (np.sqrt(5.) - 1.) / 2.

    x1 = a + (1 - tau) * (b - a)
    f1 = f(x1)

    x2 = a + tau * (b - a)
    f2 = f(x2)

    while abs(b - a) > eps:

        if display:
            print 'iteration ', iterations
            print 'x1: ', x1, 'x2: ', x2
        iterations += 1

        if(f1 > f2):
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + tau * (b - a)
            f2 = f(x2)
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (1 - tau) * (b - a)
            f1 = f(x1)

    x_ = (x1 + x2) / 2.
    return x_, iterations


def successive_parabolic_interp(u, v, w, f, eps=1e-5, display=False):
    """
    Finds the a minimum with initial values u, v, w close to the minimum.

    This function uses Successive Parabolic Interpolation with three given
    initial values u,v,w of a one dimensional function f.

    Parameters
    ----------
    u : float
        First initial value.
    v : float
        Second initial value.
    w : float
        Third initial value.
    f : function
        Function to be minimized
    eps : float
        Error tolerance.

    Returns
    -------
    x_ : float
        Local minimum of f.
    iterations : int
        Number of iterations taken to reach minimum.
    Notes
    -----
    This method will order the values such that u < v < w.

    """
    iterations = 0

    u, v, w = sorted([u, v, w])
    while True:
        p = (f(w) - f(v)) * (w - u)**2 - (f(w) - f(u)) * (w - v)**2
        q = 2. * ((f(w) - f(v)) * (w - u) - (f(w) - f(u)) * (w - v))

        x = w - (p / float(q))

        if display:
            print 'iteration ', iterations
            print 'x:', x
        iterations += 1

        if(abs(x - w) <= eps):
            break

        u = v
        v = w
        w = x

    x_ = w
    return x_, iterations


def newton1(x0, g, h, eps=1e-5, display=False):
    """
    Minimizes a one dimensional function with first derivative g
    and second derivative h.

    This method minimizes a one dimensional function f with Newton's method
    such that its first derivative is given by g and second derivaitve is
    given by h.

    Parameters
    ----------
    x0 : float
        Initial value.
    g : function
        First derivative of a function f.
    h : function
        Second derivative of a function f.
    eps : float
        Error tolerance.

    Returns
    -------
    x_ : float
        Local minimum of the function f with first and second derivatives g
        and h.
    iterations : int
        Number of iterations taken to reach minimum.
    """
    iterations = 0

    x_new = x0
    while True:
        x_old = x_new
        x_new = x_old - g(x_old) / float(h(x_old))

        if display:
            print 'iteration ', iterations
            print 'x: ', x_new
        iterations += 1

        if(abs(x_old - x_new) <= eps):
            break

    x_ = x_new
    return x_, iterations


def newtonn(x0, g, h, eps=1e-5, display=False):
    """
    Minimizes an n-dimensional function with gradient g and hessian h.

    This function will find a local minimum given the gradient g and
    hessian h of a function f.

    Parameters
    ----------
    x0 : array_like
        Initial point.
    g : function returning array_like matrix
        Gradient of a function f.
    h : function returning array_like matrix
        Hessian of a function f.

    Returns
    -------
    x_ : array_like
            Local minimum of the function f with gradient and hessian g and h.
    iterations : int
            Number of iterations taken to reach minimum.
    """
    iterations = 0

    x_new = np.array(x0, copy=True)
    while True:
        x_old = x_new
        x_new = x_old - np.dot(np.linalg.inv(h(x_old)), g(x_old))

        if display:
            print 'iteration ', iterations
            print 'x: ', x_new
        iterations += 1

        if((np.abs(x_old - x_new) < eps).all()):
            break

    x_ = x_new
    return x_, iterations


def nelder_mead(xs0, n, f, eps=1e-5,
                alpha=1., beta=2., gamma=0.5,
                sigma=0.5, display=False):
    """
    Minimizes an n-dimensional function f by passing n+1 initial values.

    This function will find a local minimum of the function f by taking as
    parameters the dimension n and n+1 initial values using Nelder-Mead's
    method.

    Parameters
    ----------
    xs0 : array_like
        Initial values which are stored in an array of array_like points.
    n : int
        Dimension of function.
    f : function
        Function to minimize.
    eps : float
        Error tolerance.
    alpha : float
        Reflection coefficient.
    beta : float
        Expansion coefficient.
    gamma : float
        Contraction coefficient.
    sigma : float
        Shrink coefficient.

    Returns
    -------
    x_ : array_like
        Local minimum of f.
    iterations : int
        Number of iterations taken to reach minimum.
    """
    iterations = 0

    xs = np.array(xs0)

    n += 1

    if(len(xs0) != n):
        raise Exception('n+1 != len( xs0 )')

    x_new = np.zeros(n - 1)
    while True:
        # order
        ps_l = []
        for x in xs:
            ps_l.append((x, f(x)))
        ps_l = np.array(sorted(ps_l, key=lambda tu: tu[1]))
        xs = ps_l[:, 0]

        # display information
        if display:
            print 'iteration ', iterations
            print 'x: ', x_new
        iterations += 1

        # check for convergence
        x_old = x_new
        x_new = xs[n - 1]
        if(np.all(np.abs(x_new - x_old) <= eps)):
            return xs[0]

        # centroid
        centroid = 0.0
        for i in range(n - 1):
            centroid += (xs[i] / (n - 1))

        # reflection
        xr = centroid + alpha * (centroid - xs[n - 1])
        if((f(xs[0]) <= f(xr)) and (f(xr) < f(xs[n - 2]))):
            xs[n - 1] = xr
            continue

        # expansion
        if(f(xr) < f(xs[0])):
            xe = centroid + beta * (xr - centroid)

            if(f(xe) < f(xr)):
                xs[n - 1] = xe
            else:
                xs[n - 1] = xr
            continue

        # contraction (the following if is redundant)
        if(f(xr) >= f(xs[n - 2])):
            xc = centroid + gamma * (xs[n - 1] - centroid)

            if(f(xc) < f(xs[n - 1])):
                xs[n - 1] = xc
                continue

        # shrink
        for j in range(1, n):
            xs[i] = xs[0] + sigma * (xs[i] - xs[0])

    x_ = xs[0]  # return best point
    return x_, iterations


def backtrack(alpha, rho, c, x, f, grad_f, p):
    x = np.array(x)
    p = np.array(p)

    lhs = f(x + alpha * p)
    rhs = f(x) + c * alpha * grad_f(x) * p

    while((lhs > rhs).all()):
        alpha = rho * alpha
        lhs = f(x + alpha * p)
        rhs = f(x) + c * alpha * grad_f(x) * p

    return alpha


def gradient_descent(x0, f, g, eps=1e-5, rho=0.5,
                     c=0.1, alpha=1e-5, display=False):
    """
    Minimizes an n-dimensional function given the function f and its gradient.

    This function finds a local minimum for a function f, given a starting
    point x0 and the function's gradient g. This method utilizes backtracking
    gradient descent to perform a line search at each iteration and adjust
    the line search parameter.

    Parameters
    ----------
    x0 : array_like
        Starting point.
    f : function
        Function to minimize.
    g : function returning an array_like matrix
        Gradient of f.
    eps : float
        Error tolerance.
    rho : float
        Search control parameter.
    c : float
        Search control parameter.
    alpha : float
        Maximum candidate step size value [1].

    Returns
    -------
    x_ : array_like
        Local minimum of f.

    References
    ----------

    [1] Dennis, J. E.; Schnabel, R. B. (1996). Numerical Methods for
     Unconstrained Optimization and Nonlinear Equations. Philadelphia:
      SIAM Publications. ISBN 978-0-898713-64-0.

    """
    iterations = 0

    x_new = x0
    while(True):
        x_old = x_new
        alpha = backtrack(alpha, rho, c, x_old, f, g, -g(x_old))
        x_new = x_old - g(x_old) * alpha

        if display:
            print 'iteration ', iterations
            print 'x: ', x_new
        iterations += 1

        if((np.abs(x_old - x_new) < eps).all()):
            break

    x_ = x_new
    return x_, iterations


def bfgs(x0, B0, f, gf, eps=1e-5, display=False):
    """
    Minimizes an n-dimensional function given the approximation for
    the hessian at x0, the function f and its gradient gf.

    This function finds a local minimum of a function f with
    gradient gf, starting point x0 and an approximation B0 of
    the hessian B(x0).

    Parameters
    ----------
    x0 : array_like
        Starting point.
    B0 : array_like
        Hessian approximation at x0.
    f : function
        Function to minimize
    gf : function returning array_like matrix
        Gradient of function.
    eps : float
        Error tolerance.

    Returns
    -------
    x_ : array_like
        Local minimum of function.
    iterations : int
        Number of iterations taken to reach minimum.
    """
    iterations = 0

    x_old = np.array(x0)
    B_old = B0

    while True:
        s = np.dot(np.linalg.inv(B_old), - gf(x_old))
        x_new = (x_old + s.T).flatten()
        y = gf(x_new) - gf(x_old)
        B_new = B_old + np.dot(y, y.T) / np.dot(y.T, s) \
            - np.dot(B_old, np.dot(s, np.dot(s.T, B_old))) / \
            np.dot(s.T, np.dot(B_old, s))

        if display:
            print 'iteration ', iterations
            print 'x: ', x_new
        iterations += 1

        if((np.abs(x_new - x_old) <= eps).all()):
            break

        x_old = x_new
        B_old = B_new

    x_ = x_new
    return x_, iterations
