"""
Tests for n-dimensional optimization methods.
"""

import numpy as np
import optimizationpack as opt


def f(x):
    x = np.array(x)
    return 100 * (x[1] - x[0]**2)**2 + (2 - x[0])**2


def grad_f(x):
    x = np.array(x)
    a = np.zeros((2))
    a[0] = -400 * x[0] * x[1] + 400 * (x[0]**3) + 2 * x[0] - 4
    a[1] = 200 * x[1] - 200 * (x[0]**2)
    return a


def hessian_f(x):
    x = np.array(x)
    a = np.zeros((2, 2))
    a[0, 0] = -400 * x[1] + 1200 * (x[0]**2) + 2
    a[0, 1] = -400 * x[0]
    a[1, 0] = a[0, 1]
    a[1, 1] = 200
    return a


xs0 = np.array([
    [-1.2, 1],
    [-0.6, 1],
    [-1.2, 0.4]
])

x0 = np.array([-1, 5])

print 'Minimum: ', opt.nelder_mead(f, xs0, 2)
print 'Minimum:', opt.newtonn(grad_f, hessian_f, x0)
print 'Minimum: ', opt.gradient_descent(f, grad_f, x0)
print 'Minimum: ', opt.bfgs(f, grad_f, x0, hessian_f(x0))
