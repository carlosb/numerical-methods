"""
Tests for one dimensional optimization methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import optimizationpack as opt


def f(x):
    """
    Example function.
    """
    return x**4 - 4 * x**2


def f1(x):
    """
    First derivative of f.
    """
    return 4 * x**3 - 8 * x


def f11(x):
    """
    Second derivative of f.
    """
    return 12 * x**2 - 8


# PLOT
x = np.linspace(-4, 4)
plt.plot(x, f(x))
plt.xlabel('$x$')
plt.ylabel('$f(x)$')

# FIND MIN
print 'Min is: ', opt.golden_section(f, -3, 3)
print 'Min is: ', opt.successive_parabolic_interp(f, -1, -2, -3, )
print 'Min is: ', opt.newton1(f1, f11, -3)

plt.show()
