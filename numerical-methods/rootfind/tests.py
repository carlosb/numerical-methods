"""
Plots a function and calculates its roots.

In order to verify we actually find the roots, we have
to plot the graph and examine the function visually.
"""

import matplotlib.pyplot as plt
import numpy as np
import rootfindpack as rootfind


def f(x):
    return x ** 2 - 3 * x


def df(x):
    return 2 * x - 3


# PLOT
x = np.linspace(-1, 4)

plt.plot(x, f(x))
plt.xlabel('$-1 < x < 4$')
plt.ylabel('$f(x)$')

print 'Root is: ', rootfind.bisection(f, -1, 1.5, eps=0.000001)
print 'Root is: ', rootfind.newton(f, df, -1)
print 'Root is: ', rootfind.newton(f, df, 1.4)
print 'Root is: ', rootfind.secant(f, -1, 1.5, eps=0.000001)
print 'Root is: ', rootfind.inv_cuadratic_interp(f, -1, 1, 1.5)
print 'Root is: ', rootfind.lin_fracc_interp(f, -1, 1, 1.5)

plt.show()
