"""
Plots a function and calculates its roots using different methods.

In order to verify we actually find the roots, we have
to plot the graph and examine the function visually.
"""

import matplotlib.pyplot as plt
import numpy as np
import rootfindpack as rootfind

plt.style.use('ggplot')


def f(x):
    return x ** 2 - 3 * x


def df(x):
    return 2 * x - 3


# PLOT
x = np.linspace(-1, 4)

plt.plot(x, f(x))
plt.xlabel('$-1 < x < 4$')
plt.ylabel('$f(x)$')

r1 = rootfind.bisection(f, -1, 1.5, eps=0.000001)
r2 = rootfind.bisection(f, 2, 3.5, eps=0.000001)

plt.plot(r1, 0, 'x', ms=20.0)
plt.plot(r2, 0, 'x', ms=20.0)

# plt.plot()

print 'Root is: ', rootfind.bisection(f, -1, 1.5, eps=0.000001)
print 'Root is: ', rootfind.bisection(f, 2, 3.5, eps=0.000001)
print 'Root is: ', rootfind.newton(f, df, -1)
print 'Root is: ', rootfind.newton(f, df, 1.4)
print 'Root is: ', rootfind.secant(f, -1, 1.5, eps=0.000001)
print 'Root is: ', rootfind.inv_cuadratic_interp(f, -1, 1, 1.5)
print 'Root is: ', rootfind.lin_fracc_interp(f, -1, 1, 1.5)

plt.show()
