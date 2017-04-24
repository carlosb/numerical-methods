import matplotlib.pyplot as plt
import numpy as np
from fftpack import *

plt.style.use('ggplot')

# DEFINE SIGNAL
def f(x):
    return np.sin(np.pi*x) + np.cos(2*np.pi*x)

# SAMPLE
nyq_freq = 2.
x = np.arange( 0, 4, float(1.)/(8. * nyq_freq) )
n = len(x)
fx = f(x)

# FFT
F = fft(fx)/n
F_half = F[0:n/2]

# PLOT SIGNAL
plt.figure()
plt.plot(x,fx)
plt.xlabel('$time$')
plt.ylabel('$f(x)$')

# PLOT FFT
plt.figure()
plt.xlabel('$freq$')
plt.ylabel('$\mathcal{F}\{f\}$')
plt.plot(  np.arange(0, (n/2)/float(4), 1/float(4)), abs(F_half))

# IFFT
iF = ifft(F)/n

# PLOT IFFT
plt.figure()
plt.plot(x,iF)
plt.xlabel('$time$')
plt.ylabel('$f(x)$')
plt.show()