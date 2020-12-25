"""
http://people.bu.edu/andasari/courses/stochasticmodeling/lecture5/stochasticlecture5.html
"""

import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

plt.style.use('bmh')


def brownian_motion(n_, t_, seed=random.randint(3, 10)):
    np.random.seed(seed)
    T = t_
    N = n_
    t = np.linspace(0, T, N)
    dt = T / (N)  # Time step

    # Preallocate arrays for efficiency:
    dX = [0] * N
    X = [0] * N

    # Initialization:
    dX[0] = np.sqrt(dt) * np.random.randn()  # Eq. (3)
    X[0] = dX[0]

    for i in range(1, N):
        dX[i] = np.sqrt(dt) * np.random.randn()  # Eq. (3)
        X[i] = X[i - 1] + dX[i]  # Eq. (4)

    return [t, X]


x = brownian_motion(10, 1)[0]
y = brownian_motion(10, 1)[1]

fig, ax = plt.subplots()
plt.xlabel('Time $t$', fontsize=14)
plt.ylabel('Random Variable $X(t)$', fontsize=14)
plt.title('1D Brownian Paths', fontsize=14)
plt.subplots_adjust(left=0.12, bottom=0.33, right=0.95, top=0.92)
p, = plt.plot(x, y, linewidth=1, color='green')

plt.axis([-0.1, 1.1, -1, 2.5])
axcolor = 'lightgoldenrodyellow'
axSlider1 = plt.axes([0.12, 0.2, 0.78, 0.02], facecolor=axcolor)
slider1 = Slider(axSlider1, valmin=2, valmax=10000, valstep=1, color='red', label='N')
plt.text(1.8, -5.6, 'N indica il numero di intervalli in cui è stato diviso il più grande intervallo [0,T]')
plt.text(1.8, -7, 'T=1 in questo caso')

axSlider2 = plt.axes([0.12, 0.15, 0.78, 0.02], facecolor=axcolor)
slider2 = Slider(axSlider2, valmin=3, valmax=8, valstep=1, color='yellow', label='seed')


def value_update(val):
    yval = brownian_motion(int(slider1.val), 1, int(slider2.val))[1]
    xval = brownian_motion(int(slider1.val), 1, int(slider2.val))[0]

    p.set_ydata(yval)
    p.set_xdata(xval)
    plt.draw()


slider1.on_changed(value_update)
slider2.on_changed(value_update)

plt.show()
