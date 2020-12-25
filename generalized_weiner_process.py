
#librerie necessarie al funzionamento:

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

plt.style.use('bmh')


def generalized_wiener_process(n_, t_, a, b,seed=6):
    np.random.seed(seed)
    T = t_
    N = n_
    t = np.linspace(0, T, N)
    dt = T / N

    # inizializzo il vettore delle variazioni della variabile X nel tempo dt
    dX = [0] * N
    X = [0] * N
    dz=[0]*N
    Z=[0]*N

#inizializzo il primo valore del vettore delle variazioni come 0
    dX[0] = 0
    X[0] = dX[0]

    for i in range(1, N):
        # la forma del moto browniano è dx = b * radq(dt) * epsilon + mu*dt
        # con epsilon una variabile casuale con distribuzione normale standard (media 0 e varianza 1)

        #per epsilon ho usato la libreria numpy che consente di fare estrazioni di numeri casuali!
        dX[i] = b * np.sqrt(dt) * np.random.randn() + a * dt
        X[i] = X[i - 1] + dX[i]


    processo=X
    return [t, processo, a]




def tendenza(y1_):
    y1 = y1_
    lx = [0, 1]
    ly = [0, y1]
    return [lx, ly]



############################
# VISUALIZZAZIONE DEL MOTO #
############################


x = generalized_wiener_process(10, 1, 0, 1)[0]
y = generalized_wiener_process(10, 1, 0, 1)[1]

x_ = tendenza(1)[0]
y_ = tendenza(1)[1]

plt.xlabel('Time $t$', fontsize=14)
plt.ylabel('Random Variable $X(t)$', fontsize=14)
plt.title('1D Brownian Paths', fontsize=14)
plt.subplots_adjust(left=0.12, bottom=0.33, right=0.95, top=0.92)

plt.axis([-0.1, 1.1, -1, 3.5])
axcolor = 'lightgoldenrodyellow'

p, = plt.plot(x, y, linewidth=1, color='green', label='Wiener process dx= a*dt + b*dz')
c, = plt.plot(x_, y_, linewidth=1, color='red', linestyle=':', marker='o', label='dx=a*dt')

leg = plt.legend()

axSlider1 = plt.axes([0.12, 0.2, 0.78, 0.02], facecolor=axcolor)
slider1 = Slider(axSlider1, valmin=2, valmax=10000, valstep=1, color='red', label='N')
axSlider2 = plt.axes([0.12, 0.15, 0.78, 0.02], facecolor=axcolor)
slider2 = Slider(axSlider2, valmin=0, valmax=5000, valstep=1, color='yellow', label='a')
axSlider3 = plt.axes([0.12, 0.10, 0.78, 0.02], facecolor=axcolor)
slider3 = Slider(axSlider3, valmin=0, valmax=100, valstep=1, color='blue', label='b')

plt.text(-4, -3, 'N numero di intervalli, mentre a indica l inclinazione del processo')
plt.text(-4, -4.5, 'a è il drift rate (/1000), b (/10) è il coeficiente di dz')


def value_update(val):
    yval = generalized_wiener_process(int(slider1.val), 1, int(slider2.val) / 1000, int(slider3.val) / 10)[1]
    xval = generalized_wiener_process(int(slider1.val), 1, int(slider2.val) / 1000, int(slider3.val) / 10)[0]
    yval2 = tendenza(int(slider2.val) / 1000)[1]
    p.set_ydata(yval)
    p.set_xdata(xval)
    c.set_ydata(yval2)
    plt.draw()


slider1.on_changed(value_update)
slider2.on_changed(value_update)
slider3.on_changed(value_update)

plt.show()
