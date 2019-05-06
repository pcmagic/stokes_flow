import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
# from sympy import *
from src import jeffery_model as jm

DoubleletStrength = np.array((1, 0, 0))
alpha = 1
B = np.array((0, 1, 0))
lbd = (alpha ** 2 - 1) / (alpha ** 2 + 1)
x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
problem = jm.SingleDoubleletJefferyProblem(B=B, DoubleletStrength=DoubleletStrength)
length = 200


def fun(zi):
    location = np.vstack((x.flatten(), y.flatten(), np.ones_like(y.flatten()) * zi))
    Jij = problem.J_matrix(location)
    JijT = Jij.transpose(1, 0, 2)
    Sij = 1 / 2 * (Jij + JijT)
    Oij = 1 / 2 * (Jij - JijT)
    Bij = Sij + lbd * Oij
    TrB2 = (Bij[0, 0, :] ** 2 + Bij[1, 1, :] ** 2 + Bij[2, 2, :] ** 2).reshape(x.shape)
    TrB3 = (Bij[0, 0, :] ** 3 + Bij[1, 1, :] ** 3 + Bij[2, 2, :] ** 3).reshape(x.shape)
    DtLine = TrB2 ** 3 - 6 * TrB3 ** 2
    return DtLine


fig, ax = plt.subplots()
p = [ax.contourf(x, y, np.log10(fun(1 / length)))]


def update(i):
    zi = (1 + i) / length * 2
    for tp in p[0].collections:
        tp.remove()
    p[0] = ax.contourf(x, y, np.log10(fun(zi)))
    return p[0].collections


ani = animation.FuncAnimation(fig, update, frames=length,
                              interval=5, blit=True, repeat=True)
# plt.show()

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=600)
ani.save('t2.mp4', writer=writer)
print('111')
