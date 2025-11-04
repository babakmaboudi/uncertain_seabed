import numpy as np
import matplotlib.pyplot as plt

import dolfin as dl
from wave import wave

def test():
    N_x=1024
    N_KL=256 
    problem = wave(N_x=512, N_KL=N_KL)

    data = np.load('./model_params/png_npz_params.npz')
    p = data['p']
    problem.compute_wave_speed( p )

    plt.figure()
    dl.plot(problem.c)
    plt.title('density')

    plt.figure()
    dl.plot(problem.rho)
    plt.title('elasticity')

    file = dl.File('rho.pvd')
    file << problem.rho

    file = dl.File('alpha.pvd')
    file << problem.c


    plt.show()

if __name__ == '__main__':
    test()

