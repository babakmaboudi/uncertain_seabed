import numpy as np
from wave import wave
from wave import wave_speed_matern
import matplotlib.pyplot as plt

def save_obs():
    N_x=512

    fmT = np.array( [10, 25, 50, 75, 100] )
    obs = []

    for freq in fmT:
        problem = wave(N_x=N_x)
        problem.initiate_load_source_xdmf(freq)

        out = problem.forward(p)
        obs.append(out)

        f, ax = plt.subplots(1)
        ax.imshow(out)
        ax.set_xlabel('x')
        ax.set_ylabel('time')
        plt.savefig('./obs/obs_costum/freq_{}.pdf'.format(freq), format='pdf', dpi=300)

    obs_true = np.array(obs)
    noise_vec = np.random.standard_normal( obs_true.shape )
    for i in range( obs_true.shape[0] ):
        noise_vec[i] /= np.linalg.norm( noise_vec[i].flatten() )

    np.savez('./obs/obs_costum/obs.npz', N_KL=N_KL, N_x=N_x, obs_true=obs_true, noise_vec=noise_vec, param_true=p )

def read_obs():
    data = np.load('./obs/obs_costum/obs.npz')
    noise_vec = data['noise_vec']
    y_true = data['obs_true']

    SNR = 10
    y_obs = []
    for i in range(y_true.shape[0]):
        print(np.linalg.norm(noise_vec[i]))
        y_obs.append( y_true[i] + np.linalg.norm(y_true[i])/SNR*noise_vec[i] )

    y_obs = np.array(y_obs)

    for i in range(y_true.shape[0]):
        f, axes = plt.subplots(1,2)
        #print(y_obs[i].shape)
        im1 = axes[0].imshow(y_true[i])
        im2 = axes[1].imshow(y_obs[i])
        plt.colorbar(im1, ax=axes[0])
        plt.colorbar(im2, ax=axes[1])
        plt.savefig('noisy_{}.pdf'.format(i),dpi=300)


if __name__ == '__main__':
    #save_obs()
    #plot_seabed()
    read_obs()