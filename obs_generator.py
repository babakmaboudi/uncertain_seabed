import numpy as np
from wave import wave
import matplotlib.pyplot as plt

def save_obs():
    N_x=512
    N_KL=256 

    fmT = np.array( [10, 25, 50, 75, 100] )
    obs = []

    p = np.random.standard_normal(N_KL)

    for freq in fmT:
        problem = wave(N_x=N_x, N_KL=N_KL)
        problem.initiate_load_source_xdmf(freq)

        out = problem.forward(p)
        obs.append(out)

        f, ax = plt.subplots(1)
        ax.imshow(out)
        ax.set_xlabel('x')
        ax.set_ylabel('time')
        plt.savefig('./obs/obs1/freq_{}.pdf'.format(freq), format='pdf', dpi=300)

    obs_true = np.array(obs)
    noise_vec = np.random.standard_normal( obs_true.shape )
    for i in range( obs_true.shape[0] ):
        noise_vec[i] /= np.linalg.norm( noise_vec[i].flatten() )

    np.savez('./obs/obs1/obs.npz', N_KL=N_KL, N_x=N_x, obs_true=obs_true, noise_vec=noise_vec, param_true=p )

def read_obs():
    data = np.load('./obs/obs1/obs.npz')
    noise_vec = data['noise_vec']
    y_true = data['obs_true']

    SNR = 10
    y_obs = []
    for i in range(y_true.shape[0]):
        y_obs.append( y_true[i] + np.linalg.norm(y_true[i])/SNR*noise_vec[i] )

    y_obs = np.array(y_obs)

    for i in range(y_true.shape[0]):
        #print(y_obs[i].shape)
        plt.imshow(y_obs[i])
        plt.savefig('noisy_{}.pdf'.format(i),dpi=300)


if __name__ == '__main__':
    #save_obs()
    read_obs()