import numpy as np
from wave import wave
from wave import wave_speed_matern
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

def plot_seabed():
    data = np.load('./obs/obs1/obs.npz')
    p_true = data['param_true']
    N_x = data['N_x']
    N_KL = data['N_KL']
    field = wave_speed_matern(N_x,N_KL)

    field.set_s(0.75)
    f,ax = plt.subplots(1, figsize = (6,3))
    field.plot_curve(p_true,ax, label='true seabed', color='blue')

    x1 = -2 * np.ones(100)
    x2 = 2 * np.ones(100)
    y = np.linspace(-1.5,1.5,100)

    ax.plot(x1,y,'r--')
    ax.plot(x2,y,'r--', label='computational domain')
    

    xx = np.linspace(-3,3,186)
    for x in xx:
        ax.plot(x,1.5,'gx')
    ax.plot(xx[0],1.5,'gx', label='sensor location')

    ax.legend()

    ax.set_xlabel('x (longitude in km)')
    ax.set_ylabel('y (depth in km)')

    plt.tight_layout()


    plt.savefig('bottom.pdf',dpi=300)



if __name__ == '__main__':
    #save_obs()
    #read_obs()
    plot_seabed()