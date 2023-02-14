import numpy as np
import matplotlib.pyplot as plt
from wave import wave
from matern import matern
import arviz

import cuqi
from cuqi.distribution import Gaussian, JointDistribution
from cuqi.sampler import pCN, MetropolisHastings

class wave_speed_matern():
    def __init__(self, N_x, N_kl, **kwargs):
        self.field = matern(N_x, num_terms=N_kl,s=6)
        self.x_grid = np.linspace(-2.001,2.001,N_x)
        self.curve = np.zeros(N_x)
        self.var = 2

    def give_curve(self, p):
        return self.var*self.field.assemble(p)

    def plot_curve(self, p, ax, label=None, color=None):
        u = self.var*self.field.assemble(p)
        x = np.linspace( -2,2, len(u) )
        ax.plot(x,u,label=label, color=color)
        ax.set_aspect('equal')
        ax.set_xlim([-2,2])
        ax.set_ylim([-1.5,1.5])

    def plot_uq(self, sample_p, ax, label=None, color=None):
        u = []
        for i in range(sample_p.shape[0]):
            u.append( self.var*self.field.assemble( sample_p[i] ) )
        u = np.array( u )

        hdi_intervals = []
        for i in range(u.shape[1]):
            local_interval = arviz.hdi( u[:,i], hdi_prob=.99 )
            hdi_intervals.append( local_interval.reshape(-1) )
        hdi_intervals = np.array(hdi_intervals)

        x = np.linspace( -2,2, len(u[0]) )
        ax.fill_between(x, hdi_intervals[:,0], hdi_intervals[:,1], alpha=0.5,color=color, label=label)

def save_obs():
    problem = wave()

    param_dim = 64
    p = np.random.standard_normal( param_dim )

    obs_true = problem.forward(p)

    noise_vec = np.random.standard_normal( obs_true.shape )
    noise_vec /= np.linalg.norm( noise_vec.flatten() )

    f, axes = plt.subplots(1,3)
    plt.sca(axes[0])
    problem.plot_wave_speed()

    plt.sca(axes[1])
    plt.imshow(obs_true.reshape(-1,72))

    plt.sca(axes[2])
    obs = obs_true + 0.2*np.linalg.norm( obs_true.flatten() )*noise_vec
    plt.imshow( obs.reshape(-1,72) )

    plt.savefig('./obs/fig2.pdf',format='pdf',dpi=300)

    np.savez('./obs/obs2.npz', param_dim=param_dim, obs_true=obs_true, noise_vec=noise_vec, param_true=p )


def run_pCN():
    print('loading observation data ...')
    obs_data = np.load('./obs/obs2.npz')
    y_true = obs_data['obs_true'].flatten()
    noise_vec = obs_data['noise_vec'].flatten()
    N = obs_data['param_dim']

    sigma = np.linalg.norm(y_true)*0.05
    sigma2 = sigma*sigma
    y_obs = y_true + sigma*noise_vec

    print('initiating forward problem ...')
    problem = wave()

    m = len(y_obs)
    Im = np.ones(m)

    p = Gaussian(np.zeros(N) , 1)
    y = Gaussian(problem.forward, sigma2*Im)

    P = JointDistribution(p,y)
    posterior = P(y=y_obs)
    sampler = pCN(posterior,x0=np.zeros(N))

    print('sampling ...')
    samples = sampler.sample_adapt(10000)

    np.savez( './stat/stat2.npz', samples=samples.samples)

def post_process():
    obs_data = np.load('./obs/obs2.npz')
    y_true = obs_data['obs_true'].flatten()
    noise_vec = obs_data['noise_vec'].flatten()
    N = obs_data['param_dim']

    stat_data = np.load('./stat/stat2.npz')
    samples = stat_data['samples'][:,4000:]
    print(samples.shape)

    mean = np.mean(samples,axis = 1)

    print(mean)
    print(obs_data['param_true'])


    f,ax = plt.subplots(1,2)
    plt.sca(ax[0])

    problem = wave()
    problem.compute_wave_speed(mean)
    problem.plot_wave_speed()

    plt.sca(ax[1])
    plt.plot(samples[0])

    plt.savefig('fig.pdf',format='pdf',dpi=300)

def post_process_curve():
    obs_data = np.load('./obs/obs1.npz')
    y_true = obs_data['obs_true'].flatten()
    noise_vec = obs_data['noise_vec'].flatten()
    N = obs_data['param_dim']
    param_true = obs_data['param_true']

    stat_data = np.load('./stat/stat1.npz')
    samples = stat_data['samples'].T
    samples = samples[4000:,:]

    speed_function = wave_speed_matern(256,64)
    
    f, ax = plt.subplots(1)

    mean = np.mean( samples, axis=0 )
    print(mean.shape)
    speed_function.plot_uq(samples,ax,'99% cred. int.')

    speed_function.plot_curve(param_true, ax, color='orange', label='true curve')
    speed_function.plot_curve(mean, ax, label='estimated curve')
    ax.legend()
    plt.savefig('fig1.pdf',format='pdf',dpi=300)





if __name__ == '__main__':
#    save_obs()
#    run_pCN()  
#    post_process()
    post_process_curve()
