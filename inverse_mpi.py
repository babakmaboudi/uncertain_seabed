import numpy as np
import matplotlib.pyplot as plt
from wave_mpi import wave
from matern import matern
import arviz

import cuqi
from cuqi.distribution import Gaussian, JointDistribution
from cuqi.sampler import pCN

from mpi4py import MPI

class forward_problem():
    def __init__(self, N_x, N_KL, freq, comm, rank, size):
        self.N_KL = N_KL
        self.problem = wave(N_x=N_x, N_KL=N_KL)
        self.problem.initiate_load_source(freq)
        self.rank = rank
        self.comm = comm
        self.size = size

    def forward_master(self, p):
        self.comm.Bcast(p, root=0)

        obs = self.problem.forward(p)
        obs_all = np.empty( [self.size, obs.shape[0], obs.shape[1]] )
        self.comm.Gather(obs, obs_all, root=0)
        return obs_all.flatten()

    def forward_slave(self):
        p = np.empty(self.N_KL)
        self.comm.Bcast(p, root=0)

        obs_all = None
        obs = self.problem.forward(p)
        self.comm.Gather(obs, obs_all, root=0)

class wave_speed_matern():
    def __init__(self, N_x, N_kl, **kwargs):
        super().__init__(**kwargs)
        self.num_terms=N_kl
        self.field = matern(N_x, L=6,num_terms=N_kl,delta=1/0.4/0.4,s=.75)
        self.x_grid = np.linspace(-3.001,3.001,N_x)
        self.curve = np.zeros(N_x)
        self.var = 5

    def give_curve(self, p):
        return self.var*self.field.assemble(p)

    def plot_curve(self, p, ax, label=None, color=None):
        u = self.var*self.field.assemble(p)
        x = np.linspace( -3,3, len(u) )
        ax.plot(x,u,label=label, color=color)
        ax.set_aspect('equal')
        #ax.set_xlim([-2,2])
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

        x = np.linspace( -3,3, len(u[0]) )
        ax.fill_between(x, hdi_intervals[:,0], hdi_intervals[:,1], alpha=0.5,color=color, label=label)

def save_obs():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N_x = 1024
    N_KL = 256
    freq_list = [10, 25, 50, 75, 100]
    fp = forward_problem(N_x, N_KL, freq_list[rank], comm, rank, size)


    if(rank == 0):
        p = np.random.standard_normal( N_KL )
        obs_true = fp.forward_master(p)

        noise_vec = np.random.standard_normal( obs_true.shape )
        noise_vec /= np.linalg.norm( noise_vec.flatten() )

        f, axes = plt.subplots(1,size)
        print(obs_true.shape)
        axes[0].imshow(obs_true[0])
        axes[0].set_xlabel('x')
        axes[0].set_xlabel('time')
        axes[0].set_title('noise-free y')

        axes[1].imshow(obs_true[1])
        axes[1].set_xlabel('x')
        axes[1].set_xlabel('time')
        axes[1].set_title('noise-free y')


        axes[2].imshow(obs_true[2])
        axes[2].set_xlabel('x')
        axes[2].set_xlabel('time')
        axes[2].set_title('noise-free y')

        axes[3].imshow(obs_true[3])
        axes[3].set_xlabel('x')
        axes[3].set_xlabel('time')
        axes[3].set_title('noise-free y')

        axes[4].imshow(obs_true[4])
        axes[4].set_xlabel('x')
        axes[4].set_xlabel('time')
        axes[4].set_title('noise-free y')

        plt.savefig('./obs/obs_multi_source_noise_free.pdf',format='pdf',dpi=300)

        noisy = obs_true + 0.05*np.linalg.norm( obs_true.flatten() )*noise_vec

        f, axes = plt.subplots(1,size)
        axes[0].imshow(noisy[0])
        axes[0].set_xlabel('x')
        axes[0].set_xlabel('time')
        axes[0].set_title('noisy y')

        axes[1].imshow(noisy[1])
        axes[1].set_xlabel('x')
        axes[1].set_xlabel('time')
        axes[1].set_title('noisy y')


        axes[2].imshow(noisy[2])
        axes[2].set_xlabel('x')
        axes[2].set_xlabel('time')
        axes[2].set_title('noisy y')

        axes[3].imshow(noisy[3])
        axes[3].set_xlabel('x')
        axes[3].set_xlabel('time')
        axes[3].set_title('noisy y')

        axes[4].imshow(noisy[4])
        axes[4].set_xlabel('x')
        axes[4].set_xlabel('time')
        axes[4].set_title('noisy y')

        plt.savefig('./obs/obs_multi_source_noisy.pdf',format='pdf',dpi=300)

        f, ax = plt.subplots(1)
        fp.problem.plot_wave_speed()
        ax.set_xlabel('x')
        ax.set_xlabel('z')
        ax.set_title('outline of the seabed')

        plt.savefig('./obs/obs_multi_source_outiline.pdf',format='pdf',dpi=300)

        np.savez('./obs/obs_multi_source.npz', N_KL=N_KL, N_x=N_x, freq=freq_list, obs_true=obs_true, noise_vec=noise_vec, param_true=p )
    else:
        fp.forward_slave()

def run_pCN():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    obs_data = np.load('./obs/obs_multi_source.npz')
    y_true = obs_data['obs_true']
    noise_vec = obs_data['noise_vec']
    N_KL = obs_data['N_KL']
    N_x = obs_data['N_x']
    freq_list = obs_data['freq']

    fp = forward_problem(N_x, N_KL, freq_list[rank], comm, rank, size)

    y = y_true.reshape( y_true.shape[0] , -1 )
    noise_vec = noise_vec.reshape( y_true.shape[0] , -1 )

    y_norm = np.linalg.norm(y, axis=1)
    sigmas = []
    Cov_diag = []
    y_obs = []
    for i in range( len(freq_list) ):
        noise_vec[i] = noise_vec[i]/np.linalg.norm( noise_vec[i] )
        sigmas.append( 0.05*np.linalg.norm( y[i] ) )
        Cov_diag.append( sigmas[-1]**2*np.ones_like( y[i] ) )
        y_obs.append( y[i] + sigmas[i]*noise_vec[i] )

    sigmas = np.array(sigmas)
    Cov_diag = np.array(Cov_diag).flatten()
    y_obs = np.array(y_obs).flatten()

    num_samples = 2000

    if(rank == 0):
        p = np.random.standard_normal( N_KL )
        #obs_true = fp.forward_master(p)

        mapping = fp.forward_master

        p = Gaussian(np.zeros(N_KL) , 1)
        y = Gaussian(fp.forward_master, Cov_diag)

        joint = JointDistribution(p,y)

        posterior = joint(y=y_obs)
        sampler = pCN(posterior,x0=np.zeros(N_KL))

        samples = sampler.sample_adapt(num_samples)

        np.savez( './stat/stat_multi_source2.npz', samples=samples.samples)
    else:
        for i in range(num_samples):
            fp.forward_slave()

def post_process():
    obs_data = np.load('./obs/obs_multi_source.npz')
    y_true = obs_data['obs_true'].flatten()
    noise_vec = obs_data['noise_vec'].flatten()
    N_KL = obs_data['N_KL']
    N_x = obs_data['N_x']
    freq = obs_data['freq']

    stat_data = np.load('./stat/stat_multi_source_hpc.npz')
    samples = stat_data['samples']#[:,3000:]
    print(samples.shape)

    mean = np.mean(samples,axis = 1)

    print(mean)
    print(obs_data['param_true'])


    f,ax = plt.subplots(1,2)
    plt.sca(ax[0])

    problem = wave(N_x=N_x, N_KL=N_KL)
    problem.initiate_load_source(freq[0])
    problem.compute_wave_speed(mean)
    problem.plot_wave_speed()

    plt.sca(ax[1])
    plt.plot(samples[0])

    plt.savefig('fig.pdf',format='pdf',dpi=300)

def post_process_curve():
    obs_data = np.load('./obs/obs_multi_source.npz')
    y_true = obs_data['obs_true'].flatten()
    noise_vec = obs_data['noise_vec'].flatten()
    N_KL = obs_data['N_KL']
    N_x = obs_data['N_x']
    freq = obs_data['freq']
    param_true = obs_data['param_true']

    stat_data = np.load('./stat/stat_multi_source_hpc.npz')
    samples = stat_data['samples'].T
    samples = samples[1000:,:]

    speed_function = wave_speed_matern(N_x,N_KL)
    
    f, ax = plt.subplots(1)

    mean = np.mean( samples, axis=0 )
    print(mean.shape)
    speed_function.plot_uq(samples,ax,'99% cred. int.')

    speed_function.plot_curve(param_true, ax, color='orange', label='true curve')
    speed_function.plot_curve(mean, ax, label='estimated curve')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.savefig('fig1.pdf',format='pdf',dpi=300)

if __name__ == '__main__':
    #save_obs()
    #run_pCN()
    #post_process()
    post_process_curve()