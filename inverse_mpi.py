import numpy as np
import matplotlib.pyplot as plt
from wave_mpi_dist_solve import wave
from mpi4py import MPI
import sys
import os

import pickle

from sampler import sampler, Gibbs, AIES_sampler_warm_up, pCN

#import cuqi
#from cuqi.distribution import Gaussian, JointDistribution, Uniform
#from cuqi.sampler import Gibbs, MH, pCN

class forward_problem():
    def __init__(self, comm, rank, N_x=512, N_KL=256, fmT = np.array([10, 25, 50, 75, 100])):
        # global communicator variables
        self.comm_world = comm
        self.rank_world = rank

        self.N_KL=N_KL
        self.fmT = fmT

        # local communicator variables for forward processing
        self.process_column_width = 2
        self.color = int( self.rank_world/self.process_column_width )
        self.key = self.rank_world%self.process_column_width

        # process list to collect final data
        p_list = []
        for i in range(self.comm_world.Get_size()):
            if( i%self.process_column_width != 0 ):
                p_list.append(i)

        # local communicator for data collection
        process_group_collect = self.comm_world.group.Excl(p_list)
        self.comm_collect = self.comm_world.Create_group(process_group_collect)

        self.problem = wave(N_x=N_x, N_KL=N_KL, comm_world=self.comm_world, color=self.color, key=self.key)
        self.problem.initiate_load_source(fmT[self.color])

    def forward_master(self, p, s=0.75):
        message = np.zeros(self.N_KL+1)
        message[:self.N_KL] = p
        message[-1] = s
        self.comm_world.Bcast(message, root=0)

        #print('in {}'.format( self.rank_world ) )
        #sys.stdout.flush()
        obs = self.problem.forward(p, s)
        #print('out {}'.format( self.rank_world ) )
        #sys.stdout.flush()

        if(self.key == 0):
            num_obs = int( self.comm_world.Get_size()/self.process_column_width)
            rcv_bf = np.empty([num_obs, obs.shape[0]*obs.shape[1]])
            self.comm_collect.Gather(obs.reshape(-1), rcv_bf, root=0)

            obs_all = rcv_bf

            #for idx, freq in enumerate(self.fmT):
            #    f,ax = plt.subplots(1)
            #    ax.imshow( obs_all[idx].reshape(obs.shape[0],obs.shape[1]) )
            #    plt.savefig('./solution_ref/obs_forward_freq_{}.pdf'.format(freq), dpi=300)

            #print('done {}'.format( self.rank_world ) )
            #sys.stdout.flush()
            return obs_all.flatten()

    def forward_slave(self):
        message = np.empty(self.N_KL+1)
        self.comm_world.Bcast(message, root=0)
        p = message[:self.N_KL]
        s = message[-1]

        #print('in {}'.format( self.rank_world ) )
        #sys.stdout.flush()
        obs = self.problem.forward(p, s)
        #print('out {}'.format( self.rank_world ) )
        #sys.stdout.flush()

        if(self.key == 0):
            num_obs = int( self.comm_world.Get_size()/self.process_column_width)
            rcv_bf = np.empty([num_obs, obs.shape[0]*obs.shape[1]])
            self.comm_collect.Gather(obs.reshape(-1), rcv_bf, root=0)

        #print('done {}'.format( self.rank_world ) )
        #sys.stdout.flush()

#class Metro(MH):
#    def step(self, x):
#        self.x0 = x
#        self.scale = 0.4
#        return self.sample(20, ).samples[:,-1]
#
#    def _print_progress(*args, **kwargs):
#        pass

#class PCN(pCN):
#    def step(self, x):
#        self.x0 = x
#        self.scale = 0.03
#        return self.sample(20).samples[:,-1]
#
#    def _print_progress(*args, **kwargs):
#        pass

def test_forward():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    problem = forward_problem(comm, rank)

    if(rank == 0):
        p = np.random.standard_normal(256)
        out = problem.forward_master(p)
        print(out.shape)
    else:
        problem.forward_slave()

def run_AIES():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    problem = forward_problem(comm, rank)
    num_samples = 250
    N_AIES = 10
    N_walkers = 4*N_AIES

    if(rank==0):
        obs_data = np.load('./obs/obs_smooth/obs.npz')
        y_true = obs_data['obs_true'].reshape(5,-1)
        noise_vec = obs_data['noise_vec']
        N_KL = obs_data['N_KL']

        y_true = y_true.reshape( y_true.shape[0] , -1 )
        noise_vec = noise_vec.reshape( y_true.shape[0] , -1 )

        y_norm = np.linalg.norm(y_true, axis=1)
        sigmas = []
        cov_diag = []
        y_obs = []
        for i in range( y_true.shape[0] ):
            sigmas.append( 0.05*np.linalg.norm( y_true[i] ) )
            cov_diag.append( sigmas[-1]**2*np.ones_like( y_true[i] ) )
            y_obs.append( y_true[i] + sigmas[i]*noise_vec[i] )

        sigmas = np.array(sigmas)
        cov_diag = np.array(cov_diag).flatten()
        y_obs = np.array(y_obs).flatten()

        np.random.seed(0)


        log_prior = lambda x: -0.5*np.sum( x**2 )
        log_like = lambda x: -0.5*np.sum( (problem.forward_master(x) - y_obs)**2/cov_diag )

        N_AIES = 10

        x0 = np.random.standard_normal([N_walkers, N_KL])
        print('warm up initiated')
        sys.stdout.flush()
        AIES = AIES_sampler_warm_up( x0, log_prior, log_like, N_AIES )
        AIES.warm_up(N_sample=4, num_windows=20)

        print('warm up finished')
        sys.stdout.flush()

        AIES.sample(num_samples)

        samples = AIES.get_samples()
        #with open('./stat/stat_smooth_AIES.pickle', 'wb') as handle:
        #    pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
        np.savez('./stat/stat_smooth_AIES.npz',samples=samples)

        AIES.print_summary()


    else:
        for i in range( 2*num_samples*N_walkers + N_walkers + 2*4*20*N_walkers ):
            problem.forward_slave()


def run_pCN():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    problem = forward_problem(comm, rank)

    num_warmup = 10000
    num_samples = 20000

    if(rank == 0):
        obs_data = np.load('./obs/obs_smooth/obs.npz')
        y_true = obs_data['obs_true'].reshape(5,-1)
        noise_vec = obs_data['noise_vec']
        N_KL = obs_data['N_KL']

        y_true = y_true.reshape( y_true.shape[0] , -1 )
        noise_vec = noise_vec.reshape( y_true.shape[0] , -1 )

        y_norm = np.linalg.norm(y_true, axis=1)
        sigmas = []
        cov_diag = []
        y_obs = []
        for i in range( y_true.shape[0] ):
            sigmas.append( 0.05*np.linalg.norm( y_true[i] ) )
            cov_diag.append( sigmas[-1]**2*np.ones_like( y_true[i] ) )
            y_obs.append( y_true[i] + sigmas[i]*noise_vec[i] )

        sigmas = np.array(sigmas)
        cov_diag = np.array(cov_diag).flatten()
        y_obs = np.array(y_obs).flatten()

        np.random.seed(0)

        x0 = np.zeros(N_KL)
        log_likelihood = lambda x: -0.5*np.sum( (problem.forward_master(x) - y_obs)**2/cov_diag )


        #s = Uniform(0.5,5)
        #p = Gaussian(np.zeros(N_KL) , 1)
        #y = Gaussian(problem.forward_master, cov_diag)

        #joint = JointDistribution(p,s,y)

        #problem.forward_master(x0)

        #pCN = sampler(x0, log_likelihood)
        pCN_sampler = pCN(x0, log_likelihood)
        pCN_sampler.scale = 0.05

        print('warm up ...')
        sys.stdout.flush()
        pCN_sampler.warm_up(num_warmup, skip_len=100)
        print('scale is: ', pCN_sampler.scale)
        print('sampling ...')
        sys.stdout.flush()
        pCN_sampler.sample(num_samples)

        samples = pCN_sampler.get_samples()
        #pCN.save_checkpoint()

        #print(pCN.get_samples())

        #posterior = joint(y=y_obs)
        #sampler = pCN(posterior,x0=np.zeros(N_KL))
        #sampler = Gibbs(posterior, {'s':Metro, 'p':PCN})

        #samples = sampler.sample(num_samples)

        np.savez( './stat/stat_smooth_pCN.npz', samples=samples)

        #saving_samples = {'p':samples[0], 's': samples[1]}

        #with open('./stat/stat_smooth_pCN.pickle', 'wb') as handle:
        #    pickle.dump(saving_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        for i in range(num_warmup + num_samples+1):
            problem.forward_slave()
        #print('{} done'.format(rank))
        #for i in range(num_samples):
        #    problem.forward_slave()

def run_Gibbs():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    problem = forward_problem(comm, rank)


    checkpoint_path = './checkpoints/Gibbs_smooth_checkpoint.pickle'

    if( os.path.isfile(checkpoint_path) == False ):
        num_warmup = 2
        warm_up_inner = 2
    else:
        num_warmup = 0
        warm_up_inner = 0

    num_samples = 2
    samples_inner = 2
    total_samples = 2 + 2*num_warmup*warm_up_inner + 2*num_samples*samples_inner

    if(rank == 0):
        obs_data = np.load('./obs/obs_smooth/obs.npz')
        y_true = obs_data['obs_true'].reshape(5,-1)
        noise_vec = obs_data['noise_vec']
        N_KL = obs_data['N_KL']

        y_true = y_true.reshape( y_true.shape[0] , -1 )
        noise_vec = noise_vec.reshape( y_true.shape[0] , -1 )

        y_norm = np.linalg.norm(y_true, axis=1)
        sigmas = []
        cov_diag = []
        y_obs = []
        for i in range( y_true.shape[0] ):
            sigmas.append( 0.05*np.linalg.norm( y_true[i] ) )
            cov_diag.append( sigmas[-1]**2*np.ones_like( y_true[i] ) )
            y_obs.append( y_true[i] + sigmas[i]*noise_vec[i] )

        sigmas = np.array(sigmas)
        cov_diag = np.array(cov_diag).flatten()
        y_obs = np.array(y_obs).flatten()

        np.random.seed(0)

        inits = [ np.zeros(N_KL), np.array([1]) ]
        log_likelihood = lambda p, s: -0.5*np.sum( (problem.forward_master(p, s) - y_obs)**2/cov_diag )


        #s = Uniform(0.5,5)
        #p = Gaussian(np.zeros(N_KL) , 1)
        #y = Gaussian(problem.forward_master, cov_diag)

        #joint = JointDistribution(p,s,y)

        #problem.forward_master(x0)

        sampler = Gibbs(inits, log_likelihood, [0.05,0.1])
        print('sampler initiated')
        sys.stdout.flush()

        if( os.path.isfile(checkpoint_path) == False ):
            print('warm up ...')
            sys.stdout.flush()
            sampler.warm_up(N_outer=num_warmup,N_inner=warm_up_inner)
            sampler.save_checkpoint(checkpoint_path)
        else:
            sampler.load_checkpoint(checkpoint_path)
            print('checkpoint loaded')

        print('sampling ...')
        sys.stdout.flush()
        sampler.sample(N_outer=num_samples, N_inner=samples_inner)
        sampler.save_checkpoint('./checkpoints/Gibbs_smooth_checkpoint_sampled.pickle')

        samples = sampler.get_samples()
        #pCN.save_checkpoint()

        #print(pCN.get_samples())

        #posterior = joint(y=y_obs)
        #sampler = pCN(posterior,x0=np.zeros(N_KL))
        #sampler = Gibbs(posterior, {'s':Metro, 'p':PCN})

        #samples = sampler.sample(num_samples)

        saving_samples = {'p':samples[0], 's': samples[1]}

        with open('./stat/stat_Gibbs_smooth.pickle', 'wb') as handle:
            pickle.dump(saving_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #np.savez( './stat/stat_no_cuqi.npz', samples=samples)
    else:
        for i in range(total_samples):
            #print(rank)
            #sys.stdout.flush()
            problem.forward_slave()
        #for i in range(num_samples):
        #    problem.forward_slave()

def run_Gibbs_out_of_prior():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    problem = forward_problem(comm, rank)


    checkpoint_path = './checkpoints/Gibbs_smooth_out_of_prior_checkpoint.pickle'

    if( os.path.isfile(checkpoint_path) == False ):
        num_warmup = 2
        warm_up_inner = 2
    else:
        num_warmup = 0
        warm_up_inner = 0

    num_samples = 2
    samples_inner = 2
    total_samples = 2 + 2*num_warmup*warm_up_inner + 2*num_samples*samples_inner

    if(rank == 0):
        obs_data = np.load('./obs/obs_smooth_out_of_prior/obs.npz')
        y_true = obs_data['obs_true'].reshape(5,-1)
        noise_vec = obs_data['noise_vec']
        N_KL = 256#obs_data['N_KL']

        y_true = y_true.reshape( y_true.shape[0] , -1 )
        noise_vec = noise_vec.reshape( y_true.shape[0] , -1 )

        y_norm = np.linalg.norm(y_true, axis=1)
        sigmas = []
        cov_diag = []
        y_obs = []
        for i in range( y_true.shape[0] ):
            sigmas.append( 0.05*np.linalg.norm( y_true[i] ) )
            cov_diag.append( sigmas[-1]**2*np.ones_like( y_true[i] ) )
            y_obs.append( y_true[i] + sigmas[i]*noise_vec[i] )

        sigmas = np.array(sigmas)
        cov_diag = np.array(cov_diag).flatten()
        y_obs = np.array(y_obs).flatten()

        np.random.seed(0)

        inits = [ np.zeros(N_KL), np.array([1]) ]
        log_likelihood = lambda p, s: -0.5*np.sum( (problem.forward_master(p, s) - y_obs)**2/cov_diag )


        #s = Uniform(0.5,5)
        #p = Gaussian(np.zeros(N_KL) , 1)
        #y = Gaussian(problem.forward_master, cov_diag)

        #joint = JointDistribution(p,s,y)

        #problem.forward_master(x0)

        sampler = Gibbs(inits, log_likelihood, [0.05,0.1])
        print('sampler initiated')
        sys.stdout.flush()

        if( os.path.isfile(checkpoint_path) == False ):
            print('warm up ...')
            sys.stdout.flush()
            sampler.warm_up(N_outer=num_warmup,N_inner=warm_up_inner)
            sampler.save_checkpoint(checkpoint_path)
        else:
            sampler.load_checkpoint(checkpoint_path)
            print('checkpoint loaded')

        print('sampling ...')
        sys.stdout.flush()
        sampler.sample(N_outer=num_samples, N_inner=samples_inner)
        sampler.save_checkpoint('./checkpoints/Gibbs_smooth_out_of_prior_checkpoint_sampled.pickle')

        samples = sampler.get_samples()
        #pCN.save_checkpoint()

        #print(pCN.get_samples())

        #posterior = joint(y=y_obs)
        #sampler = pCN(posterior,x0=np.zeros(N_KL))
        #sampler = Gibbs(posterior, {'s':Metro, 'p':PCN})

        #samples = sampler.sample(num_samples)

        saving_samples = {'p':samples[0], 's': samples[1]}

        with open('./stat/stat_Gibbs_smooth_out_of_prior.pickle', 'wb') as handle:
            pickle.dump(saving_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #np.savez( './stat/stat_no_cuqi.npz', samples=samples)
    else:
        for i in range(total_samples):
            #print(rank)
            #sys.stdout.flush()
            problem.forward_slave()
        #for i in range(num_samples):
        #    problem.forward_slave()



def dummy():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    problem = forward_problem(comm, rank)
    num_samples = 10

    if(rank == 0):
        obs_data = np.load('./obs/obs1/obs.npz')
        y_true = obs_data['obs_true'].reshape(5,-1)
        noise_vec = obs_data['noise_vec']
        N_KL = obs_data['N_KL']

        p = np.random.standard_normal(N_KL)
        s = 0.5
        out = problem.forward_master(p, s)

        out = out.reshape(5,1300,-1)

        for i in range(5):
            plt.imshow(out[i])
            plt.savefig('dummy_{}.pdf'.format(i), dpi=300)
        print(out.shape)

    else:
        problem.forward_slave()



if __name__ == '__main__':
    #run_AIES()
    #run_pCN()
    #run_Gibbs()
    run_Gibbs_out_of_prior()
    #run_Gibbs_load_checkpoint()
    #dummy()




