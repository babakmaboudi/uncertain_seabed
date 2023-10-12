import numpy as np
import matplotlib.pyplot as plt
from wave_mpi_dist_solve import wave
from mpi4py import MPI
import sys

from sampler import sampler

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

def run_pCN():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    problem = forward_problem(comm, rank)

    num_warmup = 10
    num_samples = 40

    if(rank == 0):
        obs_data = np.load('./obs/obs1/obs.npz')
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

        pCN = sampler(x0, log_likelihood)
        pCN.scale = 0.05

        print('warm up ...')
        sys.stdout.flush()
        pCN.warm_up(num_warmup, skip_len=100)
        print('sampling ...')
        sys.stdout.flush()
        pCN.sample(num_samples)

        samples = pCN.get_samples()
        #pCN.save_checkpoint()

        #print(pCN.get_samples())

        #posterior = joint(y=y_obs)
        #sampler = pCN(posterior,x0=np.zeros(N_KL))
        #sampler = Gibbs(posterior, {'s':Metro, 'p':PCN})

        #samples = sampler.sample(num_samples)

        np.savez( './stat/stat_no_cuqi.npz', samples=samples)
    else:
        for i in range(num_warmup + num_samples+1):
            problem.forward_slave()
        #for i in range(num_samples):
        #    problem.forward_slave()

def run_Gibbs_load_checkpoint(check_path='./checkpoints/checkpoint1.npz'):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    problem = forward_problem(comm, rank)
    num_samples = 100

    if(rank == 0):
        obs_data = np.load('./obs/obs1/obs.npz')
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

        s = Uniform(0.5,5)
        p = Gaussian(np.zeros(N_KL) , 1)
        y = Gaussian(problem.forward_master, cov_diag)

        joint = JointDistribution(p,s,y)

        posterior = joint(y=y_obs)

        #sampler = pCN(posterior,x0=np.zeros(N_KL))
        sampler = Gibbs(posterior, {'s':Metro, 'p':PCN})

        checkpoint_data = np.load(check_path)
        s0 = checkpoint_data['s0']
        p0 = checkpoint_data['p0']
        sampler.samplers['s'].x0 = s0
        sampler.samplers['p'].x0 = p0

        print(sampler.samplers['p'].x0)
        exit()

        samples = sampler.sample(num_samples)

        np.savez( './stat/stat_Gibbs_long_patch2.npz', samples=samples.samples)
    else:
        pass
        for i in range(num_samples):
            problem.forward_slave()


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
    run_pCN()
    #run_Gibbs_load_checkpoint()
    #dummy()




