import numpy as np
import matplotlib.pyplot as plt
import dolfin as dl
from mpi4py import MPI

from wave_mpi_dist_solve import wave

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

obs_data = np.load('./obs/obs2/obs.npz')
p = obs_data['param_true']
N_x = obs_data['N_x']
N_KL = obs_data['N_KL']
#fmT = np.array( [10, 25, 50, 75, 100] )

dist_column_width = 2
color = int( rank/dist_column_width )
key = rank%dist_column_width

p_list = []
for i in range(comm.Get_size()):
    if( i%dist_column_width != 0 ):
        p_list.append(i)

process_group_collect = comm.group.Excl(p_list)
comm_collect = comm.Create_group(process_group_collect)

problem = wave(N_x=N_x, N_KL=N_KL, comm_world=comm, key=rank)
problem.initiate_load_source(10)


obs = problem.forward(p, 0.75)
if(rank == 0):
    np.savez('out.npz',obs)
#plt.figure()
#plt.imshow(obs)
#plt.show()
