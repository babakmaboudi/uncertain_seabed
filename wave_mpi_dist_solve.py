import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from matern import matern
import dolfin as dl
from mpi4py import MPI
from progressbar import progressbar

from time import time

dl.set_log_level(50)

def boundary_diriichlet(x, on_boundary):
    return on_boundary and dl.near(x[1], 0, 1E-14)

class Boundary_W(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and dl.near(x[0], -2., 1E-14)

class Boundary_E(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and dl.near(x[0],2., 1E-14)

class Boundary_N(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and dl.near(x[1],1.5, 1E-14)

class Boundary_S(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and dl.near(x[1],-1.5, 1E-14)

class init_cond(dl.UserExpression):
    def eval(elf, values, x):
        values[0] = np.exp( -( (x[0] + 0.8)**2 + (x[1] - 0.4)**2)/(0.05**2) )
        values[0] += np.exp( -( (x[0] + 0.4 )**2 + (x[1] - 0.4)**2)/(0.05**2) )
        values[0] += np.exp( -( (x[0] )**2 + (x[1] - 0.4)**2)/(0.05**2) )
        values[0] += np.exp( -( (x[0] - 0.4)**2 + (x[1] - 0.4)**2)/(0.05**2) )
        values[0] += np.exp( -( (x[0] - 0.8)**2 + (x[1] - 0.4)**2)/(0.05**2) )

class x_boundary(dl.UserExpression):
    def eval(self, values, x):
        values[0] = x[0]

class wave_speed(dl.UserExpression):
    def __init__(self, loc, **kwargs):
        self.loc = loc
        super().__init__(**kwargs)

    def eval(self, values, x):
        if (x[1]>self.loc):
            values[0] = 1.5
        else:
            values[0] = 6.4

class source_term(dl.UserExpression):
    def __init__(self, fmT=10, num_source=5, domain_length=4, source_depth=1.4, **kwargs):
        super().__init__(**kwargs)
        self.t = 0
        self.fmT = fmT
        dx = domain_length/num_source
        self.N = num_source
        self.depth = source_depth
        self.source_loc = np.linspace( -domain_length/2 + dx/4 , domain_length/2 - dx/4, self.N )

    def print(self):
        print('')
        print((1- 2*np.pi**2*self.fmT**2*self.t**2 ) * np.exp( -np.pi**2*self.fmT**2*self.t**2  ))

    def eval(self, values, x):
        for s in self.source_loc:
            values[0] += (1- 2*np.pi**2*self.fmT**2*self.t**2 ) * np.exp( -np.pi**2*self.fmT**2*self.t**2  ) * np.exp( -( (x[0] - s )**2 + (x[1] - self.depth)**2)/(0.05**2) )

class wave_speed_matern(dl.UserExpression):
    def __init__(self, N_x, N_kl, **kwargs):
        super().__init__(**kwargs)
        self.num_terms=N_kl
        self.field = matern(N_x, L=6,num_terms=N_kl,delta=1/0.4/0.4,s=.75,load_basis=True)
        self.x_grid = np.linspace(-3.001,3.001,N_x)
        self.curve = np.zeros(N_x)
        self.var = 5

    def assemble_curve(self, p):
        self.curve = self.var*self.field.assemble(p)

    def give_curve(self, p):
        return self.var*self.field.assemble(p)

    def set_s(self, s):
        self.field.set_s(s)

    def eval(self, values, x):
        temp = (x[0] - self.x_grid)>0
        loc = ( temp[:-1] )*( ~temp[1:] )
        idx = np.argwhere(loc==True).item()
        x1 = self.x_grid[idx]
        y1 = self.curve[idx]
        x2 = self.x_grid[idx+1]
        y2 = self.curve[idx+1]

        y = ( (y2 - y1)*x[0] + y1*x2 - x1*y2 )/(x2-x1)
        if( x[1]>y ):
            values[0] = 1.5
        else:
            values[0] = 6.4

class wave_density_matern(dl.UserExpression):
    def __init__(self, N_x, N_kl, **kwargs):
        super().__init__(**kwargs)
        self.num_terms=N_kl
        self.field = matern(N_x, L=6,num_terms=N_kl,delta=1/0.4/0.4,s=.75,load_basis=True)
        #self.field = matern(N_x, L=6,num_terms=N_kl,delta=1/0.4/0.4,s=.75,load_basis=False, save_basis=True)
        self.x_grid = np.linspace(-3.001,3.001,N_x)
        self.curve = np.zeros(N_x)
        self.var = 5

    def assemble_curve(self, p):
        self.curve = self.var*self.field.assemble(p)

    def give_curve(self, p):
        return self.var*self.field.assemble(p)

    def set_s(self, s):
        self.field.set_s(s)

    def eval(self, values, x):
        temp = (x[0] - self.x_grid)>0
        loc = ( temp[:-1] )*( ~temp[1:] )
        idx = np.argwhere(loc==True).item()
        x1 = self.x_grid[idx]
        y1 = self.curve[idx]
        x2 = self.x_grid[idx+1]
        y2 = self.curve[idx+1]

        y = ( (y2 - y1)*x[0] + y1*x2 - x1*y2 )/(x2-x1)
        if( x[1]>y ):
            values[0] = 1.
        else:
            values[0] = 3.

class wave():
    def __init__(self, N_x=256, N_KL=64, comm_world=None, color=0, key=0):
        self.comm_world = comm_world
        self.local_comm = MPI.Comm.Split(comm_world,color=color, key=key)

        # uncomment to save the mesh
        #self.mesh = dl.RectangleMesh(self.comm_world, dl.Point(-3, -1.5), dl.Point(3, 1.5), 188, 94)
        #self.mesh = dl.Mesh(self.comm, './model_params/mesh_fine_extended.xml')

        #with dl.XDMFFile(self.comm_world, "./model_params/mesh_structured.xdmf") as file:
        #    file.write(self.mesh)

        # uncomment to load mesh from file
        self.mesh = dl.Mesh(self.local_comm)
        with dl.XDMFFile(self.local_comm, "./model_params/mesh_structured.xdmf") as infile:
            infile.read(self.mesh)

        # defining the function space
        self.V = dl.FunctionSpace(self.mesh,'CG', 1)
        self.dt = 0.0019

        # defining test and trial spaces
        self.t = dl.TestFunction(self.V)
        self.u = dl.TrialFunction(self.V)
        self.v = dl.TrialFunction(self.V)

        # defining the seabed curve
        self.FEM_el = self.V.ufl_element()
        self.speed_function = wave_speed_matern(N_x,N_KL,element=self.FEM_el)
        self.density_function = wave_density_matern(N_x,N_KL,element=self.FEM_el)
        self.c = dl.Function(self.V)
        self.rho = dl.Function(self.V)

        # marking domain boundaries
        boundary_markers = dl.MeshFunction('size_t',self.mesh,self.mesh.topology().dim()-1)
        boundary_markers.set_all(0)
        bound_w = Boundary_W()
        bound_w.mark(boundary_markers, 0)
        bound_e = Boundary_E()
        bound_e.mark(boundary_markers, 1)
        bound_s = Boundary_S()
        bound_s.mark(boundary_markers, 2)
        bound_n = Boundary_N()
        bound_n.mark(boundary_markers, 3)
        # defining the measure for the boundary
        self.ds = dl.Measure('ds', domain=self.mesh, subdomain_data=boundary_markers)

        # defining the Dirichlet boundary on the botton of the domain
        self.u0 = dl.Constant('0.0')
        self.zero_bc = dl.DirichletBC(self.V, self.u0, boundary_diriichlet)

        self.temp = dl.Function(self.V)

    # this function initializes wave equation with a source term and zero initial condition
    def initiate_zero_init(self, freq):
        self.source = source_term(element=self.FEM_el, fmT=freq)

        self.u_past = dl.Function( self.V )
        self.v_past = dl.Function( self.V )

        # weak form for the velocity equation
        self.a = self.rho*self.v*self.t*dl.dx 
        self.L = self.rho*self.v_past*self.t*dl.dx - self.dt/2*self.rho*self.c*self.c*dl.inner( dl.grad(self.u_past), dl.grad(self.t) )*dl.dx - self.dt/2*self.rho*self.c*self.v_past*self.t*self.ds(0) - self.dt/2*self.rho*self.c*self.v_past*self.t*self.ds(1) - self.dt/2*self.rho*self.c*self.v_past*self.t*self.ds(2) - self.dt/2*self.u0*self.t*self.ds(3) + self.dt/2*self.source*self.t*dl.dx

    # this function initializes wave equation with an initial condition and no source term
    def initiate_load_source(self, freq):
        # functions to hold the initial conditions (pressure and velocity)
        self.init_u = dl.Function( self.V )
        self.init_v = dl.Function( self.V )

        init_path = './model_params/init_state_structured_freq_{}.xdmf'.format(freq)
        file = dl.XDMFFile(self.local_comm, init_path)
        file.read_checkpoint(self.init_u, 'u_past', 0)
        file.read_checkpoint(self.init_v, 'v_past', 0)

        # extracting the indecies of the solution at the top boundary
        self.compute_boundary_indecies()

        self.u_past = dl.Function( self.V )
        self.v_past = dl.Function( self.V )

        # weak form for the velocity equation
        self.a = self.rho*self.v*self.t*dl.dx 
        self.L = self.rho*self.v_past*self.t*dl.dx - self.dt/2*self.rho*self.c*self.c*dl.inner( dl.grad(self.u_past), dl.grad(self.t) )*dl.dx - self.dt/2*self.rho*self.c*self.v_past*self.t*self.ds(0) - self.dt/2*self.rho*self.c*self.v_past*self.t*self.ds(1) - self.dt/2*self.rho*self.c*self.v_past*self.t*self.ds(2) - self.dt/2*self.u0*self.t*self.ds(3)

    # this function computes and saves a state with the source term
    def save_initial_state(self):
        self.A = dl.assemble(self.a)
        # defining conjugate gradient iterations for solving systems
        self.solver = dl.KrylovSolver('cg')

        # path to save the solution to file
        path = './solution/sol.pvd'.format()
        file = dl.File(path)

        t = 0
        time_steps=75
        # time steps
        for i in progressbar( range(time_steps) ):
            self.source.t = t
            self.stormer_verlet_step()
            t += self.dt
            if( np.mod(i,10) == 0 ):
                file << (self.u_past, i*self.dt)

        self.save_state(t, time_steps, self.source.fmT, self.source.N)

    # projecting/interpolating the wave speed function onto the FEM basis
    def compute_wave_speed(self, p, s):
        self.speed_function.set_s(s)
        self.speed_function.assemble_curve(p)
        self.density_function.set_s(s)
        self.density_function.assemble_curve(p)
        temp = dl.interpolate(self.speed_function, self.V)
        self.c.assign( temp )
        temp = dl.interpolate(self.density_function, self.V)
        self.rho.assign( temp )

    # second order symplectic integrator (Stormer-Verlet)
    def stormer_verlet_step(self):
        b = dl.assemble(self.L)
        self.solver.solve(self.A, self.temp.vector(), b)
        self.v_past.assign( self.temp )

        self.u_past.assign( self.u_past + self.dt*self.v_past )

        b = dl.assemble(self.L)
        self.solver.solve(self.A, self.temp.vector(), b)
        self.v_past.assign( self.temp )

    # this function advances the PDE in time
    def time_stepping(self, color):
        self.A = dl.assemble(self.a)
        self.solver = dl.KrylovSolver('cg')

        self.u_past.assign(self.init_u)
        self.v_past.assign(self.init_v)

        path = './solution/sol{}_multi.pvd'.format(color)
        file = dl.File(path)

        t = 0
        for i in range(2000):
            self.stormer_verlet_step()
            t += self.dt
            if( np.mod(i,10) == 0 ):
                file << (self.u_past, i*self.dt)

    # this function solves the wave equation and extract boundary measurements from snapshots
    def read_boundary(self):
        self.A = dl.assemble(self.a)
        self.solver = dl.KrylovSolver('cg')

        # uncomment to control the accuracy of the CG method
        #self.solver.parameters['absolute_tolerance'] = 1e-10
        #self.solver.parameters['relative_tolerance'] = 1e-8
        #self.solver.parameters['maximum_iterations'] = 1000

        out = []
        for i in range(2000):
            self.stormer_verlet_step()
            if(i>=700):
                out.append( self.u_past.vector().get_local()[self.bnd_idx].reshape(1,-1) )
        return np.concatenate(out, axis=0)

    # this function applies the forward operator p->y_obs
    def forward(self, p, s=0.75):
        self.compute_wave_speed(p, s)
        self.u_past.assign(self.init_u)
        self.v_past.assign(self.init_v)

        out = self.read_boundary()

        num_time_steps = out.shape[0]
        out = out.reshape(-1)

        # communicate data to the process with key=0
        if(self.local_comm.Get_rank() == 0):
            rec_buf = np.empty( num_time_steps*np.sum( self.out_size_list ), dtype='d' )
            sendcounts = num_time_steps*np.array(self.out_size_list)
        else:
            rec_buf = None
            sendcounts = None

        self.local_comm.Gatherv(sendbuf=out, recvbuf=(rec_buf, sendcounts), root=0)

        if(self.local_comm.Get_rank() == 0):
            data = []
            indecies = np.cumsum(sendcounts)
            data.append( rec_buf[:indecies[0]].reshape(num_time_steps,-1) )
            for i in range(len(sendcounts)-1):
                data.append( rec_buf[indecies[i]:indecies[i+1]].reshape(num_time_steps,-1) )

            out = np.concatenate( data, axis=1 )
            out = out[:,self.output_idx]
        return out

    # this function saves a state of the wave equation (pressure and velocity)
    def save_state(self, time, time_steps, freq, num_source):
        path = './model_params/init_state_elastic_density_freq_{}.xdmf'.format(freq)

        dl.plot(self.u_past)
        plt.savefig('init.pdf',format='pdf')
        file = dl.XDMFFile(path)
        file.write_checkpoint(self.u_past, 'u_past', 0, dl.XDMFFile.Encoding.HDF5, True)
        file.write_checkpoint(self.v_past, 'v_past', 0, dl.XDMFFile.Encoding.HDF5, True)

    # this function creates the indecies for the boundary measurement in order of x-coordinates
    def compute_boundary_indecies(self):
        FEM_el = self.V.ufl_element()
        boundary = lambda x, on_boundary: on_boundary and dl.near(x[1],1.5, 1E-14)
        u0 = dl.Constant('0.0')
        zero_bc = dl.DirichletBC(self.V, u0, boundary)

        dummy = dl.Function(self.V)
        dummy.vector().set_local( np.ones_like( dummy.vector().get_local() ) )
        zero_bc.apply( dummy.vector() )

        self.bnd_idx = np.argwhere( dummy.vector().get_local() == 0 ).flatten()
        #print(self.V.tabulate_dof_coordinates()[self.bnd_idx])
        #exit()

        x_func = x_boundary(element=FEM_el)
        x_bnd = dl.DirichletBC(self.V, x_func, boundary)
        func = dl.Function(self.V)
        x_bnd.apply( func.vector() )
        x_coords = func.vector().get_local()[self.bnd_idx]
        sorted_idx = np.argsort(x_coords)

        self.bnd_idx = self.bnd_idx[sorted_idx]
        data = self.bnd_idx.shape[0]
        data = self.local_comm.gather(data, root=0)

        if( self.local_comm.Get_rank() == 0 ):
            self.out_size_list = data
        else:
            self.out_size_list = None

        coords = np.array(self.V.tabulate_dof_coordinates()[self.bnd_idx][:,0])

        if(self.local_comm.Get_rank() == 0):
            rec_buf = np.empty( np.sum( self.out_size_list ), dtype='d' )
            sendcounts = np.array(self.out_size_list)
        else:
            rec_buf = None
            sendcounts = None

        self.local_comm.Gatherv(sendbuf=coords, recvbuf=(rec_buf, sendcounts), root=0)

        if(self.local_comm.Get_rank() == 0):
            self.output_idx = np.argsort( rec_buf )
        else:
            self.output_idx=None

def script_save_init_state():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    N_x=512
    N_KL=256
    fmT = 100
    dist_column_width = 5
    color = int( rank/dist_column_width )
    key = rank%dist_column_width

    problem = wave(N_x=N_x, N_KL=N_KL, comm_world=comm, color=color, key=key)
    #problem.initiate_load_source(100)
    p = np.empty(N_KL)
    problem.compute_wave_speed(p, s=1)
    print('{} : done with wave'.format(rank))
    problem.initiate_zero_init(fmT)
    problem.save_initial_state()

def script_save_png_npz_params():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    N_x=1024
    N_KL=256
    fmT = np.array( [10, 25, 50, 75, 100] )

    dist_column_width = 2
    color = int( rank/dist_column_width )
    key = rank%dist_column_width

    p_list = []
    for i in range(comm.Get_size()):
        if( i%dist_column_width != 0 ):
            p_list.append(i)

    process_group_collect = comm.group.Excl(p_list)
    comm_collect = comm.Create_group(process_group_collect)

    problem = wave(N_x=512, N_KL=N_KL, comm_world=comm, color=color, key=key)
    problem.initiate_load_source(fmT[color])

    #p = p = np.random.standard_normal(N_KL)
    #np.savez('png_npz_params.npz', p = p)

    if rank == 0:
        data = np.load('png_npz_params.npz')
        p = data['p']
        #p = np.random.standard_normal(N_KL)
    else:
        p = np.empty(N_KL)

    comm.Bcast(p, root=0)
    obs = problem.forward(p)

    if(key == 0):
        num_obs = int( comm.Get_size()/dist_column_width)
        rcv_bf = np.empty([num_obs, obs.shape[0]*obs.shape[1]])
        comm_collect.Gather(obs.reshape(-1), rcv_bf, root=0)
        
    if(comm.Get_rank() == 0):
        obs_all = rcv_bf

        f,axes = plt.subplots(1,2)
        axes[0].imshow( obs_all[0].reshape(obs.shape[0],obs.shape[1]) )
        axes[1].imshow( obs_all[1].reshape(obs.shape[0],obs.shape[1]) )
    else:
        obs_all = None

def test_parallel():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    N_x=512
    N_KL=256
    fmT = np.array( [10, 25, 50, 75, 100] )

    dist_column_width = 2
    color = int( rank/dist_column_width )
    key = rank%dist_column_width

    p_list = []
    for i in range(comm.Get_size()):
        if( i%dist_column_width != 0 ):
            p_list.append(i)

    process_group_collect = comm.group.Excl(p_list)
    comm_collect = comm.Create_group(process_group_collect)

    problem = wave(N_x=N_x, N_KL=N_KL, comm_world=comm, color=color, key=key)
    problem.initiate_load_source(fmT[color])

    #p = p = np.random.standard_normal(N_KL)
    #np.savez('png_npz_params.npz', p = p)

    s = 0.75

    if rank == 0:
        data = np.load('./model_params/png_npz_params.npz')
        p = data['p']
        #p = np.random.standard_normal(N_KL)
    else:
        p = np.empty(N_KL)

    comm.Bcast(p, root=0)
    obs = problem.forward(p, s)

    if(key == 0):
        num_obs = int( comm.Get_size()/dist_column_width)
        rcv_bf = np.empty([num_obs, obs.shape[0]*obs.shape[1]])
        comm_collect.Gather(obs.reshape(-1), rcv_bf, root=0)
        
    if(comm.Get_rank() == 0):
        obs_all = rcv_bf

        for idx, freq in enumerate(fmT):
            f,ax = plt.subplots(1)
            ax.imshow( obs_all[idx].reshape(obs.shape[0],obs.shape[1]) )
            plt.savefig('./solution_ref/obs_parallel_freq_{}.pdf'.format(freq), dpi=300)
            np.savez('./solution_ref/parallel_freq_{}.npz'.format(freq), obs=obs_all[idx])
    else:
        obs_all = None
if __name__ == '__main__':
    test_parallel()
    #script_save_init_state()
    #script_save_png()
    #script_save_png_npz_params()
    #test_mpi()
