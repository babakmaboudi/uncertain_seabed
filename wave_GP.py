import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from matern import matern, white_noise_convolution_1d
from petsc4py import PETSc
from slepc4py import SLEPc

from progressbar import progressbar

import dolfin as dl
#import arviz
#import mshr

dl.set_log_level(50)

def hpd_interval_numpy(samples: np.ndarray, mass: float = 0.94):
    if samples.ndim != 2:
        raise ValueError("samples must be 2D [N, M].")
    if not (0 < mass < 1):
        raise ValueError("mass must be in (0, 1).")

    N, M = samples.shape
    k = int(np.floor(mass * N))
    if k <= 0:
        raise ValueError("mass too small for given N.")
    sortx = np.sort(samples, axis=0)  # [N, M]

    if k >= N:
        return sortx[0, :], sortx[-1, :]

    # Window widths
    widths = sortx[k:, :] - sortx[:-k, :]  # [N-k, M]
    idx = np.argmin(widths, axis=0)        # [M]
    lower = sortx[idx, np.arange(M)]
    upper = sortx[idx + k, np.arange(M)]
    return lower, upper

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
        self.field = white_noise_convolution_1d(N_x, sigma=0.25)
        #self.field = matern(N_x, L=6,num_terms=N_kl,delta=1/0.4/0.4,s=.75,load_basis=True)
        #self.field = matern(N_x, L=6,num_terms=N_kl,delta=1/0.4/0.4,s=.75,load_basis=False, save_basis=True)
        self.x_grid = np.linspace(-3.001,3.001,N_x)
        self.curve = np.zeros(N_x)
        self.var = 1.

    def assemble_curve(self, p):
        self.curve = self.var*self.field.assemble(p)

    def give_curve(self, p):
        return self.var*self.field.assemble(p)

    def set_s(self, s):
        pass#self.field.set_s(s)

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

        #hdi_intervals = []
        #for i in range(u.shape[1]):
        #    local_interval = arviz.hdi( u[:,i], hdi_prob=.99 )
        #    hdi_intervals.append( local_interval.reshape(-1) )
        #hdi_intervals = np.array(hdi_intervals)
        hdi_lower, hdi_higher = hpd_interval_numpy(u)

        x = np.linspace( -3,3, len(u[0]) )
        ax.fill_between(x, hdi_lower, hdi_higher, alpha=0.5,color=color, label=label)

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
            values[0] = 1.5*(1 + 0.025*(1.5 - x[1]))
        else:
            values[0] = 6.4

class wave_speed_GP(dl.UserExpression):
    def __init__(self, N_x, N_kl, **kwargs):
        super().__init__(**kwargs)
        self.num_terms=N_kl
        self.field = white_noise_convolution_1d(N_x, sigma=0.25)
        #self.field = matern(N_x, L=6,num_terms=N_kl,delta=1/0.4/0.4,s=.75,load_basis=True)
        #self.field = matern(N_x, L=6,num_terms=N_kl,delta=1/0.4/0.4,s=.75,load_basis=False, save_basis=True)
        self.x_grid = np.linspace(-3.001,3.001,N_x)
        self.curve = np.zeros(N_x)
        self.var = 1.

    def assemble_curve(self, p):
        self.curve = self.var*self.field.assemble(p)

    def give_curve(self, p):
        return self.var*self.field.assemble(p)

    def set_s(self, s):
        pass#self.field.set_s(s)

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

        #hdi_intervals = []
        #for i in range(u.shape[1]):
        #    local_interval = arviz.hdi( u[:,i], hdi_prob=.99 )
        #    hdi_intervals.append( local_interval.reshape(-1) )
        #hdi_intervals = np.array(hdi_intervals)
        hdi_lower, hdi_higher = hpd_interval_numpy(u)

        x = np.linspace( -3,3, len(u[0]) )
        ax.fill_between(x, hdi_lower, hdi_higher, alpha=0.5,color=color, label=label)

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
            values[0] = 1.5*(1 + 0.025*(1.5 - x[1]))
        else:
            values[0] = 6.4

class wave_density_matern(dl.UserExpression):
    def __init__(self, N_x, N_kl, **kwargs):
        super().__init__(**kwargs)
        self.num_terms=N_kl
        self.field = white_noise_convolution_1d(N_x, sigma=0.25)
        #self.field = matern(N_x, L=6,num_terms=N_kl,delta=1/0.4/0.4,s=.75,load_basis=True)
        #self.field = matern(N_x, L=6,num_terms=N_kl,delta=1/0.4/0.4,s=.75,load_basis=False, save_basis=True)
        self.x_grid = np.linspace(-3.001,3.001,N_x)
        self.curve = np.zeros(N_x)
        self.var = 1

    def assemble_curve(self, p):
        self.curve = self.var*self.field.assemble(p)

    def give_curve(self, p):
        return self.var*self.field.assemble(p)

    def set_s(self, s):
        pass#self.field.set_s(s)

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
            values[0] = 1.*( 1. + 0.0044*(1.5 - x[1]) )
        else:
            values[0] = 3.

class wave():
    def __init__(self, N_x=256, N_KL=64):
        # uncomment to save mesh
        #self.mesh = RectangleMesh(Point(-3, -1.5), Point(3, 1.5), 188, 94)
        #with XDMFFile("./model_params/mesh_structured.xdmf") as file:
        #    file.write(self.mesh)

        # uncomment to load mesh from file
        self.mesh = dl.Mesh()
        with dl.XDMFFile("./model_params/mesh_structured.xdmf") as infile:
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

    def initiate_zero_init(self, freq):
        self.source = source_term(element=self.FEM_el, fmT=freq)

        #defining functions to hold the previous time-step
        self.u_past = dl.Function( self.V )
        self.v_past = dl.Function( self.V )

        self.a = self.rho*self.v*self.t*dl.dx 
        self.L = self.rho*self.v_past*self.t*dl.dx - self.dt/2*self.rho*self.c*self.c*dl.inner( dl.grad(self.u_past), dl.grad(self.t) )*dl.dx - self.dt/2*self.rho*self.c*self.v_past*self.t*self.ds(0) - self.dt/2*self.rho*self.c*self.v_past*self.t*self.ds(1) - self.dt/2*self.rho*self.c*self.v_past*self.t*self.ds(2) - self.dt/2*self.u0*self.t*self.ds(3) + self.dt/2*self.source*self.t*dl.dx

    def initiate_load_source_xdmf(self, freq):
        self.init_u = dl.Function( self.V )
        self.init_v = dl.Function( self.V )

        init_path = './model_params/init_state_elastic_density_freq_{}.xdmf'.format(freq)
        file = dl.XDMFFile(init_path)
        file.read_checkpoint(self.init_u, 'u_past', 0)
        file.read_checkpoint(self.init_v, 'v_past', 0)

        # extracting the indecies of the solution at the top boundary
        self.compute_boundary_indecies()

        self.u_past = dl.Function( self.V )
        self.v_past = dl.Function( self.V )

        self.a = self.rho*self.v*self.t*dl.dx 
        self.L = self.rho*self.v_past*self.t*dl.dx - self.dt/2*self.rho*self.c*self.c*dl.inner( dl.grad(self.u_past), dl.grad(self.t) )*dl.dx - self.dt/2*self.rho*self.c*self.v_past*self.t*self.ds(0) - self.dt/2*self.rho*self.c*self.v_past*self.t*self.ds(1) - self.dt/2*self.rho*self.c*self.v_past*self.t*self.ds(2) - self.dt/2*self.u0*self.t*self.ds(3)

    def save_initial_state(self):
        self.A = dl.assemble(self.a)
        self.solver = dl.LUSolver(self.A)

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

    # projecting the wave speed function onto the FEM basis
    def compute_wave_speed(self, p, s=0.75):
        self.speed_function.set_s(s)
        self.speed_function.assemble_curve(p)
        self.density_function.set_s(s)
        self.density_function.assemble_curve(p)
        temp = dl.interpolate(self.speed_function, self.V)
        self.c.assign( temp )
        temp = dl.interpolate(self.density_function, self.V)
        self.rho.assign( temp )

    # defining the second order symplectic integrator
    def stormer_verlet_step(self):
        b = dl.assemble(self.L)
        self.solver.solve(self.A, self.temp.vector(), b)
        self.v_past.assign( self.temp )

        self.u_past.assign( self.u_past + self.dt*self.v_past )

        b = dl.assemble(self.L)
        self.solver.solve(self.A, self.temp.vector(), b)
        self.v_past.assign( self.temp )

    # this subroutine  advances the PDE in time
    def time_stepping(self, idx):
        self.A = dl.assemble(self.a)
        self.solver = dl.LUSolver(self.A)

        self.u_past.assign( self.init_u )
        self.v_past.assign( self.init_v )

        #path = './solution/sol{}.pvd'.format(idx)
        #file = dl.File(path)

        t = 0

        sol = []
        for i in progressbar( range(2000) ):
            self.stormer_verlet_step()
            t += self.dt

            sol.append( self.u_past.vector().get_local() )
            #if( np.mod(i,10) == 0 ):
            #    file << (self.u_past, i*self.dt)

        return np.array(sol)

    def time_stepping_xdmf(self,path='./solution/sol4_ref.pvd'):
        self.A = dl.assemble(self.a)
        self.solver = dl.LUSolver(self.A)

        self.u_past.assign( self.init_u )
        self.v_past.assign( self.init_v )

        file = dl.File(path)

        t = 0
        for i in progressbar( range(2000) ):
            self.stormer_verlet_step()
            t += self.dt
            if( np.mod(i,10) == 0 ):
                file << (self.u_past, i*self.dt)

    def time_stepping_save_png(self, freq_idx):
        self.A = dl.assemble(self.a)
        self.solver = dl.LUSolver(self.A)

        self.u_past.assign( self.init_u )
        self.v_past.assign( self.init_v )

        path = './solution_smooth_png/sol_{}'.format(freq_idx) +'_{:05d}.png'
        f, ax = plt.subplots(1)

        t = 0
        idx = 0

        loc_x = np.linspace(-3,3,186)
        for i in progressbar( range(2000) ):
            self.stormer_verlet_step()

            if( np.mod(i,2) == 0 ):
                plt.sca(ax)
                dl.plot(self.u_past, mode='color', vmin=0, vmax=1.1e-4)
                ax.plot( self.speed_function.x_grid, self.speed_function.curve, color='red' , label='seabed')
                ax.set_xlabel('x')
                ax.set_ylabel('y')

                ax.plot(loc_x[0],1.5,'.b',label='snesors')
                for j in range(1,186):
                    ax.plot(loc_x[j],1.5,'.b')
                ax.legend(loc=4)
                plt.tight_layout()
                plt.savefig(path.format(idx),format='png',dpi=300)
                ax.clear()
                idx += 1

    def read_boundary(self):
        self.A = dl.assemble(self.a)
        self.solver = dl.LUSolver(self.A)

        out = []
        for i in progressbar( range(2000) ):
            self.stormer_verlet_step()
            if(i>=700):
                out.append( self.u_past.vector().get_local()[self.bnd_idx].reshape(1,-1) )
        return np.concatenate(out, axis=0)

    def compute_boundary_indecies(self):
        FEM_el = self.V.ufl_element()
        boundary = lambda x, on_boundary: on_boundary and dl.near(x[1],1.5, 1E-14)
        u0 = dl.Constant('0.0')
        zero_bc = dl.DirichletBC(self.V, u0, boundary)

        dummy = dl.Function(self.V)
        dummy.vector().set_local( np.ones_like( dummy.vector().get_local() ) )
        zero_bc.apply( dummy.vector() )
        self.bnd_idx = np.argwhere( dummy.vector().get_local() == 0 ).flatten()

        x_func = x_boundary(element=FEM_el)
        x_bnd = dl.DirichletBC(self.V, x_func, boundary)
        func = dl.Function(self.V)
        x_bnd.apply( func.vector() )
        x_coords = func.vector().get_local()[self.bnd_idx]
        sorted_idx = np.argsort(x_coords)

        self.bnd_idx = self.bnd_idx[sorted_idx]

    def forward(self, p, s=0.75):
        self.compute_wave_speed(p, s)
        self.u_past.assign( self.init_u )
        self.v_past.assign( self.init_v )

        return self.read_boundary()

    def save_wave_speed(self):
        file = File('speed.pvd')
        file << self.c

    def save_state_np(self, time, time_steps, freq, num_source):
        u_past = self.u_past.vector().get_local()
        v_past = self.v_past.vector().get_local()

        path = './model_params/init_state_elastic_density_freq_{}.npz'
        np.savez(path.format(freq), u_past_np=u_past, v_past_np=v_past, dt=self.dt, time=time, time_steps=time_steps, freq=freq, num_source=num_source)

    def save_state(self, time, time_steps, freq, num_source):
        path = './model_params/init_state_elastic_density_freq_{}.xdmf'.format(freq)

        dl.plot(self.u_past)
        plt.savefig('init.pdf',format='pdf')
        file = dl.XDMFFile(path)
        file.write_checkpoint(self.u_past, 'u_past', 0, dl.XDMFFile.Encoding.HDF5, True)
        file.write_checkpoint(self.v_past, 'v_past', 0, dl.XDMFFile.Encoding.HDF5, True)


    def load_state(self):
        data = np.load('./model_params/state_extended.npz')
        u_past = data['u_past_np']
        v_past = data['v_past_np']

        self.u_past.vector().set_local( u_past )
        self.v_past.vector().set_local( v_past )

    def plot_wave_speed(self):
        plot( self.c )

def save_png():
    N_x=1024
    N_KL=256 

    fmT = np.array( [10, 25, 50, 75, 100] )
    obs = []
    for i, freq in enumerate(fmT):
        problem = wave(N_x=512, N_KL=N_KL)
        problem.initiate_load_source_xdmf(freq)
        data = np.load('./model_params/png_npz_params.npz')
        p = data['p']
        problem.compute_wave_speed( p )

        problem.time_stepping_save_png(i)

def save_obs():
    data = np.load('./obs/obs2/obs.npz')
    p_true = data['param_true']
    N_x = data['N_x']
    N_KL = data['N_KL']

    fmT = np.array( [10, 25, 50, 75, 100] )
    obs = []
    for i, freq in enumerate(fmT):
        problem = wave(N_x=N_x, N_KL=N_KL)
        problem.initiate_load_source_xdmf(freq)
        data = np.load('./model_params/png_npz_params.npz')
        problem.compute_wave_speed( p_true )

        problem.time_stepping(i)

def save_ref_solution():
    N_x=1024
    N_KL=256 

    fmT = np.array( [10, 25, 50, 75, 100] )
    obs = []
    for freq in fmT:
        problem = wave(N_x=512, N_KL=N_KL)
        problem.initiate_load_source_xdmf(freq)
        data = np.load('./model_params/png_npz_params.npz')
        p = data['p']

        out = problem.forward(p, s=0.5)

        f, ax = plt.subplots(1)
        ax.imshow(out)
        ax.set_xlabel('x')
        ax.set_ylabel('time')
        plt.savefig('./solution_ref/obs_freq_s_50_{}.pdf'.format(freq), format='pdf', dpi=300)

        np.savez('./solution_ref/sol_freq_s_50_{}.npz'.format(freq), obs=out)

def save_solution():
    N_x=512
    N_KL=256 

    fmT = 10
    problem = wave(N_x=N_x, N_KL=N_KL)
    problem.initiate_load_source_xdmf(fmT)
    data = np.load('./model_params/png_npz_params.npz')
    p = data['p']
    problem.compute_wave_speed( p )
    problem.time_stepping_xdmf(path='./solution/sol_s_ref.pvd')

def save_solution_vector():
    data = np.load('./obs/obs2/obs.npz')
    p_true = data['param_true']
    N_x = data['N_x']
    N_KL = data['N_KL']

    fmT = 10
    problem = wave(N_x=N_x, N_KL=N_KL)
    problem.initiate_load_source_xdmf(fmT)

    problem.compute_wave_speed(p_true, 0.75)
    out = problem.time_stepping(0)

    np.savez('obs/solution.npz',sol=out)



def script():
    N_x=1024
    N_KL=256 
    problem = wave(N_x=512, N_KL=N_KL)
    #problem.propagate_with_source(freq=100)
    #problem.initiate_load_source(10)

    problem.initiate_load_source_xdmf(10)

    #p = np.random.standard_normal(N_KL)
    data = np.load('./model_params/png_npz_params.npz')
    p = data['p']
    problem.compute_wave_speed( p )
    out = problem.forward(p)

    plt.imshow(out)
    plt.savefig('obs_single.pdf')
    #problem.time_stepping_xdmf()
    #problem.time_stepping_save_png()
    #p = np.random.standard_normal(64)
    #problem.compute_wave_speed(p)
    #problem.save_wave_speed()

    #exit()
    #problem.load_state()
    #problem.time_stepping()

def script_save_init_state():
#    comm = MPI.COMM_WORLD
#    rank = comm.Get_rank()

    N_x=512
    N_KL=256
    fmT = 10
    #dist_column_width = 5
    #color = int( rank/dist_column_width )
    #key = rank%dist_column_width

    problem = wave(N_x=N_x, N_KL=N_KL)
    #problem.initiate_load_source(100)
    p = np.empty(N_KL)
    problem.compute_wave_speed(p)
    problem.initiate_zero_init(fmT)
    problem.save_initial_state()

def save_png():
    N_x=512
    N_KL=256 

    fmT = 10
    problem = wave(N_x=N_x, N_KL=N_KL)
    problem.initiate_load_source_xdmf(fmT)
    data = np.load('./model_params/png_npz_params.npz')
    p = data['p']
    problem.compute_wave_speed( p )
    problem.time_stepping_save_png(fmT)

def test_condition_number():
    N_x=512
    N_KL=256 

    fmT = 10
    problem = wave(N_x=N_x, N_KL=N_KL)
    problem.initiate_load_source_xdmf(fmT)
    data = np.load('./model_params/png_npz_params.npz')
    p = data['p']
    problem.compute_wave_speed(p, 0.5)
    A = dl.assemble(problem.a)

    A_petsc = dl.as_backend_type(A).mat()

    # Create KSP
    ksp = PETSc.KSP().create(A_petsc.getComm())
    ksp.setOperators(A_petsc)

    # Choose solver type
    ksp.setType("cg")      # use "gmres" if A is not SPD
    ksp.getPC().setType("none")  # unpreconditioned cond(A)

    # Ask PETSc to compute singular values (condition number estimate)
    ksp.setComputeSingularValues(True)
    ksp.setFromOptions()

    # Create compatible vectors
    n, _ = A_petsc.getSize()
    b = PETSc.Vec().createMPI(n, comm=A_petsc.getComm())
    x = PETSc.Vec().createMPI(n, comm=A_petsc.getComm())

    # Use a nonzero RHS (safer than all zeros)
    b.setRandom()
    b.assemble()

    # Solve
    ksp.solve(b, x)

    # Extract estimated extreme singular values
    sigma_max, sigma_min = ksp.computeExtremeSingularValues()
    cond_est = sigma_max / sigma_min
    print("Estimated condition number (2-norm):", cond_est)

def test_condition_number_with_stiffness():
    N_x=512
    N_KL=64 

    fmT = 10
    problem = wave(N_x=N_x, N_KL=N_KL)
    problem.initiate_load_source_xdmf(fmT)
    data = np.load('./model_params/png_npz_params.npz')
    p = np.random.standard_normal(64)
    problem.compute_wave_speed(p, 0.5)
    M = dl.assemble(problem.a)

    # FEniCS matrices
    k = problem.rho*problem.c*problem.c*dl.inner( dl.grad(problem.v), dl.grad(problem.t) )*dl.dx
    K = dl.assemble(k)

    M_p = dl.as_backend_type(M).mat()
    K_p = dl.as_backend_type(K).mat()

    E = SLEPc.EPS().create(comm=K_p.getComm())
    E.setOperators(K_p, M_p)
    E.setProblemType(SLEPc.EPS.ProblemType.GHEP)   # if K,M are symmetric and M SPD

    # Shift-and-invert targeting eigenvalues near 0 (smallest positive)
    st = E.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(0.0)

    # Configure the linear solver used inside ST (this matters a lot)
    ksp = st.getKSP()
    ksp.setType("cg")                     # for SPD
    pc = ksp.getPC()
    pc.setType("gamg")                    # or "hypre" if available; "ilu" in serial
    ksp.setTolerances(rtol=1e-10, max_it=2000)

    E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
    E.setTarget(0.0)
    E.setDimensions(nev=1)
    E.setTolerances(tol=1e-8, max_it=500)
    E.setFromOptions()

    E.solve()
    nconv = E.getConverged()
    print("min: nconv =", nconv, "reason =", E.getConvergedReason(), "its =", E.getIterationNumber())
    if nconv == 0:
        raise RuntimeError("No eigenpairs converged for lambda_min; try a different PC (gamg/hypre/ilu) or tolerances.")
    lam_min = E.getEigenvalue(0).real
    print("lambda_min =", lam_min)

    E = SLEPc.EPS().create(comm=K_p.getComm())
    E.setOperators(K_p, M_p)
    E.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
    E.setDimensions(nev=1)
    E.setTolerances(tol=1e-8, max_it=500)
    E.setFromOptions()

    E.solve()
    nconv = E.getConverged()
    print("max: nconv =", nconv, "reason =", E.getConvergedReason(), "its =", E.getIterationNumber())
    if nconv == 0:
        raise RuntimeError("No eigenpairs converged for lambda_max.")
    lam_max = E.getEigenvalue(0).real
    print("lambda_max =", lam_max)
    print("cond(M^{-1}K) =", lam_max/lam_min)

if __name__ == '__main__':
    #script_save_init_state()
    #save_ref_solution()
    #save_solution()
    #save_png()
    #save_obs()

    #save_solution_vector()
    #test_condition_number()
    test_condition_number_with_stiffness()
