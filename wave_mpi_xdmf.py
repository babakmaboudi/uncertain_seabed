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
        self.field = matern(N_x, L=6,num_terms=N_kl,delta=1/0.4/0.4,s=.75)
        self.x_grid = np.linspace(-3.001,3.001,N_x)
        self.curve = np.zeros(N_x)
        self.var = 5

    def assemble_curve(self, p):
        self.curve = self.var*self.field.assemble(p)

    def give_curve(self, p):
        return self.var*self.field.assemble(p)

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

class wave():
    def __init__(self, N_x=256, N_KL=64):
        # defining the mesh
        #self.mesh = UnitSquareMesh(100,100)
        #domain = mshr.Rectangle(Point(-3,-1.5), Point(3,1.5))
        #self.mesh = mshr.generate_mesh(domain, 120)

        #mesh_file = File('./model_params/mesh_fine_extended.xml')
        #mesh_file << self.mesh
        self.comm = MPI.COMM_SELF
        #self.mesh = dl.Mesh(self.comm, './model_params/mesh_fine_extended.xml')

        #with dl.XDMFFile(self.comm, "mesh.xdmf") as file:
        #    file.write(self.mesh)

        self.mesh = dl.Mesh(self.comm)
        with dl.XDMFFile(self.comm, "mesh.xdmf") as infile:
            infile.read(self.mesh)

        # defining the function space
        self.V = dl.FunctionSpace(self.mesh,'CG', 1)
        self.dt = 0.002

        # defining test and trial spaces
        self.t = dl.TestFunction(self.V)
        self.u = dl.TrialFunction(self.V)
        self.v = dl.TrialFunction(self.V)

        # defining the seabed curve
        self.FEM_el = self.V.ufl_element()
        self.speed_function = wave_speed_matern(N_x,N_KL,element=self.FEM_el)
        self.c = dl.Function(self.V)

        # marking domain boundaries
        boundary_markers = dl.MeshFunction('size_t',self.mesh,self.mesh.topology().dim()-1)
        boundary_markers.set_all(0)
        bound_w = Boundary_W()
        bound_w.mark(boundary_markers, 0)
        bound_e = Boundary_E()
        bound_e.mark(boundary_markers, 1)
        bound_e = Boundary_N()
        bound_e.mark(boundary_markers, 2)
        # defining the measure for the boundary
        self.ds = dl.Measure('ds', domain=self.mesh, subdomain_data=boundary_markers)

        # defining the Dirichlet boundary on the botton of the domain
        self.u0 = dl.Constant('0.0')
        self.zero_bc = dl.DirichletBC(self.V, self.u0, boundary_diriichlet)

        self.temp = dl.Function(self.V)

    def initiate_zero_init(self, freq):
        self.source = source_term(element=self.FEM_el, fmT=freq)

        self.u_past = dl.Function( self.V )
        self.v_past = dl.Function( self.V )

        self.a1 = self.v*self.t*dl.dx 
        self.L1 = self.v_past*self.t*dl.dx - self.dt/2*self.c*dl.inner( dl.grad(self.u_past), dl.grad(self.t) )*dl.dx - self.dt/2*self.v_past*self.t*self.ds(0) - self.dt/2*self.v_past*self.t*self.ds(1) - self.dt/2*self.u0*self.t*self.ds(2) + self.dt/2*self.source*self.t*dl.dx

        self.a2 = self.u*self.t*dl.dx 
        self.L2 = self.u_past*self.t*dl.dx + self.dt*self.v_past*self.t*dl.dx

    def initiate_load_source(self, freq):
        # loading initial condition
        #init_path = './model_params/init_state_freq_{}.npz'.format(freq)
        #data = np.load(init_path)
        #self.init_u = data['u_past_np']
        #self.init_v = data['v_past_np']

        self.init_u = dl.Function( self.V )
        self.init_v = dl.Function( self.V )

        init_path = './model_params/init_state_freq_{}.xdmf'.format(freq)
        file = dl.XDMFFile(self.comm, init_path)
        file.read_checkpoint(self.init_u, 'u_past', 0)
        file.read_checkpoint(self.init_v, 'v_past', 0)

        #dl.plot(self.init_u)
        #file = dl.File('plot_{}.pvd'.format(freq))
        #file << self.init_u
        #plt.savefig('plot_{}.pdf'.format(freq),format='pdf')
        #print('done')
        #exit()

        # extracting the indecies of the solution at the top boundary
        self.compute_boundary_indecies()

        self.u_past = dl.Function( self.V )
        self.v_past = dl.Function( self.V )

        self.a1 = self.v*self.t*dl.dx 
        self.L1 = self.v_past*self.t*dl.dx - self.dt/2*self.c*dl.inner( dl.grad(self.u_past), dl.grad(self.t) )*dl.dx - self.dt/2*self.v_past*self.t*self.ds(0) - self.dt/2*self.v_past*self.t*self.ds(1) - self.dt/2*self.u0*self.t*self.ds(2) #+ self.dt/2*self.source*self.t*dx

        self.a2 = self.u*self.t*dl.dx 
        self.L2 = self.u_past*self.t*dl.dx + self.dt*self.v_past*self.t*dl.dx

    def save_initial_state(self):
        self.A = dl.assemble(self.a1)
        #self.solver = dl.LUSolver()
        self.solver = dl.KrylovSolver('cg')

        path = './solution/sol.pvd'.format()
        file = dl.File(path)

        t = 0
        time_steps=75
        for i in progressbar(range(time_steps)):
            self.source.t = t
            self.stormer_verlet_step()
            t += self.dt
            if( np.mod(i,10) == 0 ):
                file << (self.u_past, i*self.dt)

        self.save_state(t, time_steps, self.source.fmT, self.source.N)

    # projecting the wave speed function onto the FEM basis
    def compute_wave_speed(self, p):
        self.speed_function.assemble_curve(p)
        temp = dl.interpolate(self.speed_function, self.V)
        self.c.vector().set_local( temp.vector().get_local() )

    # defining the second order symplectic integrator
    def stormer_verlet_step(self):
        b1 = dl.assemble(self.L1)
        #self.zero_bc.apply(b1)
        self.solver.solve(self.A, self.temp.vector(), b1)
        self.v_past.vector().set_local( self.temp.vector().get_local() )

        temp = self.u_past.vector().get_local() + self.dt*self.v_past.vector().get_local()
        self.u_past.vector().set_local( temp )

        b1 = dl.assemble(self.L1)
        #self.zero_bc.apply(b1)
        self.solver.solve(self.A, self.temp.vector(), b1)
        self.v_past.vector().set_local( self.temp.vector().get_local() )

    # this subroutine  advances the PDE in time
    def time_stepping(self, rank):
        self.A = dl.assemble(self.a1)
        #print(dl.as_backend_type(A).mat().isSymmetric())
        #exit()
        self.solver = dl.LUSolver()
        #self.solver = dl.KrylovSolver('cg')

        #sol = Function(self.V)

        self.u_past.vector().set_local( self.init_u )
        self.v_past.vector().set_local( self.init_v )

        path = './solution/sol{}.pvd'.format(rank)
        file = dl.File(path)

        t = 0
        for i in range(2000):
            self.stormer_verlet_step()
            t += self.dt
            if( np.mod(i,10) == 0 ):
                file << (self.u_past, i*self.dt)

    def test_solver(self):
        A = dl.assemble(self.a1)
        b1 = dl.assemble(self.L1)

        solver1 = dl.LUSolver(A)

        t = time()
        solver1.solve(self.temp.vector(), b1)
        print('LU is : {}'.format(time()-t))
        t = time()
        solver1.solve(self.temp.vector(), b1)
        print('LU is : {}'.format(time()-t))
        t = time()
        solver1.solve(self.temp.vector(), b1)
        print('LU is : {}'.format(time()-t))

        solver2 = dl.KrylovSolver('cg')
        t=time()
        solver2.solve(A, self.temp.vector(), b1)
        print('CG is : {}'.format(time()-t))
        t=time()
        solver2.solve(A, self.temp.vector(), b1)
        print('CG is : {}'.format(time()-t))
        t=time()
        solver2.solve(A, self.temp.vector(), b1)
        print('CG is : {}'.format(time()-t))

    def time_stepping_save_png(self, rank):
#        A = dl.assemble(self.a1)
#        self.solver = dl.LUSolver(A)
        self.A = dl.assemble(self.a1)
        self.solver = dl.KrylovSolver('cg')

        self.u_past.vector().set_local( self.init_u.vector().get_local() )
        self.v_past.vector().set_local( self.init_v.vector().get_local() )

        path = './solution_png/sol{}'.format(rank)+'{:05d}.png'
        f, ax = plt.subplots(1)

        t = 0
        idx = 0
        for i in progressbar( range(2000) ):
            self.stormer_verlet_step()

            if( np.mod(i,2) == 0 ):
                plt.sca(ax)
                dl.plot(self.u_past, mode='color', vmin=0, vmax=3.6e-5)
                ax.plot( self.speed_function.x_grid, self.speed_function.curve, color='red' )
                plt.tight_layout()
                plt.savefig(path.format(idx),format='png',dpi=300)
                ax.clear()
                idx += 1

    def read_boundary(self):
        A = dl.assemble(self.a1)
        self.solver = dl.LUSolver(A)

        out = []
        for i in range(2000):
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

    def forward(self, p):
        self.compute_wave_speed(p)
        self.u_past.vector().set_local( self.init_u )
        self.v_past.vector().set_local( self.init_v )

        return self.read_boundary()

    def save_wave_speed(self):
        file = dl.File('speed.pvd')
        file << self.c

    def save_state(self, time, time_steps, freq, num_source):
        #u_past = self.u_past.vector().get_local()
        #v_past = self.v_past.vector().get_local()

        path = './model_params/init_state_freq_{}.xdmf'.format(freq)
        #np.savez(path.format(freq), u_past_np=u_past, v_past_np=v_past, dt=self.dt, time=time, time_steps=time_steps, freq=freq, num_source=num_source)

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
        dl.plot( self.c )

def script_save_init_state():
    N_x=1024
    N_KL=256
    fmT = 100
    problem = wave(N_x=512, N_KL=N_KL)
    #problem.initiate_load_source(100)
    p = np.empty(N_KL)
    problem.compute_wave_speed(p)
    problem.initiate_zero_init(fmT)
    problem.save_initial_state()

def script_save_png():    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    N_x=1024
    N_KL=256
    fmT = np.array( [10, 25, 50, 75, 100] )
    problem = wave(N_x=512, N_KL=N_KL)
    problem.initiate_load_source(fmT[rank])
    if rank == 0:
        p = np.random.standard_normal(N_KL)
    else:
        p = np.empty(N_KL)

    comm.Bcast(p, root=0)
    problem.compute_wave_speed(p)
    problem.time_stepping_save_png(rank)

def script_save_png_npz_params():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    N_x=1024
    N_KL=256
    fmT = np.array( [10, 25, 50, 75, 100] )
    problem = wave(N_x=512, N_KL=N_KL)
    problem.initiate_load_source(fmT[rank])

    #p = p = np.random.standard_normal(N_KL)
    #np.savez('png_npz_params.npz', p = p)

    if rank == 0:
        data = np.load('png_npz_params.npz')
        p = data['p']
        #p = np.random.standard_normal(N_KL)
    else:
        p = np.empty(N_KL)

    comm.Bcast(p, root=0)
    problem.compute_wave_speed(p)
    problem.time_stepping_save_png(rank)


if __name__ == '__main__':
    #script_save_init_state()
    #script_save_png()
    script_save_png_npz_params()
    #test_mpi()

