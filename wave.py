import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from matern import matern

from progressbar import progressbar

from dolfin import *
import arviz
#import mshr

set_log_level(50)

def boundary_diriichlet(x, on_boundary):
    return on_boundary and near(x[1], 0, 1E-14)

class Boundary_W(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], -2., 1E-14)

class Boundary_E(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0],2., 1E-14)

class Boundary_N(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1],1.5, 1E-14)

class init_cond(UserExpression):
    def eval(elf, values, x):
        values[0] = np.exp( -( (x[0] + 0.8)**2 + (x[1] - 0.4)**2)/(0.05**2) )
        values[0] += np.exp( -( (x[0] + 0.4 )**2 + (x[1] - 0.4)**2)/(0.05**2) )
        values[0] += np.exp( -( (x[0] )**2 + (x[1] - 0.4)**2)/(0.05**2) )
        values[0] += np.exp( -( (x[0] - 0.4)**2 + (x[1] - 0.4)**2)/(0.05**2) )
        values[0] += np.exp( -( (x[0] - 0.8)**2 + (x[1] - 0.4)**2)/(0.05**2) )

class x_boundary(UserExpression):
    def eval(self, values, x):
        values[0] = x[0]

class wave_speed(UserExpression):
    def __init__(self, loc, **kwargs):
        self.loc = loc
        super().__init__(**kwargs)

    def eval(self, values, x):
        if (x[1]>self.loc):
            values[0] = 1.5
        else:
            values[0] = 6.4

class source_term(UserExpression):
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
 
class wave_speed_matern(UserExpression):
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
        # uncomment to save mesh
        #self.mesh = RectangleMesh(Point(-3, -1.5), Point(3, 1.5), 188, 94)
        #with XDMFFile("./model_params/mesh_structured.xdmf") as file:
        #    file.write(self.mesh)

        # uncomment to load mesh from file
        self.mesh = Mesh()
        with XDMFFile("./model_params/mesh_structured.xdmf") as infile:
            infile.read(self.mesh)

        # defining the function space
        self.V = FunctionSpace(self.mesh,'CG', 1)
        self.dt = 0.002

        # defining test and trial spaces
        self.t = TestFunction(self.V)
        self.u = TrialFunction(self.V)
        self.v = TrialFunction(self.V)

        # defining the seabed curve
        self.FEM_el = self.V.ufl_element()
        self.speed_function = wave_speed_matern(N_x,N_KL,element=self.FEM_el)
        self.c = Function(self.V)

        # marking domain boundaries
        boundary_markers = MeshFunction('size_t',self.mesh,self.mesh.topology().dim()-1)
        boundary_markers.set_all(0)
        bound_w = Boundary_W()
        bound_w.mark(boundary_markers, 0)
        bound_e = Boundary_E()
        bound_e.mark(boundary_markers, 1)
        bound_e = Boundary_N()
        bound_e.mark(boundary_markers, 2)
        # defining the measure for the boundary
        self.ds = Measure('ds', domain=self.mesh, subdomain_data=boundary_markers)

        # defining the Dirichlet boundary on the botton of the domain
        self.u0 = Constant('0.0')
        self.zero_bc = DirichletBC(self.V, self.u0, boundary_diriichlet)

        self.temp = Function(self.V)

    def propagate_with_source(self, freq=10, num_source=5):
        self.source = source_term(fmT=freq, num_source=num_source, element=self.FEM_el)

        #defining functions to hold the previous time-step
        self.u_past = Function( self.V )
        self.v_past = Function( self.V )

        self.a = self.v*self.t*dx 
        self.L = self.v_past*self.t*dx - self.dt/2*self.c*inner( grad(self.u_past), grad(self.t) )*dx - self.dt/2*self.v_past*self.t*ds(0) - self.dt/2*self.v_past*self.t*ds(1) - self.dt/2*self.u0*self.t*ds(2) + self.dt/2*self.source*self.t*dx

        self.compute_wave_speed( np.zeros(self.speed_function.num_terms) )

        A = assemble(self.a)
        self.solver = LUSolver(A)

        t = 0
        num_steps = 75
        for i in progressbar( range(num_steps) ):
            self.stormer_verlet_step()
            t += self.dt
            self.source.t = t

        self.save_state(t, num_steps, freq, num_source)

    def initiate_load_source_xdmf(self, freq):
        self.init_u = Function( self.V )
        self.init_v = Function( self.V )

        init_path = './model_params/init_state_structured_freq_{}.xdmf'.format(freq)
        file = XDMFFile(init_path)
        file.read_checkpoint(self.init_u, 'u_past', 0)
        file.read_checkpoint(self.init_v, 'v_past', 0)

        # extracting the indecies of the solution at the top boundary
        self.compute_boundary_indecies()

        self.u_past = Function( self.V )
        self.v_past = Function( self.V )

        self.a = self.v*self.t*dx 
        self.L = self.v_past*self.t*dx - self.dt/2*self.c*inner( grad(self.u_past), grad(self.t) )*dx - self.dt/2*self.v_past*self.t*self.ds(0) - self.dt/2*self.v_past*self.t*self.ds(1) - self.dt/2*self.u0*self.t*self.ds(2) #+ self.dt/2*self.source*self.t*dx

    # projecting the wave speed function onto the FEM basis
    def compute_wave_speed(self, p, s):
        self.speed_function.set_s(s)
        self.speed_function.assemble_curve(p)
        temp = interpolate(self.speed_function, self.V)
        self.c.vector().set_local( temp.vector().get_local() )

    # defining the second order symplectic integrator
    def stormer_verlet_step(self):
        b = assemble(self.L)
        self.solver.solve(self.temp.vector(), b)
        self.v_past.vector().set_local( self.temp.vector().get_local() )

        temp = self.u_past.vector().get_local() + self.dt*self.v_past.vector().get_local()
        self.u_past.vector().set_local( temp )

        b = assemble(self.L)
        self.solver.solve(self.temp.vector(), b)
        self.v_past.vector().set_local( self.temp.vector().get_local() )

    # this subroutine  advances the PDE in time
    def time_stepping(self):
        A = assemble(self.a)
        self.solver = LUSolver(A)

        #sol = Function(self.V)

        self.u_past.vector().set_local( self.init_u )
        self.v_past.vector().set_local( self.init_v )

        path = './solution/sol0_ref.pvd'
        file = File(path)

        t = 0
        for i in progressbar( range(2000) ):
            self.stormer_verlet_step()
            t += self.dt
            if( np.mod(i,10) == 0 ):
                file << (self.u_past, i*self.dt)

    def time_stepping_xdmf(self,path='./solution/sol4_ref.pvd'):
        A = assemble(self.a)
        self.solver = LUSolver(A)

        self.u_past.assign( self.init_u )
        self.v_past.assign( self.init_v )

        file = File(path)

        t = 0
        for i in progressbar( range(2000) ):
            self.stormer_verlet_step()
            t += self.dt
            if( np.mod(i,10) == 0 ):
                file << (self.u_past, i*self.dt)

    def time_stepping_save_png(self, freq_idx):
        A = assemble(self.a)
        self.solver = LUSolver(A)

        self.u_past.assign( self.init_u )
        self.v_past.assign( self.init_v )

        path = './solution_png/sol_{}'.format(freq_idx) +'_{:05d}.png'
        f, ax = plt.subplots(1)

        t = 0
        idx = 0

        loc_x = np.linspace(-3,3,186)
        for i in progressbar( range(2000) ):
            self.stormer_verlet_step()

            if( np.mod(i,2) == 0 ):
                plt.sca(ax)
                plot(self.u_past, mode='color', vmin=0, vmax=1.1e-4)
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
        A = assemble(self.a)
        self.solver = LUSolver(A)

        out = []
        for i in progressbar( range(2000) ):
            self.stormer_verlet_step()
            if(i>=700):
                out.append( self.u_past.vector().get_local()[self.bnd_idx].reshape(1,-1) )
        return np.concatenate(out, axis=0)

    def compute_boundary_indecies(self):
        FEM_el = self.V.ufl_element()
        boundary = lambda x, on_boundary: on_boundary and near(x[1],1.5, 1E-14)
        u0 = Constant('0.0')
        zero_bc = DirichletBC(self.V, u0, boundary)

        dummy = Function(self.V)
        dummy.vector().set_local( np.ones_like( dummy.vector().get_local() ) )
        zero_bc.apply( dummy.vector() )
        self.bnd_idx = np.argwhere( dummy.vector().get_local() == 0 ).flatten()

        x_func = x_boundary(element=FEM_el)
        x_bnd = DirichletBC(self.V, x_func, boundary)
        func = Function(self.V)
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

    def save_state(self, time, time_steps, freq, num_source):
        u_past = self.u_past.vector().get_local()
        v_past = self.v_past.vector().get_local()

        path = './model_params/init_state_structured_freq_{}.npz'
        np.savez(path.format(freq), u_past_np=u_past, v_past_np=v_past, dt=self.dt, time=time, time_steps=time_steps, freq=freq, num_source=num_source)


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

        out = problem.time_stepping_save_png(i)

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
    N_x=1024
    N_KL=256 

    fmT = 100
    problem = wave(N_x=512, N_KL=N_KL)
    problem.initiate_load_source_xdmf(fmT)
    data = np.load('./model_params/png_npz_params.npz')
    p = data['p']
    problem.compute_wave_speed( p )
    problem.time_stepping_xdmf(path='./solution/sol_s_ref.pvd')


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



if __name__ == '__main__':
    save_ref_solution()
    #save_solution()
    #save_png()