import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from matern import matern

from progressbar import progressbar

import dolfin as dl
import arviz
#import mshr

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

class wave_speed_custom(dl.UserExpression):
    def __init__(self, N_x, **kwargs):
        super().__init__(**kwargs)

        self.x_grid = np.linspace(-3.001,3.001,N_x)
        self.curve = self.make_curve(self.x_grid)
        #self.curve = np.zeros_like(self.x_grid)

    def make_curve(self,x):
        y = np.zeros_like(x)
        idx = np.array( (x>-1.2)*(x<=-.8) , dtype='int')
        np_idx = np.where(idx>0)[0]

        x_local = np.linspace(-.2,.2,sum(idx))
        y_local = -0.25*(0.5 + 0.5*np.tanh(20*x_local))

        y[np_idx] = y_local

        idx = np.array( (x>-.8)*(x<=-.7) , dtype='int')
        np_idx = np.where(idx>0)[0]

        y[np_idx] = -0.25*np.ones_like(np_idx)

        idx = np.array( (x>-.7)*(x<=-.3) , dtype='int')
        np_idx = np.where(idx>0)[0]

        x_local = np.linspace(-.2,.2,sum(idx))
        y_local = -0.25*(0.5 + 0.5*np.tanh(-20*x_local))

        y[np_idx] = y_local

        idx = np.array( (x>.15)*(x<=.85) , dtype='int')
        np_idx = np.where(idx>0)[0]

        x_local = np.linspace(-.35,.35,sum(idx))
        y_local = 0.5*(0.5 + 0.5*np.tanh(10*x_local))

        y[np_idx] = y_local

        idx = np.array( (x>.85)*(x<=1.15) , dtype='int')
        np_idx = np.where(idx>0)[0]

        y[np_idx] = 0.5*np.ones_like(np_idx)

        idx = np.array( (x>1.15)*(x<=1.85) , dtype='int')
        np_idx = np.where(idx>0)[0]

        x_local = np.linspace(-.35,.35,sum(idx))
        y_local = 0.5*(0.5 + 0.5*np.tanh(-10*x_local))

        y[np_idx] = y_local
        return y

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

    def plot_curve(self, ax, label=None, color=None):
        u = self.curve
        x = np.linspace( -3,3, len(u) )
        ax.plot(x,u,label=label, color=color)
        ax.set_aspect('equal')
        #ax.set_xlim([-2,2])
        ax.set_ylim([-1.5,1.5])

class wave_density_custom(dl.UserExpression):
    def __init__(self, N_x, **kwargs):
        super().__init__(**kwargs)

        self.x_grid = np.linspace(-3.001,3.001,N_x)
        self.curve = self.make_curve(self.x_grid)
        #self.curve = np.zeros_like(self.x_grid)

    def make_curve(self, x):
        y = np.zeros_like(x)
        idx = np.array( (x>-1.2)*(x<=-.8) , dtype='int')
        np_idx = np.where(idx>0)[0]

        x_local = np.linspace(-.2,.2,sum(idx))
        y_local = -0.25*(0.5 + 0.5*np.tanh(20*x_local))

        y[np_idx] = y_local

        idx = np.array( (x>-.8)*(x<=-.7) , dtype='int')
        np_idx = np.where(idx>0)[0]

        y[np_idx] = -0.25*np.ones_like(np_idx)

        idx = np.array( (x>-.7)*(x<=-.3) , dtype='int')
        np_idx = np.where(idx>0)[0]

        x_local = np.linspace(-.2,.2,sum(idx))
        y_local = -0.25*(0.5 + 0.5*np.tanh(-20*x_local))

        y[np_idx] = y_local

        idx = np.array( (x>.15)*(x<=.85) , dtype='int')
        np_idx = np.where(idx>0)[0]

        x_local = np.linspace(-.35,.35,sum(idx))
        y_local = 0.5*(0.5 + 0.5*np.tanh(10*x_local))

        y[np_idx] = y_local

        idx = np.array( (x>.85)*(x<=1.15) , dtype='int')
        np_idx = np.where(idx>0)[0]

        y[np_idx] = 0.5*np.ones_like(np_idx)

        idx = np.array( (x>1.15)*(x<=1.85) , dtype='int')
        np_idx = np.where(idx>0)[0]

        x_local = np.linspace(-.35,.35,sum(idx))
        y_local = 0.5*(0.5 + 0.5*np.tanh(-10*x_local))

        y[np_idx] = y_local
        return y

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

    def plot_curve(self, ax, label=None, color=None):
        u = self.curve
        x = np.linspace( -3,3, len(u) )
        ax.plot(x,u,label=label, color=color)
        ax.set_aspect('equal')
        #ax.set_xlim([-2,2])
        ax.set_ylim([-1.5,1.5])

class wave():
    def __init__(self, N_x=256):
        # uncomment to save mesh
        #self.mesh = dl.RectangleMesh(dl.Point(-3, -1.5), dl.Point(3, 1.5), 188, 94)
        self.mesh = dl.RectangleMesh(dl.Point(-3, -1.5), dl.Point(3, 1.5), 376, 188)
        #with XDMFFile("./model_params/mesh_structured.xdmf") as file:
        #    file.write(self.mesh)

        # uncomment to load mesh from file
        #self.mesh = dl.Mesh()
        #with dl.XDMFFile("./model_params/mesh_structured.xdmf") as infile:
        #    infile.read(self.mesh)

        # defining the function space
        self.V = dl.FunctionSpace(self.mesh,'CG', 1)
        #self.dt = 0.0019
        self.dt = 0.00095

        # defining test and trial spaces
        self.t = dl.TestFunction(self.V)
        self.u = dl.TrialFunction(self.V)
        self.v = dl.TrialFunction(self.V)

        # defining the seabed curve
        self.FEM_el = self.V.ufl_element()
        self.speed_function = wave_speed_custom(N_x,element=self.FEM_el)#wave_speed_matern(N_x,N_KL,element=self.FEM_el)
        self.density_function = wave_density_custom(N_x,element=self.FEM_el)#wave_density_matern(N_x,N_KL,element=self.FEM_el)
        self.c = dl.Function(self.V)
        self.rho = dl.Function(self.V)

        temp = dl.interpolate(self.speed_function, self.V)
        self.c.assign( temp )
        temp = dl.interpolate(self.density_function, self.V)
        self.rho.assign( temp )

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

        init_path = './model_params/init_state_elastic_density_fine_freq_{}.xdmf'.format(freq)
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
        time_steps=150
        # time steps
        for i in progressbar( range(time_steps) ):
            self.source.t = t
            self.stormer_verlet_step()
            #f, ax = plt.subplots(1)
            #c = dl.plot(self.u_past)
            #plt.savefig('./solution/fig{}.pdf'.format(i))
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

        path = './solution/sol{}.pvd'.format(idx)
        file = dl.File(path)

        t = 0

        #sol = []
        for i in progressbar( range(4000) ):
            self.stormer_verlet_step()
            t += self.dt

            #sol.append( self.u_past.vector().get_local() )
            if( np.mod(i,10) == 0 ):
                file << (self.u_past, i*self.dt)

        #return np.array(sol)

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

        path = './solution_png/sol_{}'.format(freq_idx) +'_{:05d}.png'
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
        #t = 0
        #ts = []
        for i in progressbar( range(4000) ):
            self.stormer_verlet_step()
            #t += self.dt
            #ts.append(t)
            if(i>=1000):
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
        self.bnd_idx = self.bnd_idx[::2]

    def forward(self, p=0, s=0.75):
        #self.compute_wave_speed(p, s)
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
        path = './model_params/init_state_elastic_density_fine_freq_{}.xdmf'.format(freq)

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

def script_save_initial_state():
    N_x=512
    
    freq = [ 50, 75, 100 ]
    for fmT in freq:
        problem = wave(N_x=N_x)
        print(fmT)
        problem.initiate_zero_init(fmT)
        problem.save_initial_state()
    #problem.initiate_load_source_xdmf(fmT)
    #problem.time_stepping(0)

def save_solution():
    N_x=512

    fmT = 10
    problem = wave(N_x=N_x)
    problem.initiate_load_source_xdmf(fmT)
    problem.time_stepping(0)


def save_costum_sginal():
    #N_x = 512
    #problem = wave(N_x=N_x)

    #fmT = 10
    #problem = wave(N_x=N_x)
    #problem.initiate_load_source_xdmf(fmT)
    ##problem.time_stepping(0)
    #out = problem.forward()
    #out = out[1::2,:]

    #plt.imshow(out)
    #plt.savefig('out2.pdf')
    #exit()

    N_x=512

    fmT = np.array( [10, 25, 50, 75, 100] )
    obs = []

    for freq in fmT:
        problem = wave(N_x=N_x)
        problem.initiate_load_source_xdmf(freq)

        out = problem.forward()
        out = out[1::2,:]
        obs.append(out)

        f, ax = plt.subplots(1)
        ax.imshow(out)
        ax.set_xlabel('x')
        ax.set_ylabel('time')
        plt.savefig('./obs/obs_costum/freq_{}.pdf'.format(freq), format='pdf', dpi=300)

    obs_true = np.array(obs)
    noise_vec = np.random.standard_normal( obs_true.shape )
    for i in range( obs_true.shape[0] ):
        noise_vec[i] /= np.linalg.norm( noise_vec[i].flatten() )

    np.savez('./obs/obs_costum/obs.npz', N_x=N_x, N_KL=256, obs_true=obs_true, noise_vec=noise_vec )


    #    problem.time_stepping(i)




if __name__ == '__main__':
    #save_solution()
    #script_save_initial_state()
    save_costum_sginal()

