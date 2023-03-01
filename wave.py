import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from matern import matern

from progressbar import progressbar

from dolfin import *
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.t = 0
        self.fmT = 10

    def eval(self, values, x):
        values[0] = (1- 2*np.pi**2*self.fmT**2*self.t**2 ) * np.exp( -np.pi**2*self.fmT**2*self.t**2  ) * np.exp( -( (x[0] + 1.8 )**2 + (x[1] - 1.4)**2)/(0.05**2) )
        values[0] += (1- 2*np.pi**2*self.fmT**2*self.t**2 ) * np.exp( -np.pi**2*self.fmT**2*self.t**2  ) * np.exp( -( (x[0] + 0.9 )**2 + (x[1] - 1.4)**2)/(0.05**2) )
        values[0] += (1- 2*np.pi**2*self.fmT**2*self.t**2 ) * np.exp( -np.pi**2*self.fmT**2*self.t**2  ) * np.exp( -( (x[0] )**2 + (x[1] - 1.4)**2)/(0.05**2) )
        values[0] += (1- 2*np.pi**2*self.fmT**2*self.t**2 ) * np.exp( -np.pi**2*self.fmT**2*self.t**2  ) * np.exp( -( (x[0] - 0.9)**2 + (x[1] - 1.4)**2)/(0.05**2) )
        values[0] += (1- 2*np.pi**2*self.fmT**2*self.t**2 ) * np.exp( -np.pi**2*self.fmT**2*self.t**2  ) * np.exp( -( (x[0] - 1.8)**2 + (x[1] - 1.4)**2)/(0.05**2) )

class wave_speed_matern(UserExpression):
    def __init__(self, N_x, N_kl, **kwargs):
        self.field = matern(N_x, num_terms=N_kl,s=6)
        self.x_grid = np.linspace(-3.001,3.001,N_x)
        self.curve = np.zeros(N_x)
        self.var = 3

        super().__init__(**kwargs)

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
    def __init__(self):
        # defining the mesh
        #self.mesh = UnitSquareMesh(100,100)
        #domain = mshr.Rectangle(Point(-3,-1.5), Point(3,1.5))
        #self.mesh = mshr.generate_mesh(domain, 60)
        #mesh_file = File('./model_params/mesh_extended.xml')
        #mesh_file << self.mesh
        self.mesh = Mesh('./model_params/mesh_extended.xml')

        # defining the function space
        self.V = FunctionSpace(self.mesh,'CG', 1)

        FEM_el = self.V.ufl_element()
        data = np.load('./model_params/state_extended.npz')
        self.init_u = data['u_past_np']
        self.init_v = data['v_past_np']

        # extracting the indecies of the solution at the top boundary
        self.compute_boundary_indecies()

        self.source = source_term(element=FEM_el)
        self.source_func = Function(self.V)

        self.dt = 0.005

        # defining test and trial spaces
        u0 = Constant('0.0')
        self.t = TestFunction(self.V)
        self.u = TrialFunction(self.V)
        self.v = TrialFunction(self.V)

        # defining the seabed curve
        N_x = 384
        N_kl = 64
        self.speed_function = wave_speed_matern(N_x,N_kl,element=FEM_el)
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
        ds = Measure('ds', domain=self.mesh, subdomain_data=boundary_markers)

        # defining the Dirichlet boundary on the botton of the domain
        self.zero_bc = DirichletBC(self.V, u0, boundary_diriichlet)

        #defining functions to hold the previous time-step
        self.u_past = Function( self.V )
        #self.u_past = interpolate(init, self.V )
        self.v_past = Function( self.V )

        self.a1 = self.v*self.t*dx 
        self.L1 = self.v_past*self.t*dx - self.dt/2*self.c*inner( grad(self.u_past), grad(self.t) )*dx - self.dt/2*self.v_past*self.t*ds(0) - self.dt/2*self.v_past*self.t*ds(1) - self.dt/2*u0*self.t*ds(2) #+ self.dt/2*self.source*self.t*dx

        self.a2 = self.u*self.t*dx 
        self.L2 = self.u_past*self.t*dx + self.dt*self.v_past*self.t*dx

        self.temp = Function(self.V)

    # projecting the wave speed function onto the FEM basis
    def compute_wave_speed(self, p):
        self.speed_function.assemble_curve(p)
        temp = interpolate(self.speed_function, self.V)
        self.c.vector().set_local( temp.vector().get_local() )

    # defining the second order symplectic integrator
    def stormer_verlet_step(self):
        b1 = assemble(self.L1)
        #self.zero_bc.apply(b1)
        self.solver.solve(self.temp.vector(), b1)
        self.v_past.vector().set_local( self.temp.vector().get_local() )

        temp = self.u_past.vector().get_local() + self.dt*self.v_past.vector().get_local()
        self.u_past.vector().set_local( temp )

        b1 = assemble(self.L1)
        #self.zero_bc.apply(b1)
        self.solver.solve(self.temp.vector(), b1)
        self.v_past.vector().set_local( self.temp.vector().get_local() )

    # this subroutine  advances the PDE in time
    def time_stepping(self):
        A = assemble(self.a1)
        self.solver = LUSolver(A)

        #sol = Function(self.V)

        path = './solution/sol.pvd'
        file = File(path)

        t = 0
        for i in progressbar( range(800) ):
            self.stormer_verlet_step()
            #t += self.dt
            #self.source.t = t
            if( np.mod(i,10) == 0 ):
                file << (self.u_past, i*self.dt)

    def read_boundary(self):
        A = assemble(self.a1)
        self.solver = LUSolver(A)

        out = []
        for i in range(700):
            self.stormer_verlet_step()
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

    def forward(self, p):
        self.compute_wave_speed(p)
        self.u_past.vector().set_local( self.init_u )
        self.v_past.vector().set_local( self.init_v )

        return self.read_boundary().flatten()
        
    def save_state(self):
        np.savez('./model_params/stat_extended.npz', u_past_np=self.u_past.vector().get_local(), v_past_np=self.v_past.vector().get_local())

    def save_wave_speed(self):
        file = File('speed.pvd')
        file << self.c

    def load_state(self):
        data = np.load('./model_params/state_extended.npz')
        u_past = data['u_past_np']
        v_past = data['v_past_np']

        self.u_past.vector().set_local( u_past )
        self.v_past.vector().set_local( v_past )

    def plot_wave_speed(self):
        plot( self.c )

if __name__ == '__main__':
    problem = wave()
    p = np.random.standard_normal(64)
    problem.compute_wave_speed(p)
    problem.save_wave_speed()

    exit()
    problem.load_state()
    problem.time_stepping()