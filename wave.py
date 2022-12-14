import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from matern import matern

from progressbar import progressbar

from dolfin import *
import mshr

set_log_level(50)

def boundary_diriichlet(x, on_boundary):
    return on_boundary and near(x[1], 0, 1E-14)

class Boundary_W(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0., 1E-14)

class Boundary_E(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0],1., 1E-14)

class Boundary_N(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1],1., 1E-14)

class init_cond(UserExpression):
    def eval(elf, values, x):
        values[0] = np.exp( -( (x[0] + 0.8)**2 + (x[1] - 0.4)**2)/(0.05**2) )
        values[0] += np.exp( -( (x[0] + 0.4 )**2 + (x[1] - 0.4)**2)/(0.05**2) )
        values[0] += np.exp( -( (x[0] )**2 + (x[1] - 0.4)**2)/(0.05**2) )
        values[0] += np.exp( -( (x[0] - 0.4)**2 + (x[1] - 0.4)**2)/(0.05**2) )
        values[0] += np.exp( -( (x[0] - 0.8)**2 + (x[1] - 0.4)**2)/(0.05**2) )

class wave_speed(UserExpression):
    def __init__(self, loc, **kwargs):
        self.loc = loc
        super().__init__(**kwargs)

    def eval(self, values, x):
        if (x[1]>self.loc):
            values[0] = 1.5
        else:
            values[0] = 6.4

class wave_speed_matern(UserExpression):
    def __init__(self, N_x, N_kl, **kwargs):
        self.field = matern(N_x, num_terms=N_kl,s=4)
        self.x_grid = np.linspace(-1.001,1.001,N_x)
        self.curve = np.zeros(N_x)

        super().__init__(**kwargs)

    def assemble_curve(self, p):
        self.curve = self.field.assemble(p)

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
        domain = mshr.Rectangle(Point(-1,-0.5), Point(1,0.5))
        self.mesh = mshr.generate_mesh(domain, 60)

        # defining the function space
        self.V = FunctionSpace(self.mesh,'CG', 1)

        FEM_el = self.V.ufl_element()
        init = init_cond(element=FEM_el)
        #dummy = Function(self.V)
        #dummy = interpolate( init, self.V )
        #file = File('init.pvd')
        #file << dummy
        #exit()

        self.dt = 0.002

        # defining test and trial spaces
        u0 = Constant('0.0')
        self.t = TestFunction(self.V)
        self.u = TrialFunction(self.V)
        self.v = TrialFunction(self.V)

        # defining the seabed curve
        N_x = 256
        N_kl = 64
        self.speed_function = wave_speed_matern(N_x,N_kl,element=FEM_el)
        p = np.random.standard_normal(N_kl)
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
        self.u_past = interpolate(init, self.V )
        self.v_past = Function( self.V )

        # defining the weakforms for the velocity and pressure equations
        self.a1 = self.v*self.t*dx 
        self.L1 = self.v_past*self.t*dx - self.dt/2*self.c*inner( grad(self.u_past), grad(self.t) )*dx - self.dt/2*self.v_past*self.t*ds(0) - self.dt/2*self.v_past*self.t*ds(1) - self.dt/2*u0*self.t*ds(2)

        self.a2 = self.u*self.t*dx 
        self.L2 = self.u_past*self.t*dx + self.dt*self.v_past*self.t*dx

        # temporary functions
        self.temp = Function(self.V)

    def compute_wave_speed(self, p):
        self.speed_function.assemble_curve(p)
        self.c = interpolate(self.speed_function, self.V)

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

        # computing the LU decomposition of the mass matrix
        A = assemble(self.a1)
        self.solver = LUSolver(A)

        for i in progressbar( range(500) ):
            # apply one time step
            self.stormer_verlet_step()

if __name__ == '__main__':
    problem = wave()

    # p is the parameters defining the random seabed curve
    p = np.random.standard_normal(64)
    problem.compute_wave_speed(p)
    
    problem.time_stepping()