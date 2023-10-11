import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt

class matern():
    def __init__(self, N, num_terms = 64,L=1, delta = 1/0.008/0.008, s=0.5, load_basis=False, save_basis=False):
        if(load_basis==False):
            dx = L/N
            diag1 = -2*np.ones(N)
            diag2 = np.ones(N-1)

            Dxx = np.diag(diag1) + np.diag(diag2,1) + np.diag(diag2,-1)
            Dxx[0,0] = -1
            Dxx[-1,-1] = -1
            Dxx /= dx**2

            M = delta*np.eye(N) - Dxx
            eig_vals, eig_vecs = linalg.eig(M)

            eig_vals = np.real( eig_vals )
            idx = np.argsort( eig_vals )

            self.eig_vals = eig_vals[ idx ]
            self.eig_vals = self.eig_vals[1:num_terms+1]

            self.weights = np.float_power( self.eig_vals , -(s+0.5) )
            self.weights /= np.linalg.norm( self.weights )

            self.eig_vecs = eig_vecs[:, idx ]
            self.eig_vecs = self.eig_vecs[:, 1:num_terms+1]

            if(save_basis==True):
                np.savez('./model_params/matern_basis_no_s.npz', weights=self.weights, eig_vecs=self.eig_vecs, eig_vals=self.eig_vals)

        else:
            matern_data = np.load('./model_params/matern_basis_no_s.npz')
            self.eig_vals = matern_data['eig_vals']
            self.weights = np.float_power( self.eig_vals , -(s+0.5) )
            self.weights /= np.linalg.norm( self.weights )
            self.eig_vecs = matern_data['eig_vecs']

    def set_s(self, s):
            self.weights = np.float_power( self.eig_vals , -(s+0.5) )
            self.weights /= np.linalg.norm( self.weights )


    def assemble(self,p):
        return self.eig_vecs@( self.weights*p )

def plot_bottom():
    N = 256
    f, axes = plt.subplots(4,1,sharey=True)
    prior = matern(512, L=6,num_terms=N,delta=1/0.4/0.4,s=.75)

    for i in range(4):
        p = np.random.standard_normal(N)
        u = prior.assemble(p)
        axes[i].plot( 5*u )
        axes[i].set_xticks([])

    plt.tight_layout()
    f.subplots_adjust(wspace=0)
    plt.show()

        

if __name__ == '__main__':
    plot_bottom()
