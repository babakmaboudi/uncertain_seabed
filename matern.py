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

class white_noise_convolution_1d:
    """
    Interface generator by convolving 1D white noise with a Gaussian kernel.

    - Domain: x in [-L/2, L/2) (periodic)
    - Output: smooth curve Gamma(x) on an x_grid of size N_x
    - Control smoothness by sigma (Gaussian std in *physical x-units*)
    - Enforce range Gamma(x) in [-target_range, target_range] by rescaling
      (or tanh if you prefer guaranteed smooth bounding)
    """
    def __init__(self, N_x, L=6.0, sigma=0.15, target_range=0.3,
                 periodic=True, bound_mode="rescale", tanh_scale=1.0, seed=None):
        """
        N_x: number of x-grid points for the interface evaluation
        L: domain length (x in [-3,3] => L=6)
        sigma: Gaussian kernel std *in x units* (e.g. 0.1 ~ pretty smooth)
        target_range: final interface lies in [-target_range, target_range]
        periodic: if True, uses periodic convolution (recommended for homogeneity)
        bound_mode: "rescale" (exactly fits max|Gamma|=target_range per-sample),
                    "tanh" (smooth bounding; close to Gaussian if tanh_scale large),
                    "clip" (hard clip)
        tanh_scale: used only if bound_mode="tanh" (larger -> less saturation)
        """
        self.N_x = int(N_x)
        self.L = float(L)
        self.sigma = float(sigma)
        self.target_range = float(target_range)
        self.periodic = bool(periodic)
        self.bound_mode = str(bound_mode)
        self.tanh_scale = float(tanh_scale)
        self.rng = np.random.default_rng(seed)

        # periodic grid in [-L/2, L/2)
        self.x_grid = np.linspace(-self.L / 2.0, self.L / 2.0, self.N_x, endpoint=False)
        self.dx = self.x_grid[1] - self.x_grid[0]

        self.kernel = self._make_gaussian_kernel()

        if self.periodic:
            # FFT of kernel for fast periodic convolution
            self.kernel_fft = np.fft.rfft(self.kernel)
        else:
            self.kernel_fft = None

    def _make_gaussian_kernel(self):
        """
        Build a discrete Gaussian kernel aligned for convolution on the periodic grid.
        For periodic convolution, we want the kernel centered at 0, represented in
        wrap-around coordinates.
        """
        # wrap-around distances from 0 on periodic grid
        x = self.x_grid
        # map x into shortest periodic distance to 0:
        dist = np.minimum(np.abs(x), self.L - np.abs(x))

        k = np.exp(-0.5 * (dist / (self.sigma + 1e-15)) ** 2)

        # normalize kernel so it acts like a smoothing average
        k /= (np.sum(k) + 1e-15)
        return k

    def sample_white_noise(self, size=None):
        """Convenience sampler."""
        n = self.N_x if size is None else int(size)
        return self.rng.standard_normal(n)

    def _convolve(self, noise_on_grid):
        if self.periodic:
            # periodic convolution via FFT
            y = np.fft.irfft(np.fft.rfft(noise_on_grid) * self.kernel_fft, n=self.N_x)
            return y
        else:
            # non-periodic convolution (zero-padding, same length)
            # (not homogeneous near boundaries)
            return np.convolve(noise_on_grid, self.kernel, mode="same")

    def _enforce_range(self, u):
        if self.bound_mode == "rescale":
            m = np.max(np.abs(u)) + 1e-15
            return (self.target_range / m) * u
        elif self.bound_mode == "tanh":
            s = (np.std(u) + 1e-15) * self.tanh_scale
            return self.target_range * np.tanh(u / s)
        elif self.bound_mode == "clip":
            return np.clip(u, -self.target_range, self.target_range)
        else:
            raise ValueError("bound_mode must be one of: 'rescale', 'tanh', 'clip'")

    def assemble(self, p):
        """
        p can be:
          - shape (N_x,): interpreted as white noise on the x_grid
          - shape (N_kl,): interpreted as white noise on a coarse grid, interpolated to N_x
        Returns curve Gamma(x_grid) in [-target_range, target_range].
        """
        p = np.asarray(p, dtype=float).reshape(-1)

        if p.size == self.N_x:
            noise = p
        else:
            # Treat p as coarse white noise on a coarse grid, then interpolate to x_grid.
            # This keeps your "parameter dimension" as len(p) = N_kl if you want that.
            Nc = p.size
            xc = np.linspace(-self.L/2.0, self.L/2.0, Nc, endpoint=False)

            # periodic interpolation: extend one point to avoid edge issues
            xc_ext = np.r_[xc, xc[0] + self.L]
            p_ext  = np.r_[p,  p[0]]

            x_ext = np.r_[self.x_grid, self.x_grid[0] + self.L]
            noise_ext = np.interp(x_ext, xc_ext, p_ext)
            noise = noise_ext[:-1]

        smooth = self._convolve(noise)
        smooth = smooth - np.mean(smooth)  # zero-mean interface height
        return self._enforce_range(smooth)

    def interpolated(self, p):
        p = np.asarray(p, dtype=float).reshape(-1)

        Nc = p.size
        xc = np.linspace(-self.L/2.0, self.L/2.0, Nc, endpoint=False)

        # periodic interpolation: extend one point to avoid edge issues
        xc_ext = np.r_[xc, xc[0] + self.L]
        p_ext  = np.r_[p,  p[0]]

        x_ext = np.r_[self.x_grid, self.x_grid[0] + self.L]
        noise_ext = np.interp(x_ext, xc_ext, p_ext)
        noise = noise_ext[:-1]

        return noise

def plot_bottom():
    N = 256
    num_samples = 10
    f, axes = plt.subplots(1)
    prior = matern(512, L=6,num_terms=N,delta=1/0.4/0.4,s=.75)

    for i in range(15):
        p = np.random.standard_normal(N)
        u = prior.assemble(p)
        axes.plot( np.linspace(-3,3,512), 5*u )
        #axes[i].set_xticks([])

    axes.set_aspect('equal')
    axes.set_xlim([-3,3])
    axes.set_ylim([-1.5,1.5])
    plt.tight_layout()
    #f.subplots_adjust(wspace=0)
    plt.show()

def plot_bottom_obs():
    obs_data = np.load('./obs/obs2/obs.npz')
    p = obs_data['param_true']
    N_x = obs_data['N_x']
    N_KL = obs_data['N_KL']
    prior = matern(N_x, L=6,num_terms=N_KL,delta=1/0.4/0.4,s=.75,load_basis=True)

    f,ax = plt.subplots(1)
    u = prior.assemble(p)
    x = np.linspace(-3,3,N_x)
    ax.plot( x, 5*u )
    ax.set_xlim([-3,3])
    ax.set_ylim(-1.5,1.5)
    ax.set_aspect('equal')
    plt.show()


def test_white_noise_convolution():
    N = 512
    N_white_noise = 64
    num_samples = 2
    f, axes = plt.subplots(1)
    prior = white_noise_convolution_1d(N, sigma=0.25)
    for i in range(num_samples):
        p = np.random.standard_normal(N_white_noise)
        u = prior.assemble(p)
        axes.plot( np.linspace(-3,3,512), u )
        #axes[i].set_xticks([])

    axes.set_aspect('equal')
    axes.set_xlim([-3,3])
    axes.set_ylim([-1.5,1.5])
    plt.tight_layout()
    #f.subplots_adjust(wspace=0)
    plt.show()


if __name__ == '__main__':
    #plot_bottom()
    test_white_noise_convolution()
