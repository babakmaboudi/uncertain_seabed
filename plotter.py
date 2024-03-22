import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from wave import wave
import dolfin as dl


def plot_seabed():
    data = np.load('./obs/obs2/obs.npz')
    p_true = data['param_true']
    N_x = data['N_x']
    N_KL = data['N_KL']
    #field = wave_speed_matern(N_x,N_KL)

    y_true = data['obs_true']
    noise_vec = data['noise_vec']

    y_norm = np.linalg.norm(y_true, axis=1)
    sigmas = []
    cov_diag = []
    y_obs = []
    for i in range( y_true.shape[0] ):
        sigmas.append( 0.05*np.linalg.norm( y_true[i] ) )
        y_obs.append( y_true[i] + sigmas[i]*noise_vec[i] )
    y_obs = np.array(y_obs)

    
    
    t = np.linspace(700*0.0019,2000*0.0019,1300)
    y_coord = np.linspace(-3,3,189)

    freq = [10, 25, 50, 75, 100]

    f,axes = plt.subplots(5,1, sharex=True, figsize=[7,5])
    for i in range(y_true.shape[0]):
        axes[i].set_ylabel(r'$x$ (km)', fontsize=8)
        im = axes[i].imshow(y_obs[i].T, extent=[700*0.0019,2000*0.0019,-3,3], aspect='auto',vmin=-0.0001, vmax=0.0001)
        axes[i].text(3.6, 1.5,r'$f^0_{} = {}$'.format(i+1,freq[i]) , fontsize=8, ha='center', va='center')

    axes[i].set_xlabel(r'$t$ (s)')
    plt.tight_layout()

    f.subplots_adjust(right=0.85)
    cbar_ax = f.add_axes([0.9, 0.15, 0.01, 0.7])
    cbar = f.colorbar(im, cax=cbar_ax, ticks=[-0.0001, 0, 0.0001])
    cbar.formatter.set_powerlimits((0, 0))

    plt.savefig('./plots/measurement.pdf')
    exit()

    field.set_s(0.75)
    f,ax = plt.subplots(1, figsize = (6,3))
    field.plot_curve(p_true,ax, label='true seabed', color='blue')

    x1 = -2 * np.ones(100)
    x2 = 2 * np.ones(100)
    y = np.linspace(-1.5,1.5,100)

    ax.plot(x1,y,'r--')
    ax.plot(x2,y,'r--', label='computational domain')
    

    xx = np.linspace(-3,3,186)
    for x in xx:
        ax.plot(x,1.5,'gx')
    ax.plot(xx[0],1.5,'gx', label='sensor location')

    ax.legend()

    ax.set_xlabel('x (longitude in km)')
    ax.set_ylabel('y (depth in km)')

    plt.tight_layout()


    plt.savefig('bottom.pdf',dpi=300)

def plot_time_snapshots():
    data = np.load('./obs/obs2/obs.npz')
    p_true = data['param_true']
    N_x = data['N_x']
    N_KL = data['N_KL']

    fmT = 10
    problem = wave(N_x=N_x, N_KL=N_KL)
    problem.compute_wave_speed( p_true )

    path = './temp/sol' +'_{:05d}.png'

    sol_data = np.load('./obs/solution.npz')
    sol = sol_data['sol']

    loc_x = np.linspace(-3,3-6/46,46)
    fig, axes = plt.subplots(2,2,sharex=True, sharey=True, figsize=[7,4])

    indecies = [200, 350, 500, 650]

    for i in range(2):
        for j in range(2):
            f = dl.Function(problem.V)
            f.vector().set_local( sol[indecies[j*2+i] ] )
            plt.sca(axes[i,j])
            im = dl.plot(f, mode='color', vmin=0, vmax=1.1e-4)
            axes[i,j].plot( problem.speed_function.x_grid, problem.speed_function.curve, color='red' , label='seabed')
            axes[i,j].set_xlabel('x')
            axes[i,j].set_ylabel('y')

            axes[i,j].plot(loc_x[0],1.5,'.b',label='snesors')
            for k in range(1,46):
                axes[i,j].plot(loc_x[k],1.5,'.b')
            


    axes[0,0].set_ylabel(r'$y$ (km)')
    axes[0,0].set_xlabel(None)
    axes[0,1].set_xlabel(None)
    axes[0,1].set_ylabel(None)
    axes[1,0].set_ylabel(r'$y$ (km)')
    axes[1,0].set_xlabel(r'$x$ (km)')
    axes[1,1].set_ylabel(None)
    axes[1,1].set_xlabel(r'$x$ (km)')
    axes[1,1].legend(loc=4)

    axes[0,0].set_title(r'(a) $t=0.38$ (s)')
    axes[0,1].set_title(r'(b) $t=0.67$ (s)')
    axes[1,0].set_title(r'(c) $t=0.95$ (s)')
    axes[1,1].set_title(r'(d) $t=1.24$ (s)')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout()

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=[-0.0001, 0, 0.0001])
    cbar.formatter.set_powerlimits((0, 0))

    plt.savefig('plots/snaps.png'.format(i),format='png',dpi=300)
            #axes[i].clear()

    #loc_x = np.linspace(-3,3-6/46,46)
    #f, ax = plt.subplots(1)

    #indecies = [200, 350, 500, 650]

    #for i in range(len(indecies)):
    
    #    f = dl.Function(problem.V)
    #    f.vector().set_local( sol[ indecies[i] ] )
    #    plt.sca(ax)
    #    dl.plot(f, mode='color', vmin=0, vmax=1.1e-4)
        #ax.plot( problem.speed_function.x_grid, problem.speed_function.curve, color='red' , label='seabed')
        #ax.set_xlabel('x')
        #ax.set_ylabel('y')

        #ax.plot(loc_x[0],1.5,'.b',label='snesors')
        #for j in range(1,46):
        #    ax.plot(loc_x[j],1.5,'.b')
        #ax.legend(loc=4)


    #    plt.tight_layout()
    #    plt.savefig('plots/snap_{}.pdf'.format(i))
    #    ax.clear()

def plot_prior_samples():
    data = np.load('./obs/obs2/obs.npz')
    p_true = data['param_true']
    N_x = data['N_x']
    N_KL = data['N_KL']

    fmT = 10
    problem = wave(N_x=N_x, N_KL=N_KL)


    problem.compute_wave_speed( p_true )
    #fig, axes = plt.subplots(4,1,sharex=True, sharey=True, figsize=[6,3])
    fig, ax = plt.subplots(1,figsize=[6,3])

    p = np.random.standard_normal( p_true.shape )
    ss = np.array([0.6, 1., 1.5, 2.])
    colors = ['red', 'blue', 'black', 'green']

    for i, s in enumerate(ss):
        problem.compute_wave_speed( p=p, s=s )
        #axes[i].plot( problem.speed_function.x_grid, problem.speed_function.curve, color='red' , label='seabed')
        problem.speed_function.plot_curve(p, ax, label=r'$s={}$'.format(s), color=colors[i])

    ax.set_xlabel(r'x (km)')
    ax.set_ylabel(r'y (km)')

    plt.legend()
    #plt.subplots_adjust(hspace=0)
    plt.tight_layout()
    plt.savefig('plots/prior.png'.format(i),format='png',dpi=300)
        



if __name__ == '__main__':
    #plot_seabed()
    #plot_time_snapshots()
    plot_prior_samples()