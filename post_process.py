import numpy as np
import matplotlib.pyplot as plt
from wave import wave_speed_matern
import pickle

import seaborn as sns


plt.rcParams['xtick.labelsize']=16
plt.rcParams['ytick.labelsize']=16


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


def post_process_pCN():

    obs_data = np.load('./obs/obs_smooth/obs.npz')
    print(obs_data.files)
    p_true = obs_data['param_true']
    N_x = obs_data['N_x']
    N_KL = obs_data['N_KL']
    field = wave_speed_matern(N_x,N_KL)

    field.set_s(0.75)

    #stat_data1 = np.load('./stat/stat_test_2_2.npz')
    #stat_data2 = np.load('./stat/stat_test_2_3.npz')
    stat_data = np.load('./stat/stat_smooth_AIES.npz')

    #samples_s = stat_data['s']
    #s_mean = np.mean(samples_s)

    s = 0.75

    field.set_s(s)
    #samples_p = stat_data['p'].T
    #samples1_p = (stat_data1['samples'])
    #samples2_p = (stat_data2['samples'])
    samples = stat_data['samples']
    

    #samples = samples[:,39,:]
    samples = samples.reshape(-1,256)
    samples = samples[8000:,:]



    f, ax = plt.subplots(1,2)
    for i in range(5):
        ax[0].plot(samples[:,i], label=r'$\beta_{}$'.format(i+1))
        ax[1].plot(samples[:,10+i], label=r'$\beta_{}$'.format(i+11))

    ax[0].set_xlabel('sample index',fontsize = 18)
    ax[0].set_ylabel('trace', fontsize = 18)
#    ax.set_xticks([0,10000,20000,30000])
#    ax.set_xticklabels(['0','10K','20K','30K'])
    ax[0].legend(fontsize = 16)
    ax[0].grid()
    ax[1].set_xlabel('sample index',fontsize = 18)
    ax[1].set_ylabel('trace', fontsize = 18)
#    ax.set_xticks([0,10000,20000,30000])
#    ax.set_xticklabels(['0','10K','20K','30K'])
    ax[1].legend(fontsize = 16)
    ax[1].grid()

    plt.tight_layout()
    #plt.savefig('./plots/trace_pCN.pdf',dpi=300)

    #samples = samples[2000:,:]

    f, ax = plt.subplots(1)

    means = np.mean(samples, axis=0)
    hdi_lower, hdi_higher = hpd_interval_numpy(samples, mass=0.95)

    f,ax = plt.subplots(1)
    indecies = np.linspace(0,255,256,endpoint=True)
    ax.plot(indecies,means)
    ax.fill_between(indecies, hdi_lower, hdi_higher, color='orange', alpha=0.3, label='94% HDI')

    #means = np.array(means)
    #intervals = np.array(intervals).T
    #intervals = intervals - means
    #intervals[0,:] = -intervals[0,:]

    #ax.errorbar(np.array(list(range(25))), means, yerr=intervals, fmt='o', elinewidth=3, label=r'95% HPD', alpha=0.7)
    #ax.scatter(np.array(list(range(25))), p_true[:25], label=r'true $\beta_j$', color='orange')
    #ax.set_xlabel(r'index $j$',fontsize = 18)
    #ax.set_ylabel(r'$\beta_j$', fontsize = 18)
    #ax.set_xticks([0,4,9,14,19,24])
    #ax.set_xticklabels([1,5,10,15,20,25])
    #ax.legend(fontsize = 16)
    #ax.grid()
    #plt.tight_layout()
    #plt.savefig('./plots/hdi_pCN.pdf',dpi=300)

    #for i in range(50):
    #    print(arviz.ess(samples_p[:,i]))
    #exit()
    

    #np.savez('./checkpoints/checkpoint1.npz',s0=samples_s[0,-1], p0=samples_p[-1,:])
    #samples_p = samples_p[300:,:]

    #plt.figure()
    #plt.plot(samples_p[:,0])
    #plt.savefig('trace.pdf')
    #exit()

    idx = np.random.permutation( samples.shape[0] )

    f,ax = plt.subplots(1, figsize=[6,3])
    field.plot_curve(samples[idx[0]],ax, label='sample', color='blue')
    field.plot_curve(samples[idx[1]],ax, label='sample', color='red')
    field.plot_curve(samples[idx[2]],ax, label='sample', color='green')
    field.plot_curve(samples[idx[3]],ax, label='sample', color='orange')
    field.plot_curve(p_true,ax, label='true seabed', color='black')
    ax.set_xlabel('x (km)',fontsize = 18)
    ax.set_ylabel('y (km)', fontsize = 18)
    ax.legend(fontsize = 12)
    plt.tight_layout()
    #plt.savefig('./plots/posterior_samples_pCN.pdf',dpi=300)


    samples_mean = np.mean(samples, axis=0)

    #print(np.mean( samples_s[0,400:]) )

    f,ax = plt.subplots(1, figsize=[6,3])
    field.plot_curve(p_true,ax, label='true seabed', color='blue')
    field.plot_curve(samples_mean,ax, label='mean seabed', color='orange')
    field.plot_uq(samples, ax, label='99% CI')
    ax.set_xlabel('x (km)',fontsize = 12)
    ax.set_ylabel('y (km)', fontsize = 12)
    #ax.set_title('seabed estimate with fixed roughness')
    ax.legend(fontsize = 12)

    plt.tight_layout()
    #plt.savefig('./plots/mean_uq_pCN.pdf',dpi=300)
    plt.show()


def post_process_gibbs():

    obs_data = np.load('./obs/obs2/obs.npz')
    print(obs_data.files)
    p_true = obs_data['param_true']
    N_x = obs_data['N_x']
    N_KL = obs_data['N_KL']
    field = wave_speed_matern(N_x,N_KL)

    field.set_s(0.75)




    with open('./stat/stat_elastic_gibbs.pickle', 'rb') as handle:
        stat_data = pickle.load(handle)

    samples_p_1 = stat_data['p']
    samples_s_1 = stat_data['s']

    with open('./stat/stat_elastic_gibbs_1.pickle', 'rb') as handle:
        stat_data = pickle.load(handle)

    samples_p_2 = stat_data['p']
    samples_s_2 = stat_data['s']

    with open('./stat/stat_elastic_gibbs_2.pickle', 'rb') as handle:
        stat_data = pickle.load(handle)

    samples_p_3 = stat_data['p']
    samples_s_3 = stat_data['s']

    with open('./stat/stat_elastic_gibbs_3.pickle', 'rb') as handle:
        stat_data = pickle.load(handle)

    samples_p_4 = stat_data['p']
    samples_s_4 = stat_data['s']

    with open('./stat/stat_elastic_gibbs_4.pickle', 'rb') as handle:
        stat_data = pickle.load(handle)

    samples_p_5 = stat_data['p']
    samples_s_5 = stat_data['s']

    #with open('./stat/stat_no_cuqi_3.pickle', 'rb') as handle:
    #    stat_data = pickle.load(handle)

    #samples_p_4 = stat_data['p']
    #samples_s_4 = stat_data['s']

    samples_p = np.concatenate([samples_p_1,samples_p_2,samples_p_3,samples_p_4,samples_p_5], axis=0)
    samples_s = np.concatenate([samples_s_1,samples_s_2,samples_s_3,samples_s_4,samples_s_5], axis=0)
    #samples_p = samples_p[300:,:]
    #samples_s = samples_s[300:,:]
    #samples_s = samples_s_3

    print(samples_p.shape)

    f, ax = plt.subplots(1)
    for i in range(5):
        ax.plot(samples_p[:,i], label=r'$\beta_{}$'.format(i+1))

    ax.set_xlabel('sample index',fontsize = 18)
    ax.set_ylabel('trace', fontsize = 18)
    ax.set_xticks([0,500,1000,1500,2000])
    ax.set_xticklabels(['0','0.5K','1K','1.5K','2K'])
    ax.legend(fontsize = 16)
    ax.grid()
    plt.tight_layout()
    plt.savefig('./plots/trace_Gibbs2.pdf',dpi=300)
    #samples_p = samples_p[15000:,:]

    f, ax = plt.subplots(1)
    ax.plot(samples_s, label=r'$s$')

    ax.set_xlabel('sample index',fontsize = 18)
    ax.set_ylabel('trace', fontsize = 18)
    ax.set_xticks([0,500,1000,1500,2000])
    ax.set_xticklabels(['0','0.5K','1K','1.5K','2K'])
    ax.legend(fontsize = 16)
    ax.grid()
    plt.tight_layout()
    plt.savefig('./plots/trace_Gibbs_s.pdf',dpi=300)

    f, ax = plt.subplots(1)

    means = []
    intervals = []
    for i in range(25):
        means.append( np.mean(samples_p[:,i]) )
        intervals.append( arviz.hdi(  samples_p[:,i], prob=0.95 ) )

    means = np.array(means)
    intervals = np.array(intervals).T
    intervals = intervals - means
    intervals[0,:] = -intervals[0,:]

    ax.errorbar(np.array(list(range(25))), means, yerr=intervals, fmt='o', elinewidth=3, label=r'95% HPD', alpha=0.7)
    ax.scatter(np.array(list(range(25))), p_true[:25], label=r'true $\beta_j$', color='orange')
    ax.set_xlabel(r'index $j$',fontsize = 18)
    ax.set_ylabel(r'$\beta_j$', fontsize = 18)
    ax.set_xticks([0,4,9,14,19,24])
    ax.set_xticklabels([1,5,10,15,20,25])
    ax.legend(fontsize = 16,loc=4)
    ax.grid()
    plt.tight_layout()
    plt.savefig('./plots/hdi_Gibbs.pdf',dpi=300)


    #f,ax = plt.subplots(1)
    #ax.plot(samples_s.reshape(-1))
    #plt.savefig('./plots/trace_s.pdf',dpi=300)
    #exit()

    # Thinning
    samples_p = samples_p[1500:,:]
    samples_s = samples_s[1000:]

    #for i in range(50):
    #    print(arviz.ess(samples_p[:,i]))

    #print(arviz.ess(samples_s.reshape(-1)))
    #exit()

    s_mean = np.mean(samples_s,axis=0)

    field.set_s(s_mean)
    #samples_p = stat_data['p'].T
    #np.savez('./checkpoints/checkpoint1.npz',s0=samples_s[0,-1], p0=samples_p[-1,:])
    #samples_p = samples_p[300:,:]

    #plt.figure()
    #plt.plot(samples_p[:,0])
    #plt.savefig('trace.pdf')
    #exit()

    #f, ax = plt.subplots(1)
    #for i in range(5):
    #    ax.plot(samples_p[:,i], label='mode no. {}'.format(i+1))

    #ax.set_xlabel('sample idx',fontsize = 18)
    #ax.set_ylabel('trace', fontsize = 18)
    #ax.legend(fontsize = 12)
    #plt.savefig('./plots/trace_gibbs.pdf',dpi=300)
    #exit()

    samples_mean = np.mean(samples_p, axis=0)

    #print(np.mean( samples_s[0,400:]) )


    #p_mean = np.mean(samples_p[8:,:],axis=0)

    #f,ax = plt.subplots(1, figsize=[6,3])
    #field.plot_curve(p_true,ax, label='true seabed', color='blue')
    #field.plot_curve(samples_mean,ax, label='mean seabed', color='orange')
    #field.plot_uq(samples_p, ax, label='99% CI')
    #ax.set_xlabel('x (km)',fontsize = 18)
    #ax.set_ylabel('y (km)', fontsize = 18)
    #ax.set_title('seabed estimate with unknown roughness')
    #ax.legend(fontsize = 12)

    #plt.tight_layout()
    #plt.savefig('./plots/curve_gibbs.pdf',dpi=300)
    samples_mean = np.mean(samples_p, axis=0)

    #print(np.mean( samples_s[0,400:]) )

    f,ax = plt.subplots(1, figsize=[6,3])
    field.plot_curve(p_true,ax, label='true seabed', color='blue')
    field.plot_curve(samples_mean,ax, label='mean seabed', color='orange')
    field.plot_uq(samples_p, ax, label='99% CI')
    ax.set_xlabel('x (km)',fontsize = 12)
    ax.set_ylabel('y (km)', fontsize = 12)
    #ax.set_title('seabed estimate with fixed roughness')
    ax.legend(fontsize = 12)
    plt.tight_layout()
    plt.savefig('./plots/mean_uq_Gibbs.pdf',dpi=300)

    #idx = np.random.permutation( samples_p.shape[0] )

    #f,ax = plt.subplots(1, figsize=[6,3])
    #field.set_s(samples_s[idx[0]])
    #field.plot_curve(samples_p[idx[0]],ax, label='sample', color='blue')
    #field.set_s(samples_s[idx[1]])
    #field.plot_curve(samples_p[idx[1]],ax, label='sample', color='red')
    #field.set_s(samples_s[idx[2]])
    #field.plot_curve(samples_p[idx[2]],ax, label='sample', color='green')
    #field.set_s(samples_s[idx[3]])
    #field.plot_curve(samples_p[idx[3]],ax, label='sample', color='orange')
    #field.set_s(0.75)
    #field.plot_curve(p_true,ax, label='true seabed', color='black')
    #ax.set_xlabel('x (km)',fontsize = 18)
    #ax.set_ylabel('y (km)', fontsize = 18)
    #ax.legend(fontsize = 12)
    #plt.tight_layout()
    #plt.savefig('./plots/posterior_samples_gibbs.pdf',dpi=300)

    f,ax = plt.subplots(1)
    #sns.kdeplot(samples_s, ax=ax, linewidth=2)
    sns.violinplot(x=samples_s.reshape(-1),ax=ax, label='KDE')
    ax.axvline(x = 0.75, color = 'orange', linewidth=2, label=r'true $s$')
    #ax.axvline(x = s_mean, color = 'g', linewidth=2)
    #ax.set_xlabel('s')
    ax.set_xlabel(r'$s$',fontsize = 18)
    ax.legend(loc=3,fontsize=18)
    ax.set_xlim([0.5,1])
    
    #exit()
    plt.tight_layout()
    plt.savefig('./plots/posterior_roughness.pdf',dpi=300)


if __name__ == '__main__':
    post_process_pCN()
    #post_process_gibbs()
