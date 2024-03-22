import numpy as np
import matplotlib.pyplot as plt
from wave import wave_speed_matern
from wave_costum_seabed import wave_speed_custom
import pickle

import seaborn as sns


def post_process_pCN():

    obs_data = np.load('./obs/obs2/obs.npz')
    print(obs_data.files)
    p_true = obs_data['param_true']
    N_x = obs_data['N_x']
    N_KL = obs_data['N_KL']
    field = wave_speed_matern(N_x,N_KL)

    field.set_s(0.75)

    stat_data1 = np.load('./stat/stat_test_2_2.npz')
    stat_data2 = np.load('./stat/stat_test_2_3.npz')

    #samples_s = stat_data['s']
    #s_mean = np.mean(samples_s)

    s = 0.75

    field.set_s(s)
    #samples_p = stat_data['p'].T
    samples1_p = (stat_data1['samples'])
    samples2_p = (stat_data2['samples'])

    print(samples1_p.shape)
    print(samples2_p.shape)

    samples_p = np.concatenate( [samples1_p, samples2_p], axis=0 )

    samples_p = samples_p[10000:,:]


    #f, ax = plt.subplots(1)
    #for i in range(5):
    #    ax.plot(samples_p[:,i], label='mode no. {}'.format(i+1))

    #ax.set_xlabel('sample idx',fontsize = 18)
    #ax.set_ylabel('trace', fontsize = 18)
    #ax.legend(fontsize = 12)
    #plt.savefig('./plots/trace_pCN.pdf',dpi=300)
    #exit()
    

    #np.savez('./checkpoints/checkpoint1.npz',s0=samples_s[0,-1], p0=samples_p[-1,:])
    #samples_p = samples_p[300:,:]

    #plt.figure()
    #plt.plot(samples_p[:,0])
    #plt.savefig('trace.pdf')
    #exit()

    idx = np.random.permutation( samples_p.shape[0] )

    f,ax = plt.subplots(1, figsize=[6,3])
    field.plot_curve(samples_p[idx[0]],ax, label='sample', color='blue')
    field.plot_curve(samples_p[idx[1]],ax, label='sample', color='red')
    field.plot_curve(samples_p[idx[2]],ax, label='sample', color='green')
    field.plot_curve(samples_p[idx[3]],ax, label='sample', color='orange')
    field.plot_curve(p_true,ax, label='true seabed', color='black')
    ax.set_xlabel('x (km)',fontsize = 18)
    ax.set_ylabel('y (km)', fontsize = 18)
    ax.legend(fontsize = 12)
    plt.tight_layout()
    plt.savefig('./plots/posterior_samples_pCN.pdf',dpi=300)


    samples_mean = np.mean(samples_p, axis=0)

    #print(np.mean( samples_s[0,400:]) )

    f,ax = plt.subplots(1, figsize=[6,3])
    field.plot_curve(p_true,ax, label='true seabed', color='blue')
    field.plot_curve(samples_mean,ax, label='mean seabed', color='orange')
    field.plot_uq(samples_p, ax, label='99% CI')
    ax.set_xlabel('x (km)',fontsize = 18)
    ax.set_ylabel('y (km)', fontsize = 18)
    ax.set_title('seabed estimate with fixed roughness')
    ax.legend(fontsize = 12)

    plt.tight_layout()
    plt.savefig('./plots/mean_uq_pCN.pdf',dpi=300)


def post_process_gibbs():

    obs_data = np.load('./obs/obs_costum/obs.npz')
    print(obs_data.files)
    #p_true = obs_data['param_true']
    N_x = obs_data['N_x']
    N_KL = obs_data['N_KL']
    field = wave_speed_custom(N_x)
    field_matern = wave_speed_matern(N_x, N_KL)

    #field.set_s(0.75)




    with open('./stat/stat_custom_gibbs.pickle', 'rb') as handle:
        stat_data = pickle.load(handle)

    samples_p_1 = stat_data['p']
    samples_s_1 = stat_data['s']

    with open('./stat/stat_custom_gibbs_1.pickle', 'rb') as handle:
        stat_data = pickle.load(handle)

    samples_p_2 = stat_data['p']
    samples_s_2 = stat_data['s']

    with open('./stat/stat_custom_gibbs_2.pickle', 'rb') as handle:
        stat_data = pickle.load(handle)

    samples_p_3 = stat_data['p']
    samples_s_3 = stat_data['s']

    #with open('./stat/stat_elastic_gibbs_3.pickle', 'rb') as handle:
    #    stat_data = pickle.load(handle)

    #samples_p_4 = stat_data['p']
    #samples_s_4 = stat_data['s']

    #with open('./stat/stat_elastic_gibbs_4.pickle', 'rb') as handle:
    #    stat_data = pickle.load(handle)

    #samples_p_5 = stat_data['p']
    #samples_s_5 = stat_data['s']

    #with open('./stat/stat_no_cuqi_3.pickle', 'rb') as handle:
    #    stat_data = pickle.load(handle)

    #samples_p_4 = stat_data['p']
    #samples_s_4 = stat_data['s']

    samples_p = np.concatenate([samples_p_1,samples_p_2,samples_p_3], axis=0)
    samples_s = np.concatenate([samples_s_1,samples_s_2,samples_s_3], axis=0)
    samples_p = samples_p[1000:,:]
    samples_s = samples_s[1000:]
    print(samples_p.shape)
    print(samples_s.shape)


    #f,ax = plt.subplots(1)
    #ax.plot(samples_s.reshape(-1))
    #plt.savefig('./plots/custom_trace_s.pdf',dpi=300)
    #exit()

    # Thinning
    #samples_p = samples_p[1000:,:]
    #samples_s = samples_s[1000:]

    s_mean = np.mean(samples_s,axis=0)
    print(s_mean)

    field_matern.set_s(s_mean)
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
    #plt.savefig('./plots/custom_trace_gibbs.pdf',dpi=300)
    #exit()

    samples_mean = np.mean(samples_p, axis=0)

    #print(np.mean( samples_s[0,400:]) )


    #p_mean = np.mean(samples_p[8:,:],axis=0)

    f,ax = plt.subplots(1, figsize=[6,3])
    field.plot_curve(ax, label='true seabed', color='blue')
    field_matern.plot_curve(samples_mean,ax, label='mean seabed', color='orange')
    field_matern.plot_uq(samples_p, ax, label='99% CI')
    ax.set_xlabel('x (km)',fontsize = 18)
    ax.set_ylabel('y (km)', fontsize = 18)
    ax.set_title('seabed estimate with unknown roughness')
    ax.legend(fontsize = 12)

    plt.tight_layout()
    plt.savefig('./plots/custom_curve_gibbs.pdf',dpi=300)

    idx = np.random.permutation( samples_p.shape[0] )

    f,ax = plt.subplots(1, figsize=[6,3])
    field_matern.set_s(samples_s[idx[0]])
    field_matern.plot_curve(samples_p[idx[0]],ax, label='sample', color='blue')
    field_matern.set_s(samples_s[idx[1]])
    field_matern.plot_curve(samples_p[idx[1]],ax, label='sample', color='red')
    field_matern.set_s(samples_s[idx[2]])
    field_matern.plot_curve(samples_p[idx[2]],ax, label='sample', color='green')
    field_matern.set_s(samples_s[idx[3]])
    field_matern.plot_curve(samples_p[idx[3]],ax, label='sample', color='orange')
    field_matern.set_s(0.75)
    field.plot_curve(ax, label='true seabed', color='black')
    ax.set_xlabel('x (km)',fontsize = 18)
    ax.set_ylabel('y (km)', fontsize = 18)
    ax.legend(fontsize = 12)
    plt.tight_layout()
    plt.savefig('./plots/custom_posterior_samples_gibbs.pdf',dpi=300)

    f,ax = plt.subplots(1, figsize=[6,4])
    sns.kdeplot(samples_s, ax=ax, linewidth=2)
    ax.axvline(x = 0.75, color = 'r', linewidth=2)
    ax.axvline(x = s_mean, color = 'g', linewidth=2)
    ax.set_xlabel('s')
    ax.legend(['seabed roughness posterior', 'true roughness', 'mean roughness'])
    ax.set_xlim([0.5,1])
    
    #exit()
    plt.tight_layout()
    plt.savefig('./plots/custom_posterior_roughness.pdf',dpi=300)


if __name__ == '__main__':
    #post_process_pCN()
    post_process_gibbs()
