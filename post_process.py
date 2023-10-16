import numpy as np
import matplotlib.pyplot as plt
from wave import wave_speed_matern


obs_data = np.load('./obs/obs1/obs.npz')
print(obs_data.files)
p_true = obs_data['param_true']
N_x = obs_data['N_x']
N_KL = obs_data['N_KL']
field = wave_speed_matern(N_x,N_KL)

field.set_s(0.75)

f,ax = plt.subplots(1)
field.plot_curve(p_true,ax, label='true seabed', color='blue')


stat_data = np.load('./stat/stat1_long_chain.npz')


#samples_s = stat_data['s']
#s_mean = np.mean(samples_s)

s_mean = 0.75

field.set_s(s_mean)
#samples_p = stat_data['p'].T
samples_p = (stat_data['samples'].T)[5000:,:]
#np.savez('./checkpoints/checkpoint1.npz',s0=samples_s[0,-1], p0=samples_p[-1,:])
#samples_p = samples_p[300:,:]

#plt.figure()
#plt.plot(samples_p[:,0])
#plt.savefig('trace.pdf')
#exit()

samples_mean = np.mean(samples_p, axis=0)

#print(np.mean( samples_s[0,400:]) )



#p_mean = np.mean(samples_p[8:,:],axis=0)

field.plot_curve(samples_mean,ax, label='mean seabed', color='orange')
field.plot_uq(samples_p, ax, label='99% CI')
ax.legend()
plt.savefig('curve_gibbs.pdf',dpi=300)



