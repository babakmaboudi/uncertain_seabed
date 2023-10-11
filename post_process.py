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


stat_data = np.load('./stat/stat_Gibbs_long.npz')

samples_s = stat_data['s']
s_mean = np.mean(samples_s)

field.set_s(s_mean)
samples_p = stat_data['p'].T
np.savez('./checkpoints/checkpoint1.npz',s0=samples_s[0,-1], p0=samples_p[-1,:])
samples_p = samples_p[300:,:]

print(np.mean( samples_s[0,400:]) )



#p_mean = np.mean(samples_p[8:,:],axis=0)

field.plot_curve(samples_p[-1],ax, label='mean seabed', color='orange')
field.plot_uq(samples_p, ax, label='99% CI')
ax.legend()
plt.savefig('curve_gibbs.pdf',dpi=300)



