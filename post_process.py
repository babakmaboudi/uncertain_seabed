import numpy as np
import matplotlib.pyplot as plt
from wave import wave_speed_matern


obs_data = np.load('./obs/obs1/obs.npz')
print(obs_data.files)
p_true = obs_data['param_true']
N_x = obs_data['N_x']
N_KL = obs_data['N_KL']
field = wave_speed_matern(N_x,N_KL)

f,ax = plt.subplots(1)
field.plot_curve(p_true,ax, label='true seabed', color='blue')


stat_data = np.load('./stat/stat1_long_chain.npz')
samples = stat_data['samples'].T
samples = samples[5000:,:]

p_mean = np.mean(samples,axis=0)

field.plot_curve(p_mean,ax, label='mean seabed', color='orange')
field.plot_uq(samples, ax, label='99% CI')
ax.legend()
plt.savefig('curve.pdf',dpi=300)



