from sampler import sampler
import numpy as np

import matplotlib.pyplot as plt

x0 = np.ones(2)
target = lambda x: -0.5*(x[0]**2/0.01 + x[1]**2/10)

samp = sampler(x0, target)
samp.warm_up(10000, skip_len=100)

samp.save_checkpoint('check.npz')

np.random.seed(0)

samp.sample(1000)
print(samp.scale)

samples1 = samp.get_samples()

samp2 = sampler(x0, target)
samp2.load_checkpoint('check.npz')

np.random.seed(0)
samp2.sample(1000)
print(samp2.scale)

samples2 = samp2.get_samples()



#samp.sample(1000, batch_size=100)
#print(samp.scale)

#samples = samp.get_samples()

f,axes = plt.subplots(2,2)
axes[0,0].plot(samples1[-1-1000:,0])
axes[0,1].plot(samples1[-1-1000:,1])
axes[1,0].plot(samples2[-1-1000:,0])
axes[1,1].plot(samples2[-1-1000:,1])
plt.show()

#axes[0].hist(samples[:,0], bins=100)
#axes[1].hist(samples[:,1], bins=100)
#plt.show()
