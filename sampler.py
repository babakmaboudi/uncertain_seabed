import numpy as np
from progressbar import progressbar

class sampler():
    def __init__(self, x0, target, scale=1):
        self.dim = len(x0)
        self.target = target

        self.current_sample = x0
        self.current_target = self.target( x0 )

        self.samples = [ x0 ]
        self.acc = [1]

        self.scale = scale

    def sample(self, N, batch_size=None):
        if(batch_size):
            batch = 0

        for i in progressbar( range(N) ):
            self.step()

            if(batch_size):
                if((i+1)%batch_size == 1):
                    self.save_checkpoint('./checkpoints/check_{}.npz'.format(batch))
                    np.savez('__sampler_cache/samples_{}.npz'.format(batch), self.samples[-1-batch_size])
                    batch += 1


    def warm_up(self, N, skip_len=None):
        if(skip_len == None):
            self.skip_len = int( N/10 )
        else:
            self.skip_len = skip_len

        update_count = 0
        for i in progressbar( range(N) ):
            self.step()
            if(  (i+1)%self.skip_len == 0 ):
                self.tune(update_count)
                update_count += 1



    def step(self):
        acc = self.pCN_step()

        self.acc.append(acc)
        self.samples.append(self.current_sample)

    def tune(self, update_count):
        hat_acc = np.mean(self.acc[-1-self.skip_len:])

        # d. compute new scaling parameter
        zeta = 1/np.sqrt(update_count+1)   # ensures that the variation of lambda(i) vanishes
        scale_temp = np.exp(np.log(self.scale) + zeta*(hat_acc-0.234))

        # update parameters
        self.scale = min(scale_temp, 1)

    def pCN_step(self):
        xi = np.random.standard_normal(self.dim)
        x_star = np.sqrt( 1 - self.scale**2 ) * self.current_sample + self.scale*xi

        target_eval_star = self.target(x_star)

        ratio = target_eval_star - self.current_target
        alpha = min(0, ratio)

        # accept/reject
        u_theta = np.log(np.random.rand())
        acc = 0
        if (u_theta <= alpha):
            self.current_sample = x_star
            self.current_target = target_eval_star
            acc = 1
        
        return acc


    def MH_step(self):
        xi = np.random.standard_normal(self.dim)
        x_star = self.current_sample + self.scale*xi

        target_eval_star = self.target(x_star)

        ratio = target_eval_star - self.current_target
        alpha = min(0, ratio)

        # accept/reject
        u_theta = np.log(np.random.rand())
        acc = 0
        if (u_theta <= alpha):
            self.current_sample = x_star
            self.current_target = target_eval_star
            acc = 1
        
        return acc

    def save_checkpoint(self, path='checkpoint.npz'):
        np.savez(path, current_sample=self.current_sample, current_target=self.current_target, scale=self.scale)

    def clear(self):
        self.current_sample = np.empty( self.dim )
        self.current_target = 0

        self.samples.clear()
        self.acc.clear()

    def load_checkpoint(self, path):
        self.clear()
        checkpoint = np.load(path)
        self.current_sample = checkpoint['current_sample']
        self.current_target = checkpoint['current_target']
        self.scale = checkpoint['scale']

    def get_samples(self):
        return np.array(self.samples)
