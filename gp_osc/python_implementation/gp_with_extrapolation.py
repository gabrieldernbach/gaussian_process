'''
Gaussian process with utilities for seemless loops.
The randomly drawn instances (interval [0,1]) of the process can be considered as wavetables.
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.pyplot.switch_backend('TKAgg')

def gentest(function, start, end, n_samples):
    x1 = np.sort(np.random.random(n_samples)) * (end - start) - start
    y1 = function(x1)
    x1s = (x1 + start) / (end - start)
    return x1s, y1

def extrapolate(x1, y1):
    x1e = np.concatenate([x1-1, x1, x1+1])
    y1e = np.concatenate([y1, y1, y1])
    return x1e, y1e

class GP():
    def __init__(self, sigma=0.3, llambda=0.01):
        self.llambda = llambda
        self.sigma = sigma
        return

    def kernel(self, a, b):
        k = np.exp(- (a[:, None] - b[None, :])**2 / (2 * self.sigma ** 2))
        k = k + np.eye(k.shape[0],k.shape[1]) * self.llambda
        return k

    def fit(self, x1, y1):
        self.x1 = x1
        self.y1 = y1
        k11 = self.kernel(x1, x1)
        self.k11i = np.linalg.inv(k11)

    def predict(self, x2):
        k12 = self.kernel(self.x1, x2)
        k22 = self.kernel(x2, x2)
        y2 = k12.T @ self.k11i @ self.y1.T
        sig2 = k22 - k12.T @ self.k11i @ k12
        return y2, sig2

    def fit_predict(self, x1, y1, x2):
        self.fit(x1,y1)
        return self.predict(x2)

    def draw_instances(self, x2, num_instances=8):
        num_samples = x2.shape[0]
        y2, sig2 = self.predict(x2)

        # generate noise with desired covariance
        U,S,V = np.linalg.svd(sig2)
        standard_sample = np.sort(np.random.randn(num_samples, num_instances),axis=1)
        sample_rot = np.sqrt(S) * U @ standard_sample

        # create window in [0,1]
        margin = int(0.2 * num_samples)
        rampin = np.tanh(np.linspace(0,1,margin)) * 1/0.761594156
        rampout = np.tanh(np.linspace(1,0,margin)) * 1/0.761594156
        sustain = np.ones(num_samples - 2 * margin)
        window = np.concatenate([rampin, sustain, rampout])[:,None]

        # add mean to windowed variance
        instances = (y2[:,None] + window * sample_rot)
        return instances

if __name__ == '__main__':

    # generate test data
    x1, y1 = gentest(np.sin, -1*np.pi, 1*np.pi, 8)
    x1e, y1e = extrapolate(x1, y1)
    x2 = np.linspace(0, 1, 1000)

    # build model
    gp = GP(sigma=0.1, llambda=0)
    y2, sig2 = gp.fit_predict(x1e, y1e, x2)

    # draw random instances
    instances = gp.draw_instances(x2, num_instances=12)
    

    plt.subplot(211)
    plt.plot(x1,y1,
            'o',mfc='None',c='C0',label='data')
    plt.plot(x2,y2,
            '-',c='C0',label='mean prediction')
    std2 = 1.96 * np.sqrt(np.diag(sig2))
    plt.plot(x2,y2-std2,
            '-.',c='C1',linewidth=0.5,label='95 percentile')
    plt.plot(x2,y2+std2
            ,'-.',c='C1',linewidth=0.5)
    plt.legend(); plt.title('mean prediction given data')

    plt.subplot(212)
    plt.plot(instances)
    plt.title('random instances drawn from the posterior distribution')

    plt.tight_layout()
    plt.show()
