'''
Stochastic Wavetable Synthesis by GP Model Interpolation
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
import sounddevice

def gentest(function, start, end, n_samples):
    '''
    evaluate function on set of random point in domain [start, end]
    and rescale domain to live inside [0, 1]^d afterwards
    '''
    x1 = np.sort(np.random.random(n_samples)) * (end - start) - start
    y1 = function(x1)
    x1s = (x1 + start) / (end - start)
    return x1s, y1

def extrapolate(x1, y1):
    '''
    extrapolate supporting points periodically to constrain
    the model to become periodic in domain [0, 1]
    '''
    x1e = np.concatenate([x1-1, x1, x1+1])
    y1e = np.concatenate([y1, y1, y1])
    return x1e, y1e

class GP():
    def __init__(self, sigma=0.3, llambda=0.01):
        '''
        initialize length scale llambda (effective upper bound of frequncy)
        and regularization sigma (effectively introduces noise), 
        but stabalizes possibly ill posed regression tasks (close points impossible to cover)
        '''
        self.llambda = llambda
        self.sigma = sigma
        return

    def kernel(self, a, b):
        '''
        squared exponential kernel as used in nadarya watson estimator
        results in radial basis function as similarity measure
        '''
        k = np.exp(- (a[:, None] - b[None, :])**2 / (2 * self.sigma ** 2))
        k = k + np.eye(k.shape[0],k.shape[1]) * self.llambda
        # k = np.minimum(a[:,None],b[None,:]) # use spline kernel alternatively
        return k

    def fit(self, x1, y1):
        '''
        setup the inverse kernel matrix and save the model supporting points
        '''
        self.x1 = x1
        self.y1 = y1
        k11 = self.kernel(x1, x1)
        self.k11i = np.linalg.inv(k11)

    def predict(self, x2):
        '''
        evaluate the model on possibly unseen data points which is 
        by design a smooth function in x2 and periodic at the ends
        '''
        k12 = self.kernel(self.x1, x2)
        k22 = self.kernel(x2, x2)
        y2 = k12.T @ self.k11i @ self.y1.T
        sig2 = k22 - k12.T @ self.k11i @ k12
        return y2, sig2

    def fit_predict(self, x1, y1, x2):
        '''convenience wrapper for joint training and classification'''
        self.fit(x1,y1)
        return self.predict(x2)

class OSC():
    '''
    oscillator module acting on a model Gaussian Process assumed 
    to be periodic on the domain [0, 1],
    it aids wavetables generation and rendering of interpolated sequences
    '''
    def __init__(self, model, num_samples):
        self.num_samples = num_samples
        
        # extract mean oscillator and de-whitening matrix from model
        x = np.linspace(0, 1, num_samples)
        self.mean, cov = model.predict(x)
        U, S, V = np.linalg.svd(cov)
        self.dewhiten = np.sqrt(S) * U
        
        # initialize window
        self.window = self.window_builder()

    def window_builder(self, relative_margin=0.2):
        # create window for ramp on osicalltion variance
        margin = int(relative_margin * self.num_samples)
        rampin = np.tanh(np.linspace(0,1,margin)) * 1/0.761594156
        rampout = np.tanh(np.linspace(1,0,margin)) * 1/0.761594156
        sustain = np.ones(self.num_samples - 2 * margin)
        window = np.concatenate([rampin, sustain, rampout])
        return window

    def draw_wavetable(self, num_instances):
        # generate standard normal gaussian and dewhiten 
        standard_normal = np.random.randn(self.num_samples, num_instances)
        colored_normal = self.dewhiten @ standard_normal
        
        # add mean to windowed and dewhitened gaussian
        instances = self.mean[:,None] + self.window[:,None] * colored_normal
        return instances # wave_len x num_instances

    def generate_interpolated_sequence(self, num_instances, blend_len=96000):
        table = self.draw_wavetable(num_instances)
        
        blend = np.linspace(0, 1, blend_len // 2)
        ramp = np.append(blend, blend[::-1])
        X = np.tile(table.T, blend_len // self.num_samples) * ramp
        
        Xa = np.pad(X, ((0,1), (0, 0)))
        Xb = np.pad(X, ((1,0), (0, 0)))

        X = Xa[:,:blend_len//2] + Xb[:,(blend_len//2):]
        X = X.ravel()
        return X
        


if __name__ == '__main__':

    # generate test data
    x1, y1 = gentest(function = np.sin,
                     start = -np.pi,
                     end = np.pi,
                     n_samples = 4)
    x1e, y1e = extrapolate(x1, y1)

    # build model
    gp = GP(sigma=0.1, llambda=0.00001)
    x2 = np.linspace(0, 1, 1000)
    y2, sig2 =  gp.fit_predict(x1e, y1e, x2)

    # oscillator
    osc = OSC(gp, num_samples=1000)
    instances = osc.draw_wavetable(num_instances=8) # for plotting
    sequenceL = osc.generate_interpolated_sequence(num_instances=20, blend_len=96000)
    sequenceR = osc.generate_interpolated_sequence(num_instances=20, blend_len=96000)
    sequence = np.stack((sequenceL,sequenceR),axis=1)
    
    def play(X):
        fs = 48000
        sounddevice.play(X,fs)
        sounddevice.wait()
    
    play(sequence)

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
