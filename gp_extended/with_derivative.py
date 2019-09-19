import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

class GP:
    
    def __init__(self,kernelwidth,alpha,x1d=None,y1d=None):
        self.kernelwidth = kernelwidth
        self.alpha = alpha

    def kernel(self,x,y):
        return self.alpha * np.exp(-(x-y)**2 / self.kernelwidth)

    def dkernel(self,x,y):
        dk = -(x-y)
        kernel = self.alpha / self.kernelwidth * np.exp(-(x-y)**2 / self.kernelwidth)
        return dk * kernel

    def ddkernel(self,x,y):
        ddk = 1 - 1 / self.kernelwidth * (x-y)**2
        kernel = self.alpha / self.kernelwidth * np.exp(-(x-y)**2 / self.kernelwidth)
        return ddk * kernel

    def fit(self,x1,y1):
        self.x1 = x1
        self.y1 = y1
        K11 = self.kernel(x1,x1.T)
        self.K11i = np.linalg.inv(K11)
        
    def fit_derivative(self,x1,y1,x1d,y1d): 
        self.x1 = x1
        self.y1 = y1
        self.x1d = x1d
        self.y1d = y1d
        K11 = self.kernel(x1,x1.T)
        K11d = self.dkernel(x1,x1d.T)
        K11dd = self.ddkernel(x1d,x1d.T)
        upper = np.concatenate((K11,K11d),axis=1)
        lower = np.concatenate((K11d.T,K11dd),axis=1)
        K11D = np.concatenate((upper,lower),axis=0)
        K11D += np.diag(np.random.randn(K11D.shape[0]) * 1e-3)
        self.K11Dinv = np.linalg.inv(K11D)

    def predict(self,x2):
        k12 = self.kernel(self.x1,x2.T)
        k22 = self.kernel(x2,x2.T)
        y2 = k12.T @ self.K11i @ self.y1
        cov2 = k22 - k12.T @ self.K11i @ k12
        return y2, cov2

    def predict_derivative(self,x2):
        k12 = self.kernel(x2,self.x1.T)
        k12d = self.dkernel(x2,self.x1d.T)
        k12D = np.concatenate((k12,k12d),axis=1)
        k22 = self.kernel(x2,x2.T)
        y1D = np.concatenate((self.y1,self.y1d),axis=0)

        y2d = k12D @ self.K11Dinv @ y1D
        cov2d = k22 - k12D @ self.K11Dinv @ k12D.T
        return y2d, cov2d

    @staticmethod
    def compute_neg_loglikelihood(_params, _x1, _y1):
        kernelwidth = _params[0]
        alpha = _params[1]
        
        K11 = alpha * np.exp(-(_x1 - _x1.T)**2 / kernelwidth)
        K11inv = np.linalg.inv(K11 + 0.001*np.eye(K11.shape[0]))
        sign, logdet = np.linalg.slogdet(K11)

        fak = _x1.shape[0] / 2 * np.log(2 * np.pi)
        
        log_likelihood = 0.5 * _y1.T @ K11inv @ _y1 + 0.5 * logdet + fak
        print(log_likelihood)

        return log_likelihood.ravel()

    def update_hyperparams(self,x1,y1):
        self.x1 = x1
        self.y1 = y1

        print('starting with\n')
        print(f'Updated kernelwidth to {self.kernelwidth} and alpha to {self.alpha}\n')
        print(f'The Loglikelihood is {self.compute_neg_loglikelihood([self.kernelwidth,self.alpha],x1,y1)}\n')

        x0 = [self.kernelwidth, self.alpha]
        bounds = [(1e-5, None), (1e-5, None)]

        solution = scipy.optimize.minimize(
                self.compute_neg_loglikelihood,
                x0,
                args=(self.x1, self.y1),
                bounds = bounds)

        self.kernelwidth = solution.x[0]
        self.alpha = solution.x[1]

        print(f'Updated kernelwidth to {self.kernelwidth} and alpha to {self.alpha}\n')
        print(f'The Loglikelihood is {self.compute_neg_loglikelihood([self.kernelwidth,self.alpha],x1,y1)}\n')



#x1 = np.random.random((7,1)).reshape((-1,1))
#y1 = np.sin(x1) * x1
#y1d = -(np.sin(x1) + x1 * np.cos(x1))
x1 = np.array([0,0.2,0.3,0.5,0.9,1]).reshape((-1,1)
y1 = np.array([0,1,0.5,-1,0.5,0]).reshape((-1,1))
x1d = np.array([0,0.2,1]).reshape((-1,1))
y1d = np.array([3,100,-3]).reshape((-1,1))
#x1d = np.array([0,1]).reshape((-1,1))
#y1d = np.array([-100,100]).reshape((-1,1))

#x1 = np.array([[2, 4, 7, 10, 13]]).reshape((-1, 1))
#y1 = x1*np.sin(x1)
#x1d = np.array([[2, 5, 8, 11, 15]]).reshape((-1, 1))
#y1d = -(np.sin(x1d) + x1d*np.cos(x1d))

x2 = np.linspace(0,1,1000).reshape((-1,1));

kernelwidth = 0.8
alpha = 1
gp = GP(kernelwidth,alpha)
gp.fit_derivative(x1,y1,x1d,y1d)
y2, cov2 = gp.predict_derivative(x2)

gp.update_hyperparams(x1,y1)

if 1:
    plt.plot(x1,y1,'C0o',fillstyle='none')
    plt.plot(x2,y2,'C0')
    plt.plot(x2, y2[:,0] - 1.96 * np.diag(cov2),c='C1',linewidth=0.3)
    plt.plot(x2, y2[:,0] + 1.96 * np.diag(cov2),c='C1',linewidth=0.3)
            
    plt.show()
