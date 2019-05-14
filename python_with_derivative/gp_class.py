import numpy as np
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
        k11 = self.kernel(x1,x1.T)
        self.k11i = np.linalg.inv(k11)
        
    def fit_derivative(self,x1,y1,x1d,y1d): 
        self.x1 = x1
        self.y1 = y1
        self.x1d = x1d
        self.y1d = y1d
        k11 = self.kernel(x1,x1.T)
        k11d = self.dkernel(x1,x1d.T)
        k11dd = self.ddkernel(x1d,x1d.T)
        upper = np.concatenate((k11,k11d),axis=1)
        lower = np.concatenate((k11d.T,k11dd),axis=1)
        k11D = np.concatenate((upper,lower),axis=0)
        k11D += np.diag(np.random.randn(k11D.shape[0]) * 1e-3)
        self.K11Dinv = np.linalg.inv(k11D)

    def predict(self,x2):
        k12 = self.kernel(self.x1,x2.T)
        k22 = self.kernel(x2,x2.T)
        y2 = k12.T @ self.k11i @ self.y1
        cov2 = k22 - k12.T @ self.k11i @ k12
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

#x1 = np.random.random((7,1)).reshape((-1,1))
#y1 = np.sin(x1) * x1
#y1d = -(np.sin(x1) + x1 * np.cos(x1))
x1 = np.array([0,0.2,0.3,0.5,0.9,1]).reshape((-1,1))
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

kernelwidth = 0.01
alpha = 1
gp = GP(kernelwidth,alpha)
gp.fit_derivative(x1,y1,x1d,y1d)
y2, cov2 = gp.predict_derivative(x2)

plt.plot(x1,y1,'C0o',fillstyle='none')
plt.plot(x2,y2,'C0')
plt.plot(x2, y2[:,0] - 1.96 * np.diag(cov2),c='C1',linewidth=0.3)
plt.plot(x2, y2[:,0] + 1.96 * np.diag(cov2),c='C1',linewidth=0.3)
        
plt.show()
