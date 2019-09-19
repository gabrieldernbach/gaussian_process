import numpy as np
from scipy.optimize import minimize

x1 = np.array([-1, -0.9, -0.4, -0.25, 0.5, 1]);
y1 = np.array([0, -1, 2, 0.5, 2, 0]);

x2 = np.linspace(-1,1,1000);

sigma = 0.3
llambda = 0


def kernel(x,y,sigma,llambda):
    k = np.exp(-(x[:,None] - y[None,:])**2 / (2*sigma**2))
    k = k + np.eye(k.shape[0],k.shape[1]) * llambda
    return k

def gaussian_process(x2,x1,y1,sigma,llambda):
    k11 = kernel(x1,x1,sigma,llambda)
    k12 = kernel(x1,x2,sigma,llambda)
    k22 = kernel(x2,x2,sigma,llambda)

    k11i = np.linalg.inv(k11)
    y2 = k12.T @ k11i @ y1.T
    sig2 = k22 - k12.T @ k11i @ k12
    return y2, sig2




## Optimiztion
def compute_neg_loglikelihood(params,x1,y1):
    sigma = params[0]
    llambda = params[1]

    K = kernel(x1,x1,sigma,llambda)
    sing, logdet = np.linalg.slogdet(K)
    K_inv = np.linalg.inv(K + 0.0001 * np.eye(K.shape[0]))

    assert(np.isfinite(logdet))

    log_likelihood = 0.5 * y1 @ K_inv @ y1 + 0.5 * logdet + x1.shape[0]/2*np.log(2*np.pi)

    return log_likelihood.ravel()


def update_hyperparams(x1,y1,sigma,llambda):
    params = [sigma, llambda]
    bounds = [(1e-5, None),(1e-5, None)]

    solution = minimize(
            compute_neg_loglikelihood,
            params,
            args=(x1,y1),
            bounds = bounds)

    new_sigma = solution.x[0]
    new_llambda = solution.x[1]

    return new_sigma, new_llambda

print(sigma,llambda,update_hyperparams(x1,y1,sigma,llambda))
sigma, llambda = update_hyperparams(x1,y1,sigma,llambda)

y2,sig2 = gaussian_process(x2,x1,y1,sigma,llambda)
instances8 = np.random.multivariate_normal(y2,sig2,8)

if 1:
    ## Plot Result
    import matplotlib.pyplot as plt
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
    plt.plot(instances8.T)
    plt.title('random instances drawn from the posterior distribution')

    plt.tight_layout()
    plt.show()

