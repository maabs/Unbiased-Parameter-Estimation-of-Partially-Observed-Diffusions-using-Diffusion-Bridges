#%%
#import PF_functions_def as pff
#import bridge as bdg
from scipy.stats import ncx2
import multiprocessing
import time
from scipy.special import gamma as gamma_den, iv 
from scipy.stats import gamma
import scipy.stats as ss
import math
import numpy as np
import matplotlib.pyplot as plt 
#import progressbar
from scipy import linalg as la
from scipy.sparse import identity
from scipy.sparse import rand
from scipy.sparse import diags
from scipy.sparse import triu

import copy
#from sklearn.linear_model import LinearRegression
from scipy.stats import ortho_group
from scipy.stats import multivariate_normal


t0=0
T=10
the1,the2,the3,the4=2.397, 4.429e-3, 0.84, 17.36
theta=[the1,the2,the3]
the4=np.array([[10]])
theta_aux=1
sigma=the3
sigma_aux=sigma
l=10
d=1
N=50
x0=0+np.zeros(N)
x_p=0+np.zeros(N)

dim=1
seed=1 
# t0,x0,T,x_p,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,l,d,N,seed
#start=time.time()
n_tests=1000
"""
for i in range(n_tests):
    pars_0=[np.array([the1,the2,the3]),the3]
    [A_til,fi_til,r_pars,H_pars,\
    atdp]=bdg.update_log_functs(pars_0,pars_0,t0,x0,x0,T,x_p,x_p,levels=1)
    x=bdg.Bridge_1d_path(t0,x0,T,x_p,bdg.b_log,[the1,the2,the3],bdg.Sig_gbm_1d,sigma,bdg.b_log_aux,A_til,bdg.Sig_aux_gbm_1d,\
    fi_til,bdg.r_log_normal,r_pars,bdg.H_log_normal,\
    H_pars,l,d,N,seed,non_seeded=True)[0]
    end=time.time()
"""


#print("The time is: ",end-start)

    
def EM_logistic(t0,x0,T,the1,the2,the3,l,N,seed):

    Dt=1/(2**l)
    steps=int((T-t0)/(Dt))
    
    x=np.zeros((steps+1,N))
    x[0]=x0+np.zeros(N)

    dW=np.sqrt(Dt)*np.random.normal(0,1,(steps,N))
    for i in range(1,steps+1):
        #x[i]=x[i-1]+(the3**2/2+the1-the2*x[i-1])*x[i-1] *Dt+dW[i-1]*the3*x[i-1]
        x[i]=x[i-1]+(the1/the3-the2/the3*np.exp(the2*x[i-1])*Dt+dW[i-1])
    
    return x


start=time.time()
#"""
for i in range(n_tests):
    #x=bdg.EM_log(t0,x0,T,bdg.b_log,[the1,the2,the3],bdg.Sig_gbm_1d,the4,l,N,seed)
    x=EM_logistic(t0,x0,T,the1,the2,the3,l,N,seed)
end=time.time()
print("The time is: ",end-start)