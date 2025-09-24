# This file is made to test the bridge sampling algorithm
#%%
# 

import math
import PF_functions_def as pff
# We import the module PF_functions_def such that it has 
# several important function from the project 
#https://github.com/maabs/Multilevel-for-Diffusions-
#Observed-via-Marked-Point-Processes **
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_den
from scipy.special import digamma as digamma
from scipy.special import iv 
from scipy.stats import ncx2

from scipy import linalg as la
import copy
from scipy.stats import gamma
from scipy.sparse import identity
from scipy.sparse import rand
from scipy.sparse import diags
from scipy.sparse import triu

#from sklearn.linear_model import LinearRegression
from scipy.stats import ortho_group
import time
from scipy.stats import norm
import scipy.stats as ss

#from scipy.stats import multivariate_normal#%%
# The conversion for the dimension and what they mean in the previous project **
# is (2,times,n_pars,samples,samples,resampling_times,DIM,dim,dim)

# Note: the second "samples" is for particles sampled by the stride function. 
# Note: DIM is used for derivatives while dim is used for the dimension of the hidden model
# or the diffusion matrix in the case dim,dim


# The conversion for the order of the argumetns of the functions is
# T,xin,b_ou,A,Sig_ou,fi,Grad_b,b_numb_par,obs,obs_time,l,d,N,dim,resamp_coef,g_den,Grad_Log_g_den,g_par,\
# Lambda,Grad_Lambda,Grad_Log_Lambda, Lamb_par,Lamb_par_numb,step_0,beta)
# In this sense it's better to define classes instead of function since allows for
# a more organized way of passing the arguments.

#%%
# QUESTIONS: 
# 1. How do we generalize this for the whole project 
#        1.1. I think the main concern is to keep a 
#             convention for the order of the dimensions of 
#             the arrays and the functionality of the 
#             coefficients, i.e., how the coefficient 
#             are computing depending on the dimension of the 
#             problem.
#        1.2  Another concern is the arrangement of the parameters to estimate,
#             some of them are going to be in form of a matrix, others in form of
#             a vector, and others in form of a scalar. In the previous project 
"""
              we used variables specifying the dimension of the problem. This 
              is almost inevitable given the nature of the current coefficient 
              functions that we define. In order to improve this we could 
              pass the argumetns of the coefficient function as a vector. But 
              the trouble of doing that sounds more complicated than just leaving it
              as it is. 
        1.3.  Should we include estimation of parameters that affect both the 
              observation and the diffusion process? In this case some problems might
              arise in the broadcasing of different gradients that have different rank.
              e.g. the gradient of diffusion might be a rank 3 array, while the gradient
              drift might be a rank 2 array.       

 2. Is it better to use the coefficients as classes?
        2.1.  What is the advantage of using classes w.r.t. to funcitons? 
              we can define methods of those classes, which methods? ploting 
              methods is an option, but it's not good enough to go through the
              trouble of defining a class.
               
The plan now is check the function for SDA of the previous project and 
adapt it to the current project.

"""

#%%
# NOTATION ALONG THE CODE
"""
As in the paper "guided proposal for..." we used the following notation:
-X is going to correspond the UNCONDITONED diffusion process
-X^{\star} is going to represent the CONDITIONED diffusion process, i.e. the
bridge
-X^{\circ} (x_pr) is going to represent the PROPOSAL diffusion process
-\tilde{X} is going to represent the AUXILIAR diffusion process
- p_hat is going to represent the proposal stride, which is different to the AUXILIAR
stride, which is the stride of the auxiliar process.
- p_tilde is going to represent the auxiliar stride, which is different to the PROPOSAL stride.
IS IMPORTANT TO NOTE THAT THE DIFFUSION TERM OF THE AUXILIAR DIFFUSION MUST BE THE SAME AT THE 
END OF THE BRIDGE, I.E., \sigma(x')=\tilde{\sigma}(x').
"""

#%% 


def coef(x, y): 
    # number of observations/points 
    n = np.size(x) 
  
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) 
  
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x 
  
    # calculating regression coefficients 
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 
    
    return np.asarray((b_0, b_1)) 


def get_deltas(data):

    # Function that takes an array of data and returns the deltas of the data
    # ARGUMENTS:
    # data: is a rank 1 array with dimension (N)
    # OUTPUTS:
    # deltas: is a rank 1 array with dimension (N-1)

    Delta_l=np.min(data[1:]-  data[:-1])/2
    intervals=data[1:]-  data[:-1]
    partits=np.rint(intervals/Delta_l)
    deltas=intervals/partits
    return deltas



def cut(T,lmax,l,v):

    # Function that takes an array v of lenght  T*2**lmax+1 and returns the array v cutted at the level l,
    # i.e., it takes the every 2**l-th element of v.

    # ARGUMENTS:
    # T: is the final time of the process
    # lmax: is the maximum level of the process
    # l: is the level at which we want to cut the array
    # v: is the array that we want to cut

    ind = np.arange(T*2**l+1,dtype=int)
    rtau = 2**(lmax-l)
    #print(lmax-l,ind,ind*rtau)
    w = v[ind*rtau]
    return(w)


# In the following we have a list of fuctions cathegorized by the role they play in the
# algorithm, e.g. drift, diffusion, gradient of the log of the kernel of the auxiliar process, etc.
#%%

# INITIAL DISTRIBUTIONS

def gamma_sampling(dist_pars,N):

    # Function that samples from a gamma distribution
    # ARGUMENTS:
    # N: number of samples
    # dist_pars: (the1,the2,theta3) are the parameters used in
    # the red kangaroo project.
    
    the1,the2,the3= dist_pars
    w=the3**2/2+the1
    xi=the2
    sigma=the3

    alpha=2*w/sigma**2-1
    theta=sigma**2/(2*xi)

    return np.random.gamma(alpha,theta,N)

"""
dist_pars=[1,1,1]
samples=gamma_sampling(dist_pars,10000000)

plt.hist(samples, bins=1000, density=True, alpha=0.5, label="Sample Histogram")


the1,the2,the3= dist_pars
w=the3**2/2+the1
xi=the2
sigma=the3
alpha=2*w/sigma**2-1
theta=sigma**2/(2*xi)

# 3. Overlay the theoretical Gamma PDF
#    We'll create an x-grid that spans the sample range
x_vals = np.linspace(0, samples.max(), 200)
pdf_vals = gamma.pdf(x_vals, a=alpha, scale=theta)
plt.plot(x_vals, pdf_vals, 'r-', lw=2, label="Gamma PDF")
# Add labels and legend
plt.title(f"Gamma Sampling Test (alpha={alpha}, theta={theta})")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
# 4. Show the plot
plt.show()
"""

#%%




#%%
#  DRIFS FUNCITONS
#%%


## COMMENTS ON THE DRIFT FUNCTION.

# The drift function comes in two forms, the first form is the drift of the original diffusion process,
# this process doesn't include time, thus the parameters of these drifts funciton are only two, on the 
# other hand, the drifts of the auxiliar process might include time, thus the parameters of these drifts are
# three. 



def b_log(x,b_pars):

    # Function for the drift of the logistic process.
    # ARGUMENTS:
    # x: rank 1 array with dimension (N)
    # b_pars: (the1,the2,theta3) are the parameters of the logistic process
    # OUTPUTS:
    # the drift of the logistic process evaluated at x.
    the1,the2,the3=b_pars

    return (the3**2/2+the1-the2*x)*x 

def b_log_aux(t,x,b_aux_pars):

    # Function for the drift of the auxiliar process that mimics the logistic process.

    # ARGUMENTS:
    # x: rank 1 array with dimension (N)
    # b_aux_pars: (a,b) each one being a rank 1 array with dimension (N)

    # OUTPUTS:
    # x*\bar{g}(t) where \bar{g}(x) is the linearization of the function g(x)=(the3**2/2+the1-the2*x)

    a,b=b_aux_pars
    return (a*t+b)*x





def b_ou_1d(x,A):   
    # Returns the drift "vector" evaluated at x
    # ARGUMENTS: x is a rank 1 or 2 array with dims (1), (N) or (N,N)
    # and A is a rank 0
    # OUTPUTS: the product x*A, so the rank of x is keep for the output.
        mult=x*A
        #mult=np.array(mult)*10
        return mult

def b_gbm(x,A):
    # Returns the drift "vector" evaluated at x
    # ARGUMENTS: x is a rank 1 (N)
    # OUTPUTS: drift with the shape of x.
    mult=x*A
    return mult

def b_gbm_aux(t,x,A):

    # Returns the drift "vector" evaluated at x
    # ARGUMENTS: x is a rank 1 (N)
    # OUTPUTS: drift with the shape of x.
    mult=x*A
    return mult

def b_ou_aux(t,x,A):
    # Returns the drift "vector" evaluated at x
    # ARGUMENTS: x is a rank 1 or 2 array with dims (1), (N) or (N,N)
    # and A is a rank 0
    # OUTPUTS: the product x*A, so the rank of x is keep for the output.
        
        mult=x*A
        #mult=np.array(mult)*10
        return mult




def b_artificial_1d(t,x,A):
    

    return np.zeros(x.shape)


def b_gbm_1d(x,mu):
    # Returns the drift "vector" evaluated at x
     # ARGUMENTS: x is a rank 1 or 2 array with dims (1), (N) or (N,N)
    # and sigma is a rank 0
    
    # OUTPUTS: drfit with the shape of x.

    mult=x*mu
        #mult=np.array(mult)*10
    return mult


def b_ou(x,A):
    # Returns the drift "vector" evaluated at x
    # ARGUMENTS: x is a rank two array with dimensions where the first dimension 
    # corresponds to the number of particles and the second to the dimension
    # of the probelm. The second argument is A, which is a squared rank 2 array 
    # with dimension of the dimesion of the problem
    # ARGUMENTS:
    # ** x: is either a rank 1, 2 or 3 array, with rank (dim), (N,dim) or (N,N,dim)
    # ** respectively.
    # ** A: is a rank 2 array with dimension (dim,dim)
    
    # OUTPUTS: 
    # ** mult is either a rank 1, 2 or 3 array, with rank (dim), (N,dim) or (N,N,dim)
    # ** depending on the rank of x.
        
        mult=x@(A.T)
        #mult=np.array(mult)*10
        return mult

def b_til(t,x,A):
    return pff.b_gbm(x,A)

def b_til_1d(t,x,A):
    return b_gbm_1d(x,A)


#%%
# DIFFUSIONS

## COMMENTS ON THE DRIFT FUNCTION.

# The diffusion function comes in two forms, the first form is the diffusion of the original diffusion process,
# this process doesn't include time, thus the parameters of these diffusion funciton are only two, on the 
# other hand, the drifts of the auxiliar process might include time, thus the parameters of these diffusions are
# three. 



def Sig_til(t,x,fi_til):
        #[fi,t0,x0,T,x_pr]
        return pff.Sig_gbm(x_pr,fi)*(t-t0)/((T-t0))+pff.Sig_gbm(x0,fi)*(T-t)/(T-t0)



#| | | | |
#v v v v v
# The function below needs a bid of work.
def Sig_til_1d(t,x,fi_til):
        #[fi,t0,x0,T,x_pr]

        return Sig_gbm_1d(x_pr,fi)*(t-t0)/((T-t0))+Sig_gbm_1d(x0,fi)*(T-t)/(T-t0)

def Sig_ou_1d(x,sigma):
    # This is the function for the diffusion of the OU process
    # ARGUMENTS: x is a rank 0 or 1 or 2 array with dims (1) (N) or (N,N)
    # and sigma is a rank 0 
    # OUTPUTS: just the scalar sigma
    
    return sigma



def Sig_ou_aux(t,x,sigma):
    # This is the function for the diffusion of the OU process
    # ARGUMENTS: x is a rank 0 or 1 or 2 array with dims (1) (N) or (N,N)
    # and sigma is a rank 0 
    # OUTPUTS: just the scalar sigma
    
    return sigma


def Sig_alpha(t,x,pars):
    [alpha,sigma, T]=pars
    return sigma*(t/T)**(alpha)

def Sig_gbm_1d(x,sigma):
    # Returns the diffussion "vector" evaluated at x
    # ARGUMENTS: x is a rank 1 or 2 array with dims (1), (N) or (N,N)
    # and sigma is a rank 0
    
    # OUTPUTS: diffusion term, which has the shape of x.
    return sigma*x

def Sig_aux_gbm_1d(t,x,sigma):
    # Returns the diffussion "vector" evaluated at x
    # ARGUMENTS: x is a rank 1 or 2 array with dims (1), (N) or (N,N)
    # and sigma is a rank 0
    
    # OUTPUTS: diffusion term, which has the shape of x.
    return sigma*x


def Sig_ou(x,sigma):
    # This is the function for the diffusion of the OU process
    # ARGUMENTS:
    # ** x: is either a rank 1, 2 or 3 array, with rank (dim), (N,dim) or (N,N,dim)
    # ** respectively.
    # ** sigma: is a rank 2 array with dimension (dim,dim)
    # OUTPUTS:
    # ** sig is either a rank 2, 3 or 4 array, with rank (dim,dim), (N,dim,dim)
    # ** or (N,N,dim,dim) respectively, depending on the rank of x.
    shape= np.concatenate((x.shape,np.array([sigma.shape[0]])))
    sig=np.zeros(shape)+sigma
    return sig
#%%
# r's OR GRADIENTS OF THE LOG OF THE KERNEL OF THE AUXILIAR PROCESS

# note: The sequence of arguments "t,x,T,x_pr" is used thorought the document and 
# it keeps this order obiquous to the reader.


def r_log_normal(t,x,T,x_pr,pars):

    
    # This is the function for the gradient w.r.t. x of the log of the kernel

    # ARGUMENT:

    # ** a,b,fi=pars
    # ** a and b are arrays each with dimension (N) 
    # ** fi is a scalar

    # OUTPUTS:
    # ** r is a rank 1 array with dimension (N)

    a,b,fi=pars
    mu=np.log(x )+ a*T**2/2+ (b-fi**2/2)*T-(a*t**2/2+ (b-fi**2/2)*t)

    r=-(mu-np.log(x_pr))/(x*fi**2*(T-t))

    return r




def r_normal_1d(t,x,T,x_pr,pars,crossed=False):
    #t, x, T, x_pr, pars, pars_numb

    # This is the function for the gradient w.r.t. x of the log of the kernel
    # of the auxiliar process
    # The normal variable taht this function represent is one dimensional
    # ARGUMENTS:
    # ** x_p: is a rank 1  with dim (N) 
    # ** x: is a rank 1 or 2 array with dim (N) or (N,N), respectively.
    # ** This is the value of the process at time t.
    # ** t and T are such that the auxiliar transition is X_T=x_pr|X_t=x
    # ** pars: include sd, the standard deviation function, and sd_pars which are the parameters
    # ** of the standard deviation function
    #
    # OUTPUTS:
    # ** grad_x_log_p is a rank 1 or 2 array with dims (N) and (N,N)
    # ** respectively, depending on the rank of x.

    [sd,sd_pars]=pars
    #trans_st(t,alpha,sigma[0,0]))**2
    if crossed==True:
        grad_x_log_p=-(x-x_pr[np.neaxis,:])/(sd(t,x,T,x_pr,sd_pars)**2)
    else:
        grad_x_log_p=-(x-x_pr)/(sd(t,x,T,x_pr,sd_pars)**2)
    return grad_x_log_p


def r_quasi_normal_1d(t,x,T,x_pr,pars,crossed=False):
    #t, x, T, x_pr, pars, pars_numb
    
    # This is the function for the gradient w.r.t. x of the log of the kernel
    # of the auxiliar process
    # This function made to compute the r for the o.u. process, given that is the mean of the variable
    # is shifted.
    # IMPORTANT! In the case where crossed=True we need to choose a function sd accordingly to this. Current
    # sd function defined in this code don't have this charactheristic.

    # ARGUMENTS:
    # ** x_p: is a rank 1 array, with rank (N)
    # ** x: is a rank 1 or 2 array with rank (N) or (N,N), respectively. 
    # ** This is the value of the process at time t0.
    # ** t and T are such that the auxiliar transition is the distribution of X_T=x_pr|X_t=x
    # ** pars: include sd, the standard deviation function, and sd_pars which are the parameters
    # ** of the standard deviation function
    #
    # OUTPUTS:
    # ** grad_x_log_p is the shape of x.
    # ** depending on the boolean crossed.

    [sd,sd_pars]=pars
    [mu,sigma]=sd_pars
    Phi=np.exp(mu*(T-t))
    #trans_st(t,alpha,sigma[0,0]))**2


    
    grad_x_log_p=-(x*Phi-x_pr)*Phi/(sd(t,x,T,x_pr,sd_pars)**2)

    
    #print("the diff is",x*Phi-x_pr)
    #print("and the sd is",sd(t,x,T,x_pr,sd_pars)**2)
    return grad_x_log_p



def r_quasi_normal(t,x,T,x_pr,pars,crossed=False):
    #t, x, T, x_pr, pars, pars_numb
    # THIS FUNCTION IS MADE ONLY FOR dim=1, larger dimensions will require a
    # a more careful definition of the gradient.
    # This is the function for the gradient w.r.t. x of the log of the kernel
    # of the auxiliar process
    # The normal variable taht this function represent is one dimensional
    # ARGUMENTS:
    # ** x_p: is a rank 2 array, with rank (N,dim)
    # ** x: is a rank 2 array with rank (N,dim) 
    # ** This is the value of the auxiliar process at time t0.
    # ** t and T are such that the auxiliar transition is X_T=x_pr|X_t=x
    # ** pars: include sd, the standard deviation function, and sd_pars which are the parameters
    # ** of the standard deviation function
    #
    # OUTPUTS:
    # ** grad_x_log_p is a rank 2 array (N,DIM) or a rank 3 array (N,N,DIM)
    # ** depending on the boolean crossed.

    [sd,sd_pars]=pars
    [sigma,mu,t0,x0]=sd_pars
    Phi=np.exp(mu*(T-t))
    #trans_st(t,alpha,sigma[0,0]))**2
    if crossed==True:
        grad_x_log_p=-(x[:,np.newaxis]*Phi-x_pr)/(sd(t,x,T,x_pr,sd_pars)**2)
    else:
        grad_x_log_p=-Phi*(x*Phi-x_pr)/(sd(t,x,T,x_pr,sd_pars)**2)
        #print("the diff is",x*Phi-x_pr)
        #print("and the sd is",sd(t,x,T,x_pr,sd_pars)**2)
    return grad_x_log_p

# the following funciton was defined in the early stages of the project, it's not used in the current
# version, we keep it since it's part of some of the testing of other functions through the document.

def r_normal(t,x,T,x_pr,pars,crossed=False):
    #t, x, T, x_pr, pars, pars_numb
    # THIS FUNCTION IS MADE ONLY FOR dim=1, larger dimensions will require a
    # a more careful definition of the gradient.
    # This is the function for the gradient w.r.t. x of the log of the kernel
    # of the auxiliar process
    # The normal variable taht this function represent is one dimensional
    # ARGUMENTS:
    # ** x_p: is a rank 2 array, with rank (N,dim)
    # ** x: is a rank 2 array with rank (N,dim) 
    # ** This is the value of the auxiliar process at time t0.
    # ** t and T are such that the auxiliar transition is X_T=x_pr|X_t=x
    # ** pars: include sd, the standard deviation function, and sd_pars which are the parameters
    # ** of the standard deviation function
    #
    # OUTPUTS:
    # ** grad_x_log_p is a rank 2 array (N,N,DIM)
    # ** respectively, depending on the rank of x.

    [sd,sd_pars]=pars
    #trans_st(t,alpha,sigma[0,0]))**2
    if crossed==True:
        grad_x_log_p=-(x[:,np.newaxis]-x_pr)/(sd(t,x,T,x_pr,sd_pars)**2)
    else:
        grad_x_log_p=-(x-x_pr)/(sd(t,x,T,x_pr,sd_pars)**2)
    return grad_x_log_p
#%%
# HESSIANS OF THE LOG OF THE KERNEL OF THE AUXILIAR PROCESS

def H_log_normal(t,x,T,x_pr,pars):

    
    # This is the function for the gradient w.r.t. x of the log of the kernel

    # ARGUMENT:

    # ** a,b,fi=pars
    # ** a and b are arrays each with dimension (N) 
    # ** fi is a scalar

    # OUTPUTS:
    # ** r is a rank 1 array with dimension (N)

    a,b,fi=pars
    mu=np.log(x )+ a*T**2/2+ (b-fi**2/2)*T-(a*t**2/2+ (b-fi**2/2)*t)

    H=-(mu-np.log(x_pr)-1)/(x**2*fi**2*(T-t))

    return H

def H_normal_1d(t,x,T,x_pr,pars):
    # This is the function for the Hessian (not the negative of the hessian) w.r.t. x of the log of the kernel
    # of the auxiliar process
    # The normal variable taht this function represent is one dimensional
    # ARGUMENTS:
    # ** x_p: is a rank 1 array, with dim (N)
    # ** x: is a rank 2 or 3 array with dim (N) and (N,N), respectively.
    # ** This is the value of the process at time t0.
    # ** t and T are such that the auxiliar transition is X_T=x_pr|X_t=x
    # ** pars: include sd, the standard deviation function, and sd_pars which are the parameters
    # ** of the standard deviation function

    # OUTPUTS:
    # ** Hess_x_log_p is a rank 2 array (N,N) or a rank 1 array (N)
    # ** respectively, depending on the rank of x.
    [sd,sd_pars]=pars
    Phi=np.exp(theta*(T-t))

    hess_x_log_p=(np.zeros(x.shape)-1)/(sd(t,x,T,x_pr,sd_pars)**2)
 
 
    return hess_x_log_p

def H_normal(t,x,T,x_pr,pars,crossed=False):
    # THIS FUNCTION IS MADE ONLY FOR dim=1, larger dimensions will require a
    # a more careful definition of the Hessian.
    # This is the function for the Hessian (not the negative of the hessian) w.r.t. x of the log of the kernel
    # of the auxiliar process
    # The normal variable taht this function represent is one dimensional
    # ARGUMENTS:
    # ** x_p: is a rank 2 array, with rank (N,dim)
    # ** x: is a rank 2 array with rank (N,dim) 
    # ** This is the value of the auxiliar process at time t0.
    # ** t and T are such that the auxiliar transition is X_T=x_pr|X_t=x
    # ** pars: include sd, the standard deviation function, and sd_pars which are the parameters
    # ** of the standard deviation function

    # OUTPUTS:
    # ** Hess_x_log_p is a rank 3 array (N,N,DIM) or a rank 2 array (N,DIM )
    # ** respectively, depending on the rank of x.
    [sd,sd_pars]=pars
    if crossed==True:
        hess_x_log_p=(np.zeros((x.shape[0],x.shape[0],x.shape[-1]))-1)/(sd(t,x,T,x_pr,sd_pars)**2)
    else:
        hess_x_log_p=(np.zeros((x.shape[0],x.shape[-1]))-1)/(sd(t,x,T,x_pr,sd_pars)**2)
    return hess_x_log_p

def H_quasi_normal(t,x,T,x_pr,pars,crossed=False):

    # This is the function for the Hessian (not the negative of the hessian) w.r.t. x of the log of the kernel
    # of the auxiliar process
    # The normal variable taht this function represent is one dimensional
    # ARGUMENTS:
    # ** x_p: is a rank 1 array, with dim (N)
    # ** x: is a rank 2 or 3 array with dim (N) and (N,N), respectively.
    # ** This is the value of the process at time t0.
    # ** t and T are such that the auxiliar transition is X_T=x_pr|X_t=x
    # ** pars: include sd, the standard deviation function, and sd_pars which are the parameters
    # ** of the standard deviation function

    # OUTPUTS:
    # ** Hess_x_log_p is a rank 2 array (N,N) or a rank 1 array (N)
    # ** respectively, depending on the rank of x.
    [sd,sd_pars,theta]=pars
    Phi=np.exp(theta*(T-t))
    hess_x_log_p=Phi**2*(np.zeros(x.shape)-1)/(sd(t,x,T,x_pr,sd_pars)**2)
 
    return hess_x_log_p



#%%

# GRADIENTS OF THE LOG OF THE KERNEL OF THE AUXILIAR PROCESS

# The gradiends of the function will have the same parameters of the funcitons.

def Grad_t_b_ou_1d(x,A):

    # This is the function for the gradient wrt the parameters of the drift of the OU process
    # ARGUMENTS:
    # ** x: is either a rank 0, 1, or 2 array, with rank (1)  (N), (N,N) 
    # ** respectively for the lastest.
    # ** A: is a rank 0 array with dimension 
    # OUTPUTS:
    # ** grad_t is is the same rank as x.
    return x

def Grad_t_Sig_ou_1d(x,sigma):
    
    # This is the function for the gradient wrt the parameters of the diffusion of the OU process

    # ARGUMENTS:
    # ** x: is either a rank 0, 1, or 2 with rank (1), (N) or (N,N)
    # ** respectively.
    # ** sigma: rank zero array
    # OUTPUTS:
    # ** grad_t has the shape of x. 
    
    grad_t=np.zeros(x.shape)+1
    return grad_t


def Grad_t_b_ou(x,A):

    # This is the function for the gradient wrt the parameters of the drift of the OU process
    # THIS FUNCTION IS JUST FOR dim=1, in this case n_par=1
    # ARGUMENTS:
    # ** x: is either a rank 1, 2 or 3 array, with rank (dim), (N,dim) or (N,N,dim)
    # ** respectively.
    # ** A: is a rank 2 array with dimension (dim,dim)
    # OUTPUTS:
    # ** grad_t is either a rank 2, 3 or 4 array, with rank (n_par,dim), (n_par,N,dim)
    # ** or (n_par,N,N,dim) respectively, depending on the rank of x.
    #x_ranks=np.array(x.shape)-x.shape
    #print(x_ranks)
    #x_ranks=x_ranks-x_ranks+1
    #x_ranks[-1]=x.shape[-1]
    #print(x_ranks)
    #grad_t=np.tile(x,x_ranks)
    return x[np.newaxis]


def Grad_t_Sig_ou(x,sigma_pars):
    [sigma,n_pars_s]=sigma_pars
    # This is the function for the gradient wrt the parameters of the diffusion of the OU process
    # THIS FUNCTION IS JUST FOR dim=1, in this case n_par_s=1
    # FOR MORE GENERAL DISTRIBUTIONS WE NEED TO RELATE THE n_par_s WITH THE DIMENSION OF THE
    # DRIFT SOMEHOW.
    # ARGUMENTS:
    # ** x: is either a rank 1, 2 or 3 array, with rank (dim), (N,dim) or (N,N,dim)
    # ** respectively.
    # ** sigma: is a rank 2 array with dimension (dim,dim)
    # OUTPUTS:
    # ** grad_t is either a rank 3, 4 or 5 array, with rank (n_pars_s,dim,dim), (n_pars_s,N,dim,dim)
    # ** or (n_par_s,N,N,dim,dim) respectively, depending on the rank of x.
    
    grad_t=np.zeros(np.concatenate(([n_pars_s],x.shape,np.array([sigma.shape[0]]))))+1
    return grad_t
#TEST FOR Grad_t_Sig_ou
"""
sigma=np.array([[2]])
n_pars_s= 1
x=np.array([[[1],[1]],[[1],[5]]])
sigma_pars=[sigma,n_pars_s]
grad_t=Grad_t_Sig_ou(x,sigma_pars)
print(grad_t)
"""


def Grad_x_Sig_ou(x,sigma):
    # This is the function for the gradient wrt to x, of the diffusion of the OU process
    # ARGUMENTS:
    # ** x: is either a rank 1, 2 or 3 array, with rank (dim), (N,dim) or (N,N,dim)
    # ** respectively.
    # ** sigma: is a rank 2 array with dimension (dim,dim)
    # OUTPUTS:
    # ** grad_x is either a rank 3, 4 or 5 array, with rank (dim,dim,dim), (N,dim,dim,dim)
    # ** or (N,N,dim,dim,dim) respectively, depending on the rank of x.
    shape= np.concatenate( (x.shape,sigma.shape))
    grad_x=np.zeros(shape)
    return grad_x



def Grad_x_b_ou_1d(x,A):
    # This is the function for the gradient wrt to x, of the drift of the OU process
    # ARGUMENTS:
    # ** x: is either a rank 0, 1, or 2 array, with rank (1)  (N), (N,N) 
    # ** respectively for the lastest.
    # ** A: is a rank 0 array with dimension 
    # OUTPUTS:
    # ** grad_x has the shape of x.
   
    shape= x.shape
    #print(x.shape,shape)
    grad_x=np.zeros(shape)+A
    return grad_x



def Grad_x_Sig_ou_1d(x,sigma):
    
    # ARGUMENTS:
    # ** x: is either a rank 0, 1 or 2 or 3 array with rank (1), (N) or (N,N)
    # ** respectively.
    # ** sigma: is a rank 0 array
    # OUTPUTS:
    # ** grad_x is has the shape of x.
    shape= x.shape
    grad_x=np.zeros(shape)
    return grad_x


def Grad_x_b_ou(x,A):
    # This is the function for the gradient wrt to x, of the drift of the OU process
    # ARGUMENTS:
    # ** x: is either a rank 1, 2 or 3 array, with rank (dim), (N,dim) or (N,N,dim) 
    # ** respectively.
    # ** A: is a rank 2 array with dimension (dim,dim)
    # OUTPUTS:
    # ** grad_x is either a rank 2, 3 or 4 array, with rank (dim,dim), (N,dim,dim)
    # ** or (N,N,dim,dim) respectively, depending on the rank of x.
    shape= np.concatenate((x.shape,np.array([A.shape[0]])))
    #print(x.shape,shape)
    grad_x=np.zeros(shape)+A.T
    return grad_x



def Grad_log_gamma_in_dist(x,pars):


    # THis function computes the gradient of the log of the gamma distribution
    # with respect to the parameters \theta_1, \theta_2, \theta_3; the gamama distriubtion
    # depends on them in the following way:
    # alpha=2*the1/the3**2, theta=the3**2/(2*the2), where the parametrization of the gamma distriubtion is
    # f(x|alpha,theta)=1/(Gamma(alpha)*)
    # 

    the1,the2,the3=pars

    alpha=2*the1/the3**2
    theta=the3**2/(2*the2)

    part_alpha=np.log(x)-np.log(theta)-digamma(alpha)
    part_theta=x/theta**2-alpha/theta

    part_the1=part_alpha*alpha/the1 
    part_the2=-part_theta*theta/the2
    part_the3=part_alpha*(-2)*alpha/the3+part_theta*(2)*theta/the3

    return part_the1,part_the2,part_the3

def Grad_log_g_nbino(x, y, the4):
    """
    Compute the partial derivatives of the log of the Negative Binomial PMF
    with respect to p and r, given a single data point k >= 0.

    Parameterization:
       log f(k; p, r) = log Gamma(k+r) - log Gamma(k+1) - log Gamma(r)
                        + r * ln(1-p) + k * ln(p).

    Returns
    -------
    dlog_dp : float
        The partial derivative of log f(k; p, r) w.r.t. p
    dlog_dr : float
        The partial derivative of log f(k; p, r) w.r.t. r

    Notes
    -----
    - k is the number of "failures" until r successes.
    - p in (0,1) is the probability of failure in each Bernoulli trial
      (according to one common Negative Binomial convention).
    - r > 0 is the (possibly non-integer) target number of successes.
    """
    p=the4/(the4+x)
    r=the4
    dp_dthe4=1/(the4+x)-the4/(the4+x)**2
    # Partial w.r.t. p
    
    dlog_dp_1 = -y[0]/(1 - p) + r/p
    dlog_dp_2 = -y[1]/(1 - p) + r/p

    # Partial w.r.t. r
    
    dlog_dr_2 = digamma(y[1] + r) - digamma(r) + np.log(p)
    dlog_dr_1 = digamma(y[0] + r) - digamma(r) + np.log(p)



    dlog_dthe4_1=dlog_dp_1*dp_dthe4+dlog_dr_1
    dlog_dthe4_2=dlog_dp_2*dp_dthe4+dlog_dr_2
    return dlog_dthe4_1+dlog_dthe4_2

def Grad_trans_log_normal(t,x,T,x_pr,pars):

    # This function computes the gradient of the transition density of the auxiliary process
    # Since we only need the partial with respect to \theta_3 this is the only one we compute.
    # Note that the mu depends on theta3 through a and b, turns out that this dependence cancels out
    # so the derivative of mu is zeor w.r.t. theta3.

    # ARGUMENTS:
    # pars=[a,b,fi]:  a and b are the result of evaluating the drift 
    # of the logarithmic sde at differetn spots, these parameters are 
    # related to the time linearization of the auxiliar process.

    # OUTPUT:
    # Partial derivative of the transition density of the auxiliar process with respect to \theta_3.
     

    a,b,fi=pars
    mu=np.log(x)+ a*T**2/2+ (b-fi**2/2)*T-(a*t**2/2+ (b-fi**2/2)*t)

    return (mu-np.log(x_pr))**2/(fi**3*(T-t))-1/fi



def Grad_log_G(x,y,pars):
    sigma_y_s=pars
    return (1/2)*(x-y)**2/sigma_y_s**2-1/(2*sigma_y_s)

def Grad_log_G_new(x,y,pars):
    sigma_y_s=pars**2
    return (1/2)*(x-y)**2/sigma_y_s**2-1/(2*sigma_y_s)

#%%
def Grad_log_aux_trans_ou(t,x,T,x_pr,pars):

    # This function computes the derivative of the transition density of the auxiliar process
    # when this is a OU process. The derivative is w.r.t. the parameters of the OU process
    # with the particularity that the derivate w.r.t. the square of the diffusion term instead of 
    # the diffusion term

    # ARGUMENTS: pars include the rank zero parameter sigma
    # and theta.
    # t,x,T,x_pr are the usual bunch.

    # OUTPUTS: the derivative of the transition density of the auxiliar process, rank zero
    # with respect to the square of the diffusion term.
    [theta,sigma_s]=pars
    Sigma_s=(sigma_s/(2*theta))*(np.exp(2*theta*(T-t))-1)   
    der_theta=-1/(2*Sigma_s)*(x*np.exp(theta*(T-t))-x_pr)*(T-t)*x*np.exp(theta*(T-t))\
    +1/(2*Sigma_s)*((x*np.exp(theta*(T-t))-x_pr)**2/(Sigma_s)-1)*(sigma_s/(2*theta**2))\
    *(1-np.exp(2*theta*(T-t))-2*theta*(T-t)*np.exp(2*theta*(T-t)))
    der_sigma_s=(1/(2*Sigma_s*sigma_s))*(x*np.exp(theta*(T-t))-x_pr)**2 - 1/(2*sigma_s)
    return np.array([der_theta,der_sigma_s])

def Grad_log_aux_trans_ou_new(t,x,T,x_pr,pars):

    # This function computes the derivative of the transition density of the auxiliar process
    # when this is a OU process. The derivative is w.r.t. the parameters of the OU process.
    # As opposed to Grad_log_aux_trans_ou, this function´s argument includes the 
    # diffusion term instead of the square
    # of the diffusion term.

    # ARGUMENTS: pars include the rank zero parameter sigma
    # and theta.
    # t,x,T,x_pr are the usual bunch.

    # OUTPUTS: the derivative of the transition density of the auxiliar process, rank zero
    # with respect to the square of the diffusion term.
    [theta,sigma]=pars
    sigma_s=sigma**2
    
    Sigma_s=(sigma_s/(2*theta))*(np.exp(2*theta*(T-t))-1)   
    # We don't need it
    der_theta=0
    der_sigma_s=(1/(2*Sigma_s*sigma_s))*(x*np.exp(theta*(T-t))-x_pr)**2 - 1/(2*sigma_s)
    return np.array([der_theta,der_sigma_s*2*np.sqrt(sigma_s)])

# Test for Grad_d

#sigma_y_s=.1
#x=np.array([0,2,3])
#y=np.array([1,1.5,3.1])
#print(Grad_log_G(x,y,sigma_y_s))

#Test for Grad_aux_trans_ou
#t=1
#T=3
#x=np.array([1,2,3])
#x_pr=np.array([-2,1.3,4.4])
#theta=-0.3
#sigma_s=0.1
#pars=[theta,sigma_s]
#print(Grad_log_aux_trans_ou(t,x[0],T,x_pr[0],pars))


#%%

# Standar deviations 



def ou_sd(t,x,T,x_pr,pars):

    # This function computes the standard deviation of 
    # the transition of the ou process, i.e. x_T=x'|x_t=x

    # ARGUMENTS: pars include the rank zero parameter sigma
    # and theta.

    # OUTPUTS: the standard deviation of the transition, rank zero


    [theta,sigma]=pars
    return sigma*np.sqrt((np.exp(2*theta*(T-t))-1)/(2*theta))


def trans_st(t,alpha,sigma):
    # This function was used in the testing stage and it's suitable for the current 
    # version of the project.
    return sigma*np.sqrt((T**(2*alpha+1)-t**(2*alpha+1))/((2*alpha+1)*T**(2*alpha)))

# We create the new version of the function trans_st to account for a change of parameters 
# in the funciton making it more general
def new_trans_st(t,x,T,x_pr,pars):
    # This function computes the standard deviation of a process 
    # where the transition is given by x_T=x_pr|x_t=x and the process
    # has diffusion function sigma*((t/T)**alpha) and zero drift.

    # ARGUMENTS: pars include the rank zero parameter sigma
    # and alpha.

    # OUTPUTS: the standard deviation of the transition, rank zero

    [alpha,sigma]=pars
    return sigma*np.sqrt((T**(2*alpha+1)-t**(2*alpha+1))/((2*alpha+1)*T**(2*alpha)))


def brow_aux_sd(t,x,T,x_pr,pars):

    # This function computes the standard deviation of a process that is a Brownian motion
    # with diffusion sigma*x_pr and zero drift. Note that in this case x_pr is treated as 
    # an static parameter of the process.

    # ARGUMENTS: pars include the rank zero parameter sigma and x_pr.

    # OUTPUTS: the standard deviation of the transition, rank zero

    sigma=pars
    return sigma*x_pr*np.sqrt((T-t))


def gbm_aux_sd(t,x,T,x_pr,pars):
    # This function is called gbm_axu_sd because it is the standard deviation of the auxiliar process
    # for the GBM.
    # ARGUMENTS:
    # t,x,T,x_pr are the usual parameters
    # sigma is a rank 2 array with dimension (dim,dim) in this case

    # It's prossible that in order to evaluate the backward transition kernel 
    # the rank of x will be 3 with dimensions (N,N,dim),
    # In that case a minor modification will be needed 
    # for A,C,D,F, Until then we keep this simple presentation.

    # OUTPUTS:
    # 
    # b
    [sigma,mu,t0,x0]=pars
    # Compute A
    A = (sigma@(x_pr.T-x0.T)).T/((T-t0))
    #print("A is ",A)
    # Compute C
    C =(sigma@(x0.T*T-x_pr.T*t0)).T/((T-t0))

    Phi=np.exp(mu*(T-t))

    """
    print("A is", A)
    print("C is", C)
    print("D is", D)
    print("F is", F)
    """
   

    u=t 

    xit=-(( (2 * C**2 * mu**2 + 2 * A * C * mu * (1 + 2 * mu * u) \
    + A**2 * (1 + 2 * mu * u + 2 * mu**2 * u**2))) / (4 * mu**3))
    

    u=T 

    xiT=-np.exp(-2*mu*(T-t))*(( (2 * C**2 * mu**2 + 2 * A * C * mu * (1 + 2 * mu * u) \
    + A**2 * (1 + 2 * mu * u + 2 * mu**2 * u**2))) / (4 * mu**3))

    return Phi*np.sqrt(xiT-xit)


def gbm_aux_sd_1d(t,x,T,x_pr,pars):
    # This function is called gbm_axu_sd because it is the standard deviation of the auxiliar process
    # for the GBM.
    # ARGUMENTS:
    # t,x,T,x_pr are the usual parameters
    # sigma is a rank 0 array.

    # It's prossible that in order to evaluate the backward transition kernel 
    # the rank of x will be 3 with dimensions (N,N,dim),
    # In that case a minor modification will be needed 
    # for A,C,D,F, Unitil then we keep this simple presentation.

    # OUTPUTS:
    # Phi*np.sqrt(xiT-xit): same rank as x_pr.
    # 
    [sigma,mu,t0,x0]=pars
    # Compute A
    A = (sigma*(x_pr-x0))/((T-t0))
    #print("A is ",A)
    # Compute C
    C =(sigma*(x0*T-x_pr*t0))/((T-t0))

    Phi=np.exp(mu*(T-t))

    """
    print("A is", A)
    print("C is", C)
    print("D is", D)
    print("F is", F)
    """
   

    u=t 

    xit=-(( (2 * C**2 * mu**2 + 2 * A * C * mu * (1 + 2 * mu * u) \
    + A**2 * (1 + 2 * mu * u + 2 * mu**2 * u**2))) / (4 * mu**3))
    

    u=T 

    xiT=-np.exp(-2*mu*(T-t))*(( (2 * C**2 * mu**2 + 2 * A * C * mu * (1 + 2 * mu * u) \
    + A**2 * (1 + 2 * mu * u + 2 * mu**2 * u**2))) / (4 * mu**3))

    return Phi*np.sqrt(xiT-xit)




def alpha_trans_sd(t,x_pr,sd_pars):
    # This is the function for the standard deviation of the auxiliar process
    # when the proposal transition is gaussian. Notice that in some instances 
    # the parameters on x', that is not the case of these funciton
    # nevertheless we include this to follow the conversion. 
    [alpha,sigma]=sd_pars
        
    return sigma*np.sqrt((T**(2*alpha+1)-t**(2*alpha+1))/((2*alpha+1)*T**(2*alpha)))


def new_alpha_trans_sd(t,x,T,x_pr,sd_pars):
    # This is the function for the standard deviation of the auxiliar process
    # when the proposal transition is gaussian. Notice that in some instances 
    # the parameters on x', that is not the case of these funciton
    # nevertheless we include this to follow the conversion. 
    [alpha,sigma]=sd_pars
        
    return sigma*np.sqrt((T**(2*alpha+1)-t**(2*alpha+1))/((2*alpha+1)*T**(2*alpha)))


#%%

# generators of realization 

def gen_gen_data_1d(T,x0,l,collection_input): #After generator of general data
    
    # Function that generates euler maruyama samples of a difussion process
    
    # ARGUMENTS: T: final time of the discretization
    # x0: Initial position of the difussion, rank 0 array of 
    # l: level of discretization, the time step of this discretization is 2^{-l}
    # collection input: dim is the dimension of the difussion, b is a drift 
    # function that takes on two arguments, the diffusion array at an specific time
    # and the parameters of the functions A. The diffusion function of the process is 
    # Sig, which takes on the same first argument as b and the second argument is 
    # fi.
    # OUTPUTS: A rank 1 array of dimensions T*2**l with the Euler-Maruyama
    # discretization.
    [ b,A,Sig,fi]=collection_input
    J=T*(2**l)
    I=identity(1).toarray()
    #I_o=identity(dim_o).toarray()
    tau=2**(-l)
    v=np.zeros(J+1)    
    #v[0]=np.random.multivariate_normal(m0,C0,(1)).T
    v[0]=x0



    for j in range(J):
        ## truth
        #print(np.shape(Sig(v[j],fi)),np.shape(b(v[j],A)))
        #print(b(v[j],A).shape,Sig(v[j],fi).shape)
        v[j+1] = v[j]+b(v[j],A)*tau + np.sqrt(tau)*(np.random.normal(0,1))*(Sig(v[j],fi))
        ## observation
        
    return v


def g_normal_1d(x,sd):

    # Sampling function: it samples observations as normal samples
    # with mean x and covariance cov:

    # ARGUMENTS:
    # x: Realization of the process at the times of the observation,
    # rank 2 with dimensions (p,dim), where p is the number of observations.
    # cov: covariance of the gaussian distribution. 
    return x+np.random.normal(0,sd,x.shape)


def g_normal(x,cov):

    # Sampling function: it samples observations as normal samples
    # with mean x and covariance cov:

    # ARGUMENTS:
    # x: Realization of the process at the times of the observation,
    # rank 2 with dimensions (p,dim), where p is the number of observations.
    # cov: covariance of the gaussian distribution. 
    dim=cov.shape[0]

    return x+np.random.multivariate_normal(np.zeros(dim),cov,x.shape[0])



def gen_gen_data(T,x0,l,collection_input): #After generator of general data
    # parameters [dim,dim_o, b_ou,A,Sig_ou,fi,ht,H]
    # Function that generates euler maruyama samples of a difussion process
    # ARGUMENTS: T: final time of the discretization
    # x0: Initial position of the difussion, rank 1 array of dimension dim
    # l: level of discretization, the time step of this discretization is 2^{-l}
    # collection input: dim is the dimension of the difussion, b is a drift 
    # function that takes on two arguments, the diffusion array at an specific time
    # and the parameters of the functions A. The diffusion function of the process is 
    # Sig, which takes on the same first argument as b and the second argument is 
    # fi.
    # OUTPUTS: A rank two array of dimensions (T*2**l,dim) with the Euler-Maruyama
    # discretization.
    [dim, b,A,Sig,fi]=collection_input
    J=T*(2**l)
    I=identity(dim).toarray()
    #I_o=identity(dim_o).toarray()
    tau=2**(-l)
    v=np.zeros((J+1,dim))    
    #v[0]=np.random.multivariate_normal(m0,C0,(1)).T
    v[0]=x0



    for j in range(J):
        ## truth
        #print(np.shape(Sig(v[j],fi)),np.shape(b(v[j],A)))
        #print(b(v[j],A).shape,Sig(v[j],fi).shape)
        v[j+1] = v[j]+b(v[j],A)*tau + np.sqrt(tau)*(np.random.multivariate_normal(np.zeros(dim),I))@(Sig(v[j],fi).T)
        ## observation
        
    return v



def gen_obs(x,g,g_par):

    # Function that generates observations from a given process.
    # ARGUMENTS: x: is a vector with the values of the diffusion
    # at certain regular times with rank two and dimension(p,dim),
    # where dim is the dimension of the diff process.
    # g: is a function that generates the samples accordingly to a given kernel,
    # its arguments are the true position of the signal and the hyperparameters of 
    # the function. 
    # OUTPUTS: We have 3 outputs 
    # obs: the observations generated, with the same dimensions (we assume that in the 
    # simulations) as x.
    obs=np.zeros(x.shape)
    for i in range(len(x)):
        obs[i]=g(x[i],g_par)


    return obs


#%%

# Density functions


def trans_prop_log_normal(t,x,T,x_pr,pars):


    # This function computes the transition of the log_normal distribution
    # Its differernce with "trans_log_normal" is that this function is not
    # tailored for the linealization in time of the drift. 

    # ARGUMENTS:

    the1,the2,the3=pars
    mu=(the1)*(T-t)+np.log(x)
    s=the3*np.sqrt(T-t)
    return ss.lognorm.pdf(x_pr,s=s,scale=np.exp(mu))


def trans_prop_log_normal_2(t,x,T,x_pr,pars):

    # This function is different from trans_prop_log_normal in the scale, 
    # the difference comes from taking a different approach when approximating 
    # the original transition density, instead of approximating the drift as 
    # x(the3^2/2+the1) we approximate it as x(the3^2/2+the1-the2*(mean of the invariant distribution))

    # This function computes the transition of the log_normal distribution
    # Its differernce with "trans_log_normal" is that this function is not
    # tailored for the linealization in time of the drift. 

    # ARGUMENTS:

    the1,the2,the3=pars
    mu=np.log(x)
    s=the3*np.sqrt(T-t)
    return ss.lognorm.pdf(x_pr,s=s,scale=np.exp(mu))




def trans_log_normal(t,x,T,x_pr,pars):

    # This function is made in order to use it a the transtion dsitribution
    # for individual paths, meaning that the endpoint of each paths determine 
    # a and b.

    # ARGUMENTS:
    # pars=[a,b,fi]:  a and b are the result of evaluating the drift 
    # of the logarithmic sde at differetn spots, these parameters are 
    # related to the time linearization of the auxiliar process.

    # OUTPUT:
    # the evaluation of the transition density of the sistem x_pr at time T conditioned
    # on x at time t. 

    a,b,fi=pars
    mu=np.log(x)+ a*T**2/2+ (b-fi**2/2)*T-(a*t**2/2+ (b-fi**2/2)*t)
    s=fi*np.sqrt(T-t)
    return ss.lognorm.pdf(x_pr,s=s,scale=np.exp(mu))


def trans_log_normal_test(t,x,T,x_pr,pars):

    # This function the transtion probability of a lognormal distribution


    s=pars
    return ss.lognorm.pdf(x_pr,s=s,scale=x)


#%%




def trans_noncentral_chisquare(t0,x0,T,x_pr,pars):
    """
    This function computes the transition density of a noncentral p(x/c)/c
    where p() is the noncentral chi square distribution with parameters
    df and lam. This distribution is the transtion of the process
    x_T|x_t0=x0, where x_t= (the1-the2*x_t)*dx_t+the3*sqrt(x_t)dB_t

    ARGUMENTS:
    t0,x0,T,x_pr are the usual parameters
    pars=[the1,the2,the3]

    OUTPUTS:
    The transition
     """
    the1,the2,the3=pars
    d=T-t0

    c=the3**2*(1-np.exp(-the2*d))/(4*the2)
    df=4*the1/the3**2
    lam=4*the2*np.exp(-the2*d)*x0/(the3**2*(1-np.exp(-the2*d)))
    # Convert x to a NumPy array for vectorized evaluation
    x = np.asarray(x_pr/c, dtype=float)
    pdf_vals = np.zeros_like(x)

    pdf_vals=ncx2.pdf(x, df, lam)
    return pdf_vals/c
#%%
"""
t0=1.4
N=3
x0=np.random.chisquare(4,N)
T=3.2
x_pr=np.random.chisquare(4,3)
x_pr=np.array([1.9,3.2,2])
x0=1.4
the1=2
the2=1
the3=0.5
dist_params=[the1,the2,the3]

trans_noncentral_chisquare(t0,x0,T,x_pr,dist_params)
"""
#%%




def g_den_1d(y, x,   sd  ,crossed=False):
    # This function computes the conditional density of y given x
    # of a normal distribution.
    # y: rank 0 array 
    # x: rank 1 array with dimensions (N)
    # sd: standar deviation of the normal distribution
    
    # OUTPUT: gs is a rank 1 array with dimensions (N) withe the likelihood values

    if crossed==False:  
        mean=x
        diffs=y-mean
        exponent = -0.5 * (diffs**2/sd**2)
        gs=np.exp(exponent)/np.sqrt(2*np.pi*sd**2)
        return gs
    else:
        mean=x
        diffs=y-mean
        exponent = -0.5 * (diffs**2/sd**2)
        gs=np.exp(exponent)/np.sqrt(2*np.pi*sd**2)
        
        return gs[np.newaxis,:]+np.zeros((x.shape[0],x.shape[0]))


def log_g_nbino_den(y,x,the4):

    # This function computes the log likelihood of the negative binomial distribution
    # More specifically, we compute two densities of the nb distribution with parameters
    # the4 and p=the4/(the4+x) for the values y[0] and y[1], the mean of this distribution
    # is x and variance x*(the4+x)/(the4)

    # ARGUMENTS: y is a rank 1 array with dimensions (2), x is a rank 1 array with dim N
    # the4 is a rank 0 array
    
    # OUTPUT: the log likelihood of the negative binomial distribution
    # with rank 1 and dim N
    p=the4/(the4+x)
    den1=ss.nbinom.pmf(y[0],the4,p)
    den2=ss.nbinom.pmf(y[1],the4,p)

    return np.log(den1)+np.log(den2)

def log_g_normal_den(y,x,sd,crossed=False):
    # function that computes the  log likelihood for each individual 
    # observation for the normal distribution with mean y and 
    # standard deviation sd 
    # ARGUMENTS: y is a rank 0 array, x is a rank 1 or two array with dimensions (N)
    # or (N,N) respectively. sd is the standard deviation of the normal distribution. 
    if crossed==False:
        mean=x
        diffs=y-mean
        return -0.5*(diffs**2/sd**2)-0.5*np.log(2*np.pi*sd**2)
    else:
        mean=x
        diffs=y-mean
        probs= -0.5 * (diffs**2/sd**2)-0.5*np.log(2*np.pi*sd**2)
        return probs[np.newaxis,:]+np.zeros((x.shape[0],x.shape[0]))


def aux_trans_den_lin_t_diff_1d(t0,x0,T,x_pr,pars,crossed=False):

    # This function computes the density of the auxiliar transition
    # ARGUMENTS:
    # t,x,T,x_pr are the usual parameters
    # pars=[aux_sd,sigma,mu,t0,x0]
    # OUTPUT: Transitions densities of the auxiliar process
    # which is a r.v. 

    if crossed==False:
        
        [aux_sd,mu,sigma,t0,x0]=pars
        aux_trans_den=norm(loc=x0*np.exp(mu*(T-t0)),scale=aux_sd(t0,x0,T,x_pr,[mu,sigma]))
        return aux_trans_den.pdf(x_pr)

    else:
        
        [aux_sd,mu,sigma,t0,x0]=pars
        aux_trans_den=norm(loc=(x0[:,np.newaxis]+np.zeros((x0.shape[0],x0.shape[0])))*np.exp(mu*(T-t0)),scale=aux_sd(t0,x0,T,x_pr,[mu,sigma]))
        return aux_trans_den.pdf(x_pr[np.newaxis,:]+np.zeros((x0.shape[0],x0.shape[0])))

#Test for the crossed=True fo aux_trans_den_lin_t_diff_1d

"""
t0=0
x0=np.array([4,1,-4.5])
T=1
x_pr=np.array([10,3.2,-.5])
pars=[ou_sd,1,0.5,t0,x0]
print(aux_trans_den_lin_t_diff_1d(t0,x0,T,x_pr,pars,crossed=False))
"""

def ou_trans_den(t0,x0,T,x_pr,pars,crossed=False):
    # This function computes the density of the  transition
    # of an ou process with 

    # NOTE: This function is an special case of the funciton aux_trans_den_lin_t_diff_1d. 

    # ARGUMENTS:
    # t,x,T,x_pr are the usual parameters
    # pars=[mu,sigma], whihc are the parameter mu and sigma of the SDE.
    
    # OUTPUT:
    # The transition density of the ou process, a rank 1 or 2 array with dimensions (N) or (N,N)
    # depending on the argument crossed. 
    #print(pars)
    if crossed==False:
        [mu,sigma]=pars
        rv=norm(loc=x0*np.exp(mu*(T-t0)),scale=ou_sd(t0,x0,T,x_pr,pars) )
        prob=rv.pdf(x_pr)

    else:
        [mu,sigma]=pars
        rv=norm(loc=(x0[:,np.newaxis]+np.zeros((x0.shape[0],x0.shape[0])))*np.exp(mu*(T-t0)),scale=ou_sd(t0,x0,T,x_pr,pars) )
        prob=rv.pdf(x_pr[np.newaxis,:]+np.zeros((x0.shape[0],x0.shape[0])))


    return prob
"""
t0=0
x0=np.array([1,2.])
T=1
x_pr=np.array([12,2.])
pars=[0.5,2]
print(ou_trans_den(t0,x0,T,x_pr,pars))
"""
#%%

def aux_trans_den_alpha(t0,x0,T,x_pr,pars):

    # This function computes the density of the auxiliar transition
    # when the auxiliary process's SDE is dx_t=sigma*(t/T)^alpha*dW_t
    # ARGUMENTS:
    # t,x,T,x_pr are the usual parameters
    # pars=[alpha,sigma], whihc are the parameter alpha and sigma of the SDE.

    # OUTPUT: Transitions densities of the auxiliar process

    aux_trans=norm(loc=x0,scale=new_alpha_trans_sd(t0,x0,T,x_pr,pars))
    return aux_trans.pdf(x_pr)



def prop_trans_den_lognormal_1d(t0,x0,T,x_pr,sample_pars,crossed=False):
    # This function computes the density of the proposal transition of a lognormal distribution
    # ARGUMENTS:
    # t,x,T,x_pr are the usual parameters
    # pars=[prop_sd,sigma,mu,t0,x0]
    # OUTPUT:
    if crossed==False:
        
        [mu,sigma_prop]=sample_pars
        prop_trans_den=(np.exp(-(np.log(x_pr)-np.log(x0)-(mu-sigma_prop**2/2)*(T-t0))**2/(2*sigma_prop**2*(T-t0)))\
        /(np.sqrt(2*np.pi)*sigma_prop*np.sqrt(T-t0)*x_pr))
    else:
        [mu,sigma_prop]=sample_pars
        prop_trans_den=(np.exp(-(np.log(x_pr[np.newaxis,:])-np.log(x0[:,np.newaxis])-(mu-sigma_prop**2/2)*(T-t0))**2/(2*sigma_prop**2*(T-t0)))\
        /(np.sqrt(2*np.pi)*sigma_prop*np.sqrt(T-t0)*x_pr[np.newaxis,:]))

    return  prop_trans_den
#%%
#Test for the prop_trans_den_lognormal_1d function
"""
t0=0 
T=2
x0=np.array([1,2,3])
x_pr=np.array([1.5,0.2,3.5])
sample_pars=[0.5,2]
print(prop_trans_den_lognormal_1d(t0,x0,T,x_pr,sample_pars,crossed=True))
"""
#%%

def g_den(y, x,   g_par  ):
    # Compute the conditional density of y given x
    # y: rank 2 array with dimensions (# of observations, dim)
    # x: rank 3 or rank 4 array with dimensions (# of observations, particles, dim)
    # (# of observations, particles, particles, dim)
    # g_par: parameters of the density (rank 2 array with dimensions (dim, dim))
    # Compute the mean of the multivariate normal distribution
    # OUTPUT: gs is a rank 3 or rank 4 depending on the rank of x with dimensions
    # (# of observations, particles) or (# of observations, particles, particles)
    # respectively.
    if x.ndim==3:
        mean = x
        # Compute the probabilities using the multivariate normal distribution and matrix multiplication
        diffs = y[:, np.newaxis, :] - mean
        exponent = -0.5 * np.sum(diffs @ np.linalg.inv(g_par) * (diffs), axis=2)
        gs = np.exp(exponent) / np.sqrt((2 * np.pi) ** mean.shape[-1] * np.linalg.det(g_par))


    if x.ndim==4:
        diffs=y[:,np.newaxis,np.newaxis,:]-x
        exponent = -0.5 * np.sum(np.dot(diffs, np.linalg.inv(g_par)) * (diffs), axis=-1)
        gs = np.exp(exponent) / np.sqrt((2 * np.pi) ** x.shape[-1] * np.linalg.det(g_par))
    return gs



#%%
# Sampling of proposal transitions

"""
def trans_prop_log_normal(t,x,T,x_pr,pars):


    # This function computes the transition of the log_normal distribution
    # Its differernce with "trans_log_normal" is that this function is not
    # tailored for the linealization in time of the drift. 

    # ARGUMENTS:

    the1,the2,the3,fi=pars
    mu=(the3**2/2-the1-fi**2/2)(T-t)+np.log(x)
    s=fi*np.sqrt(T-t)
    return ss.lognorm.pdf(x_pr,s=s,scale=np.exp(mu))


"""

def sampling_prop_log_normal(x0,N,d,sample_pars):

    the1,the2,the3=sample_pars
    mu=(the1)*(d)+np.log(x0)
    s=the3*np.sqrt(d)
    return np.random.lognormal(mean=mu,sigma=s,size=N)

def sampling_prop_log_normal_2(x0,N,d,sample_pars):

    the1,the2,the3=sample_pars
    mu=np.log(x0)
    s=the3*np.sqrt(d)
    return np.random.lognormal(mean=mu,sigma=s,size=N)




def sampling_CIR(x0,N,d,sample_pars):

    # This function samples from the transition of an OU process.

    # ARGUMENTS:
    # x0: rank 0 array with dimensions (dim)
    # N: number of samples
    # d: temporal interval where we compute one iteration of the PF
    # sample_pars: 

    # OUTPUT: 
    # rank 1 array with dimensions (N) with the samples of the proposal transition.

    the1,the2,the3=sample_pars

    c=the3**2*(1-np.exp(-the2*d))/(4*the2)
    df=4*the1/the3**2
    lamb=4*the2*np.exp(-the2*d)*x0/(the3**2*(1-np.exp(-the2*d)))
    chi=np.random.noncentral_chisquare(df,lamb,N)
    return c*chi

def sampling_log_normal(x0,N,d,sample_pars):
    
    s=sample_pars
    # Function to test the particle filter of the lognormal process.

    return np.random.lognormal(mean=np.log(x0),sigma=s,size=N)

#%%







def sample_lognorm(x0,N,d,sample_pars):
    [mu,sigma_prop]=sample_pars
    x_pr=np.random.lognormal(np.log(x0)+(mu-sigma_prop**2/2)*(d),sigma_prop*np.sqrt(d),(N,1))    
    return x_pr



def sampling_alpha_trans_props(x0,N,d,sample_pars):

    #THIS FUNCTION SAMPLES THE PROPOSAL TRANSITION, \hat{\rho}.

    # ARGUMENTS:
    # x0: rank 0 array with dimensions (dim)
    # N: number of samples
    # d: temporal interval where we compute one iteration of the PF
    # sample_pars: parameters of the proposal distribution, which are the parameter
    # alpha and the diffusion term of the auxiliar distribution

    # OUTPUT: 
    # rank 1 array with dimensions (N) with the samples of the proposal transition.

    [alpha,sigma]=sample_pars
    #(t,x,T,x_pr,pars)
    return np.random.normal(x0,sigma*np.sqrt(d/(2*alpha+1)))


def sampling_ou(x0,N,d,sample_pars):

    # This function samples from the transition of an OU process.

    # ARGUMENTS:
    # x0: rank 0 array with dimensions (dim)
    # N: number of samples
    # d: temporal interval where we compute one iteration of the PF
    # sample_pars: parameters of the proposal distribution, which are the parameter
    # theta and the diffusion term of the auxiliar distribution

    # OUTPUT: 
    # rank 1 array with dimensions (N) with the samples of the proposal transition.


    [theta,sigma]=sample_pars
    #(t,x,T,x_pr,pars)
    return np.random.normal(x0*np.exp(theta*d),ou_sd(0,x0,d,x0,[theta,sigma]))


# test  sampling_alpha_trans_props
#N=4
#x0=np.random.normal(0,1,N)
#d=3
#sample_pars=[0.5,2]
#print(sampling_alpha_trans_props(x0,N,d,sample_pars))



#%%
# Bridges

def Bridge(t0,x0,T,x_p,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,l,d,N,dim,crossed=False):

    # This function samples the bridge process (the approximated one, using the auxiliar process) 
    # using the Euler-Maruyama method and additionally computes 
    # the integral of the bridge process leading to the change of measure w.r.t. the original bridge process.
    
    # ARGUMENTS: 

    # t0: initial time of the bridge process
    # x0: initial position of the bridge process
    # T: final time of the bridge process
    # x_p: final position of the bridge process
    # b: drift of the origianl process
    # A: parameters of the drift of the original process
    # Sig: diffusion of the original process
    # fi: parameters of the diffusion of the original process
    # b_til: drift of the auxiliar process
    # A_til: parameters of the drift of the auxiliar process
    # Sig_til: diffusion of the auxiliar process
    # fi_til: parameters of the diffusion of the auxiliar process
    # r: function that computes the gradient of the log of the kernel of the auxiliar process
    # r_pars: parameters of the function r
    # H: function that computes the Hessian of the log of the kernel of the auxiliar process
    # H_pars: parameters of the function H
    # l: level of discretization
    # d: temporal interval where we compute one iteration of the PF
    # N: number of samples
    # dim: dimension of the process

    # OUTPUTS:

    # x: rank 3 array with dimensions (steps+1,N,dim) with the bridge process
    # int_G: rank 2 array with dimensions (N,N) with the integral of the bridge process
    # or rank 1 array with dimensions (N) if crossed=False
    # x_test(specific for testing): rank 2 array with dimensions (N,dim) with the bridge process at time t_test

    
    steps=int((2**(l))*d)
    dt=1./2**l
    x=np.zeros((steps+1,N,dim))
    x[0]=x0
    x[-1]=x_p
    if crossed==True:
        int_G=np.zeros((N,N))
    else:
        int_G=np.zeros(N)
    #print("int_G shape is: ",int_G.shape)
    I=identity(dim).toarray()
    

    for i in np.array(range(steps-1)):

        t=t0+(i)*dt
        r_n=r(t,x[i],T,x_p,r_pars,crossed=crossed) # rank 2 o 3 with dims (N,dim) or (N,N,dim)
        # respectively, depending on the boolean crossed.
        # 1) what do I need from a_tilde? 2) How can I get it in terms of Sig_tilde?
        # 3) what are the specifications of Sig_tilde? 3) answer: the same as Sig.
        # 1) answer:
        Sig_tilde=Sig_til(t,x[i],fi_til)
        Sig_reali=Sig(x[i],fi)
        if crossed==False:
            a_tilde=np.einsum('nij,njl->nil',Sig_tilde,Sig_tilde) # rank 3 array with dimensions (N,dim,dim)
            a=np.einsum('nij,njl->nil',Sig_reali,Sig_reali)
        else:
            a_tilde=np.einsum('nmij,nmjl->nmil',Sig_tilde,Sig_tilde)
            a=np.einsum('nmij,nmjl->nmil',Sig_reali,Sig_reali)
            # rank 4 array with dimensions (N,N,dim,dim)
        
        # for dim=1
        drift=b(x[i],A)+(a*r_n[:,:,np.newaxis])[:,:,0]

        # more generally
        #drift=b(x[i],A)+np.einsum('nij,nj->ni',a,r_n)
        #print("r_n's shape is ",(((a-a_tilde)*(-H_normal(t,x[i],T,x_pr,H_pars,crossed=crossed)[:,:,np.newaxis]\
        #-(r_n**2)[:,:,np.newaxis]))[:,:,0]).shape)
        x[i+1]=x[i]+drift*dt+Sig(x[i],fi)[:,0]*np.sqrt(dt)*np.random.normal(0,1,(N,1))
        int_G=int_G+dt*((b(x[i],A)-b_til(t,x[i],A_til))*r_n\
        -(1/2)*((a-a_tilde)*(-H(t,x[i],T,x_pr,H_pars,crossed=crossed)[:,:,np.newaxis]\
        -(r_n**2)[:,:,np.newaxis]))[:,:,0])[:,0]
        #print("int_G shape is: ",int_G.shape)
        #print("int_G first term is ", int_G[0])
        if t==t_test:
            x_test=x[i]
        
         
       
    return x,int_G,x_test

def EM(t0,x0,T,b,A,Sig,fi,l,N,seed):

    # This function computes the Euler-Maruyama discretization of a process
    Dt=(T-t0)/(2**l)
    steps=int((T-t0)/(Dt))
    x=np.zeros((2**l+1,N))
    x[0]=x0+np.zeros(N)
    for i in range(1,steps+1):
        x[i]=x[i-1]+b(x[i-1],A)*Dt+np.sqrt(Dt)*np.random.normal(0,1,N)*Sig(x[i-1],fi)

    return x

    
def EM_logistic(t0,x0,T,the1,the2,the3,l,N,seed,in_c=False):


    stat=time.time()
    Dt=1/(2**l)
    steps=int((T-t0)/(Dt))
    
    x=np.zeros((steps+1,N))
    x[0]=x0+np.zeros(N)
    if in_c==True:
        dW=np.sqrt(Dt)*np.random.normal(0,1,(steps,N))
        for i in range(1,steps+1):
            #x[i]=x[i-1]+(the3**2/2+the1-the2*x[i-1])*x[i-1] *Dt+dW[i-1]*the3*x[i-1]
            x[i]=x[i-1]+(the1/the3-the2/the3*np.exp(the2*x[i-1])*Dt+dW[i-1])
    else:
        for i in range(1,steps+1):
            x[i]=x[i-1]+(the1/the3-the2/the3*np.exp(the2*x[i-1]))*Dt+np.sqrt(Dt)*np.random.normal(0,1,N)
        
    time= time.time()-stat
    return time




def Bridge_1d_path(t0,x0,T,x_p,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,l,d,N,seed\
    ,crossed=False,backward=False,j=False,fd=False,N_pf=False,cond_seed=False,an=False,Dt=False,\
    non_seeded=False):
    
    # This function computes the bridge process for a one dimensional process, it has 
    # several instances of conditionals rendering different outcomes.

    # The cases are: 
    # backward: Boolean that indicates if we are computing the backward transition kernel, in this 
    # case only one sample of the N independent brownian motions is needed.
    # conditonal seed: variable for the case where backward sample is implemented.
    # j: index of the sample that is needed in the case of the backward transition kernel and other
    # cases to retriever the desired sample.
    # it is included as a keyword argument.
    # fd: Boolean that indicates if we are computing the finite difference of the gradient of the
    # log of the kernel of the auxiliar process. In this case, the seed is used to retrieve the
    # correct sample of the brownian motion.
    # crossed: This conditional is vestige of a failed generalization of the project, since it was 
    # in most of the function I decided not to remove it, simply not referencing in the evaluation
    # of the code so it doesn't affect the performance.
    # Dt: In the case we have irregular observations in time Dt is the time step of the discretization.


    
    # the drift and diffusion are b and Sig, respectively, and they take
    # x(a (N) dimensional) and A as arguments for the drift and x and fi for the diffusion.
    # the level of discretization l, the distance of resampling, the number of
    # particles N.
    
    # b_til,A_til,Sig_til,fi_til, are the analogous functions for the auxiliar process.
    # a difference is that their arguments depend on (t,x) for the drift and (t,x,fi_til) for the diffusion.
    # r is the function that computes the gradient of the log of the kernel of the auxiliar process
    # and it takes (t,x,T,x_pr,r_pars) as arguments.
    # H is the function that computes the Hessian of the log of the kernel of the auxiliar process
    # and it takes (t,x,T,x_pr,H_pars) as arguments. 
    # N_pf is used in the specific case where we use to bridge function to compute the finite difference
    # in this case N=1 and N_pf represents the number of the particle filter. It is needed in order to
    # retrieve the correct seeded sample.
    # corresponds to the number of the particle corresponding to the sample seed, or cond_seed in the case
    # of the forward iteration that is not fd. 
    # fd is used as a keyword that labels the process of the computation of int_G for the finite difference of 
    # the gradient.
    # The argument conditional_seed is used to obtain the sample of the int_G value of the 
    # conditional path. Previously, the value of int_G was store from it's original computations
    # so this complication was avoided. Nevertheless, in the implementation of the SGD, the parameters might change,
    # thus the value of the stored int_G loses validity.


    # OUTCOMES: int_G is the integral of the gradient of the log of the kernel of the auxiliar process
    # and depending on crossed it can be a rank 1 or 2 array with dimensions (N) or (N,N).
    # x_test is either a rank 1 or 2 array with dimensions (N) or (N,N) that represents the value of the process
    # at time t_test.

    if Dt==False:
        steps=int((2**(l))*d)
        dt=1./2**l
    else:
        dt=Dt
        steps=int((T-t0)/dt)
    
    if crossed==True:
        int_G=np.zeros((N,N))
        x0=x0[:,np.newaxis]+np.zeros((N,N))
        #x=np.zeros((steps+1,N,N))
        #x[0]=x0[:,np.newaxis]
        #x[-1]=x_p[np.newaxis,:]
    else:
        int_G=np.zeros(N)
        x0=x0+np.zeros(N)
        #x[0]=x0
        #x[-1]=x_p
    #print("int_G shape is: ",int_G.shape)
    #I=identity(dim).toarray()

    x_path=np.zeros((steps+1,N))
    x_path[0]=x0
    x_prev=x0

    x_the=0
    x_the_prev=0
    x_sig=0
    x_sig_prev=0
    int_psi_the=0
    int_psi_sig=0
    

    for i in np.array(range(steps-1)):

        t=t0+(i)*dt
        r_n=r(t,x_prev,T,x_p,r_pars) # rank 2 o 3 with dims (N,dim) or (N,N,dim)
        # respectively, depending on the boolean crossed.
        # 1) what do I need from a_tilde? 2) How can I get it in terms of Sig_tilde?
        # 3) what are the specifications of Sig_tilde? 3) answer: the same as Sig.
        # 1) answer:
        Sig_tilde=Sig_til(t,x_prev,fi_til)
        Sig_reali=Sig(x_prev,fi)
        a_tilde=Sig_tilde**2
        a=Sig_reali**2
        
        # for dim=1
        #print("b, x_prev and A are: ")
        #print("i is: ",i)
        #print(b,x_prev, A)
        drift=b(x_prev,A)+(a*r_n)
        # more generally
        #drift=b(x[i],A)+np.einsum('nij,nj->ni',a,r_n)
        #print("r_n's shape is ",(((a-a_tilde)*(-H_normal(t,x[i],T,x_pr,H_pars,crossed=crossed)[:,:,np.newaxis]\
        #-(r_n**2)[:,:,np.newaxis]))[:,:,0]).shape)
        #print("individual seeds are: ", seed+i)
        if backward==False and fd==False and cond_seed !=False:
            #print("seed is: ", seed)
            
            samp=np.sqrt(dt)*np.random.default_rng(seed=seed+i).normal(0,1,N)
            samp[0]=np.sqrt(dt)*np.random.default_rng(seed=cond_seed+i).normal(0,1,N)[j]
            x=x_prev+drift*dt+Sig_reali*samp

        elif an==True:

            samp=np.sqrt(dt)*np.random.default_rng(seed=seed+i).normal(0,1,N_pf)
            
            x=x_prev+drift*dt+Sig_reali*samp[j]
            x_the=x_the_prev+x_the_prev*(A+a*H(t,x_prev,T,x_p,H_pars))*dt\
            +x_prev*dt 
            x_sig=x_sig_prev+x_sig_prev*(A+a*H(t,x_prev,T,x_p,H_pars))*dt\
            +samp[j] 
            Sigma_s=(fi**2/(2*A_til))*(np.exp(A_til*(T-t))-1)
            int_psi_the=int_psi_the+dt*(-x_prev*(x_prev*np.exp(A_til*(T-t))-x_p)*np.exp(A_til*(T-t))/Sigma_s\
            - x_the_prev*(A-A_til)* np.exp(A_til*(T-t))*(2*x_prev*np.exp(A_til*(T-t))-x_p)\
            /Sigma_s)
            int_psi_sig=int_psi_sig+dt*(2*(b(x_prev,A)-b_til(t,x_prev,A_til))*(x_prev*np.exp(A_til*(T-t))-x_p)\
            *np.exp(A_til*(T-t))/(fi*Sigma_s)\
            - x_sig_prev*(A-A_til)*np.exp(A_til*(T-t))*(2*x_prev*np.exp(A_til*(T-t))-x_p)\
            /Sigma_s)

            if mid_step==i:
                x_the_mid=x_the
                x_sig_mid=x_sig

        
        elif backward==False and fd==False and cond_seed ==False:
            #print("seed is: ", seed)
            x=x_prev+drift*dt+Sig_reali*np.sqrt(dt)*np.random.default_rng(seed=seed+i).normal(0,1,N)
            x_path[i+1]=np.copy(x)
            #print("brownian is: ", np.random.default_rng(seed=seed+i).normal(0,1,N))

        elif non_seeded==True:

            x=x_prev+drift*dt+Sig_reali*np.sqrt(dt)*np.random.normal(0,1,N)
            x_path[i+1]=x
            

        elif backward==True:
            samp=np.random.default_rng(seed=seed+i).normal(0,1,N)
            samp[:]=samp[j]
            #print("samp is ", samp)
            #print("j is: ", j)
            
            x=x_prev+drift*dt+Sig_reali*np.sqrt(dt)*samp
        else:
            samp=np.random.default_rng(seed=seed+i).normal(0,1,N_pf)
            #print("correct conditional ")
            #print("l is ",l )
            #print("Sig_reali shape is: ", Sig_reali.shape)
            #print("drift shape is: ", drift.shape)
            #print("x[i] shape is", x[i].shape)  
            x=x_prev+drift*dt+Sig_reali*np.sqrt(dt)*samp[j]

        int_G=int_G+dt*((b(x_prev,A)-b_til(t,x_prev,A_til))*r_n\
        -(1/2)*((a-a_tilde)*(-H(t,x_prev,T,x_p,H_pars)\
        -(r_n**2))))
        x_prev=x

        if an==True:
            x_the_prev=x_the
            x_sig_prev=x_sig
        #print("int_G first term is ", int_G[0])
        #print("term's shape is ", (-H(t,x[i],T,x_pr,H_pars)) .shape)
        #print("int_G shape is: ",int_G.shape)

        """
        x[i+1]=x[i]+drift*dt+Sig(x[i],fi)[:,0]*np.sqrt(dt)*np.random.normal(0,1,(N,1))
        int_G=int_G+dt*((b(x[i],A)-b_til(t,x[i],A_til))*r_n\
        -(1/2)*((a-a_tilde)*(-H(t,x[i],T,x_pr,H_pars,crossed=crossed)[:,:,np.newaxis]\
        -(r_n**2)[:,:,np.newaxis]))[:,:,0])[:,0]
        """
    if an==False:

        return x_path,int_G
    else:
        return int_G,int_psi_the,int_psi_sig,x_the_mid,x_sig_mid
    


def Bridge_1d(t0,x0,T,x_p,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,l,d,N,seed\
    ,crossed=False,backward=False,j=False,fd=False,N_pf=False,cond_seed=False,an=False,Dt=False):
    
    # This function computes the bridge process for a one dimensional process, it has 
    # several instances of conditionals rendering different outcomes.

    # The cases are: 
    
    # backward: Boolean that indicates if we are computing the backward transition kernel, in this 
    # case only one sample of the N independent brownian motions is needed.
    # conditonal seed: variable for the case where backward sample is implemented.
    # j: index of the sample that is needed in the case of the backward transition kernel and other
    # cases to retriever the desired sample.
    # it is included as a keyword argument.
    # fd: Boolean that indicates if we are computing the finite difference of the gradient of the
    # log of the kernel of the auxiliar process. In this case, the seed is used to retrieve the
    # correct sample of the brownian motion.
    # crossed: This conditional is vestige of a failed generalization of the project, since it was 
    # in most of the function I decided not to remove it, simply not referencing in the evaluation
    # of the code so it doesn't affect the performance.
    # Dt: In the case we have irregular observations in time Dt is the time step of the discretization.


    
    # the drift and diffusion are b and Sig, respectively, and they take
    # x(a (N) dimensional) and A as arguments for the drift and x and fi for the diffusion.
    # the level of discretization l, the distance of resampling, the number of
    # particles N.
    
    # b_til,A_til,Sig_til,fi_til, are the analogous functions for the auxiliar process.
    # a difference is that their arguments depend on (t,x) for the drift and (t,x,fi_til) for the diffusion.
    # r is the function that computes the gradient of the log of the kernel of the auxiliar process
    # and it takes (t,x,T,x_pr,r_pars) as arguments.
    # H is the function that computes the Hessian of the log of the kernel of the auxiliar process
    # and it takes (t,x,T,x_pr,H_pars) as arguments. 
    # N_pf is used in the specific case where we use to bridge function to compute the finite difference
    # in this case N=1 and N_pf represents the number of the particle filter. It is needed in order to
    # retrieve the correct seeded sample.
    # corresponds to the number of the particle corresponding to the sample seed, or cond_seed in the case
    # of the forward iteration that is not fd. 
    # fd is used as a keyword that labels the process of the computation of int_G for the finite difference of 
    # the gradient.
    # The argument conditional_seed is used to obtain the sample of the int_G value of the 
    # conditional path. Previously, the value of int_G was store from it's original computations
    # so this complication was avoided. Nevertheless, in the implementation of the SGD, the parameters might change,
    # thus the value of the stored int_G loses validity.


    # OUTCOMES: int_G is the integral of the gradient of the log of the kernel of the auxiliar process
    # and depending on crossed it can be a rank 1 or 2 array with dimensions (N) or (N,N).
    # x_test is either a rank 1 or 2 array with dimensions (N) or (N,N) that represents the value of the process
    # at time t_test.

    if Dt==False:
        steps=int((2**(l))*d)
        dt=1./2**l
    else:
        dt=Dt
        steps=int((T-t0)/dt)
    
    if crossed==True:
        int_G=np.zeros((N,N))
        x0=x0[:,np.newaxis]+np.zeros((N,N))
        #x=np.zeros((steps+1,N,N))
        #x[0]=x0[:,np.newaxis]
        #x[-1]=x_p[np.newaxis,:]
    else:
        int_G=np.zeros(N)
        x0=x0+np.zeros(N)
        #x[0]=x0
        #x[-1]=x_p
    #print("int_G shape is: ",int_G.shape)
    #I=identity(dim).toarray()


    x_prev=x0

    x_the=0
    x_the_prev=0
    x_sig=0
    x_sig_prev=0
    int_psi_the=0
    int_psi_sig=0
    

    for i in np.array(range(steps-1)):

        t=t0+(i)*dt
        r_n=r(t,x_prev,T,x_p,r_pars) # rank 2 o 3 with dims (N,dim) or (N,N,dim)
        # respectively, depending on the boolean crossed.
        # 1) what do I need from a_tilde? 2) How can I get it in terms of Sig_tilde?
        # 3) what are the specifications of Sig_tilde? 3) answer: the same as Sig.
        # 1) answer:
        Sig_tilde=Sig_til(t,x_prev,fi_til)
        Sig_reali=Sig(x_prev,fi)
        a_tilde=Sig_tilde**2
        a=Sig_reali**2
        
        # for dim=1
        #print("b, x_prev and A are: ")
        #print("i is: ",i)
        #print(b,x_prev, A)
        drift=b(x_prev,A)+(a*r_n)
        # more generally
        #drift=b(x[i],A)+np.einsum('nij,nj->ni',a,r_n)
        #print("r_n's shape is ",(((a-a_tilde)*(-H_normal(t,x[i],T,x_pr,H_pars,crossed=crossed)[:,:,np.newaxis]\
        #-(r_n**2)[:,:,np.newaxis]))[:,:,0]).shape)
        #print("individual seeds are: ", seed+i)
        if backward==False and fd==False and cond_seed !=False:
            #print("seed is: ", seed)
            
            samp=np.sqrt(dt)*np.random.default_rng(seed=seed+i).normal(0,1,N)
            samp[0]=np.sqrt(dt)*np.random.default_rng(seed=cond_seed+i).normal(0,1,N)[j]
            x=x_prev+drift*dt+Sig_reali*samp

        elif an==True:

            samp=np.sqrt(dt)*np.random.default_rng(seed=seed+i).normal(0,1,N_pf)
            
            x=x_prev+drift*dt+Sig_reali*samp[j]
            x_the=x_the_prev+x_the_prev*(A+a*H(t,x_prev,T,x_p,H_pars))*dt\
            +x_prev*dt 
            x_sig=x_sig_prev+x_sig_prev*(A+a*H(t,x_prev,T,x_p,H_pars))*dt\
            +samp[j] 
            Sigma_s=(fi**2/(2*A_til))*(np.exp(A_til*(T-t))-1)
            int_psi_the=int_psi_the+dt*(-x_prev*(x_prev*np.exp(A_til*(T-t))-x_p)*np.exp(A_til*(T-t))/Sigma_s\
            - x_the_prev*(A-A_til)* np.exp(A_til*(T-t))*(2*x_prev*np.exp(A_til*(T-t))-x_p)\
            /Sigma_s)
            int_psi_sig=int_psi_sig+dt*(2*(b(x_prev,A)-b_til(t,x_prev,A_til))*(x_prev*np.exp(A_til*(T-t))-x_p)\
            *np.exp(A_til*(T-t))/(fi*Sigma_s)\
            - x_sig_prev*(A-A_til)*np.exp(A_til*(T-t))*(2*x_prev*np.exp(A_til*(T-t))-x_p)\
            /Sigma_s)

            if mid_step==i:
                x_the_mid=x_the
                x_sig_mid=x_sig

        
        elif backward==False and fd==False and cond_seed ==False:
            #print("seed is: ", seed)
            x=x_prev+drift*dt+Sig_reali*np.sqrt(dt)*np.random.default_rng(seed=seed+i).normal(0,1,N)
            #print("brownian is: ", np.random.default_rng(seed=seed+i).normal(0,1,N))

        elif backward==True:
            samp=np.random.default_rng(seed=seed+i).normal(0,1,N)
            samp[:]=samp[j]
            #print("samp is ", samp)
            #print("j is: ", j)
            
            x=x_prev+drift*dt+Sig_reali*np.sqrt(dt)*samp
        else:
            samp=np.random.default_rng(seed=seed+i).normal(0,1,N_pf)
            #print("correct conditional ")
            #print("l is ",l )
            #print("Sig_reali shape is: ", Sig_reali.shape)
            #print("drift shape is: ", drift.shape)
            #print("x[i] shape is", x[i].shape)  
            x=x_prev+drift*dt+Sig_reali*np.sqrt(dt)*samp[j]

        int_G=int_G+dt*((b(x_prev,A)-b_til(t,x_prev,A_til))*r_n\
        -(1/2)*((a-a_tilde)*(-H(t,x_prev,T,x_p,H_pars)\
        -(r_n**2))))
        x_prev=x

        if an==True:
            x_the_prev=x_the
            x_sig_prev=x_sig
        #print("int_G first term is ", int_G[0])
        #print("term's shape is ", (-H(t,x[i],T,x_pr,H_pars)) .shape)
        #print("int_G shape is: ",int_G.shape)

        """
        x[i+1]=x[i]+drift*dt+Sig(x[i],fi)[:,0]*np.sqrt(dt)*np.random.normal(0,1,(N,1))
        int_G=int_G+dt*((b(x[i],A)-b_til(t,x[i],A_til))*r_n\
        -(1/2)*((a-a_tilde)*(-H(t,x[i],T,x_pr,H_pars,crossed=crossed)[:,:,np.newaxis]\
        -(r_n**2)[:,:,np.newaxis]))[:,:,0])[:,0]
        """
    if an==False:

        return int_G
    else:
        return int_G,int_psi_the,int_psi_sig,x_the_mid,x_sig_mid
    



def Bridge_1d_opt(t0,x0,T,x_p,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,l,d,N,seed\
    ,crossed=False,backward=False,j=False,fd=False,N_pf=False,cond_seed=False,an=False,Dt=False,\
    in_c=False,mid_step=False):
    
    # This function computes the bridge process for a one dimensional process, it has 
    # several instances of conditionals rendering different outcomes.

    # The cases are: 
    
    # backward: Boolean that indicates if we are computing the backward transition kernel, in this 
    # case only one sample of the N independent brownian motions is needed.
    # conditonal seed: variable for the case where backward sample is implemented.
    # j: index of the sample that is needed in the case of the backward transition kernel and other
    # cases to retriever the desired sample.
    # it is included as a keyword argument.
    # fd: Boolean that indicates if we are computing the finite difference of the gradient of the
    # log of the kernel of the auxiliar process. In this case, the seed is used to retrieve the
    # correct sample of the brownian motion.
    # crossed: This conditional is vestige of a failed generalization of the project, since it was 
    # in most of the function I decided not to remove it, simply not referencing in the evaluation
    # of the code so it doesn't affect the performance.
    # Dt: In the case we have irregular observations in time Dt is the time step of the discretization.


    
    # the drift and diffusion are b and Sig, respectively, and they take
    # x(a (N) dimensional) and A as arguments for the drift and x and fi for the diffusion.
    # the level of discretization l, the distance of resampling, the number of
    # particles N.
    
    # b_til,A_til,Sig_til,fi_til, are the analogous functions for the auxiliar process.
    # a difference is that their arguments depend on (t,x) for the drift and (t,x,fi_til) for the diffusion.
    # r is the function that computes the gradient of the log of the kernel of the auxiliar process
    # and it takes (t,x,T,x_pr,r_pars) as arguments.
    # H is the function that computes the Hessian of the log of the kernel of the auxiliar process
    # and it takes (t,x,T,x_pr,H_pars) as arguments. 
    # N_pf is used in the specific case where we use to bridge function to compute the finite difference
    # in this case N=1 and N_pf represents the number of the particle filter. It is needed in order to
    # retrieve the correct seeded sample.
    # corresponds to the number of the particle corresponding to the sample seed, or cond_seed in the case
    # of the forward iteration that is not fd. 
    # fd is used as a keyword that labels the process of the computation of int_G for the finite difference of 
    # the gradient.
    # The argument conditional_seed is used to obtain the sample of the int_G value of the 
    # conditional path. Previously, the value of int_G was store from it's original computations
    # so this complication was avoided. Nevertheless, in the implementation of the SGD, the parameters might change,
    # thus the value of the stored int_G loses validity.


    # OUTCOMES: int_G is the integral of the gradient of the log of the kernel of the auxiliar process
    # and depending on crossed it can be a rank 1 or 2 array with dimensions (N) or (N,N).
    # x_test is either a rank 1 or 2 array with dimensions (N) or (N,N) that represents the value of the process
    # at time t_test.

    if Dt==False:
        steps=int((2**(l))*d)
        dt=1./2**l
    else:
        dt=Dt
        steps=int((T-t0)/dt)
    
    if crossed==True:
        int_G=np.zeros((N,N))
        x0=x0[:,np.newaxis]+np.zeros((N,N))
        #x=np.zeros((steps+1,N,N))
        #x[0]=x0[:,np.newaxis]
        #x[-1]=x_p[np.newaxis,:]
    else:
        int_G=np.zeros(N)
        x0=x0+np.zeros(N)
        #x[0]=x0
        #x[-1]=x_p
    #print("int_G shape is: ",int_G.shape)
    #I=identity(dim).toarray()


    x_prev=x0

    x_the=0
    x_the_prev=0
    x_sig=0
    x_sig_prev=0
    int_psi_the=0
    int_psi_sig=0

    if backward==False and fd==False and cond_seed !=False:

        for i in np.array(range(steps-1)):

            t=t0+(i)*dt
            r_n=r(t,x_prev,T,x_p,r_pars) # rank 2 o 3 with dims (N,dim) or (N,N,dim)
            Sig_tilde=Sig_til(t,x_prev,fi_til)
            Sig_reali=Sig(x_prev,fi)
            a_tilde=Sig_tilde**2
            a=Sig_reali**2
            drift=b(x_prev,A)+(a*r_n)
            samp=np.sqrt(dt)*np.random.default_rng(seed=seed+i).normal(0,1,N)
            samp[0]=np.sqrt(dt)*np.random.default_rng(seed=cond_seed+i).normal(0,1,N)[j]
            x=x_prev+drift*dt+Sig_reali*samp
            int_G=int_G+dt*((b(x_prev,A)-b_til(t,x_prev,A_til))*r_n\
            -(1/2)*((a-a_tilde)*(-H(t,x_prev,T,x_p,H_pars)\
            -(r_n**2))))
            x_prev=x

    elif backward==False and fd==False and cond_seed ==False:

        dW=np.sqrt(dt)*np.random.default_rng(seed=seed).normal(0,1,(steps-1,N))
        for i in np.array(range(steps-1)):
            t=t0+(i)*dt
            r_n=r(t,x_prev,T,x_p,r_pars) # rank 2 o 3 with dims (N,dim) or (N,N,dim)
            Sig_tilde=Sig_til(t,x_prev,fi_til)
            Sig_reali=Sig(x_prev,fi)
            a_tilde=Sig_tilde**2
            a=Sig_reali**2
            drift=b(x_prev,A)+(a*r_n)
            #x=x_prev+drift*dt+Sig_reali*np.sqrt(dt)*np.random.default_rng(seed=seed+i).normal(0,1,N)
            x=x_prev+drift*dt+Sig_reali*dW[i]

            int_G=int_G+dt*((b(x_prev,A)-b_til(t,x_prev,A_til))*r_n\
            -(1/2)*((a-a_tilde)*(-H(t,x_prev,T,x_p,H_pars)\
            -(r_n**2))))
            x_prev=x
        
    elif backward==True:

        for i in np.array(range(steps-1)):
            t=t0+(i)*dt
            r_n=r(t,x_prev,T,x_p,r_pars) # rank 2 o 3 with dims (N,dim) or (N,N,dim)
            Sig_tilde=Sig_til(t,x_prev,fi_til)
            Sig_reali=Sig(x_prev,fi)
            a_tilde=Sig_tilde**2
            a=Sig_reali**2
            drift=b(x_prev,A)+(a*r_n)

            samp=np.random.default_rng(seed=seed+i).normal(0,1,N)
            samp[:]=samp[j]
            x=x_prev+drift*dt+Sig_reali*np.sqrt(dt)*samp

            int_G=int_G+dt*((b(x_prev,A)-b_til(t,x_prev,A_til))*r_n\
            -(1/2)*((a-a_tilde)*(-H(t,x_prev,T,x_p,H_pars)\
            -(r_n**2))))
            x_prev=x

    else:

        for i in np.array(range(steps-1)):
            t=t0+(i)*dt
            r_n=r(t,x_prev,T,x_p,r_pars) # rank 2 o 3 with dims (N,dim) or (N,N,dim)
            Sig_tilde=Sig_til(t,x_prev,fi_til)
            Sig_reali=Sig(x_prev,fi)
            a_tilde=Sig_tilde**2
            a=Sig_reali**2
            drift=b(x_prev,A)+(a*r_n)

            samp=np.random.default_rng(seed=seed+i).normal(0,1,N_pf)
            x=x_prev+drift*dt+Sig_reali*np.sqrt(dt)*samp[j]

            int_G=int_G+dt*((b(x_prev,A)-b_til(t,x_prev,A_til))*r_n\
            -(1/2)*((a-a_tilde)*(-H(t,x_prev,T,x_p,H_pars)\
            -(r_n**2))))
            x_prev=x

    return int_G
    
    


def C_Bridge_1d(t0,x0_0,x0_1,T,x_p_0,x_p_1,b,A_0,A_1,Sig,fi_0,fi_1,b_til,A_til_0,A_til_1,\
    Sig_til,fi_til_0,fi_til_1,r,r_pars_0,r_pars_1,H,H_pars_0,H_pars_1,l,d,N,seed\
    ,crossed=False,backward=False,j_0=False,j_1=False,fd=False,N_pf=False,\
    cond_seed_0=False,cond_seed_1=False,Dt=False):
    # This function computes the coupled version of the bridge process with the specific cases. 

    # ARGUMENTS: the argument of the Kenel x0 rank 1 dims (dim) 
    # the drift and diffusion are b and Sig, respectively, and they take
    # x(either a (N) dimensional or (N,N) dimensional array) and A as arguments for the drift and x and fi for the diffusion.
    # the level of discretization l, the distance of resampling, the number of
    # particles N.
    # Grad_b is a function that takes (x,A) as argument and computes the gradnient of b wrt the 
    # parameters A, and evaluates it a (x,A).
    # b_til,A_til,Sig_til,fi_til, are the analogous functions for the auxiliar process.
    # a difference is that their arguments are (t,x) for the drift and (t,x,fi_til) for the diffusion.
    # r is the function that computes the gradient of the log of the kernel of the auxiliar process
    # and it takes (t,x,T,x_pr,r_pars) as arguments.
    # H is the function that computes the Hessian of the log of the kernel of the auxiliar process
    # and it takes (t,x,T,x_pr,H_pars) as arguments.
    # crossed is the boolean that indicates if we need the computations for the crossed terms that 
    # are needed for the smoother. 
    # N_pf is used in the specific case where we use to bridge function to compute the finite difference
    # in this case N=1 and N_pf represents the number of the particle filter. It is needed in order to
    # retrieve the correct seeded sample.
    # x is either a rank 2 or 3 array with dimensions (steps+1,N) or (steps+1,N,N)
    # corresponds to the number of the particle corresponding to the sample seed, or cond_seed in the case
    # of the forward iteration that is not fd. 
    # fd is used as a keyword that labels the process of the computation of int_G for the finite difference of 
    # the gradient.
    # The argument conditional_seed is used to obtain the sample of the int_G value of the 
    # conditional path. Previously, the value of int_G was store from it's original computations
    # so this complication was avoided. Nevertheless, in the implementation of the SGD, the parameters might change,
    # thus the value of the stored int_G loses validity.


    # OUTCOMES: int_G is the integral of the gradient of the log of the kernel of the auxiliar process
    # and depending on crossed it can be a rank 1 or 2 array with dimensions (N) or (N,N).
    # x_test is either a rank 1 or 2 array with dimensions (N) or (N,N) that represents the value of the process
    # at time t_test.
    #print("l, and d are: ")
    #print(l,d)
    if Dt==False:
        steps_0=int((2**(l-1))*d)
        #steps_1=int((2**(l))*d)
        dt_1=1./2**l
        dt_0=1./2**(l-1)        
    else:
        dt_1=Dt
        dt_0=Dt*2
        steps_0=int((T-t0)/dt_0)
        
        
    if crossed==True:
        int_G_1=np.zeros((N,N))
        int_G_0=np.zeros((N,N))
        
        x0_1=x0_1[:,np.newaxis]+np.zeros((N,N))
        x0_0=x0_0[:,np.newaxis]+np.zeros((N,N))
        #x=np.zeros((steps+1,N,N))
        #x[0]=x0[:,np.newaxis]
        #x[-1]=x_p[np.newaxis,:]
    else:
        int_G_0=np.zeros(N)
        x0_0=x0_0+np.zeros(N)
        int_G_1=np.zeros(N)
        x0_1=x0_1+np.zeros(N)
        #x_1_s=np.zeros((steps_0*2+1,N))
        #x_1_s[0]=x0_1
        #x_1_s[-1]=x_p_1
        #x_0_s=np.zeros((steps_0+1,N))
        #x_0_s[0]=x0_0
        #x_0_s[-1]=x_p_0
        #x[0]=x0
        #x[-1]=x_p
    #print("int_G shape is: ",int_G.shape)
    #I=identity(dim).toarray()
    x_prev_0=np.copy(x0_0)
    x_prev_1=np.copy(x0_1)
    dW=np.zeros((2,N))
    for i in np.array(range(steps_0-1)):
        for s in range(2):
            t_1=t0+(2*i+s)*dt_1
            r_n_1=r(t_1,x_prev_1,T,x_p_1,r_pars_1) # rank 2 o 3 with dims (N,dim) ofr (N,N,dim)
            # respectively, depending on the boolean crossed.
            # 1) what do I need from a_tilde? 2) How can I get it in terms of Sig_tilde?
            # 3) what are the specifications of Sig_tilde? 3) answer: the same as Sig.
            # 1) answer:
            Sig_tilde_1=Sig_til(t_1,x_prev_1,fi_til_1)
            Sig_reali_1=Sig(x_prev_1,fi_1)
            a_tilde_1=Sig_tilde_1**2
            a_1=Sig_reali_1**2
            drift_1=b(x_prev_1,A_1)+(a_1*r_n_1)

            if backward==False and fd==False and cond_seed_1 !=False:
                #print("seed is: ", seed)
                dW[s]=np.sqrt(dt_1)*np.random.default_rng(seed=seed+2*i+s).normal(0,1,N)
                dW[s,0]=np.sqrt(dt_1)*(np.random.default_rng(seed=cond_seed_1+2*i+s).normal(0,1,N))[j_1]
                x_1=x_prev_1+drift_1*dt_1+Sig_reali_1*dW[s]

            elif backward==False and fd==False and cond_seed_1 ==False:
                #print("seed is: ", seed)
                dW[s]=np.sqrt(dt_1)*np.random.default_rng(seed=seed+2*i+s).normal(0,1,N)
                x_1=x_prev_1+drift_1*dt_1+Sig_reali_1*dW[s]
                #print("brownian is: ", np.random.default_rng(seed=seed+i).normal(0,1,N))
            elif backward==True:
                dW[s]=np.sqrt(dt_1)*np.random.default_rng(seed=cond_seed_1+2*i+s).normal(0,1,N)[j_1]
                #dW[s,:]=dW[s,j_1]
                #print("samp is ", samp)
                #print("j is: ", j)
                x_1=x_prev_1+drift_1*dt_1+Sig_reali_1*dW[s]
            elif fd!=False:
                #print("Correct conditional 1")
                dW[s]=np.sqrt(dt_1)*np.random.default_rng(seed=cond_seed_1+2*i+s).normal(0,1,N_pf)
                #print("Sig_reali shape is: ", Sig_reali.shape)
                #print("drift shape is: ", drift.shape)
                #print("x[i] shape is", x[i].shape)  
                x_1=x_prev_1+drift_1*dt_1+Sig_reali_1*dW[s,j_1]

            int_G_1=int_G_1+dt_1*((b(x_prev_1,A_1)-b_til(t_1,x_prev_1,A_til_1))*r_n_1\
            -(1/2)*((a_1-a_tilde_1)*(-H(t_1,x_prev_1,T,x_p_1,H_pars_1)\
            -(r_n_1**2))))
            x_prev_1=x_1    
            #x_1_s[2*i+s+1]=x_1
        
        t_0=t0+(i)*dt_0
        r_n_0=r(t_0,x_prev_0,T,x_p_0,r_pars_0) # rank 2 o 3 with dims (N,dim) or (N,N,dim)
        # respectively, depending on the boolean crossed.
        # 1) what do I need from a_tilde? 2) How can I get it in terms of Sig_tilde?
        # 3) what are the specifications of Sig_tilde? 3) answer: the same as Sig.
        # 1) answer:
        Sig_tilde_0=Sig_til(t_0,x_prev_0,fi_til_0)
        Sig_reali_0=Sig(x_prev_0,fi_0)
        a_tilde_0=Sig_tilde_0**2
        a_0=Sig_reali_0**2
        # for dim=1
        #print("b, x_prev and A are: ")
        #print("i is: ",i)
        #print(b,x_prev, A)
        drift_0=b(x_prev_0,A_0)+(a_0*r_n_0)
        # more generally
        #drift=b(x[i],A)+np.einsum('nij,nj->ni',a,r_n)
        #print("r_n's shape is ",(((a-a_tilde)*(-H_normal(t,x[i],T,x_pr,H_pars,crossed=crossed)[:,:,np.newaxis]\
        #-(r_n**2)[:,:,np.newaxis]))[:,:,0]).shape)
        #print("individual seeds are: ", seed+i)
        if backward==False and fd==False and cond_seed_0!=False:
            #print("INCORRECT IMPLEMENTATION")
            #print("seed is: ", seed)
            dW_0=np.sum(dW,axis=0)
            # the following line is necessary since the cond_seed_0 might be different from cond_seed_1.
            # similarly, j_0 might be different from j_1.
            dW_0[0]=np.sqrt(dt_1)*((np.random.default_rng(seed=cond_seed_0+2*i).normal(0,1,N))[j_0]\
            +(np.random.default_rng(seed=cond_seed_0+2*i+1).normal(0,1,N))[j_0])
            x_0=x_prev_0+drift_0*dt_0+Sig_reali_0*dW_0

        elif backward==False and fd==False and cond_seed_0==False:
            
            x_0=x_prev_0+drift_0*dt_0+Sig_reali_0*np.sum(dW,axis=0)
            #print("brownian is: ", np.random.default_rng(seed=seed+i).normal(0,1,N))
        elif backward==True:
            #print("INCORRECT IMPLEMENTATION")
            dW_0=(np.sqrt(dt_1)*np.random.default_rng(seed=cond_seed_0+2*i).normal(0,1,N)+\
            np.sqrt(dt_1)*np.random.default_rng(seed=cond_seed_0+2*i+1).normal(0,1,N))[j_0]
            
            #dW_0[:]=dW_0[j_0]
            #print("samp is ", samp)
            #print("j is: ", j)
            
            x_0=x_prev_0+drift_0*dt_0+Sig_reali_0*dW_0

        elif fd !=False:
            #print("Correct conditional 0")
            dW_0=(np.sqrt(dt_1)*np.random.default_rng(seed=cond_seed_0+2*i).normal(0,1,N_pf)+\
            np.sqrt(dt_1)*np.random.default_rng(seed=cond_seed_0+2*i+1).normal(0,1,N_pf))[j_0]
            x_0=x_prev_0+drift_0*dt_0+Sig_reali_0*dW_0

        int_G_0=int_G_0+dt_0*((b(x_prev_0,A_0)-b_til(t_0,x_prev_0,A_til_0))*r_n_0\
        -(1/2)*((a_0-a_tilde_0)*(-H(t_0,x_prev_0,T,x_p_0,H_pars_0)\
        -(r_n_0**2))))
        x_prev_0=x_0
        #x_0_s[i+1]=x_0
        #print("int_G first term is ", int_G[0])
        #print("term's shape is ", (-H(t,x[i],T,x_pr,H_pars)).shape)
        #print("int_G shape is: ",int_G.shape)
        """
        x[i+1]=x[i]+drift*dt+Sig(x[i],fi)[:,0]*np.sqrt(dt)*np.random.normal(0,1,(N,1))
        int_G=int_G+dt*((b(x[i],A)-b_til(t,x[i],A_til))*r_n\
        -(1/2)*((a-a_tilde)*(-H(t,x[i],T,x_pr,H_pars,crossed=crossed)[:,:,np.newaxis]\
        -(r_n**2)[:,:,np.newaxis]))[:,:,0])[:,0]
        """
    # The following lines are added since the last computation of the euler maruyama scheme is 
    # at differernt times for the levels l and l-1. This is due to the fact that the process is a 
    # bridge and the final value of the discretization is already determined.
    ##################
    i=steps_0-1
    s=0
    t_1=t0+(2*i+s)*dt_1
    r_n_1=r(t_1,x_prev_1,T,x_p_1,r_pars_1) # rank 2 o 3 with dims (N,dim) or (N,N,dim)
    # respectively, depending on the boolean crossed.
    # 1) what do I need from a_tilde? 2) How can I get it in terms of Sig_tilde?
    # 3) what are the specifications of Sig_tilde? 3) answer: the same as Sig.
    # 1) answer:
    Sig_tilde_1=Sig_til(t_1,x_prev_1,fi_til_1)
    Sig_reali_1=Sig(x_prev_1,fi_1)
    a_tilde_1=Sig_tilde_1**2
    a_1=Sig_reali_1**2
    drift_1=b(x_prev_1,A_1)+(a_1*r_n_1)
    if backward==False and fd==False and cond_seed_1 !=False:
        #print("seed is: ", seed)
        dW[s]=np.sqrt(dt_1)*np.random.default_rng(seed=seed+2*i+s).normal(0,1,N)
        dW[s,0]=np.sqrt(dt_1)*np.random.default_rng(seed=cond_seed_1+2*i+s).normal(0,1,N)[j_1]
        x_1=x_prev_1+drift_1*dt_1+Sig_reali_1*dW[s]
    elif backward==False and fd==False and cond_seed_1 ==False:
        #print("seed is: ", seed)
        dW[s]=np.sqrt(dt_1)*np.random.default_rng(seed=seed+2*i+s).normal(0,1,N)
        x_1=x_prev_1+drift_1*dt_1+Sig_reali_1*dW[s]
        #print("brownian is: ", np.random.default_rng(seed=seed+i).normal(0,1,N))
    elif backward==True:
        dW[s]=np.sqrt(dt_1)*np.random.default_rng(seed=seed+2*i+s).normal(0,1,N)
        dW[s]=dW[s,j_1]
        #print("samp is ", samp)
        #print("j is: ", j)
        x_1=x_prev_1+drift_1*dt_1+Sig_reali_1*dW[s]

    elif fd!=False:
        dW[s]=np.sqrt(dt_1)*np.random.default_rng(seed=cond_seed_1+2*i+s).normal(0,1,N_pf)
        #print("Sig_reali shape is: ", Sig_reali.shape)
        #print("drift shape is: ", drift.shape)
        #print("x[i] shape is", x[i].shape)  
        x_1=x_prev_1+drift_1*dt_1+Sig_reali_1*dW[s,j_1]

    int_G_1=int_G_1+dt_1*((b(x_prev_1,A_1)-b_til(t_1,x_prev_1,A_til_1))*r_n_1\
    -(1/2)*((a_1-a_tilde_1)*(-H(t_1,x_prev_1,T,x_p_1,H_pars_1)\
    -(r_n_1**2))))
    x_prev_1=x_1    
    #x_1_s[2*i+s+1]=x_1
    
    return int_G_0,int_G_1
#%%

#print(1e-41!=False)



#%%
"""
# Step 1: Create a NumPy array of random numbers
N = 10  # Size of the array
rng = np.random.default_rng(seed=42)  # Create a random number generator
rand_vec = rng.normal(0, 1, N)  # Generate an array of N random numbers

print("Original array:")
print(rand_vec)

# Step 2: Select a component from the array (e.g., the component at index 5)
index = 5
selected_value = rand_vec[index]

# Step 3: Set all elements of the array to the value of the selected component
rand_vec[:] = selected_value

print("Modified array:")
print(rand_vec)
"""
#%%

#%%
#print(np.random.default_rng(seed=0).normal(0,1,4))
#np.random.default_rng(seed=0).normal(0,1,4)



#%%

def Bridge_cheap(t0,x0,T,x_p,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,l,d,N,crossed=False):
    # This is the function for the transition Kernel M(x,du): R^{d_x}->P(E_l)
    # ARGUMENTS: the argument of the Kenel x0 rank 1 dims (dim) 
    # the drift and diffusion are b and Sig, respectively, and they take
    # x(either a (N) dimensional or (N,N) dimensional array) and A as arguments for the drift and x and fi for the diffusion.
    # the level of discretization l, the distance of resampling, the number of
    # particles N.
    # Grad_b is a function that takes (x,A) as argument and computes the gradnient of b wrt the 
    # parameters A, and evaluates it a (x,A).
    # b_til,A_til,Sig_til,fi_til, are the analogous functions for the auxiliar process.
    # a difference is that their arguments are (t,x) for the drift and (t,x,fi_til) for the diffusion.
    # r is the function that computes the gradient of the log of the kernel of the auxiliar process
    # and it takes (t,x,T,x_pr,r_pars) as arguments.
    # H is the function that computes the Hessian of the log of the kernel of the auxiliar process
    # and it takes (t,x,T,x_pr,H_pars) as arguments.
    # crossed is the boolean that indicates if we need the computations for the crossed terms that 
    # are needed for the smoother. 

    # OUTCOMES: int_G is the integral of the gradient of the log of the kernel of the auxiliar process
    # and depending on crossed it can be a rank 1 or 2 array with dimensions (N) or (N,N).
    # x_test is either a rank 1 or 2 array with dimensions (N) or (N,N) that represents the value of the process
    # at time t_test.
    # x is either a rank 2 or 3 array with dimensions (steps+1,N) or (steps+1,N,N)
    
    steps=int((2**(l))*d)
    dt=1./2**l
        
    if crossed==True:
        int_G=np.zeros((N,N))
        x0=x0[:,np.newaxis]+np.zeros((N,N))
    else:
        int_G=np.zeros(N)
        x0=x0+np.zeros(N)
        
    #print("int_G shape is: ",int_G.shape)
    #I=identity(dim).toarray()
        
    x_prev=x0

    for i in np.array(range(steps-1)):

        t=t0+(i)*dt
        r_n=r(t,x_prev,T,x_p,r_pars,crossed=crossed) # rank 2 o 3 with dims (N,dim) or (N,N,dim)
        # respectively, depending on the boolean crossed.
        # 1) what do I need from a_tilde? 2) How can I get it in terms of Sig_tilde?
        # 3) what are the specifications of Sig_tilde? 3) answer: the same as Sig.
        # 1) answer:
        Sig_tilde=Sig_til(t,x_prev,fi_til)
        Sig_reali=Sig(x_prev,fi)
        a_tilde=Sig_tilde**2
        a=Sig_reali**2
        
        # for dim=1
        drift=b(x_prev,A)+(a*r_n)

        # more generally
        #drift=b(x[i],A)+np.einsum('nij,nj->ni',a,r_n)
        #print("r_n's shape is ",(((a-a_tilde)*(-H_normal(t,x[i],T,x_pr,H_pars,crossed=crossed)[:,:,np.newaxis]\
        #-(r_n**2)[:,:,np.newaxis]))[:,:,0]).shape)
        x=x_prev+drift*dt+Sig_reali*np.sqrt(dt)*np.random.normal(0,1,N)

        int_G=int_G+dt*((b(x_prev,A)-b_til(t,x_prev,A_til))*r_n\
        -(1/2)*((a-a_tilde)*(-H(t,x_prev,T,x_pr,H_pars)\
        -(r_n**2))))

        x_prev=x
        #print("int_G first term is ", int_G[0])
        #print("term's shape is ", (-H(t,x[i],T,x_pr,H_pars)).shape)
        #print("int_G shape is: ",int_G.shape)

        """
        x[i+1]=x[i]+drift*dt+Sig(x[i],fi)[:,0]*np.sqrt(dt)*np.random.normal(0,1,(N,1))
        int_G=int_G+dt*((b(x[i],A)-b_til(t,x[i],A_til))*r_n\
        -(1/2)*((a-a_tilde)*(-H(t,x[i],T,x_pr,H_pars,crossed=crossed)[:,:,np.newaxis]\
        -(r_n**2)[:,:,np.newaxis]))[:,:,0])[:,0]
        """
        

        
        

    return int_G

"""
x=np.array([1.2,3])
print(g_normal_1d(x,sd))
"""

#aux_trans_den=norm(loc=x0*np.exp(mu*(T-t0)),scale=gbm_aux_sd_1d(t0,x0,T,x_pr,[sigma,mu,t0,x0]))
#dPdP=np.exp(int_G)*aux_trans_den.pdf(x_pr)/prop_trans_den
#%%

# IN THE FOLLOWING LINES I WILL DEFINE A FUNCTION FOR THE MAXIMUM COUPLING, THIS FUNCTION WILL IS SPECIFIC 
# FOR NORMAL DISTRIBUTIONS.




def multi_samp_exp(W,N,x,dim): 
    # from multinomial sampling order exponential
    # Given probability weights (normalized)
    # it uses multinomial resampling, and constructs the set of resampled
    # particles. 

    # ARGUMENTS: W: Normalized weight
    # N: number of samples to get 
    # x: rank 1 array of the positions of the N particles
    # OUTPUTs: part: is a N dimentional array where its value in the ith position
    # represents the number of times that particle was sampled.
    # x_new: is the new set of resampled particles.
    #print(len(W),W,np.sum(W))

    W_cum=np.cumsum(W)
    exps=np.random.exponential(1,N+1)
    exps_cum=np.cumsum(exps)
    os=exps_cum/exps_cum[-1]
    j=0
    part_samp=np.zeros(N,dtype=int)
    for i in range(N):
        w=W_cum[i]
        while w>os[j] and j<N:
            #print("w is",w)
            #print("os is",os[j])
            part_samp[j]=i
            #new update
            j+=1
    x_resamp=x[part_samp]
    return [part_samp,x_resamp]
"""
W=np.array([0.1,0.2,0.3,0.4])
W=W/np.sum(W)
x=np.array([1,2,3,4])
N=1
dim=1
samples=10000
sample_store=np.zeros((samples,N))
for sample in range(samples):
    sample_store[sample],x_resamp=multi_samp_exp(W,N,x,dim)
sample_store=sample_store.flatten()
plt.hist(sample_store)
"""

#%%



#%%


def multi_samp_coup_exp(W,N,x0,x1,dim): 

    # This functions samples from W (using multi_samp_exp) and evaluates the sample on x0 and x1
    # ARGUMENTS: W: Normalized weights with dimension N (number of particles)
    # x: rank 2 array of the positions of the N particles, its dimesion is
    # N.
    # OUTPUTs: part: is a N dimentional array where its value in the ith position
    # represents the number of times that particle was sampled.
    # x_new: is the new set of resampled particles.
    
    W_cum=np.cumsum(W)
    exps=np.random.exponential(1,N+1)
    exps_cum=np.cumsum(exps)
    os=exps_cum/exps_cum[-1]
    j=0
    part_samp=np.zeros(N,dtype=int)
    
    for i in range(N):
        w=W_cum[i]
        while w>os[j] and j<N:
            #print("w is",w)
            #print("os is",os[j])
            part_samp[j]=i
            #new update
            j+=1
    #print(part_samp)
    x0_new=x0[part_samp]
    x1_new=x1[part_samp]
    #return [part, x0_new,x1_new]
    return [part_samp,x0_new,x1_new] #here we add 1 bc it is par_lab are thought 
    # as python labels, meaning that they start with 0.

def multi_samp_coup(W,N,x0,x1,dim): #from multinomial sampling
    # This function does 2 things, given probability weights (normalized)
    # it uses multinomial resampling, and constructs the set of resampled 
    # particles.
    # ARGUMENTS: W: Normalized weights with dimension N (number of particles)
    # x: rank 2 array of the positions of the N particles, its dimesion is
    # N.
    # OUTPUTs: part: is a N dimentional array where its value in the ith position
    # represents the number of times that particle was sampled.
    # x_new: is the new set of resampled particles.
    
    part_samp=np.random.choice(len(W),size=N,p=W) #particles resampled 
    #print(part_samp)
    x0_new=x0[part_samp]
    x1_new=x1[part_samp]
    #return [part, x0_new,x1_new]
    return [part_samp,x0_new,x1_new] #here we add 1 bc it is par_lab are thought 
    # as python labels, meaning that they start with 0.
    


def multi_samp(W,N,x,dim): #from multinomial sampling
    # This function does 2 things, given probability weights (normalized)
    # it uses multinomial resampling, and constructs the set of resampled 
    # particles.
    # ARGUMENTS: W: Normalized weight
    # N: number of samples to get 
    # x: rank 1 array of the positions of the N particles
    # OUTPUTs: part: is a N dimentional array where its value in the ith position
    # represents the number of times that particle was sampled.
    # x_new: is the new set of resampled particles.
    #print(len(W),W,np.sum(W))
    part_samp=np.random.choice(len(W),size=N,p=W,replace=True) #particles resampled 
    #print(part_samp)
    x_resamp=x[part_samp]
    return [part_samp,x_resamp] #here we add 1 bc it is par_lab are thought 

def max_coup_multi(w0,w1,N,x0,x1,dim):
    
    # This function does 2 things, given probability weights (normalized)
    # it uses systematic resampling, and  constructs the set of resampled 
    # particles.
    # ARGUMENTS: w0: Normalized weights with dimension N (number of particles)
    # w1: Normalized weights with dimension N (number of particles)
    # x1: rank 2 array of the positions of the N particles, its dimesion is
    # N, where dim is the dimension of the problem.
    # OUTPUTs: part: is a N dimentional array where its value in the ith position
    # represents the number of times that particle was sampled.
    # x_new: is the new set of resampled particles.
    wmin=np.minimum(w0, w1)
    r=np.min(np.array([np.sum(wmin),1.0]))
    wmin_den=wmin/r
    #np.random.seed(seed_val)
    bino=np.random.binomial(N,r)
    part=np.zeros(N,dtype=int)
    part0=np.zeros(N,dtype=int)
    part1=np.zeros(N,dtype=int)
    x0_new=np.zeros(N)
    x1_new=np.zeros(N)
    

    if bino>0:
        [part_prox,x0_new_prox,x1_new_prox]=multi_samp_coup_exp(wmin_den,N,x0,x1,dim)
        part[:bino]=np.random.choice(part_prox,size=bino,replace=True)
        part0[:bino]=part[:bino]
        part1[:bino]=part[:bino]
        x0_new[:bino]=x0[part[:bino]]
        x1_new[:bino]=x1[part[:bino]]
        
    #print(w0,w1,wmin,"dists")
    #w4_den=(w0-wmin)/(1-r)
    #print(w4_den)
    if bino<N:
        w4_den=(w0-wmin)/np.sum((w0-wmin))
        [part0_prox,x0_new_prox]=multi_samp_exp(w4_den,N,x0,dim)
        part0[bino:]= np.random.choice(part0_prox,size=N-bino,replace=True)
        x0_new[bino:]=x0[part0[bino:]]
        w5_den=(w1-wmin)/np.sum((w1-wmin))
        #w5_den=(w1-wmin)/(1-r)
        [part1_prox,x1_new_prox]=multi_samp_exp(w5_den,N,x1,dim)
        part1[bino:]= np.random.choice(part1_prox,size=N-bino,replace=True)
        x1_new[bino:]=x1[part1[bino:]]

    return [part0,part1,x0_new,x1_new]

#%%

#Testing the function max_coup_multi
"""
w0=np.array([.1,.2,.3,.4])
w1=np.array([.4,.3,.2,.1])
#w1=w0
B=10000
N=4
samples=np.zeros((2,B,N))

for i in range(B):

    part0,part1,x0_new,x1_new=max_coup_multi(w0,w1,N,np.array([1,2,3,4]),np.array([1,2,3,4]),1)
    samples[0,i]=x0_new
    samples[1,i]=x1_new

samples=samples.reshape((2,B*N))
x=samples[0]
y=samples[1]

# Create a 2D histogram
plt.hist2d(x, y, bins=4, cmap='Blues',density=True)  # Adjust 'bins' for resolution

# Add a color bar to indicate the count density
plt.colorbar(label='Counts in bin')

# Add labels and a title to the plot
plt.xlabel('X axis label')
plt.ylabel('Y axis label')
plt.title('2D Histogram')

# Display the plot
plt.show()



from matplotlib.gridspec import GridSpec

# Example data: create a (B, 2) array with random data
#B = 1000
#data = np.random.randn(B, 2)  # Normally distributed data for demonstration

# Extract x and y data points from the (B, 2) array
#x = data[:, 0]  # The first column for x-values
#y = data[:, 1]  # The second column for y-values

# Set up the figure with GridSpec to arrange the plots
fig = plt.figure(figsize=(8, 8))
gs = GridSpec(4, 4, fig)
# Create the main 2D histogram plot in the center
ax_main = fig.add_subplot(gs[1:4, 0:3])
h2d = ax_main.hist2d(x, y, bins=30, cmap='Blues',density=True)
ax_main.set_xlabel('X axis label')
ax_main.set_ylabel('Y axis label')

# Add the color bar for the 2D histogram
plt.colorbar(h2d[3], ax=ax_main, orientation='vertical', label='Counts in bin')

# Create the marginal histogram for the x-axis at the top
ax_x_marginal = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
ax_x_marginal.hist(x, bins=30, color='gray')
ax_x_marginal.axis('off')  # Hide the axis for a cleaner look

# Create the marginal histogram for the y-axis on the right
ax_y_marginal = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
ax_y_marginal.hist(y, bins=30, color='gray', orientation='horizontal')
ax_y_marginal.axis('off')  # Hide the axis for a cleaner look

# Display the plot
plt.tight_layout()
plt.show()
"""

#%%
def max_coup_multi_sing_samp(w0,w1,x0,x1,dim):
    
    # This function does 2 things, given probability weights (normalized)
    # it uses systematic resampling, and  constructs the set of resampled 
    # particles.
    # ARGUMENTS: w0: Normalized weights with dimension N (number of particles)
    # w1: Normalized weights with dimension N (number of particles)
    # x1: rank 2 array of the positions of the N particles, its dimesion is
    # N, where dim is the dimension of the problem.
    # OUTPUTs: part: is a N dimentional array where its value in the ith position
    # represents the number of times that particle was sampled.
    # x_new: is the new set of resampled particles.
    
    
    
    wmin=np.minimum((w0),(w1))
    #lg_wmin=np.minimum(np.log(w0),np.log(w1))
    
    r=np.min(np.array([np.sum(wmin),1.]))
    #wmin_den=pff.norm_logweights(lg_wmin)
    wmin_den=wmin/r
    N=1
    #np.random.seed(seed_val)
    bino=np.random.binomial(N,r)

    part=np.zeros(N,dtype=int)
    part0=np.zeros(N,dtype=int)
    part1=np.zeros(N,dtype=int)
    x0_new=np.zeros(N)
    x1_new=np.zeros(N)
    if bino==1:
        [part[:bino],x0_new[:bino],x1_new[:bino]]=\
        multi_samp_coup(wmin_den,bino,x0,x1,dim)
        part0[:bino]=part[:bino]
        part1[:bino]=part[:bino]
    else:
        w4_den=(w0-wmin)/(1-r)
        [part0[bino:],x0_new[bino:]]=multi_samp(w4_den,N-bino,x0,dim)
        w5_den=(w1-wmin)/(1-r)
        [part1[bino:],x1_new[bino:]]=multi_samp(w5_den,N-bino,x1,dim)
    return [part0,part1,x0_new,x1_new]
#%%
"""
w0=np.array([.1,.2,.3,.4])
w1=np.array([.4,.3,.2,.1])
#w1=w0
B=10000
N=1
samples=np.zeros((2,B))

for i in range(B):

    part0,part1,x0_new,x1_new=max_coup_multi_sing_samp(w0,w1,np.array([1,2,3,4]),np.array([1,2,3,4]),1)
    samples[0,i]=x0_new
    samples[1,i]=x1_new

samples=samples.reshape((2,B*N))
x=samples[0]
y=samples[1]

# Create a 2D histogram
plt.hist2d(x, y, bins=4, cmap='Blues',density=True)  # Adjust 'bins' for resolution

# Add a color bar to indicate the count density
plt.colorbar(label='Counts in bin')

# Add labels and a title to the plot
plt.xlabel('X axis label')
plt.ylabel('Y axis label')
plt.title('2D Histogram')

# Display the plot
plt.show()


from matplotlib.gridspec import GridSpec

# Example data: create a (B, 2) array with random data
#B = 1000
#data = np.random.randn(B, 2)  # Normally distributed data for demonstration

# Extract x and y data points from the (B, 2) array
#x = data[:, 0]  # The first column for x-values
#y = data[:, 1]  # The second column for y-values

# Set up the figure with GridSpec to arrange the plots
fig = plt.figure(figsize=(8, 8))
gs = GridSpec(4, 4, fig)
# Create the main 2D histogram plot in the center
ax_main = fig.add_subplot(gs[1:4, 0:3])
h2d = ax_main.hist2d(x, y, bins=30, cmap='Blues',density=True)
ax_main.set_xlabel('X axis label')
ax_main.set_ylabel('Y axis label')

# Add the color bar for the 2D histogram
plt.colorbar(h2d[3], ax=ax_main, orientation='vertical', label='Counts in bin')

# Create the marginal histogram for the x-axis at the top
ax_x_marginal = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
ax_x_marginal.hist(x, bins=30, color='gray')
ax_x_marginal.axis('off')  # Hide the axis for a cleaner look

# Create the marginal histogram for the y-axis on the right
ax_y_marginal = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
ax_y_marginal.hist(y, bins=30, color='gray', orientation='horizontal')
ax_y_marginal.axis('off')  # Hide the axis for a cleaner look

# Display the plot
plt.tight_layout()
plt.show()
"""
#%%
 
# what do kind of normal distriubtion we need for the OU process? the one I have is "sampling_ou", it is the 
# distribution of the OU process, which the auxiliar parameters, in other words it is the auxiliar distributions.
# what kind of convention are we going to have regasrding the levels? we will use the number 0 and 1, 0 corresponds 
# to the level l-1 and 1 to the level l.
# if we want this class of function general, what parameters do we have to include? 

def max_coup_normal_reflec(x0,x1,N,d,sample_pars):
    # This funciton is specific for the reflexion method (Synchronous Pairwise Reflection Maximal Couplings), 
    # which constrains sigma0=sigma1. This method can be found inhttps://arxiv.org/pdf/1708.03625 alg 2. 
    # Following the convention of sample_funct we choose the above parameters.
    # ARGUMENTS:
    # x0: rank 0 array with dimensions (dim)
    # x1: rank 0 array with dimensions (dim)
    # sample pars is going to have the structure [theta0,sigma0,theta1,sigma1]
    # where each one of those parameters are scalars.
    # N: number of samples
    # d: temporal interval where we compute one iteration of the PF
    
    # OUTPUT: 
    # 2 rank 1 arrays with dimensions (N) with the samples of the proposal transitions.
    [theta0,sigma0,theta1,sigma1]=sample_pars
    #(t,x,T,x_pr,pars)
    x_pr0=np.random.normal(x0*np.exp(theta0*d),ou_sd(0,x0,d,x0,[theta0,sigma0]))

    u=(x0*np.exp(theta0*d)-x1*np.exp(theta1*d))

    # since we also need the ou_sd to be different (since theta0 and theta1 are different) 
    # we stop building this fucntion, we proceed to define the function that works with rejection sampling
    # https://arxiv.org/pdf/1708.03625


def rej_max_coup_ou(x0,x1,N,d,sample_pars):

    # This function is made in order incorporate the maximum coupling.
    # The algorithm is a rejection sampler  https://arxiv.org/pdf/1708.03625 alg 2.
    # The fucntion is specific for ou proposal densities.
    # The nature of the sampler is random, applying universal function to this samplers 
    # might difficult and that's why we include the for loop. 

    # ARGUMENTS:
    # x0: rank 0 array with dimensions (dim)
    # x1: rank 0 array with dimensions (dim)
    # sample pars is going to have the structure [theta0,sigma0,theta1,sigma1]
    # where each one of those parameters are scalars.
    # N: number of samples
    # d: temporal interval where we compute one iteration of the PF

    # OUTPUT: 
    # 2 rank 1 arrays with dimensions (N) with the samples of the proposal transitions.
    [theta0,sigma0,theta1,sigma1]=sample_pars
    x_pr0=np.random.normal(x0*np.exp(theta0*d),ou_sd(0,x0,d,x0,[theta0,sigma0]))
    rv0=norm(loc=x0*np.exp(theta0*d),scale=ou_sd(0,x0,d,x0,[theta0,sigma0]))
    px=rv0.pdf(x_pr0)
    rv1=norm(loc=x1*np.exp(theta1*d),scale=ou_sd(0,x1,d,x1,[theta1,sigma1]))
    w=np.random.uniform(0,px)
    x_pr1=x_pr0.copy()
    qx=rv1.pdf(x_pr0)
    for i in range(N):
        if w[i]>qx[i]:
            #print(x_pr1[i])
            x_pr1[i]=np.random.normal(x1[i]*np.exp(theta1*d),ou_sd(0,x1[i],d,x1[i],[theta1,sigma1]))
            rv1i=norm(loc=x1[i]*np.exp(theta1*d),scale=ou_sd(0,x1[i],d,x1[i],[theta1,sigma1]))
            rv0i=norm(loc=x0[i]*np.exp(theta0*d),scale=ou_sd(0,x0[i],d,x0[i],[theta0,sigma0]))
            qy=rv1i.pdf(x_pr1[i])
            w_s=np.random.uniform(0,qy)
            #print(w_s)
            py=rv0i.pdf(x_pr1[i])
            while w_s <= py:
                x_pr1[i]=np.random.normal(x1[i]*np.exp(theta1*d),ou_sd(0,x1[i],d,x1[i],[theta1,sigma1]))
                qy=rv1i.pdf(x_pr1[i])
                w_s=np.random.uniform(0,qy)
                py=rv0i.pdf(x_pr1[i])
    return x_pr0,x_pr1


def rej_max_coup_CIR(x0,x1,N,d,sample_pars):

    # This function is made in order incorporate the maximum coupling.
    # The algorithm is a rejection sampler  https://arxiv.org/pdf/1708.03625 alg 2.
    # The fucntion is specific for ou proposal densities.
    # The nature of the sampler is random, applying universal function to this samplers 
    # might difficult and that's why we include the for loop. 

    # ARGUMENTS:
    # x0: rank 0 array with dimensions (dim)
    # x1: rank 0 array with dimensions (dim)
    # sample pars is going to have the structure [theta0,sigma0,theta1,sigma1]
    # where each one of those parameters are scalars.
    # N: number of samples
    # d: temporal interval where we compute one iteration of the PF

    # OUTPUT: 
    # 2 rank 1 arrays with dimensions (N) with the samples of the proposal transitions.



    """
    
    the1,the2,the3=sample_pars

    c=the3**2*(1-np.exp(-the2*d))/(4*the2)
    df=4*the1/the3**2
    lamb=4*the2*np.exp(-the2*d)*x0/(the3**2*(1-np.exp(-the2*d)))
    chi=np.random.noncentral_chisquare(df,lamb,N)
    return c*chi
    """

    [the1_0,the2_0,the3_0,the1_1,the2_1,the3_1]=sample_pars
    #[theta0,sigma0,theta1,sigma1]=sample_pars

    x_pr0=sampling_CIR(x0,N,d,[the1_0,the2_0,the3_0])
    #x_pr0=np.random.normal(x0*np.exp(theta0*d),ou_sd(0,x0,d,x0,[theta0,sigma0]))
    rv0=trans_noncentral_chisquare(0,x0,d,x_pr0,[the1_0,the2_0,the3_0])
    #rv0=norm(loc=x0*np.exp(theta0*d),scale=ou_sd(0,x0,d,x0,[theta0,sigma0]))
    px=rv0 
    # px=rv0.pdf(x_pr0)
    rv1=trans_noncentral_chisquare(0,x1,d,x_pr0,[the1_1,the2_1,the3_1])
    w=np.random.uniform(0,px)
    x_pr1=x_pr0.copy()
    qx=rv1
    for i in range(N):
        if w[i]>qx[i]:
            
            #x_pr1[i]=np.random.normal(x1[i]*np.exp(theta1*d),ou_sd(0,x1[i],d,x1[i],[theta1,sigma1]))
            x_pr1[i]=sampling_CIR(x1[i],1,d,[the1_1,the2_1,the3_1])[0]
            #rv1i=norm(loc=x1[i]*np.exp(theta1*d),scale=ou_sd(0,x1[i],d,x1[i],[theta1,sigma1]))
            rv1i=trans_noncentral_chisquare(0,x1[i],d,x_pr1[i],[the1_1,the2_1,the3_1])
            #rv0i=norm(loc=x0[i]*np.exp(theta0*d),scale=ou_sd(0,x0[i],d,x0[i],[theta0,sigma0]))
            rv0i=trans_noncentral_chisquare(0,x0[i],d,x_pr1[i],[the1_0,the2_0,the3_0])

            qy=rv1i
            w_s=np.random.uniform(0,qy)
            #print(w_s)
            py=rv0i
            while w_s <= py:
                #x_pr1[i]=np.random.normal(x1[i]*np.exp(theta1*d),ou_sd(0,x1[i],d,x1[i],[theta1,sigma1]))
                x_pr1[i]=sampling_CIR(x1[i],1,d,[the1_1,the2_1,the3_1])[0]
                qy=trans_noncentral_chisquare(0,x1[i],d,x_pr1[i],[the1_1,the2_1,the3_1])

                w_s=np.random.uniform(0,qy)
                
                py=trans_noncentral_chisquare(0,x0[i],d,x_pr1[i],[the1_0,the2_0,the3_0])
    return x_pr0,x_pr1

def rej_max_coup_log_normal(x0,x1,N,d,sample_pars):

    [the1_0,the2_0,the3_0,the1_1,the2_1,the3_1]=sample_pars
    #[theta0,sigma0,theta1,sigma1]=sample_pars

    x_pr0=sampling_prop_log_normal(x0,N,d,[the1_0,the2_0,the3_0])
    rv0=trans_prop_log_normal(0,x0,d,x_pr0,[the1_0,the2_0,the3_0])
    px=rv0
    # px=rv0.pdf(x_pr0)
    rv1=trans_prop_log_normal(0,x1,d,x_pr0,[the1_1,the2_1,the3_1])
    w=np.random.uniform(0,px)
    x_pr1=x_pr0.copy()
    qx=rv1
    for i in range(N):
        if w[i]>qx[i]:
            
            #x_pr1[i]=np.random.normal(x1[i]*np.exp(theta1*d),ou_sd(0,x1[i],d,x1[i],[theta1,sigma1]))
            x_pr1[i]=sampling_prop_log_normal(x1[i],1,d,[the1_1,the2_1,the3_1])[0]
            #rv1i=norm(loc=x1[i]*np.exp(theta1*d),scale=ou_sd(0,x1[i],d,x1[i],[theta1,sigma1]))
            rv1i=trans_prop_log_normal(0,x1[i],d,x_pr1[i],[the1_1,the2_1,the3_1])
            #rv0i=norm(loc=x0[i]*np.exp(theta0*d),scale=ou_sd(0,x0[i],d,x0[i],[theta0,sigma0]))
            rv0i=trans_prop_log_normal(0,x0[i],d,x_pr1[i],[the1_0,the2_0,the3_0])
            qy=rv1i
            w_s=np.random.uniform(0,qy)
            #print(w_s)
            py=rv0i
            while w_s <= py:
                #x_pr1[i]=np.random.normal(x1[i]*np.exp(theta1*d),ou_sd(0,x1[i],d,x1[i],[theta1,sigma1]))
                x_pr1[i]=sampling_prop_log_normal(x1[i],1,d,[the1_1,the2_1,the3_1])[0]
                qy=trans_prop_log_normal(0,x1[i],d,x_pr1[i],[the1_1,the2_1,the3_1])
                w_s=np.random.uniform(0,qy)
                
                py=trans_prop_log_normal(0,x0[i],d,x_pr1[i],[the1_0,the2_0,the3_0])
    return x_pr0,x_pr1


def rej_max_coup_log_normal_2(x0,x1,N,d,sample_pars):

    [the1_0,the2_0,the3_0,the1_1,the2_1,the3_1]=sample_pars
    #[theta0,sigma0,theta1,sigma1]=sample_pars

    x_pr0=sampling_prop_log_normal_2(x0,N,d,[the1_0,the2_0,the3_0])
    rv0=trans_prop_log_normal_2(0,x0,d,x_pr0,[the1_0,the2_0,the3_0])
    px=rv0
    # px=rv0.pdf(x_pr0)
    rv1=trans_prop_log_normal_2(0,x1,d,x_pr0,[the1_1,the2_1,the3_1])
    w=np.random.uniform(0,px)
    x_pr1=x_pr0.copy()
    qx=rv1
    for i in range(N):
        if w[i]>qx[i]:
            
            #x_pr1[i]=np.random.normal(x1[i]*np.exp(theta1*d),ou_sd(0,x1[i],d,x1[i],[theta1,sigma1]))
            x_pr1[i]=sampling_prop_log_normal_2(x1[i],1,d,[the1_1,the2_1,the3_1])[0]
            #rv1i=norm(loc=x1[i]*np.exp(theta1*d),scale=ou_sd(0,x1[i],d,x1[i],[theta1,sigma1]))
            rv1i=trans_prop_log_normal_2(0,x1[i],d,x_pr1[i],[the1_1,the2_1,the3_1])
            #rv0i=norm(loc=x0[i]*np.exp(theta0*d),scale=ou_sd(0,x0[i],d,x0[i],[theta0,sigma0]))
            rv0i=trans_prop_log_normal_2(0,x0[i],d,x_pr1[i],[the1_0,the2_0,the3_0])
            qy=rv1i
            w_s=np.random.uniform(0,qy)
            #print(w_s)
            py=rv0i
            while w_s <= py:
                #x_pr1[i]=np.random.normal(x1[i]*np.exp(theta1*d),ou_sd(0,x1[i],d,x1[i],[theta1,sigma1]))
                x_pr1[i]=sampling_prop_log_normal_2(x1[i],1,d,[the1_1,the2_1,the3_1])[0]
                qy=trans_prop_log_normal_2(0,x1[i],d,x_pr1[i],[the1_1,the2_1,the3_1])
                w_s=np.random.uniform(0,qy)
                
                py=trans_prop_log_normal_2(0,x0[i],d,x_pr1[i],[the1_0,the2_0,the3_0])
    return x_pr0,x_pr1


# TEst for rej_max_coup_log_normal

#import numpy as np
#import matplotlib.pyplot as plt
#
#def joint_plot_with_marginals(N, pars, bins=30):
#    #"""
#    #Creates a single figure:
#    #  - Bottom-left area: scatter of (x, y) from f(N, pars)
#    #  - Top edge: histogram of x + overlay of g1(x)
#    #  - Right edge: histogram of y + overlay of g2(y)
#    #Parameters
#    #----------
#    #N : int
#    #    Number of samples
#    #pars : dict or similar
#    #    Parameters passed to sampling function f
#    #f : function
#    #    f(N, pars) -> (x, y), each of shape (N,).
#    #g1 : function
#    #    g1(x_grid) -> PDF for x, same shape as x_grid.
#    #g2 : function
#    #    g2(y_grid) -> PDF for y, same shape as y_grid.
#    #bins : int
#    #    Number of histogram bins (both x and y).
#    #"""
#
#    # 1) Generate data from the 2D distribution
#    x, y = rej_max_coup_log_normal(x0,x1,N,d,pars) # each should be shape (N,)
#
#    # Create the figure and define a GridSpec layout
#    fig = plt.figure(figsize=(8, 8))
#    gs = fig.add_gridspec(4, 4)
#
#    # Main scatter axis (bottom-left 3x3 section)
#    ax_scatter = fig.add_subplot(gs[1:4, 0:3])
#    
#    # Top histogram axis (top row, aligned with scatter's x-axis)
#    ax_histx = fig.add_subplot(gs[0, 0:3], sharex=ax_scatter)
#    
#    # Right histogram axis (right column, aligned with scatter's y-axis)
#    ax_histy = fig.add_subplot(gs[1:4, 3], sharey=ax_scatter)
#
#    # ---- Main scatter plot (joint distribution) ----
#    ax_scatter.scatter(x, y, alpha=0.5)
#    ax_scatter.set_xlabel("X")
#    ax_scatter.set_ylabel("Y")
#    ax_scatter.set_title("Joint distribution")
#
#    # Hide tick labels on the top axis (we only want them on the main axis)
#    ax_histx.tick_params(axis="x", labelbottom=False)
#    
#    # Hide tick labels on the right axis (we only want them on the main axis)
#    ax_histy.tick_params(axis="y", labelleft=False)
#
#    # ---- X histogram (top) ----
#    ax_histx.hist(x, bins=bins, density=True, alpha=0.6)
#    # Overlay the marginal PDF g1(x) on the same axis
#    x_grid = np.linspace(x.min(), x.max(), 200)
#    ax_histx.plot(x_grid, trans_prop_log_normal(0,x0[1],d,x_grid,pars[:3]), lw=2)
#    
#    # ---- Y histogram (right) ----
#    # We'll make it horizontal so the shape matches the Y distribution
#    ax_histy.hist(y, bins=bins, density=True, alpha=0.6, orientation='horizontal')
#    # Overlay the marginal PDF g2(y), but we "flip" it: plot PDF on the x-axis, y on the y-axis
#    y_grid = np.linspace(y.min(), y.max(), 200)
#    ax_histy.plot(trans_prop_log_normal(0,x1[1],d,y_grid,pars[3:]), y_grid, lw=2)
#
#    # Adjust layout so subplots fit nicely
#    fig.tight_layout()
#    plt.show()
#
#
## -------------------------------------------------------------------
## Example usage with dummy functions (replace these with your own!)
## -------------------------------------------------------------------
#
#def f_example(N, pars):
#    #"""
#    #Returns N samples (x, y) from some 2D distribution.
#    #Here we just sample from a correlated bivariate normal for demonstration.
#    #"""
#    mean = [0, 0]
#    cov = [[1, 0.8], [0.8, 1]]  # Correlation 0.8
#    xy = np.random.multivariate_normal(mean, cov, size=N)
#    x, y = xy[:, 0], xy[:, 1]
#    return x, y
#
#def g1_example(x):
#    #"""
#    #Marginal PDF for X (example: standard normal).
#    #"""
#    return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2)
#
#def g2_example(y):
#    #"""
#    #Marginal PDF for Y (example: standard normal).
#    #"""
#    return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * y**2)
#
#
#if __name__ == "__main__":
#    N = 50000
#    the1_0,the2_0,the3_0,the1_1,the2_1,the3_1=2.1,1.2,0.5,2,1,0.5
#    pars =   [the1_0,the2_0,the3_0,the1_1,the2_1,the3_1]
#    x0=10+np.zeros(N)
#
#    x1=15+np.zeros(N)
#    d=1
#    joint_plot_with_marginals(N, pars, bins=500)

def rej_max_coup_gamma_in_dist(sample_pars,N):


    [the1_0,the2_0,the3_0,the1_1,the2_1,the3_1]=sample_pars

    w_0=the3_0**2/2+the1_0
    xi_0=the2_0
    sigma_0=the3_0

    alpha_0=2*w_0/sigma_0**2-1
    theta_0=sigma_0**2/(2*xi_0)

    w_1=the3_1**2/2+the1_1
    xi_1=the2_1
    sigma_1=the3_1

    alpha_1=2*w_1/sigma_1**2-1
    theta_1=sigma_1**2/(2*xi_1)

    
    #[theta0,sigma0,theta1,sigma1]=sample_pars

    x_pr0=gamma_sampling([the1_0,the2_0,the3_0],N)
    rv0=gamma(a=alpha_0,scale=theta_0)
    #rv0=trans_prop_log_normal(0,x0,d,x_pr0,[the1_0,the2_0,the3_0])
    px=rv0.pdf(x_pr0)
    # px=rv0.pdf(x_pr0)
    rv1=gamma(a=alpha_1,scale=theta_1)
    w=np.random.uniform(0,px)
    x_pr1=x_pr0.copy()
    qx=rv1.pdf(x_pr0)
    for i in range(N):
        if w[i]>qx[i]:
            
            #x_pr1[i]=np.random.normal(x1[i]*np.exp(theta1*d),ou_sd(0,x1[i],d,x1[i],[theta1,sigma1]))
            x_pr1[i]=gamma_sampling([the1_1,the2_1,the3_1],1)
            #rv1i=norm(loc=x1[i]*np.exp(theta1*d),scale=ou_sd(0,x1[i],d,x1[i],[theta1,sigma1]))
            rv1i=rv1
            #rv1i=trans_prop_log_normal(0,x1[i],d,x_pr1[i],[the1_1,the2_1,the3_1])
            #rv0i=norm(loc=x0[i]*np.exp(theta0*d),scale=ou_sd(0,x0[i],d,x0[i],[theta0,sigma0]))
            #rv0i=trans_prop_log_normal(0,x0[i],d,x_pr1[i],[the1_0,the2_0,the3_0])
            rv0i=rv0
            qy=rv1i.pdf(x_pr1[i])
            w_s=np.random.uniform(0,qy)
            #print(w_s)
            py=rv0i.pdf(x_pr1[i])
            while w_s <= py:
                #x_pr1[i]=np.random.normal(x1[i]*np.exp(theta1*d),ou_sd(0,x1[i],d,x1[i],[theta1,sigma1]))
                x_pr1[i]=gamma_sampling([the1_1,the2_1,the3_1],1)
                qy=rv1i.pdf(x_pr1[i])
                w_s=np.random.uniform(0,qy)                
                py=rv0i.pdf(x_pr1[i])
    return x_pr0,x_pr1


# Test for rej_max_coup_gamma_in_dist

#import numpy as np
#import matplotlib.pyplot as plt
#
#def joint_plot_with_marginals(N, pars, bins=30):
#    #"""
#    #Creates a single figure:
#    #  - Bottom-left area: scatter of (x, y) from f(N, pars)
#    #  - Top edge: histogram of x + overlay of g1(x)
#    #  - Right edge: histogram of y + overlay of g2(y)
#    #Parameters
#    #----------
#    #N : int
#    #    Number of samples
#    #pars : dict or similar
#    #    Parameters passed to sampling function f
#    #f : function
#    #    f(N, pars) -> (x, y), each of shape (N,).
#    #g1 : function
#    #    g1(x_grid) -> PDF for x, same shape as x_grid.
#    #g2 : function
#    #    g2(y_grid) -> PDF for y, same shape as y_grid.
#    #bins : int
#    #    Number of histogram bins (both x and y).
#    #"""
#
#    # 1) Generate data from the 2D distribution
#    x, y = rej_max_coup_gamma_in_dist(pars,N) # each should be shape (N,)
#
#    # Create the figure and define a GridSpec layout
#    fig = plt.figure(figsize=(8, 8))
#    gs = fig.add_gridspec(4, 4)
#
#    # Main scatter axis (bottom-left 3x3 section)
#    ax_scatter = fig.add_subplot(gs[1:4, 0:3])
#    
#    # Top histogram axis (top row, aligned with scatter's x-axis)
#    ax_histx = fig.add_subplot(gs[0, 0:3], sharex=ax_scatter)
#    
#    # Right histogram axis (right column, aligned with scatter's y-axis)
#    ax_histy = fig.add_subplot(gs[1:4, 3], sharey=ax_scatter)
#
#    # ---- Main scatter plot (joint distribution) ----
#    ax_scatter.scatter(x, y, alpha=0.5)
#    ax_scatter.set_xlabel("X")
#    ax_scatter.set_ylabel("Y")
#    ax_scatter.set_title("Joint distribution")
#
#    # Hide tick labels on the top axis (we only want them on the main axis)
#    ax_histx.tick_params(axis="x", labelbottom=False)
#    
#    # Hide tick labels on the right axis (we only want them on the main axis)
#    ax_histy.tick_params(axis="y", labelleft=False)
#
#    # ---- X histogram (top) ----
#    ax_histx.hist(x, bins=bins, density=True, alpha=0.6)
#    # Overlay the marginal PDF g1(x) on the same axis
#    x_grid = np.linspace(x.min(), x.max(), 200)
#
#    ax_histx.plot(x_grid, gamma.pdf(x_grid,a=alpha_0,scale=theta_0), lw=2)
#    
#    # ---- Y histogram (right) ----
#    # We'll make it horizontal so the shape matches the Y distribution
#    ax_histy.hist(y, bins=bins, density=True, alpha=0.6, orientation='horizontal')
#    # Overlay the marginal PDF g2(y), but we "flip" it: plot PDF on the x-axis, y on the y-axis
#    y_grid = np.linspace(y.min(), y.max(), 200)
#    ax_histy.plot(gamma.pdf(y_grid,a=alpha_1,scale=theta_1), y_grid, lw=2)
#
#    # Adjust layout so subplots fit nicely
#    fig.tight_layout()
#    plt.show()
#
#
## -------------------------------------------------------------------
## Example usage with dummy functions (replace these with your own!)
## -------------------------------------------------------------------
#
#def f_example(N, pars):
#    #"""
#    #Returns N samples (x, y) from some 2D distribution.
#    #Here we just sample from a correlated bivariate normal for demonstration.
#    #"""
#    mean = [0, 0]
#    cov = [[1, 0.8], [0.8, 1]]  # Correlation 0.8
#    xy = np.random.multivariate_normal(mean, cov, size=N)
#    x, y = xy[:, 0], xy[:, 1]
#    return x, y
#
#def g1_example(x):
#    #"""
#    #Marginal PDF for X (example: standard normal).
#    #"""
#    return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2)
#
#def g2_example(y):
#    #"""
#    #Marginal PDF for Y (example: standard normal).
#    #"""
#    return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * y**2)
#
#
#if __name__ == "__main__":
#    N = 50000
#    the1_0,the2_0,the3_0,the1_1,the2_1,the3_1=2.1,1.2,0.5,2,1,0.5
#    pars =   [the1_0,the2_0,the3_0,the1_1,the2_1,the3_1]
#
#    w_0=the3_0**2/2+the1_0
#    xi_0=the2_0
#    sigma_0=the3_0
#    alpha_0=2*w_0/sigma_0**2-1
#    theta_0=sigma_0**2/(2*xi_0)
#
#    w_1=the3_1**2/2+the1_1
#    xi_1=the2_1
#    sigma_1=the3_1
#    alpha_1=2*w_1/sigma_1**2-1
#    theta_1=sigma_1**2/(2*xi_1)
#
#
#    d=1
#    joint_plot_with_marginals(N, pars, bins=300)




#%%


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_samples_and_marginals(f, pdf_x, pdf_y, N=1000, bins=30):
    """
    1) Generates N samples from f(N) -> (X, Y), where X and Y are arrays of shape (N,).
    2) Plots the (X, Y) scatter in the main panel.
    3) Plots histograms of X (top) and Y (right) and overlays the user-provided PDFs.
    
    Parameters
    ----------
    f : function
        A function f(N) that returns (X, Y) each of shape (N,).
        Each (X[i], Y[i]) is presumably correlated. 
    pdf_x : function
        A function pdf_x(x_array) -> array_like of same shape as x_array,
        giving the theoretical marginal density of X.
    pdf_y : function
        A function pdf_y(y_array) -> array_like of same shape as y_array,
        giving the theoretical marginal density of Y.
    N : int
        Number of samples to draw from f.
    bins : int
        Number of bins for the marginal histograms.
    """
    x0=np.zeros(N)+103
    x1=np.zeros(N)+103
    the1=2.1
    the2=1.2
    the3=0.5
    the1_y=2
    the2_y=1
    the3_y=0.5
    the4=1
    sample_pars=[the1,the2,the3,the1_y,the2_y,the3_y]
    w_0=the3**2/2+the1
    xi_0=the2
    sigma_0=the3

    alpha_0=2*w_0/sigma_0**2-1
    theta_0=sigma_0**2/(2*xi_0)

    w_1=the3_y**2/2+the1_y
    xi_1=the2_y
    sigma_1=the3_y

    alpha_1=2*w_1/sigma_1**2-1
    theta_1=sigma_1**2/(2*xi_1)




    d=2
    # 1) Generate samples
    X, Y = rej_max_coup_gamma_in_dist(sample_pars,N)  # X, Y each of shape (N,)

    # 2) Set up figure with GridSpec to place:
    #    - main scatter,
    #    - top histogram,
    #    - right histogram
    fig = plt.figure(figsize=(8,8))
    gs = GridSpec(4, 4, figure=fig)

    # Main scatter: from row=1..4, col=0..3
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    # Top hist: row=0, col=0..3, share x with main
    ax_xhist = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    # Right hist: row=1..4, col=3, share y with main
    ax_yhist = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

    # 3) Main scatter plot
    ax_main.scatter(X, Y, alpha=0.5, s=20)
    ax_main.set_xlabel("X")
    ax_main.set_ylabel("Y")

    # 4) Histogram of X (top)
    ax_xhist.hist(X, bins=bins, density=True, alpha=0.5, color="C0")
    # Evaluate pdf_x on a grid covering X range
    x_grid = np.linspace(X.min(), X.max(), 300)
    r_gamma_0=gamma(a=alpha_0,scale=theta_0)
    #px =trans_prop_log_normal(0,x0[0],d,x_grid,[the1,the2,the3])  #pdf_x(x_grid)  # user function
    px=r_gamma_0.pdf(x_grid)
    # Overlay the PDF on top histogram
    ax_xhist.plot(x_grid, px, 'r-', lw=2, label="pdf_x")
    ax_xhist.legend()
    # Hide X ticklabels on top hist to reduce clutter
    plt.setp(ax_xhist.get_xticklabels(), visible=False)
    ax_xhist.set_ylabel("Density (X)")

    # 5) Histogram of Y (right), horizontally oriented
    ax_yhist.hist(Y, bins=bins, density=True, alpha=0.5, color="C0", orientation="horizontal")
    # Evaluate pdf_y on a grid covering Y range
    y_grid = np.linspace(Y.min(), Y.max(), 300)
    r_gamma_1=gamma(a=alpha_1,scale=theta_1)
    #px =trans_prop_log_normal(0,x0[0],d,x_grid,[the1,the2,the3])  #pdf_x(x_grid)  # user function
    py=r_gamma_1.pdf(y_grid)

    #py = trans_prop_log_normal(0,x1[0],d,y_grid,[the1_y,the2_y,the3_y])  # user function
    # Plot pdf_y vertically: (pdf, y_grid)
    ax_yhist.plot(py, y_grid, 'r-', lw=2, label="pdf_y")
    ax_yhist.legend()
    ax_yhist.set_xlabel("Density (Y)")
    # Hide Y ticklabels
    plt.setp(ax_yhist.get_yticklabels(), visible=False)

    # 6) Tight layout / show
    plt.tight_layout()
    plt.show()
# ----------------------------------------------------------------------
# Example usage
"""
if __name__ == "__main__":
    import math
    # Suppose you have some correlation generator f(N):
    def f_example(N):
        # For demonstration: create correlated (X, Y).
        # E.g., X ~ N(0,1), Y = 0.5*X + sqrt(0.75)*Z, Z~N(0,1).
        X = np.random.randn(N)
        Z = np.random.randn(N)
        Y = 0.5*X + math.sqrt(0.75)*Z
        return X, Y
    
    # Suppose the known marginals are N(0,1) for X, and also N(0,1) for Y
    # (just an example--in reality you'd define your actual pdfs).
    def pdf_x(x):
        # standard normal pdf
        return (1.0/np.sqrt(2*math.pi))*np.exp(-x**2/2)
    def pdf_y(y):
        # standard normal pdf
        return (1.0/np.sqrt(2*math.pi))*np.exp(-y**2/2)

    # Now make the plot
    plot_samples_and_marginals(f_example, pdf_x, pdf_y, N=100000, bins=300)
"""

#%%





#%%
# Here we have the functions that update the parameters of the drift and diffusion that 
# depend on the intial and transitioned values. This function is to be used at every step of 
# the particle filter.



def g(x,g_pars):

    # This function is made such that b(x)=x*g(x) where b(x) is
    # the drift of the logistic sde.

    the1,the2,the3=g_pars

    return the3**2/2+the1-the2*x

def update_log_functs(pars_0,pars_1,t0,x0_0,x0_1,T,x_pr_0,x_pr_1,levels=2):

    # This is a draft function that doesn't include the updated parameters of the 
    # finite difference. A more complete function would include them
    # This function updates the drift and diffusion parameters of the auxiliar process.
    # The dependance on x0, and x_pr is such that we can create a linearization in time.

    if levels==2:

        A_0,fi_0=pars_0
        A_1,fi_1=pars_1
        the3_0=A_0[2]
        the3_1=A_1[2]
        if the3_0!=fi_0:
            raise ValueError("Parameters that are equals are not equal")

        if the3_1!=fi_1:
            raise ValueError("Parameters that are equals are not equal")

        a_0= g(x0_0,A_0)/(t0-T)+g(x_pr_0,A_0)/(T-t0)
        a_1= g(x0_1,A_1)/(t0-T)+g(x_pr_1,A_1)/(T-t0)
        b_0=-g(x0_0,A_0)*T/(t0-T)-g(x_pr_0,A_0)*t0/(T-t0)
        b_1=-g(x0_1,A_1)*T/(t0-T)-g(x_pr_1,A_1)*t0/(T-t0)

        A_til_0=[a_0,b_0]
        A_til_1=[a_1,b_1]
        fi_til_0=the3_0
        fi_til_1=the3_1
        r_pars_0=[a_0,b_0,fi_0]
        r_pars_1=[a_1,b_1,fi_1]
        H_pars_0=[a_0,b_0,fi_0]
        H_pars_1=[a_1,b_1,fi_1]
        atdp_0=[a_0,b_0,fi_0]
        atdp_1=[a_1,b_1,fi_1]   

        return [A_til_0,A_til_1,fi_til_0,fi_til_1,r_pars_0,\
        r_pars_1,H_pars_0,H_pars_1,atdp_0,atdp_1]

    elif levels==1:
        A,fi=pars_0
        the3=A[2]
        if the3!=fi:
            raise ValueError("Parameters that are equals are not equal")
        a= g(x0_0,A)/(t0-T)+g(x_pr_0,A)/(T-t0)
        b=-g(x0_0,A)*T/(t0-T)-g(x_pr_0,A)*t0/(T-t0)
        # the parameters of the b_log function are 
        A=pars_0
        # the parameters of the b_log_aux function are
        A_til=[a,b]
        fi_til=the3
        r_pars=[a,b,fi]
        H_pars=[a,b,fi]        
        atdp=[a,b,fi]
    # we are going to return 
        return [A_til,fi_til,r_pars,H_pars,\
        atdp]


 #%%
# The following set of functions is build to generalize the update 
# of parameters in the Coupled SGD algorithm. It's function is basically 
# to have as arguments the direct update of the parameters (updated via SGD)
# and consequently update other parameters that depend on them. As example we have 
# sample pars, r_pars, H_pars, etc.


def update_pars_ou(pars_0,pars_1,up_pars,levels=2):

    # This function updates the parameters of the OU process

    # ARGUMENTS: pars_0 and pars_1 contain the parameters 
    # A_0,fi_0,g_den_par_0 and A_1,fi_1,g_den_par_1 respectively.
    # up_pars is the fd_rate.
    
    # DISCLAIMER: Additional structure might be included 
    # in the definitions of the update pars functions given that 
    # by default up_pars is just fd_rate.

    # OUTPUT: parameters necessary for the evaluation of the function C_SGD_bridge.

    
    if levels==2:
        
        fd_rate=up_pars
        A_0,fi_0,g_den_par_0=pars_0
        A_1,fi_1,g_den_par_1=pars_1


        A_fd_0=A_0+fd_rate
        A_fd_1=A_1+fd_rate*0.5
        fi_fd_0=fi_0+fd_rate
        fi_fd_1=fi_1+fd_rate*0.5
        # For the testing case we only change the 
        # auxiliar parameters by adding 0.2 to the drift
        # constant.
        A_til_0=A_0+0.2
        A_til_1=A_1+0.2
        fi_til_0=fi_0
        fi_til_1=fi_1
        fi_til_fd_0=fi_fd_0
        fi_til_fd_1=fi_fd_1
        r_pars_0=[ou_sd,[A_til_0,fi_til_0]]
        r_pars_fd_0=[ou_sd,[A_til_0,fi_til_fd_0]]
        H_pars_0=[ou_sd,[A_til_0,fi_til_0],A_til_0]
        H_pars_fd_0=[ou_sd,[A_til_0,fi_til_fd_0],A_til_0]
        ind_prop_trans_par_0=[A_til_0,fi_til_0]
        atdp_0=[A_til_0,fi_til_0]
        r_pars_1=[ou_sd,[A_til_1,fi_til_1]]
        r_pars_fd_1=[ou_sd,[A_til_1,fi_til_fd_1]]
        H_pars_1=[ou_sd,[A_til_1,fi_til_1],A_til_1]
        H_pars_fd_1=[ou_sd,[A_til_1,fi_til_fd_1],A_til_1]
        ind_prop_trans_par_1=[A_til_1,fi_til_1]
        atdp_1=[A_til_1,fi_til_1]
        sample_pars=[A_til_0,fi_til_0,A_til_1,fi_til_1]

        return [A_fd_0,A_fd_1,fi_fd_0,fi_fd_1,A_til_0,A_til_1,fi_til_0,fi_til_1,fi_til_fd_0,\
        fi_til_fd_1,r_pars_0,r_pars_1,\
        r_pars_fd_0,r_pars_fd_1,H_pars_0,H_pars_1,H_pars_fd_0,H_pars_fd_1,\
        sample_pars,g_den_par_0,g_den_par_1,atdp_0,atdp_1,\
        ind_prop_trans_par_0,ind_prop_trans_par_1]
    
    if levels==1:

        fd_rate=up_pars
        A_0,fi_0,g_den_par_0=pars_0
        
        # For the testing case we only change the 
        # auxiliar parameters by adding 0.2 to the drift
        # constant.
        A_fd_0=A_0+fd_rate
        
        fi_fd_0=fi_0+fd_rate
        
        A_til_0=A_0+0.2
        
        fi_til_0=fi_0
        
        fi_til_fd_0=fi_fd_0
        
        r_pars_0=[ou_sd,[A_til_0,fi_til_0]]
        r_pars_fd_0=[ou_sd,[A_til_0,fi_til_fd_0]]
        H_pars_0=[ou_sd,[A_til_0,fi_til_0],A_til_0]
        H_pars_fd_0=[ou_sd,[A_til_0,fi_til_fd_0],A_til_0]
        
        atdp_0=[A_til_0,fi_til_0]
        sample_pars=[A_til_0,fi_til_0]
        return [A_fd_0,fi_fd_0,A_til_0,fi_til_0,fi_til_fd_0,\
        r_pars_0,r_pars_fd_0,H_pars_0,H_pars_fd_0,sample_pars,g_den_par_0,\
        atdp_0]


def update_pars_logis(pars_0,pars_1,up_pars,levels=2):

    # This function updates the parameters of the OU process

    # ARGUMENTS: pars_0 and pars_1 contain the parameters 
    # A_0,fi_0,g_den_par_0 and A_1,fi_1,g_den_par_1 respectively.
    # up_pars is the fd_rate.
    
    # DISCLAIMER: Additional structure might be included 
    # in the definitions of the update pars functions given that 
    # by default up_pars is just fd_rate.

    # OUTPUT: parameters necessary for the evaluation of the function C_SGD_bridge.

    
    if levels==2:
        
        fd_rate=up_pars
        dist_params_0,g_den_par_0=pars_0
        dist_params_1,g_den_par_1=pars_1

        
        # For the testing case we only change the 
        # auxiliar parameters by adding 0.2 to the drift
        # constant.
        the1_0, the2_0, the3_0=dist_params_0
        the1_1, the2_1, the3_1=dist_params_1
        dist_params_fd_0=np.array([[the1_0+fd_rate,the2_0,the3_0],\
        [the1_0,the2_0+fd_rate,the3_0],[the1_0,the2_0,the3_0+fd_rate]])

        dist_params_fd_1=np.array([[the1_1+fd_rate*0.5,the2_1,the3_1],\
        [the1_1,the2_1+fd_rate*0.5,the3_1],[the1_1,the2_1,the3_1+fd_rate*0.5]])
        the3_fd_0=the3_0+fd_rate
        the3_fd_1=the3_1+fd_rate*0.5
        in_dist_pars=np.array([the1_0,the2_0,the3_0,the1_1,the2_1,the3_1])
        A_fd_0=dist_params_fd_0
        A_fd_1=dist_params_fd_1
        fi_fd_0=the3_fd_0
        fi_fd_1=the3_fd_1
        sample_pars=in_dist_pars
        ind_prop_trans_par_0=[the1_0,the2_0,the3_0]   
        ind_prop_trans_par_1=[the1_1,the2_1,the3_1]
        
        return [in_dist_pars,A_fd_0,A_fd_1,fi_fd_0,fi_fd_1,sample_pars,\
        ind_prop_trans_par_0,ind_prop_trans_par_1]
    
    if levels==1:

        fd_rate=up_pars
        dist_params,g_den_par=pars_0
        
        # For the testing case we only change the 
        # auxiliar parameters by adding 0.2 to the drift
        # constant.
        the1, the2, the3=dist_params
        dist_params_fd=np.array([[the1+fd_rate,the2,the3],\
        [the1,the2+fd_rate,the3],[the1,the2,the3+fd_rate]])

        the3_fd=the3+fd_rate
        in_dist_pars_0=dist_params 
        A_fd_0=dist_params_fd
        fi_fd_0=the3_fd
        sample_pars=dist_params
        
        return [in_dist_pars_0,A_fd_0,fi_fd_0,sample_pars]

            
#%%
# Test for rej_max_coup_ou
"""
x0=np.array([0,.5,2.2])
x1=np.array([0.5,-0.1,2.3])

N=len(x0)
d=2
sample_pars=np.array([-0.1,1.2,-0.11,0.8],dtype=float)

np.random.seed(0)
samples= 100000
start=time.time()
x0_samples=np.zeros((samples,3))
x1_samples=np.zeros((samples,3))

for i in range(samples):
    x0_samples[i],x1_samples[i]=rej_max_coup_ou(x0,x1,N,d,sample_pars)
    
end=time.time()
print(end-start)
"""
#%%
"""
rv0=norm(loc=x0*np.exp(sample_pars[0]*d),scale=ou_sd(0,x0,d,x0,[sample_pars[0],sample_pars[1]]))
rv1=norm(loc=x1*np.exp(sample_pars[2]*d),scale=ou_sd(0,x1,d,x1,[sample_pars[2],sample_pars[3]]))
bins=np.arange(-8,8,0.3)
grid=np.arange(-8,8,2**(-8))
three_grid=np.zeros((3,len(grid)))+grid
print(three_grid.shape)
den0=rv0.pdf(three_grid.T)
den1=rv1.pdf(three_grid.T)
"""
#%%
"""
i=1
plt.hist(x0_samples[:,i],bins=bins,density=True)
plt.plot(grid,den0[:,i])
plt.hist(x1_samples[:,i],bins=bins,density=True)
plt.plot(grid,den1[:,i])
"""
# The results for the maximum coupling of this function were succesful. A more generalized 
# version in this function can be persued if we include as parameters a way of computing the 
# densities and samples of the distributions .
#%%
"""

coup_samples=np.zeros((2,samples,N))
coup_samples[0]=x0_samples
coup_samples[1]=x1_samples
i=2
x=x0_samples[:,i]
y=x1_samples[:,i]



from matplotlib.gridspec import GridSpec

# Example data: create a (B, 2) array with random data
#B = 1000
#data = np.random.randn(B, 2)  # Normally distributed data for demonstration

# Extract x and y data points from the (B, 2) array
#x = data[:, 0]  # The first column for x-values
#y = data[:, 1]  # The second column for y-values

# Set up the figure with GridSpec to arrange the plots
fig = plt.figure(figsize=(8, 8))
gs = GridSpec(4, 4, fig)
# Create the main 2D histogram plot in the center
ax_main = fig.add_subplot(gs[1:4, 0:3])
h2d = ax_main.hist2d(x, y, bins=100, cmap='Blues',density=True)
ax_main.set_xlabel('X axis label')
ax_main.set_ylabel('Y axis label')

# Add the color bar for the 2D histogram
plt.colorbar(h2d[3], ax=ax_main, orientation='vertical', label='Counts in bin')

# Create the marginal histogram for the x-axis at the top
ax_x_marginal = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
ax_x_marginal.hist(x, bins=100, color='gray')
ax_x_marginal.axis('off')  # Hide the axis for a cleaner look

# Create the marginal histogram for the y-axis on the right
ax_y_marginal = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
ax_y_marginal.hist(y, bins=100, color='gray', orientation='horizontal')
ax_y_marginal.axis('off')  # Hide the axis for a cleaner look

# Display the plot
plt.tight_layout()
plt.show()

"""




#%%
"""
# Test for Bridge function with crossed parameters
N=5
x0_sca=1.2
np.random.seed(3)
x0=np.random.normal(0,1,N)
l=4
alpha=0
T=10
t0=0
l_d=1
d=2**(l_d)
theta=0.2
sigma=1.2
#sigma_aux=0.2
theta_aux=theta+0.2
sigma_aux=sigma+0.1
#print(theta)
collection_input=[ b_ou_1d,theta,Sig_ou_1d,sigma]
resamp_coef=1
l_max=6
x_true=gen_gen_data_1d(T,x0_sca,l_max,collection_input)
x_reg=cut(T,l,-l_d,x_true)[1:]
# x_reg stands for x regular 
cov=2e0
obs=gen_obs(x_reg,g_normal_1d,cov)
np.random.seed(3)
#print(x_pr)
n_steps=int(2**l*(T-t0))
dt=2**(-l)
x=np.zeros((n_steps+1,N))
x[0]=x0
x_pr=np.random.normal(x0,np.sqrt(sigma**2*(T-t0)),N)
x[-1]=x_pr
int_G=np.zeros(N)
t_test=2.

bridge_false=Bridge_cheap(t0,x0,T,x_pr,b_ou_1d,theta,Sig_ou_1d,sigma,b_ou_aux,theta_aux,\
Sig_ou_aux,sigma_aux,r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,\
[ou_sd,[theta_aux,sigma_aux],theta_aux],l,d,N,crossed=False)

N=5
x0_sca=1.2
np.random.seed(3)
x0=np.random.normal(0,1,N)
l=4
alpha=0
T=10
t0=0
l_d=1
d=2**(l_d)
theta=0.2
sigma=1.2
#sigma_aux=0.2
theta_aux=theta+0.2
sigma_aux=sigma+0.1
#print(theta)
collection_input=[ b_ou_1d,theta,Sig_ou_1d,sigma]
resamp_coef=1
l_max=6
x_true=gen_gen_data_1d(T,x0_sca,l_max,collection_input)
x_reg=cut(T,l,-l_d,x_true)[1:]
# x_reg stands for x regular 
cov=2e0
obs=gen_obs(x_reg,g_normal_1d,cov)
np.random.seed(3)
#print(x_pr)
n_steps=int(2**l*(T-t0))
dt=2**(-l)
x=np.zeros((n_steps+1,N))
x[0]=x0
x_pr=np.random.normal(x0,np.sqrt(sigma**2*(T-t0)),N)
x[-1]=x_pr
int_G=np.zeros(N)
t_test=2.

bridge_true=Bridge_cheap(t0,x0,T,x_pr,b_ou_1d,theta,Sig_ou_1d,sigma,b_ou_aux,theta_aux,\
Sig_ou_aux,sigma_aux,r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,\
[ou_sd,[theta_aux,sigma_aux],theta_aux],l,d,N,crossed=True)

print(bridge_true.shape)
print(bridge_false.shape)

diff=bridge_true-bridge_false
print(bridge_true)
print(bridge_false)
print(diff.diagonal())


# RESULTS: The bridge for crossing=True seems to be working, the diagonal of the difference
# is zero as expected.
"""

#%%
# Kalman filter, this funciton is originally from the Un_cox_PF_functions_def.py file

def KF(xin,dim,dim_o,K,G,H,D,obs):

    #This function computes the Kalman Filter observations at arbitrary times
    # and a time indendent linear setting, i.e.
    #X_t=KX_{t-1}+ G W_t
    #Y_t=HX_t+DV_t
    
    # Suppose you have an ou process with sde dx_t=theta x_t+sigma dB_t,
    # we can write it as discrete state space model as follows:
    # X_t=exp(theta)X_{t-1}+  sigma*sqrt([exp(2*theta)-1]/(2*theta)) W_t
    # so that K=exp(theta) and G=sigma*sqrt([exp(2*theta)-1]/(2*theta))

    #ARGUMENTS: 
    #obs: is a rank 2 array with dimensions (number of observations, dim_o)
    #xin: rank 1 array with dimensions (dim)
    #OUTPUTS:
    #x: rank two array with dimensions (T+1,dim)
    T=obs.shape[0]
    x=np.zeros((T+1,dim))
    P=np.zeros((T+1,dim,dim))

    x[0]=xin
    I=identity(dim).toarray()
    P[0]=0

    for i in range(T):
        xh=K@x[i]
        #print("xh",xh)
        Ph=K@P[i]@(K.T)+(G@(G.T))
        #print("Ph",Ph)
        y=obs[i]-H@xh
        #print("y",y)
        S=H@Ph@(H.T)+(D@(D.T))
        
        Kg=Ph@(H.T)@la.inv(S)
        x[i+1]=xh+Kg@y
        P[i+1]=(I-Kg@H)@Ph

    x_smooth=np.zeros((T+1,dim))
    x_smooth[0]=x[0]
    x_smooth[-1]=x[-1]  
    P_smooth=np.zeros((T+1,dim,dim))
    P_smooth[-1]=P[-1]

    for i in range(T-1):
        t=T-i-1
        Gamma=P[t]@(K.T)@la.inv(K@(P[t]@(K.T))+G@(G.T))
        x_smooth[t]=x[t]+Gamma@(x_smooth[t+1]-K@x[t])
        P_smooth[t]=P[t]+Gamma@(P_smooth[t+1]-(K@P[t]@(K.T)+G@(G.T)))@Gamma.T

    return x,x_smooth

def KF_Grad_lik(xin,dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S):


    # DISCLAIMER: Even though some steps to generalize this function to multiple dimensions are taken,
    # the function can only be applied to one dimension.

    #This function computes the Kalman Filter observations at arbitrary times
    # and a time indendent linear setting, i.e.
    #X_t=KX_{t-1}+ G W_t
    #Y_t=HX_t+DV_t
    # For this specific funciton we have the convention R=GG^T and S=DD^T

    # Suppose you have an ou process with sde dx_t=theta x_t+sigma dB_t,
    # we can write it as discrete state space model as follows:
    # X_t=exp(theta)X_{t-1}+  sigma*sqrt([exp(2*theta)-1]/(2*theta)) W_t
    # so that K=exp(theta) and G=sigma*sqrt([exp(2*theta)-1]/(2*theta))

    #ARGUMENTS: S
    #obs: is a rank 2 array with dimensions (number of observations, dim_o)
    #xin: rank 1 array with dimensions (dim)
    # Grads_K is a rank 3 array with dimension (3,dim,dim), it represetns the partial derivatives 
    # of K w.r.t. the 3 different theta, sigma and sigma_obs. 
    # Grad_R and Grad_S have the same interpretation of Grad_K but for R and S. 
    # theta, sigma and sigma_obs represent the parameters of the OU and the Gaussian observations.
    #OUTPUTS:
    #x: rank two array with dimensions (T+1,dim)

    T=obs.shape[0]
    x=np.zeros((T+1,dim))
    P=np.zeros((T+1,dim,dim))
    Grad_x=np.zeros((3,T+1,dim))
    Grad_P=np.zeros((3,T+1,dim,dim))
    x[0]=xin
    I=identity(dim).toarray()
    P[0]=0
    Grad_x[:,0]=0
    Grad_P[:,0]=0
    Grad_log_lik=0

    for i in range(T):
        xh=K@x[i]
        #print("xh",xh)
        Grad_xh=Grad_K@x[i]+Grad_x[:,i]@(K.T)
        # the above expression makes sure that the resulting array is 
        # a rank 2 array with dimensions (3,dim)
        Ph=K@P[i]@(K.T)+(G@(G.T))
        Grad_Ph=2*Grad_K@P[i]@(K.T)\
        +np.einsum('ij,kjl,lm->kim',K,Grad_P[:,i],K.T)+Grad_R

        #print("Ph",Ph)
        y=obs[i]-H@xh
        #print("y",y)
        S_old=H@Ph@(H.T)+(D@(D.T))
        # This S_old is name like that so it is not confused with the S matrix coming form the 
        # stochastic part of the observations.
        #aux_der=np.einsum('ij,kjl,lm->kim',H,Grad_Ph,H.T)
        Grad_S_old=Grad_S+np.einsum('ij,kjl,lm->kim',H,Grad_Ph,H.T)
        # since H is constant the above equation only has to terms on the rhs.
        Kg=Ph@(H.T)@la.inv(S_old)
        Grad_Kg=Grad_Ph@(H.T)@la.inv(S_old)\
        -np.einsum('ij,kjl->kil',Kg@la.inv(S_old),Grad_S_old)
        x[i+1]=xh+Kg@y
        Grad_x[:,i+1]=Grad_xh+Grad_Kg@y-(Grad_xh@(Kg.T))
        P[i+1]=(I-Kg@H)@Ph
        Grad_P[:,i+1]=np.einsum('ij,kjl->kil',I-Kg@H,Grad_Ph)\
        -Grad_Kg@H@Ph
        grad_log_lik_y=(-1/2)*( -np.einsum("i,ki->k",2*(obs[i]-H@xh)@(la.inv(S_old))@H,Grad_xh)[:,np.newaxis,np.newaxis]\
        -np.einsum("i,kji->kji",(obs[i]-H@xh)**2@(la.inv(S_old)**2),Grad_S_old )\
        + Grad_S_old@la.inv(S_old))
        #print((np.einsum("i,ki->k",2*(obs[i]-H@xh)@(la.inv(S_old))@H,Grad_xh)[:,np.newaxis,np.newaxis]).shape)
        Grad_log_lik+=grad_log_lik_y

    x_smooth=np.zeros((T+1,dim))
    x_smooth[0]=x[0]
    x_smooth[-1]=x[-1]  
    P_smooth=np.zeros((T+1,dim,dim))
    P_smooth[-1]=P[-1]

    for i in range(T-1):
        t=T-i-1
        Gamma=P[t]@(K.T)@la.inv(K@(P[t]@(K.T))+G@(G.T))
        x_smooth[t]=x[t]+Gamma@(x_smooth[t+1]-K@x[t])
        P_smooth[t]=P[t]+Gamma@(P_smooth[t+1]-(K@P[t]@(K.T)+G@(G.T)))@Gamma.T


    return x,x_smooth, Grad_log_lik

#%%

def sampling(cumulative):
    # Given a cumulative distribution this function samples from it.

    # ARGUMENTS:
    # cumulative: rank 1 array with dimensions (s), where s is the number of
    # states of the distribution.
    # OUTPUT:
    # p_sampled: integer from 0 to s-1 that represents the sampled state.
    u=np.random.uniform()
    for i in range(len(cumulative)):
        if cumulative[i]>u:
            p_sampled=i
            break
    return p_sampled


def P_l_cumu_gen(P_0,s,beta,l0):

    # This function generates the cumulative distribution of a pdf which has a number of 
    # states s. Let q=4, the states specifically are l0 to s+l0-1, the pdf is an arbitrary number P_0
    # in (0,1) for l=l0 and the pdf is proportional to (q+l-l0)log(q+l-l0)^2*2^{-beta*l} for l=l0+1 to s-1+l0.
    
    # ARGUMENTS:
    # P_0: arbitrary probability of l_0
    # s: number of states of the distribution
    # beta: parameter of the distribution
    # l0: initial state of the distribution, must be l0 >= 0.

    # OUTPUT:
    # cumulative: rank 1 array with dimensions (s), where s is the number of
    # states of the distribution.
    q=4
    eLes=np.arange(l0,s+l0)
    P=(1-P_0)/np.sum((q+eLes[1:]-l0)*np.log(q+eLes[1:]-l0)**2/2**(beta*eLes[1:]))
    pdf=np.zeros(s)
    pdf[0]=P_0
    pdf[1:]=P*(q+eLes[1:]-l0)*np.log(q+eLes[1:]-l0)**2/2**(beta*eLes[1:])
    cumulative=np.cumsum(pdf)

    return cumulative


def P_p_cumu_gen(P_0,pmax,beta,s0):

    # This function generates the cumulative distribution of a pdf which has a number of 
    # states s. The states specifically are from 0 to pmax, the pdf is an arbitrary number P_0
    # in (0,1) for the state 0 the rest of the pdf is proportional to (p+q)log(p+q)^2*2^{-beta*p} for p=1 to p=pmax.
    
    # ARGUMENTS:
    # P_0: arbitrary probability of the state 0
    # beta: parameter of the distribution
    # pmax: maximum state
    # s0: number of steps of the state 0. 

    # OUTPUT:
    # cumulative: rank 1 array with dimensions (s), where s is the number of
    # states of the distribution.
    q=4
    ePes=np.array(range(pmax+1))
    P=(1-P_0)/np.sum((ePes[1:]+q)*np.log(ePes[1:]+q)**2/(s0*2**(ePes[1:]))**(beta))
    pdf=np.zeros(pmax+1)
    pdf[0]=P_0
    pdf[1:]=P*(ePes[1:]+q)*np.log(ePes[1:]+q)**2/(s0*2**(ePes[1:]))**(beta)
    cumulative=np.cumsum(pdf)

    return cumulative


#%%

# Testing of cumulative distribution
"""
beta=1
print(P_l_cumu_gen(0.5,20,beta,1))
"""

#%%
    
#print(np.arange(1,5))
#%%






#%%
#x=np.array([1.2,3])
#print(x.shape)


#%%

# TEST 1: OU process where the proposal distribution is a brownian motion with
"""
T=2
x0=3
l=10
A=0.05
sigma=0.45
collection_input=[b_ou_1d,A,Sig_ou_1d,sigma]
print(gen_gen_data_1d(T,x0,l,collection_input).shape)
times=np.linspace(0,T,T*2**l+1)
plt.plot(times,gen_gen_data_1d(T,x0,l,collection_input))
"""

#%%
# IN THIS PART WE CHECK THE DIFFUSION BRIDGE SAMPLING ALGORITHM (NOT THE 
# PARTICLE FILTER YET)

# TEST 1: OU process where the proposal distribution is a brownian motion with 
# variance equal to the diffusion of the OU process. 
# The proposal transition is the same as the transition of the auxiliar process but with 
# with a different diffusion term, in a notation error I call that diffusion term 
# sigma_aux.
# We where able to use the bridge to sample from a distribution.
# Parameters and initial conditions
"""
x0=np.array([1.2])
T=3
t0=0
theta=np.array([[0.3]])
sigma=np.array([[0.2]])
sigma_aux=np.array([[0.2]])
theta_aux=np.array([[0.]])
"""
#%%
# Sampling from the stride distribution (auxiliar distribtuon such that resembles 
# as much as possible the transition of the original diffusion process from time t0
# to time T)
"""
np.random.seed(1)
x_pr=np.random.normal(x0,np.sqrt(sigma_aux[0]**2*(T-t0)),1)
print(x_pr)
l=8
n_steps=2**l*(T-t0)
dt=2**(-l)
samples=100
x=np.zeros((n_steps+1,samples,1))
x[0]=x0
x_pr=np.random.normal(x0,np.sqrt(sigma_aux**2*(T-t0)),(samples,1))
x[-1]=x_pr
int_G=np.zeros(samples)
t_test=2.

for i in range(n_steps-1):

    t=t0+(i)*dt
    #rv=norm(loc=x_pr,scale=np.sqrt(sigma**2*(T-t)))
    drift=pff.b_ou(x[i],-theta)+pff.Sig_ou(x[i],sigma)[:,0]**(2)\
    *(x_pr-x[i])/(sigma[0]**2*(T-t))
    
    x[i+1]=x[i]+drift*dt+pff.Sig_ou(x[i],sigma)[:,0]*np.sqrt(dt)*np.random.normal(0,1,(samples,1))
    int_G=int_G+(theta[0,0]*x[i+1]*(x[i+1]-x_pr)*dt/(sigma[0,0]**2*(T-t)))[:,0]
    if t==t_test:
        x_2=x[i]
    
    #print(x_pr)

"""

#%%

"""
rv_aux_1=norm(loc=x0,scale=np.sqrt(sigma_aux**2*(T-t0)))
rv_aux=norm(loc=x0,scale=np.sqrt(sigma**2*(T-t0)))
dPdP=np.exp(int_G)*rv_aux.pdf(x_pr[:,0])/rv_aux_1.pdf(x_pr[:,0])
print(int_G.shape)
plt.plot(np.linspace(t0,T,n_steps+1),x[:,:100,0])
#print(x[-2,0   ],x[-1,0])

print(x0*np.exp(-theta*t_test))
print(np.mean(dPdP*x_2[:,0]))
"""
# %%

"""
print(dPdP.shape)
print(sigma**2*(1-np.exp(-2*theta*(t_test-t0)))/(2*theta))
print(np.mean(dPdP*x_2[:,0]**2)-np.mean(dPdP*x_2[:,0])**2)
print(sigma**2*(1-np.exp(-2*theta*(t_test-t0)))/(2*theta)+x0**2*np.exp(-2*theta*(t_test-t0)))
print(np.mean(dPdP*x_2[:,0]**2))
"""

#%%
# 1D version
#########################################################################################################
#########################################################################################################
# IN THIS PART WE CHECK THE DIFFUSION BRIDGE SAMPLING ALGORITHM (NOT THE 
# PARTICLE FILTER YET)

# TEST 1: OU process where the proposal distribution is a brownian motion with 
# variance equal to the diffusion of the OU process. 
# The proposal transition is the same as the transition of the auxiliar process but with 
# with a different diffusion term, in a notation error I call that diffusion term 
# sigma_aux.
# We where able to use the bridge to sample from a distribution.
# Parameters and initial conditions

"""
x0=1.2
T=3
t0=0
theta=0.3
sigma=0.2
sigma_aux=0.2
theta_aux=0.
print(theta)
"""
#%%
# Sampling from the stride distribution (auxiliar distribtuon such that resembles 
# as much as possible the transition of the original diffusion process from time t0
# to time T)
"""
np.random.seed(1)
x_pr=np.random.normal(x0,np.sqrt(sigma_aux**2*(T-t0)),1)
print(x_pr)
l=4
n_steps=2**l*(T-t0)
dt=2**(-l)
samples=5000
x=np.zeros((n_steps+1,samples))
x[0]=x0
x_pr=np.random.normal(x0,np.sqrt(sigma_aux**2*(T-t0)),samples)
x[-1]=x_pr
int_G=np.zeros(samples)
t_test=2.
"""
#%%
"""
for i in range(n_steps-1):

    t=t0+(i)*dt
    #rv=norm(loc=x_pr,scale=np.sqrt(sigma**2*(T-t)))
    drift=b_ou_1d(x[i],-theta)+Sig_ou_1d(x[i],sigma)**(2)\
    *(x_pr-x[i])/(sigma**2*(T-t))
    
    x[i+1]=x[i]+drift*dt+Sig_ou_1d(x[i],sigma)*np.sqrt(dt)*np.random.normal(0,1,samples)
    int_G=int_G+(theta*x[i+1]*(x[i+1]-x_pr)*dt/(sigma**2*(T-t)))
    if t==t_test:
        x_2=x[i]
    
    #print(x_pr)
"""


#%%
"""
rv_aux_1=norm(loc=x0,scale=np.sqrt(sigma_aux**2*(T-t0)))
rv_aux=norm(loc=x0,scale=np.sqrt(sigma**2*(T-t0)))
dPdP=np.exp(int_G)*rv_aux.pdf(x_pr)/rv_aux_1.pdf(x_pr)
print(int_G.shape)
plt.plot(np.linspace(t0,T,n_steps+1),x[:,:100])
#print(x[-2,0   ],x[-1,0])

print(x0*np.exp(-theta*t_test))
print(np.mean(dPdP*x_2))
"""
# %%

"""
print(dPdP.shape)
print(sigma**2*(1-np.exp(-2*theta*(t_test-t0)))/(2*theta))
print(np.mean(dPdP*x_2**2)-np.mean(dPdP*x_2)**2)
print(sigma**2*(1-np.exp(-2*theta*(t_test-t0)))/(2*theta)+x0**2*np.exp(-2*theta*(t_test-t0)))
print(np.mean(dPdP*x_2**2))
"""

# end of 1D version
#########################################################################################################
#########################################################################################################

#%%

# NOW WE TEST 
    # TEST 2: In this test we generalize the diffusion term to be proportional to a polynomial function of the time
    # considering that at the final time both diffusion terms of the original and the auxiliar process coincide.
    # we define a function to compute the standard deviation of the auxiliar process in terms of alpha.


#%%
"""
x0=np.array([1.2])
T=4
t0=0
theta=np.array([[.13]])
sigma=np.array([[0.2]])
sigma_aux=np.array([[0.2]])
alpha=0.5
x0=np.array([1.2])
T=4
t0=0
theta=np.array([[.13]])
sigma=np.array([[0.2]])
sigma_aux=np.array([[0.2]])
alpha=0.5
"""
#%%
# Sampling from the stride distribution (auxiliar distribtuon such that resembles 
# as much as possible the transition of the original diffusion process from time t0
# to time T)
"""
np.random.seed(7)
# x_pr is sampled accordingly to the proposal transition, we 
# want it to be as close to the transition of the auxiliar process
# as possible.

l=7
n_steps=2**l*(T-t0)
dt=2**(-l)
samples=1000000
x=np.zeros((n_steps+1,samples,1))
x[0]=x0
x_pr=np.random.normal(x0,trans_st(0,alpha,sigma[0,0]),(samples,1))
x[-1]=x_pr
int_G=np.zeros(samples)
t_test=3.


for i in range(n_steps-1):
    t=t0+(i)*dt
    #rv=norm(loc=x_pr,scale=np.sqrt(sigma**2*(T-t)))
    drift=pff.b_ou(x[i],-theta)+pff.Sig_ou(x[i],sigma)[:,0]**(2)\
    *(-1)*(x[i]-x_pr)/(trans_st(t,alpha,sigma[0,0])**2)
    x[i+1]=x[i]+drift*dt+pff.Sig_ou(x[i],sigma)[:,0]*np.sqrt(dt)*np.random.normal(0,1,(samples,1))
    int_G=int_G+dt*(theta[0,0]*x[i]*(x[i]-x_pr)/(trans_st(t ,alpha,sigma[0,0])**2)\
    -(1/2)*pff.Sig_ou(x[i],sigma)[:,0]**(2)*(1-(t/T)**(2*alpha))*\
    (1/trans_st(t,alpha,sigma[0,0])**2-(x[i]-x_pr)**2/trans_st(t,alpha,sigma[0,0])**4))[:,0]
    if t==t_test:
        print("it's copying them at time ",t)
        x_2=x[i]
    #print(x_pr)
"""     
#%%
# in the following we compute the change of measure 
"""
prop_trans=norm(loc=x0,scale=trans_st(0,alpha,sigma[0,0]))
aux_trans=norm(loc=x0,scale=trans_st(0,alpha,sigma[0,0]))
dPdP=np.exp(int_G)*aux_trans.pdf(x_pr[:,0])/prop_trans.pdf(x_pr[:,0])
print(int_G.shape)
plt.plot(np.linspace(t0,T,n_steps+1),x[:,:100,0])
#print(x[-2,0   ],x[-1,0]3)
"""
#%%
"""
# mean of the OU process at time t_test
print((x0*np.exp(-theta*t_test)))
print(np.mean(dPdP*x_2[:,0]))
print(np.mean(dPdP*x_2[:,0])/(x0*np.exp(-theta*t_test)))
# second moment of the OU process at time t_test
print((sigma**2*(1-np.exp(-2*theta*(t_test-t0)))/(2*theta)+x0**2*np.exp(-2*theta*(t_test-t0))))
print(np.mean(dPdP*x_2[:,0]**2))
print(np.mean(dPdP*x_2[:,0]**2)/(sigma**2*(1-np.exp(-2*theta*(t_test-t0)))/(2*theta)+x0**2*np.exp(-2*theta*(t_test-t0))))
"""
#%%

# 1D version
#########################################################################################################
#########################################################################################################
"""
x0=np.array(1.2)
T=4
t0=0
theta=np.array(.13)
sigma=np.array(0.2)
sigma_aux=np.array(0.2)
alpha=0.5

x0=np.array(1.2)
T=4
t0=0
theta=np.array(.13)
sigma=np.array(0.2)
sigma_aux=np.array(0.2)
alpha=0.5
"""
#%%
# Sampling from the stride distribution (auxiliar distribtuon such that resembles 
# as much as possible the transition of the original diffusion process from time t0
# to time T)
"""
np.random.seed(7)
# x_pr is sampled accordingly to the proposal transition, we 
# want it to be as close to the transition of the auxiliar process
# as possible.
l=7
n_steps=2**l*(T-t0)
dt=2**(-l)
samples=1000000
x=np.zeros((n_steps+1,samples))
x[0]=x0
x_pr=np.random.normal(x0,trans_st(0,alpha,sigma),(samples))
x[-1]=x_pr
int_G=np.zeros(samples)
t_test=3.
for i in range(n_steps-1):
    t=t0+(i)*dt
    #rv=norm(loc=x_pr,scale=np.sqrt(sigma**2*(T-t)))
    drift=b_ou_1d(x[i],-theta)+Sig_ou_1d(x[i],sigma)**(2)\
    *(-1)*(x[i]-x_pr)/(trans_st(t,alpha,sigma)**2)
    x[i+1]=x[i]+drift*dt+Sig_ou_1d(x[i],sigma)*np.sqrt(dt)*np.random.normal(0,1,(samples))
    int_G=int_G+dt*(theta*x[i]*(x[i]-x_pr)/(trans_st(t ,alpha,sigma)**2)\
    -(1/2)*Sig_ou_1d(x[i],sigma)**(2)*(1-(t/T)**(2*alpha))*\
    (1/trans_st(t,alpha,sigma)**2-(x[i]-x_pr)**2/trans_st(t,alpha,sigma)**4))
    if t==t_test:
        print("it's copying them at time ",t)
        x_2=x[i]
    #print(x_pr)
     """   
#%%
# in the following we compute the change of measure 
"""
prop_trans=norm(loc=x0,scale=trans_st(0,alpha,sigma))
aux_trans=norm(loc=x0,scale=trans_st(0,alpha,sigma))
dPdP=np.exp(int_G)*aux_trans.pdf(x_pr)/prop_trans.pdf(x_pr)
print(int_G.shape)
plt.plot(np.linspace(t0,T,n_steps+1),x[:,:100])
#print(x[-2,0   ],x[-1,0]3)
"""
#%%
# mean of the OU process at time t_test
"""
print((x0*np.exp(-theta*t_test)))
print(np.mean(dPdP*x_2))
print(np.mean(dPdP*x_2)/(x0*np.exp(-theta*t_test)))
# second moment of the OU process at time t_test
print((sigma**2*(1-np.exp(-2*theta*(t_test-t0)))/(2*theta)+x0**2*np.exp(-2*theta*(t_test-t0))))
print(np.mean(dPdP*x_2**2))
print(np.mean(dPdP*x_2**2)/(sigma**2*(1-np.exp(-2*theta*(t_test-t0)))/(2*theta)+x0**2*np.exp(-2*theta*(t_test-t0))))
"""

# end of 1D version
#########################################################################################################
#########################################################################################################
#%%
# In the following we will recreate the previous experiment with predefined functions,
# this will help us latter to make sure we introduce no errors in the code.

"""
x0=np.array([1.2])
T=4
t0=0
theta=np.array([[.13]])
sigma=np.array([[0.2]])
sigma_aux=np.array([[0.2]])
alpha=0.5
"""
#%%
"""
np.random.seed(7)
# x_pr is sampled accordingly to the proposal transition, we 
# want it to be as close to the transition of the auxiliar process
# as possible.
l=7
n_steps=2**l*(T-t0)
dt=2**(-l)
samples=1000000
x=np.zeros((n_steps+1,samples,1))
x[0]=x0
x_pr=np.random.normal(x0,trans_st(0,alpha,sigma[0,0]),(samples,1))
x[-1]=x_pr
int_G=np.zeros(samples)
t_test=3.

for i in range(n_steps-1):
    t=t0+(i)*dt
    #rv=norm(loc=x_pr,scale=np.sqrt(sigma**2*(T-t)))
    drift=pff.b_ou(x[i],-theta)+pff.Sig_ou(x[i],sigma)[:,0]**(2)\
    *r_normal(t,x[i],T,x_pr,[new_trans_st,[alpha,sigma[0,0]]],crossed=False)
    #new_trans_st(t,x,T,x_pr,pars)
    #[gbm_aux_sd,[sigma,mu,t0,x0]]
    #(t,x,T,x_pr,pars,crossed=False)
    #trans_st(t,alpha,sigma[0,0])
    x[i+1]=x[i]+drift*dt+pff.Sig_ou(x[i],sigma)[:,0]*np.sqrt(dt)*np.random.normal(0,1,(samples,1))
    int_G=int_G+dt*((pff.b_ou(x[i],-theta)-pff.b_ou(x[i],np.array([[0]])))\
    *r_normal(t,x[i],T,x_pr,[new_trans_st,[alpha,sigma[0,0]]],crossed=False)\
    #-(1/2)*pff.Sig_ou(x[i],sigma)[:,0]**(2)*(1-(t/T)**(2*alpha))*\
    # (1/trans_st(t,alpha,sigma[0,0])**2-(x[i]-x_pr)**2/trans_st(t,alpha,sigma[0,0])**4))[:,0]
    -(1/2)*(pff.Sig_ou(x[i],sigma)[:,0]**(2)-pff.Sig_ou(x[i],sigma*(t/T)**(alpha))[:,0]**(2))*\
    (-H_normal(t,x[i],T,x_pr,[new_trans_st,[alpha,sigma[0,0]]])\
    -r_normal(t,x[i],T,x_pr,[new_trans_st,[alpha,sigma[0,0]]],crossed=False)**2))[:,0]
    if t==t_test:
        print("it's copying them at time ",t)
        x_2=x[i]
"""
    #print(x_pr)
#%%
"""
prop_trans=norm(loc=x0,scale=trans_st(0,alpha,sigma[0,0]))
aux_trans=norm(loc=x0,scale=trans_st(0,alpha,sigma[0,0]))
dPdP=np.exp(int_G)*aux_trans.pdf(x_pr[:,0])/prop_trans.pdf(x_pr[:,0])
print(int_G.shape)
n_points=10
plt.scatter(np.zeros(n_points)+T,x_pr[:n_points,0])
plt.plot(np.linspace(t0,T,n_steps+1),x[:,:n_points,0])
"""
#print(x[-2,0   ],x[-1,0]3)
#%%
# mean of the OU process at time t_test
"""
print((x0*np.exp(-theta*t_test)))
print(np.mean(dPdP*x_2[:,0]))
print(np.mean(dPdP*x_2[:,0])/(x0*np.exp(-theta*t_test)))
# second moment of the OU process at time t_test
print((sigma**2*(1-np.exp(-2*theta*(t_test-t0)))/(2*theta)+x0**2*np.exp(-2*theta*(t_test-t0))))
print(np.mean(dPdP*x_2[:,0]**2))
print(np.mean(dPdP*x_2[:,0]**2)/(sigma**2*(1-np.exp(-2*theta*(t_test-t0)))/(2*theta)+x0**2*np.exp(-2*theta*(t_test-t0))))
"""
# The results of this experiments are positive, the values expected for both 
# the mean and the second mmoment are comparable. This version is a bit more generalizable but 
# have to define yet a couple of function to make it completely generalizable, these functions are

# - Proposal sampler, this function is equal to the auxiliar sampler when available.
# - These function must have the following arguments: t, x, T, x_pr, pars, pars_numb
# - Drift and diff terms of the auxiliar process
# - grad of the log density must be in terms of 4 parameters: t, x, T, x_pr, pars, pars_numb
# - Hessian of log density must be in terms of 4 parameters: t, x, T, x_pr, pars, pars_numb

# %%

#1D version
#########################################################################################################
#########################################################################################################

"""
x0=np.array(1.2)
T=4
t0=0
theta=np.array(.13)
sigma=np.array(0.2)
sigma_aux=np.array(0.2)
alpha=0.5
"""
#%%
"""
np.random.seed(7)
# x_pr is sampled accordingly to the proposal transition, we 
# want it to be as close to the transition of the auxiliar process
# as possible.
l=7
n_steps=2**l*(T-t0)
dt=2**(-l)
samples=1000
x=np.zeros((n_steps+1,samples))
x[0]=x0
x_pr=np.random.normal(x0,trans_st(0,alpha,sigma),(samples))
x[-1]=x_pr
int_G=np.zeros(samples)
t_test=3.
"""
#%%
"""
for i in range(n_steps-1):
    t=t0+(i)*dt
    #rv=norm(loc=x_pr,scale=np.sqrt(sigma**2*(T-t)))
    drift=b_ou_1d(x[i],-theta)+Sig_ou_1d(x[i],sigma)**(2)\
    *r_normal_1d(t,x[i],T,x_pr,[new_trans_st,[alpha,sigma]],crossed=False)
    #new_trans_st(t,x,T,x_pr,pars)
    #[gbm_aux_sd,[sigma,mu,t0,x0]]
    #(t,x,T,x_pr,pars,crossed=False)
    #trans_st(t,alpha,sigma[0,0])
    x[i+1]=x[i]+drift*dt+Sig_ou_1d(x[i],sigma)*np.sqrt(dt)*np.random.normal(0,1,samples)
    int_G=int_G+dt*((b_ou_1d(x[i],-theta)-b_ou_1d(x[i],0))\
    *r_normal_1d(t,x[i],T,x_pr,[new_trans_st,[alpha,sigma]],crossed=False)\
    #-(1/2)*pff.Sig_ou(x[i],sigma)[:,0]**(2)*(1-(t/T)**(2*alpha))*\
    # (1/trans_st(t,alpha,sigma[0,0])**2-(x[i]-x_pr)**2/trans_st(t,alpha,sigma[0,0])**4))[:,0]
    -(1/2)*(Sig_ou_1d(x[i],sigma)**(2)-Sig_ou_1d(x[i],sigma*(t/T)**(alpha))**(2))*\
    (-H_normal_1d(t,x[i],T,x_pr,[new_trans_st,[alpha,sigma]])\
    -r_normal_1d(t,x[i],T,x_pr,[new_trans_st,[alpha,sigma]],crossed=False)**2))
    if t==t_test:
        print("it's copying them at time ",t)
        x_2=x[i]
    #print(x_pr)
"""
#%%
"""
prop_trans=norm(loc=x0,scale=trans_st(0,alpha,sigma))
aux_trans=norm(loc=x0,scale=trans_st(0,alpha,sigma))
dPdP=np.exp(int_G)*aux_trans.pdf(x_pr)/prop_trans.pdf(x_pr)
print(int_G.shape)
n_points=10
plt.scatter(np.zeros(n_points)+T,x_pr[:n_points])
plt.plot(np.linspace(t0,T,n_steps+1),x[:,:n_points])
#print(x[-2,0   ],x[-1,0]3)
"""
#%%
"""
# mean of the OU process at time t_test
print((x0*np.exp(-theta*t_test)))
print(np.mean(dPdP*x_2))
print(np.mean(dPdP*x_2)/(x0*np.exp(-theta*t_test)))
# second moment of the OU process at time t_test
print((sigma**2*(1-np.exp(-2*theta*(t_test-t0)))/(2*theta)+x0**2*np.exp(-2*theta*(t_test-t0))))
print(np.mean(dPdP*x_2**2))
print(np.mean(dPdP*x_2**2)/(sigma**2*(1-np.exp(-2*theta*(t_test-t0)))/(2*theta)+x0**2*np.exp(-2*theta*(t_test-t0))))
"""

# end of 1D version
#########################################################################################################
#########################################################################################################

#%%
# We need to define the function of the coefficient, and its derivatives. Additionally,
# we need to define the kernel of the auxiliar process, it log and its gradient and hermitian matrix.
# and the gradient of the previous w.r.t. the parameters.
# we need the derivative of the observation likelihood function w.r.t. the parameters.

# OU PROCESS


# TEST FOR Gread_t_b_ou
"""
A=np.array([[2]])
x = np.array([[[1],[2]], [[3],[34]]])
grad_t=Grad_t_b_ou(x,A)
print(grad_t)
"""

#TEST FOR Grad_x_b_ou

"""
A=np.array([[2,3],[4,5]])
x = np.array([[[1,2],[1,20]], [[3,34],[3,3]]])
x=np.array([[1,2]])
print(x.shape)
grad_t=Grad_x_b_ou(x,A)
print(grad_t)
"""


# TEST FOR Grad_x_Sig_ou
"""
sigma=np.array([[2]])
n_pars_s= 1
x=np.array([[[1],[1]],[[1],[5]]])
sigma_pars=[sigma,n_pars_s]
grad_x=Grad_x_Sig_ou(x,sigma)
print(grad_x.shape)
"""


#%%
# DEFINITION OF THE LOG OF THE KERNEL OF THE AUXILIAR PROCESS, AND ITS DIFFERENT DERIVATIVES 
# The kernet in this case is the normal distribution with constant covariance matrix, meaning 
# that it doesn't depend on x_p, as we need in some instances.


# we don't really need the log_p_normal
"""
def log_p_normal(x,x_p,sigma_aux,crossed=False):
    # This is the function for the log of the kernel of the auxiliar process
    # The normal variable taht this function represent is one dimensional
    # ARGUMENTS:
    # ** x_p: is a rank 2 array, with rank (N,dim)
    # ** x: is a rank 2 array with rank (N,dim) 
    # ** This is the value of the auxiliar process at time t0.
    # ** theta: is a ge
    # OUTPUTS:
    # ** log_p is a rank 2 array (N,N)
    # ** respectively, depending on the rank of x.
    if crossed==True:
        log_p=-(x[np.newaxis]-x_p[:,np.newaxis])**2/(2*sigma_aux**2)-np.log(2*np.pi*sigma_aux**2)/2
    else:
        log_p=-(x-x_p)**2/(2*sigma_aux**2)-np.log(2*np.pi*sigma_aux[[0]]**2)/2
    return log_p

"""
#Test for log_p_normal
"""
x_p=np.array([0,2,5])
x=np.array([0,3,6])
sigma_aux=np.array([[2]])
normal_rv=norm(loc=x_p[np.newaxis],scale=sigma_aux[[0]])
densities=normal_rv.pdf(x[:,np.newaxis])
print(densities)
print(np.exp(log_p_normal(x,x_p,sigma_aux)))
"""



#%%

# TEST FOR r_normal
"""
x_p=np.array([0,2])
x_p=x_p[:,np.newaxis]
x=np.array([0,3])
x=x[:,np.newaxis]
alpha=0
sigma=2
t=1
print(t,T)
sd_pars=[alpha,sigma]
pars=[alpha_trans_sd,sd_pars]
#alpha_trans_sd(t,x_pr,sd_pars)
print(r_normal(t,x,T,x_p,pars,crossed=False))
"""
#%%


# TEST FOR H_normal
"""
x_p=np.array([0,2])
x_p=x_p[:,np.newaxis]
x=np.array([0,3])
x=x[:,np.newaxis]
alpha=0
sigma=2
t=1
print(t,T)
sd_pars=[alpha,sigma]
pars=[new_alpha_trans_sd,sd_pars]
#alpha_trans_sd(t,x_pr,sd_pars)
print(H_normal(t,x,T,x_p,pars,crossed=True))



"""

#%%
# In the following we define the standard deviation of the gaussian process obtained by having 
# a process with diffusion term linear in time, and constant in terms of x, and a drift term linear in
# terms of x. 
"""
x_p=np.array([-4,2])
x_p=x_p[:,np.newaxis]
x0=np.array([1,3])
x0=x0[:,np.newaxis]
print(x_p)
mu=np.array([[0.2]])
sigma=np.array([[0.2]])
print(theta)
T=2
t=1
print(t,T)
x=x0-x0
print(gbm_aux_sd(t,x,T,x_p,[sigma,mu,0,x0]))
"""

#%%
# TEST FOR THE 1D VERSION
#########################################################################################################
"""
x_p=np.array([-4,2])
x0=np.array([1,3])
print(x_p)
mu=np.array(0.2)
sigma=np.array(0.2)
print(theta)
T=2
t=1
print(t,T)
x=x0-x0
print(gbm_aux_sd_1d(t,x,T,x_p,[sigma,mu,0,x0]))
"""
# END OF TEST FOR THE 1D VERSION
#########################################################################################################


#%%
# TEST FOR THE BRIDGE FUNCTION USING GBM

# This test is designed to be constructed in the most general way, being able to use it for
# the whole project, we will be using a proposal transition that has the same for as the 
# auxiliar transition. 

# DEFINITION OF PARAMETERS
"""
x0=np.array([1.2])
T=3
t0=0
mu=np.array([[.2]])
sigma=np.array([[0.1]])

# The SDE has the form dX_t=mu*X_t*dt+sigma*X_t*dW_t 

sigma_prop=np.array([[0.3]])
fi=[sigma,np.array([[1]])]
# COMPUTATION

np.random.seed(14)
l=5
n_steps=2**l*(T-t0)
dt=2**(-l)
samples=500000
x=np.zeros((n_steps+1,samples,1))
x[0]=x0
x_pr=np.random.lognormal(np.log(x0)+(mu-sigma_prop**2/2)*(T-t0),sigma_prop*np.sqrt(T-t0),(samples,1))
x[-1]=x_pr
int_G=np.zeros(samples)
#t_test=T-2*dt
t_test=2
#gbm_aux_sd(t,x,T,x_pr,sigma,b)
#[sd,sd_pars]=pars
#r_normal(t,x[0],T,x_pr,[gbm_aux_sd,[sigma,mu]],crossed=False)
"""
#%%
"""
for i in range(n_steps-1):
    t=t0+(i)*dt
    r_n=r_quasi_normal(t,x[i],T,x_pr,[gbm_aux_sd,[sigma,mu,t0,x0]],crossed=False)
    #r_n=r_normal(t,x[i],T,x_pr,[brow_aux_sd,sigma],crossed=False)
    a_tilde=(pff.Sig_gbm(x_pr,fi)[:,0]*(t-t0)/((T-t0))\
    +pff.Sig_gbm(x[0],fi)[:,0]*(T-t)/(T-t0))**2
    if i==1:
        print("a_tilde is: ",a_tilde.shape)
    drift=pff.b_gbm(x[i],mu)+pff.Sig_gbm(x[i] ,fi)[:,0]**(2)\
    *r_n
    #print((pff.Sig_gbm(x[i],fi)[:,0]).shape)
    x[i+1]=x[i]+drift*dt+pff.Sig_gbm(x[i],fi)[:,0]*np.sqrt(dt)*np.random.normal(0,1,(samples,1))
    int_G=int_G+dt*((pff.b_gbm(x[i],mu)-pff.b_gbm(x[i],mu))*r_n\
    #-(1/2)*pff.Sig_ou(x[i],sigma)[:,0]**(2)*(1-(t/T)**(2*alpha))*\
    # (1/trans_st(t,alpha,sigma[0,0])**2-(x[i]-x_pr)**2/trans_st(t,alpha,sigma[0,0])**4))[:,0]
    -(1/2)*(pff.Sig_gbm(x[i],fi)[:,0]**(2)-a_tilde)*\
    (-H_quasi_normal(t,x[i],T,x_pr,[gbm_aux_sd,[sigma,mu,t0,x0],mu])\
    -r_n**2))[:,0]
    if t==t_test:
        print("it's copying them at time ",t)
        x_2=x[i]
"""
#%%
"""
prop_trans_den=(np.exp(-(np.log(x_pr)-np.log(x0)-(mu-sigma_prop**2/2)*(T-t0))**2/(2*sigma_prop**2*(T-t0)))\
/(np.sqrt(2*np.pi)*sigma_prop*np.sqrt(T-t0)*x_pr))[:,0]
aux_trans_den=norm(loc=x0*np.exp(mu*(T-t0)),scale=gbm_aux_sd(t0,x0,T,x_pr,[sigma,mu,t0,x0])[:,0])
dPdP=np.exp(int_G)*aux_trans_den.pdf(x_pr[:,0])[0]/prop_trans_den
n_points=10
plt.scatter(np.zeros(n_points)+T,x_pr[:n_points,0])
plt.plot(np.linspace(t0,T,n_steps+1),x[:,:n_points,0])
"""#print(x[-2,0   ],x[-1,0]3)
#%%
"""
print((x0*np.exp(mu*(t_test-t0))))
print(np.mean(dPdP*x_2[:,0]))
print(np.mean(dPdP*x_2[:,0])/(x0*np.exp(mu*(t_test-t0))))
# second moment of the OU process at time t_test
print(  x0**2*np.exp(2*mu*(t_test-t0))*(np.exp(sigma**2*(t_test-t0))))
print(np.mean(dPdP*x_2[:,0]**2))
print(np.mean(dPdP*x_2[:,0]**2)/(x0**2*np.exp(2*mu*(t_test-t0))*(np.exp(sigma**2*(t_test-t0)))))
"""
#%%
# RESULTS OF THE TEST: The test was carried out by defining a general notation that includes
# as arguments (t,x,T,x_p,pars), that includes the current time and state, the final time and state 
# and some additional parameters called pars which are specific to each function. 
# The process used was GBM, we tested the general functions, the auxiliar process depending on the final
# time, and the importance sampling of the transition proposal. The test were also carried out for singular(in time)
# values of the state. The results were positive in all cases. Further tests can be carried out to check the weak and 
# strong error of the approximations as the monte carlo rate of the process. 

#%%
# TEST FOR THE 1D VERSION
#########################################################################################################

# DEFINITION OF PARAMETERS
"""
x0=np.array(1.2)
T=4
t0=0
mu=np.array(.2)
sigma=np.array(0.1)

# The SDE has the form dX_t=mu*X_t*dt+sigma*X_t*dW_t 

sigma_prop=np.array(0.3)
fi=sigma
# COMPUTATION

np.random.seed(14)
l=7
n_steps=2**l*(T-t0)
dt=2**(-l)
samples=5000000
x=np.zeros((n_steps+1,samples))
x[0]=x0
x_pr=np.random.lognormal(np.log(x0)+(mu-sigma_prop**2/2)*(T-t0),sigma_prop*np.sqrt(T-t0),(samples))
x[-1]=x_pr
int_G=np.zeros(samples)
t_test=T-2*dt
theta=mu
"""
#gbm_aux_sd(t,x,T,x_pr,sigma,b)
#[sd,sd_pars]=pars
#r_normal(t,x[0],T,x_pr,[gbm_aux_sd,[sigma,mu]],crossed=False)

#%%
"""
for i in range(n_steps-1):
    t=t0+(i)*dt
    r_n=r_quasi_normal_1d(t,x[i],T,x_pr,[gbm_aux_sd_1d,[sigma,mu,t0,x0]])
    #r_n=r_normal(t,x[i],T,x_pr,[brow_aux_sd,sigma],crossed=False)
    a_tilde=(Sig_gbm_1d(x_pr,fi)*(t-t0)/((T-t0))\
    +Sig_gbm_1d(x[0],fi)*(T-t)/(T-t0))**2
    drift=b_gbm_1d(x[i],mu)+Sig_gbm_1d(x[i] ,fi)**(2)\
    *r_n
    #print((pff.Sig_gbm(x[i],fi)[:,0]).shape)
    x[i+1]=x[i]+drift*dt+Sig_gbm_1d(x[i],fi)*np.sqrt(dt)*np.random.normal(0,1,(samples))
    int_G=int_G+dt*((b_gbm_1d(x[i],mu)-b_gbm_1d(x[i],mu))*r_n\
    #-(1/2)*pff.Sig_ou(x[i],sigma)[:,0]**(2)*(1-(t/T)**(2*alpha))*\
    # (1/trans_st(t,alpha,sigma[0,0])**2-(x[i]-x_pr)**2/trans_st(t,alpha,sigma[0,0])**4))[:,0]
    -(1/2)*(Sig_gbm_1d(x[i],fi)**(2)-a_tilde)*\
    (-H_quasi_normal(t,x[i],T,x_pr,[gbm_aux_sd_1d,[sigma,mu,t0,x0],mu])\
    -r_n**2))
    if t==t_test:
        print("it's copying them at time ",t)
        x_2=x[i]
"""
#%%
"""
prop_trans_den=(np.exp(-(np.log(x_pr)-np.log(x0)-(mu-sigma_prop**2/2)*(T-t0))**2/(2*sigma_prop**2*(T-t0)))\
/(np.sqrt(2*np.pi)*sigma_prop*np.sqrt(T-t0)*x_pr))
aux_trans_den=norm(loc=x0*np.exp(mu*(T-t0)),scale=gbm_aux_sd_1d(t0,x0,T,x_pr,[sigma,mu,t0,x0]))
dPdP=np.exp(int_G)*aux_trans_den.pdf(x_pr)/prop_trans_den
n_points=10
plt.scatter(np.zeros(n_points)+T,x_pr[:n_points])
plt.plot(np.linspace(t0,T,n_steps+1),x[:,:n_points])
"""
#print(x[-2,0   ],x[-1,0]3)
#%%
"""
print((x0*np.exp(mu*(t_test-t0))))
print(np.mean(dPdP*x_2))
print(np.mean(dPdP*x_2)/(x0*np.exp(mu*(t_test-t0))))
# second moment of the OU process at time t_test
print(  x0**2*np.exp(2*mu*(t_test-t0))*(np.exp(sigma**2*(t_test-t0))))
print(np.mean(dPdP*x_2**2))
print(np.mean(dPdP*x_2**2)/(x0**2*np.exp(2*mu*(t_test-t0))*(np.exp(sigma**2*(t_test-t0)))))

"""
# END OF THE TEST FOR THE 1D VERSION
#########################################################################################################
#%%


# PARTICLE FILTER.

# In this case we need new structures:
# - Generator of observations: check
# - Generator of realizations: check
# - Likelihood of individual observtions: check
# - Function that samples the paths given the initial 
#   and final values of the diffusion: in construction



# TESTS: One of the tests can be the comparison of the PF
# of the previous project (with regular timed observations)
# or in the case of OU and Gaussian observations we can check with
# KF. 


# Sampling form the 



# TEST FOR g_normal
"""
x=np.array([[0],[1],[2]])
cov=np.array([[1]])
print(g_normal(x,cov))
"""


#Test for the Bridge function

# DEFINITION OF PARAMETERS
"""
x0=np.array([1.2])
T=2
t0=0
mu=np.array([[.2]])
sigma=np.array([[0.1]])
# The SDE has the form dX_t=mu*X_t*dt+sigma*X_t*dW_t 
sigma_prop=np.array([[0.3]])
fi=[sigma,np.array([[1]])]
# COMPUTATION
np.random.seed(14)
l=5
N=50000
x_pr=np.random.lognormal(np.log(x0)+(mu-sigma_prop**2/2)*(T-t0),sigma_prop*np.sqrt(T-t0),(N,1))
t_test=1
d=T-t0
n_steps=2**l*d

dim=1
"""
#%%
"""
prop_trans_den=(np.exp(-(np.log(x_pr)-np.log(x0)-(mu-sigma_prop**2/2)*(T-t0))**2/(2*sigma_prop**2*(T-t0)))\
/(np.sqrt(2*np.pi)*sigma_prop*np.sqrt(T-t0)*x_pr))[:,0]
aux_trans_den=norm(loc=x0*np.exp(mu*(T-t0)),scale=gbm_aux_sd(t0,x0,T,x_pr,[sigma,mu,t0,x0])[:,0])
dPdP=np.exp(int_G)*aux_trans_den.pdf(x_pr[:,0])[0]/prop_trans_den
n_points=10
plt.scatter(np.zeros(n_points)+T,x_pr[:n_points,0])
plt.plot(np.linspace(t0,T,n_steps+1),x[:,:n_points,0])
"""
#print(x[-2,0   ],x[-1,0]3)
#%%
"""
print((x0*np.exp(mu*(t_test-t0))))
print(np.mean(dPdP*x_test[:,0]))
print(np.mean(dPdP*x_test[:,0])/(x0*np.exp(mu*(t_test-t0))))
# second moment of the OU process at time t_test
print(  x0**2*np.exp(2*mu*(t_test-t0))*(np.exp(sigma**2*(t_test-t0))))
print(np.mean(dPdP*x_test[:,0]**2))
print(np.mean(dPdP*x_test[:,0]**2)/(x0**2*np.exp(2*mu*(t_test-t0))*(np.exp(sigma**2*(t_test-t0)))))
"""
#%%
# We want to create a 1D version of this algorithm, so it is much more easy to compute and we don't have to worry
# about dimensions that we were not going to compute anyway. 

# TEST FOR  THE 1D VERSION
####################v###########################################################################################
####################v###########################################################################################


#Test for the Bridge function
"""
# DEFINITION OF PARAMETERS
x0=np.array(1.2)
T=2
t0=0
mu=np.array(.2)
sigma=np.array(0.1)
# The SDE has the form dX_t=mu*X_t*dt+sigma_t*dW_t 
sigma_prop=np.array(0.3)
fi=sigma
# COMPUTATION
np.random.seed(14)
l=5
N=500

x_pr=np.random.lognormal(np.log(x0)+(mu-sigma_prop**2/2)*(T-t0),sigma_prop*np.sqrt(T-t0),N)
t_test=1
d=T-t0
n_steps=2**l*d
"""
#%%
"""
prop_trans_den=(np.exp(-(np.log(x_pr)-np.log(x0)-(mu-sigma_prop**2/2)*(T-t0))**2/(2*sigma_prop**2*(T-t0)))\
/(np.sqrt(2*np.pi)*sigma_prop*np.sqrt(T-t0)*x_pr))
aux_trans_den=norm(loc=x0*np.exp(mu*(T-t0)),scale=gbm_aux_sd_1d(t0,x0,T,x_pr,[sigma,mu,t0,x0]))
dPdP=np.exp(int_G)*aux_trans_den.pdf(x_pr)/prop_trans_den
n_points=10
plt.scatter(np.zeros(n_points)+T,x_pr[:n_points])
plt.plot(np.linspace(t0,T,n_steps+1),x[:,:n_points])
"""
#print(x[-2,0   ],x[-1,0]3)
#%%
"""
print((x0*np.exp(mu*(t_test-t0))))
print(np.mean(dPdP*x_test))
print(np.mean(dPdP*x_test)/(x0*np.exp(mu*(t_test-t0))))
# second moment of the OU process at time t_test
print(  x0**2*np.exp(2*mu*(t_test-t0))*(np.exp(sigma**2*(t_test-t0))))
print(np.mean(dPdP*x_test**2))
print(np.mean(dPdP*x_test**2)/(x0**2*np.exp(2*mu*(t_test-t0))*(np.exp(sigma**2*(t_test-t0)))))
"""

# END OF TEST FOR  THE 1D VERSION
###############################################################################################################
###############################################################################################################



#%%
#  we need to desing functions that sample the jump, what arguments do we need generaly? 

# PARTICLE FILTER FUNCTION

def PF_bridge(t0,x0,T,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,\
    sample_funct,sample_pars,obs,log_g_den,g_den_par, aux_trans_den,atdp,\
    prop_trans_den, resamp_coef, l, d,N,seed,crossed=False):
    #(T,xin,b_ou,A,Sig_ou,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par):
    
    
    
    # ARGUMENTS: the argument of the Kenel x0 rank 1 dims (dim) 
    # the drift and diffusion are b and Sig, respectively, and they take
    # x(either a (N) dimensional or (N,N) dimensional array) and A as arguments for the drift and x and fi for the diffusion.
    # the level of discretization l, the distance of resampling, the number of
    # particles N.
    # Grad_b is a function that takes (x,A) as argument and computes the gradnient of b wrt the 
    # parameters A, and evaluates it a (x,A).
    # b_til,A_til,Sig_til,fi_til, are the analogous functions for the auxiliar process.
    # a difference is that their arguments are (t,x) for the drift and (t,x,fi_til) for the diffusion.
    # r is the function that computes the gradient of the log of the kernel of the auxiliar process
    # and it takes (t,x,T,x_pr,r_pars) as arguments.
    # H is the function that computes the Hessian of the log of the kernel of the auxiliar process
    # and it takes (t,x,T,x_pr,H_pars) as arguments.
    # crossed is the boolean that indicates if we need the computations for the crossed terms that 
    # are needed for the smoother.

    log_weights=np.zeros((int(T/d),N))
    x_pr=np.zeros((int(T/d),N))
    int_Gs=np.zeros((int(T/d),N))                      
    x_new=x0
    for i in range(int(T/d)):

        tf=t0+(i+1)*d
        ti=t0+(i)*d
        x_pr[i]=sample_funct(x_new,N,d,sample_pars)
        # what parameters do we need in order to make the auxiliar density general?
        # x_new,  d, t, x_pr,tf
        # aux_trans_den(t0,x0,T,x_pr,atdp)
        # atdp stands for auxiliar transition density parameters. 
        int_G=Bridge_1d(ti,x_new,tf,x_pr[i],b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,\
        r,r_pars,H,H_pars,l,d,N, seed+i*int(int(2**l*d-1)),crossed=False)
        int_Gs[i]=int_G
        #print(xi.shape)
        #print(yi,obti-i*d,)
        #print(x_new,xi)
        #print(xi)
        #Things that could be wrong
        # observations, x_new, weights
        #observations seem to be fine

        #xs[2**l*d*i:2**l*d*(i+1)]=x[:-1]

        #print("other parameteres are:",ti,x_new,tf,x_pr[i] )
        #print("atdp is ", atdp)
        #print("object is: ", aux_trans_den(ti,x_new,tf,x_pr[i],atdp))
        log_weights[i]=int_G+log_g_den(obs[i],x_pr[i],g_den_par,crossed=crossed)\
        +np.log(aux_trans_den(ti,x_new,tf,x_pr[i],atdp,crossed=crossed))-np.log(prop_trans_den(ti,x_new,tf,x_pr[i],sample_pars,crossed=crossed))
        weights=pff.norm_logweights(log_weights[i])
        #print(yi,weights)
        #seed_val=i
        #print(weights.shape)
        x_last=x_pr[i]
        ESS=0
        #print(ESS,resamp_coef*N)
        if ESS<resamp_coef*N:
            #print("resampling at time ",i)
        #if True==False:
            #[part0,part1,x0_new,x1_new]=max_coup_sr(w0,w1,N,xi0[-1],xi1[-1],dim)
            #print(x_new.shape)
            
            [part_resamp, x_new]=multi_samp_exp(weights,N,x_last,1)
            log_weights[:i+1]=log_weights[:i+1,part_resamp]
            int_Gs[:i+1]=int_Gs[:i+1,part_resamp]
            x_pr[:i+1]=x_pr[:i+1,part_resamp]
                
            #x_new=multi_samp_exp(weights,N,x_last,1)[1]
            #print(x_new.shape)
        else:
            #print("time is",i)
            x_new=x_last
            if i< int(T/d)-1:
                log_weights[i+1]=log_weights[i]
        #print(i)
        
       #x_new=sr(weights,N,x_pf[i],dim)[1]
    #weights=np.reshape(norm_logweights(log_weights,ax=1),(int(T/d),N,1))
    #pf=np.sum(weights*x_pf,axis=1)
    #Filter
    #spots=np.arange(d_steps,2**l*T+1,d_steps,dtype=int)
    #x_pf=x[spots]
    #weights=norm_logweights(log_weights,ax=1)

    #print(x_pf.shape,weights.shape)
    #suma=np.sum(x_pf[:,:,1]*weights,axis=1)
    return [log_weights,int_Gs,x_pr]




def Gen_PF_bridge(in_dist,in_dist_pars, b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,\
    update_func,sample_funct,sample_pars,obs,obs_times,log_g_den,g_den_par, aux_trans_den,atdp,\
    prop_trans_den, resamp_coef, l, d,N,seed): 
    
    # ARGUMENTS: 
    # Intead of T(as in the function PF_bidge) we use the lenght of the observations obs_time. 
    # the argument of the Kenel x0 rank 1 dims (dim) 
    # the drift and diffusion are b and Sig, respectively, and they take
    # x(either a (N) dimensional or (N,N) dimensional array) and A as arguments for the drift and x and fi for the diffusion.
    # the level of discretization l, the distance of resampling, the number of
    # particles N.
    # Grad_b is a function that takes (x,A) as argument and computes the gradnient of b wrt the 
    # parameters A, and evaluates it a (x,A).
    # b_til,A_til,Sig_til,fi_til, are the analogous functions for the auxiliar process.
    # a difference is that their arguments are (t,x) for the drift and (t,x,fi_til) for the diffusion.
    # r is the function that computes the gradient of the log of the kernel of the auxiliar process
    # and it takes (t,x,T,x_pr,r_pars) as arguments.
    # H is the function that computes the Hessian of the log of the kernel of the auxiliar process
    # and it takes (t,x,T,x_pr,H_pars) as arguments.
    # crossed is the boolean that indicates if we need the computations for the crossed terms that 
    # are needed for the smoother.
    T=len(obs_times)*d # Here we change this in order to simplify the changes that we make to 
    # the funciton PF_bridge in order to get the current function.
    log_weights=np.zeros((int(T/d),N))
    x_pr=np.zeros((int(T/d),N))
    int_Gs=np.zeros((int(T/d),N))
    indices=np.zeros((int(T/d)-1,N))
    #sampling   
    np.random.seed(seed)                   
    x_new=in_dist(in_dist_pars,N)
    x_pr[0]= copy.deepcopy(x_new)
    log_weights[0]=log_g_den(obs[0],x_pr[0],g_den_par)
    # resampling
    weights=pff.norm_logweights(log_weights[0])
    [part_resamp, x_new]=multi_samp_exp(weights,N,x_new,1)
    log_weights[0]=log_weights[0,part_resamp]
    #int_Gs[0]=int_Gs[0,part_resamp]
    x_pr[0]=x_pr[0,part_resamp]

    # here we make the first step which is different from the rest.
    Deltas=get_deltas(obs_times)
    intervals=obs_times[1:]-obs_times[:-1]
    
    for i in range(int(T/d)-1):
        tf=obs_times[i+1]
        ti=obs_times[i]
        np.random.seed(seed+(i+1)*int(2**l-1))
        x_pr[i+1]=sample_funct(x_new,N,tf-ti,sample_pars)
        pars_0=[A,fi]
        [A_til,fi_til,r_pars,H_pars,\
        atdp]=update_func(pars_0,pars_0,ti,x_new,x_new,tf,x_pr[i+1],x_pr[i+1],levels=1)
        # what parameters do we need in order to make the auxiliar density general?
        # x_new,  d, t, x_pr,tf
        # aux_trans_den(t0,x0,T,x_pr,atdp)
        # atdp stands for auxiliar transition density parameters. 
        # Here we need to update A_til and fi_til to consider both x0 and x_pr
        int_G=Bridge_1d(ti,x_new,tf,x_pr[i+1],b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,\
        r,r_pars,H,H_pars,l,d,N, seed,crossed=False,Dt=Deltas[i]/2**l)
        int_Gs[i]=int_G
        seed+=int(2**l*intervals[i]/Deltas[i]-1)
        #print(xi.shape)
        #print(yi,obti-i*d,)
        #print(x_new,xi)
        #print(xi)
        #Things that could be wrong
        # observations, x_new, weights
        #observations seem to be fine
        #xs[2**l*d*i:2**l*d*(i+1)]=x[:-1]

        #print("other parameteres are:",ti,x_new,tf,x_pr[i] )
        #print("atdp is ", atdp)
        #print("object is: ", aux_trans_den(ti,x_new,tf,x_pr[i],atdp))
        log_weights[i+1]=int_G+log_g_den(obs[i+1],x_pr[i+1],g_den_par)\
        +np.log(aux_trans_den(ti,x_new,tf,x_pr[i+1],atdp))-np.log(prop_trans_den(ti,x_new,tf,x_pr[i+1],sample_pars))
        weights=pff.norm_logweights(log_weights[i+1])
        #print(yi,weights)
        #seed_val=i
        #print(weights.shape)
        x_last=x_pr[i+1]
        ESS=0
        #print(ESS,resamp_coef*N)
        if ESS<resamp_coef*N:
            #print("resampling at time ",i)
        #if True==False:
            #[part0,part1,x0_new,x1_new]=max_coup_sr(w0,w1,N,xi0[-1],xi1[-1],dim)
            #print(x_new.shape)
            indices[i]=np.copy(np.arange(N))
            [part_resamp, x_new]=multi_samp_exp(weights,N,x_last,1)
            log_weights[:i+2]=log_weights[:i+2,part_resamp]
            int_Gs[:i+2]=int_Gs[:i+2,part_resamp]
            x_pr[:i+2]=x_pr[:i+2,part_resamp]
            indices[:i+1]=indices[:i+1,part_resamp]

        else:
            #print("time is",i)
            x_new=x_last
            if i< int(T/d)-1:
                log_weights[i+1]=log_weights[i]
        #print(i)
        
       #x_new=sr(weights,N,x_pf[i],dim)[1]
    #weights=np.reshape(norm_logweights(log_weights,ax=1),(int(T/d),N,1))
    #pf=np.sum(weights*x_pf,axis=1)
    #Filter
    #spots=np.arange(d_steps,2**l*T+1,d_steps,dtype=int)
    #x_pf=x[spots]
    #weights=norm_logweights(log_weights,ax=1)

    #print(x_pf.shape,weights.shape)
    #suma=np.sum(x_pf[:,:,1]*weights,axis=1)
    return [log_weights,int_Gs,x_pr,indices]


def C_PF_bridge(t0,x0,T,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,\
    max_sample_funct,sample_pars,obs,log_g_den,g_den_par, aux_trans_den,atdp,\
    prop_trans_den,ind_prop_trans_par, resamp_coef, l, d,N,seed,crossed=False):
    #(T,xin,b_ou,A,Sig_ou,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par):
    
    # ARGUMENTS: the argument of the Kenel x0 rank 1 dims (dim) 
    # the drift and diffusion are b and Sig, respectively, and they take
    # x(either a (N) dimensional or (N,N) dimensional array) and A as arguments for the drift and x and fi for the diffusion.
    # the level of discretization l, the distance of resampling, the number of
    # particles N.
    # Grad_b is a function that takes (x,A) as argument and computes the gradnient of b wrt the 
    # parameters A, and evaluates it a (x,A).
    # b_til,A_til,Sig_til,fi_til, are the analogous functions for the auxiliar process.
    # a difference is that their arguments are (t,x) for the drift and (t,x,fi_til) for the diffusion.
    # r is the function that computes the gradient of the log of the kernel of the auxiliar process
    # and it takes (t,x,T,x_pr,r_pars) as arguments.
    # H is the function that computes the Hessian of the log of the kernel of the auxiliar process
    # and it takes (t,x,T,x_pr,H_pars) as arguments.
    # crossed is the boolean that indicates if we need the computations for the crossed terms that 
    # are needed for the smoother.
    # ind_prop_trans_par is created bcs sample_pars cannot be used by prop_trans_den anymore since it 
    # has max_sample_funct deals with couplings 
    # it's worth noting that the function max_sample_funct is a function that samples from the
    # proposal transition, it is labeled like that to obtain the maximum coupling of the proposal   

    log_weights_0=np.zeros((int(T/d),N))
    log_weights_1=np.zeros((int(T/d),N))
    x_pr_0=np.zeros((int(T/d),N))
    x_pr_1=np.zeros((int(T/d),N))
    #xs=np.zeros((int(T*2**l*d),N))
    int_Gs_0=np.zeros((int(T/d),N))                      
    int_Gs_1=np.zeros((int(T/d),N))                      
    x_new_0=x0
    seeds_0_wp=np.zeros((int(T/d),2,N),dtype=int)
    seeds_1_wp=np.zeros((int(T/d),2,N),dtype=int)
    x_new_1=x0
    for i in range(int(T/d)):
        tf=t0+(i+1)*d
        ti=t0+(i)*d
        x_pr_0[i],x_pr_1[i]=max_sample_funct(x_new_0,x_new_1,N,d,sample_pars)
        # what parameters do we need in order to make the auxiliar density general?
        # x_new,  d, t, x_pr,tf
        # aux_trans_den(t0,x0,T,x_pr,atdp)
        # atdp stands for auxiliar transition density parameters. 
        seeds_0_wp[i,1,:]=np.copy(np.arange(N))
        seeds_1_wp[i,1,:]=np.copy(np.arange(N))
        seeds_0_wp[i,0,:]=seed+i*int(2**l*d-1)
        seeds_1_wp[i,0,:]=seed+i*int(2**l*d-1)
        """
        (t0,x0_0,x0_1,T,x_p_0,x_p_1,b,A_0,A_1,Sig,fi_0,fi_1,b_til,A_til_0,A_til_1,\
        Sig_til,fi_til_0,fi_til_1,r,r_pars_0,r_pars_1,H,H_pars_0,H_pars_1,l,d,N,seed\
        ,crossed=False,backward=False,j_0=False,j_1=False,fd=False,N_pf=False,cond_seed_0=False,cond_seed_1=False):
        """
        int_G_0,int_G_1=C_Bridge_1d(ti,x_new_0,x_new_1,tf,x_pr_0[i],x_pr_1[i],b,A,A,Sig,fi,fi,b_til,A_til,A_til,\
        Sig_til,fi_til,fi_til,r,r_pars,r_pars,H,H_pars,H_pars,l,d,N,seed+i*int(2**l*d-1))
        #int_G=Bridge_1d(ti,x_new,tf,x_pr[i],b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,\
        #r,r_pars,H,H_pars,l,d,N, seed+i*int(int(2**l*d-1)),crossed=False)
        #int_Gs[i]=int_G
        #print(xi.shape)
        #print(yi,obti-i*d,)
        #print(x_new,xi)
        #print(xi)
        #Things that could be wrong
        # observations, x_new, weights
        #observations seem to be fine

        #xs[2**l*d*i:2**l*d*(i+1)]=x[:-1]

        #print("other parameteres are:",ti,x_new,tf,x_pr[i] )
        #print("atdp is ", atdp)
        #print("object is: ", aux_trans_den(ti,x_new,tf,x_pr[i],atdp))

        log_weights_0[i]=int_G_0+log_g_den(obs[i],x_pr_0[i],g_den_par,crossed=crossed)\
        +np.log(aux_trans_den(ti,x_new_0,tf,x_pr_0[i],atdp,crossed=crossed))-np.log(prop_trans_den(ti,x_new_0,tf,x_pr_0[i],ind_prop_trans_par,crossed=crossed))
        weights_0=pff.norm_logweights(log_weights_0[i])
        
        log_weights_1[i]=int_G_1+log_g_den(obs[i],x_pr_1[i],g_den_par,crossed=crossed)\
        +np.log(aux_trans_den(ti,x_new_1,tf,x_pr_1[i],atdp,crossed=crossed))-np.log(prop_trans_den(ti,x_new_1,tf,x_pr_1[i],ind_prop_trans_par,crossed=crossed))
        weights_1=pff.norm_logweights(log_weights_1[i])

        """ 
        log_weights[i]=log_weights[i]+int_G+log_g_den(obs[i],x_pr[i],g_den_par,crossed=crossed)\
        +np.log(aux_trans_den(ti,x_new,tf,x_pr[i],atdp,crossed=crossed))-np.log(prop_trans_den(ti,x_new,tf,x_pr[i],sample_pars,crossed=crossed))
        weights=pff.norm_logweights(log_weights[i])
        """
        #print(yi,weights)
        #seed_val=i 
        #print(weights.shape)
        x_last_0=x_pr_0[i]
        x_last_1=x_pr_1[i]
        
        #ESS=1/np.sum(weights**2)
        #ESS=1/np.sum(weights**2)
        
        """
        [part0,part1,x0_new,x1_new]=max_coup_multi(w0,w1,N,xi0[-1],xi1[-1],dim)
        """
        [part_resamp_0,part_resamp_1, x_new_0,x_new_1]=max_coup_multi\
        (weights_0,weights_1,N,x_last_0,x_last_1,1)
        #print(part_resamp_0,part_resamp_1)
        log_weights_0[:i+1]=log_weights_0[:i+1,part_resamp_0]
        int_Gs_0[:i+1]=int_Gs_0[:i+1,part_resamp_0]
        x_pr_0[:i+1]=x_pr_0[:i+1,part_resamp_0]
        log_weights_1[:i+1]=log_weights_1[:i+1,part_resamp_1]
        int_Gs_1[:i+1]=int_Gs_1[:i+1,part_resamp_1]
        x_pr_1[:i+1]=x_pr_1[:i+1,part_resamp_1]

        seeds_0_wp[:i+1]=np.copy(seeds_0_wp[:i+1,:,part_resamp_0])
        seeds_1_wp[:i+1]=np.copy(seeds_1_wp[:i+1,:,part_resamp_1])
        
        #x_new=multi_samp_exp(weights,N,x_last,1)[1]
        #print(x_new.shape)
        
        
       #x_new=sr(weights,N,x_pf[i],dim)[1]
    #weights=np.reshape(norm_logweights(log_weights,ax=1),(int(T/d),N,1))
    #pf=np.sum(weights*x_pf,axis=1)
    #Filter
    #spots=np.arange(d_steps,2**l*T+1,d_steps,dtype=int)
    #x_pf=x[spots]
    #weights=norm_logweights(log_weights,ax=1)

    #print(x_pf.shape,weights.shape)
    #suma=np.sum(x_pf[:,:,1]*weights,axis=1)
    return [log_weights_0,log_weights_1,int_Gs_0,int_Gs_1,x_pr_0,x_pr_1,seeds_0_wp,seeds_1_wp]

def Gen_C_PF_bridge(in_dist,in_dist_pars, b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,\
    update_func,max_sample_funct,sample_pars,obs,obs_times,log_g_den,g_den_par, aux_trans_den,atdp,\
    prop_trans_den, ind_prop_trans_par,resamp_coef, l, d,N,seed):

    #(t0,x0,T,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,\
    #max_sample_funct,sample_pars,obs,log_g_den,g_den_par, aux_trans_den,atdp,\
    #prop_trans_den,ind_prop_trans_par, resamp_coef, l, d,N,seed,crossed=False):
    
    # ARGUMENTS: 

    T=len(obs_times)*d
    log_weights_0=np.zeros((int(T/d),N))
    log_weights_1=np.zeros((int(T/d),N))

    x_pr_0=np.zeros((int(T/d),N))
    x_pr_1=np.zeros((int(T/d),N))

    int_Gs_0=np.zeros((int(T/d)-1,N))                      
    int_Gs_1=np.zeros((int(T/d)-1,N))
    
    np.random.seed(seed)                   
    x_new_0,x_new_1=in_dist(in_dist_pars,N)
    x_pr_0[0],x_pr_1[0]= copy.deepcopy(x_new_0),copy.deepcopy(x_new_1)
    log_weights_0[0]=log_g_den(obs[0],x_pr_0[0],g_den_par)
    log_weights_1[0]=log_g_den(obs[0],x_pr_1[0],g_den_par)
    # resampling
    weights_0=pff.norm_logweights(log_weights_0[0])
    weights_1=pff.norm_logweights(log_weights_1[0])
    [part_resamp_0,part_resamp_1, x_new_0,x_new_1]=max_coup_multi\
        (weights_0,weights_1,N,x_new_0,x_new_1,1)
    log_weights_0[0]=log_weights_0[0,part_resamp_0]
    log_weights_1[0]=log_weights_1[0,part_resamp_1]
    #int_Gs[0]=int_Gs[0,part_resamp]
    x_pr_0[0]=x_pr_0[0,part_resamp_0]

    x_pr_1[0]= x_pr_1[0,part_resamp_1]

    seeds_0_wp=np.zeros((int(T/d)-1,2,N),dtype=int)
    seeds_1_wp=np.zeros((int(T/d)-1,2,N),dtype=int)


    Deltas=get_deltas(obs_times)
    intervals=obs_times[1:]-obs_times[:-1]
    
    for i in range(int(T/d)-1):
        tf=obs_times[i+1]
        ti=obs_times[i]
        x_pr_0[i+1],x_pr_1[i+1]=max_sample_funct(x_new_0,x_new_1,N,tf-ti,sample_pars)
        np.random.seed(seed)
        pars_0=[A,fi]
        pars_1=[A,fi]
        [A_til_0,A_til_1,fi_til_0,fi_til_1,r_pars_0,r_pars_1,H_pars_0,H_pars_1,\
         atdp_0,atdp_1]=update_func(pars_0,pars_1,ti,x_new_0,x_new_1,tf,x_pr_0[i+1],x_pr_1[i+1],levels=2)
        #[A_til,fi_til,r_pars,H_pars,\
        #atdp]=update_func(pars_0,pars_0,ti,x_new,x_new,tf,x_pr[i+1],x_pr[i+1],levels=1)

        # what parameters do we need in order to make the auxiliar density general?
        # x_new,  d, t, x_pr,tf
        # aux_trans_den(t0,x0,T,x_pr,atdp)
        # atdp stands for auxiliar transition density parameters. 
        seeds_0_wp[i,1,:]=np.copy(np.arange(N))
        seeds_1_wp[i,1,:]=np.copy(np.arange(N))
        seeds_0_wp[i,0,:]=seed
        seeds_1_wp[i,0,:]=seed
        """
        (t0,x0_0,x0_1,T,x_p_0,x_p_1,b,A_0,A_1,Sig,fi_0,fi_1,b_til,A_til_0,A_til_1,\
        Sig_til,fi_til_0,fi_til_1,r,r_pars_0,r_pars_1,H,H_pars_0,H_pars_1,l,d,N,seed\
        ,crossed=False,backward=False,j_0=False,j_1=False,fd=False,N_pf=False,cond_seed_0=False,cond_seed_1=False):
        """
        int_G_0,int_G_1=C_Bridge_1d(ti,x_new_0,x_new_1,tf,x_pr_0[i+1],x_pr_1[i+1],b,A,A,Sig,fi,fi,b_til,A_til_0,A_til_1,\
        Sig_til,fi_til_0,fi_til_1,r,r_pars_0,r_pars_1,H,H_pars_0,H_pars_1,l,d,N,seed,Dt=Deltas[i]/2**l)
        seed+=int(2**l*intervals[i]/2**l-1)
        #int_G=Bridge_1d(ti,x_new,tf,x_pr[i],b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,\
        #r,r_pars,H,H_pars,l,d,N, seed+i*int(int(2**l*d-1)),crossed=False)
        #int_Gs[i]=int_G
        #print(xi.shape)
        #print(yi,obti-i*d,)
        #print(x_new,xi)
        #print(xi)
        #Things that could be wrong
        # observations, x_new, weights
        #observations seem to be fine

        #xs[2**l*d*i:2**l*d*(i+1)]=x[:-1]

        #print("other parameteres are:",ti,x_new,tf,x_pr[i] )
        #print("atdp is ", atdp)
        #print("object is: ", aux_trans_den(ti,x_new,tf,x_pr[i],atdp))

        log_weights_0[i+1]=int_G_0+log_g_den(obs[i+1],x_pr_0[i+1],g_den_par)\
        +np.log(aux_trans_den(ti,x_new_0,tf,x_pr_0[i+1],atdp_0))-np.log(prop_trans_den(ti,x_new_0,tf,x_pr_0[i+1],ind_prop_trans_par))
        weights_0=pff.norm_logweights(log_weights_0[i+1])
        
        log_weights_1[i+1]=int_G_1+log_g_den(obs[i+1],x_pr_1[i+1],g_den_par)\
        +np.log(aux_trans_den(ti,x_new_1,tf,x_pr_1[i+1],atdp_1))-np.log(prop_trans_den(ti,x_new_1,tf,x_pr_1[i+1],ind_prop_trans_par))
        weights_1=pff.norm_logweights(log_weights_1[i+1])

        """ 
        log_weights[i]=log_weights[i]+int_G+log_g_den(obs[i],x_pr[i],g_den_par,crossed=crossed)\
        +np.log(aux_trans_den(ti,x_new,tf,x_pr[i],atdp,crossed=crossed))-np.log(prop_trans_den(ti,x_new,tf,x_pr[i],sample_pars,crossed=crossed))
        weights=pff.norm_logweights(log_weights[i])
        """
        #print(yi,weights)
        #seed_val=i 
        #print(weights.shape)
        x_last_0=x_pr_0[i+1]
        x_last_1=x_pr_1[i+1]
        
        #ESS=1/np.sum(weights**2)
        #ESS=1/np.sum(weights**2)
        
        """
        [part0,part1,x0_new,x1_new]=max_coup_multi(w0,w1,N,xi0[-1],xi1[-1],dim)
        """
        [part_resamp_0,part_resamp_1, x_new_0,x_new_1]=max_coup_multi\
        (weights_0,weights_1,N,x_last_0,x_last_1,1)
        #print(part_resamp_0,part_resamp_1)
        log_weights_0[:i+2]=log_weights_0[:i+2,part_resamp_0]
        int_Gs_0[:i+2]=int_Gs_0[:i+2,part_resamp_0]
        x_pr_0[:i+2]=x_pr_0[:i+2,part_resamp_0]
        log_weights_1[:i+2]=log_weights_1[:i+2,part_resamp_1]
        int_Gs_1[:i+2]=int_Gs_1[:i+2,part_resamp_1]
        x_pr_1[:i+2]=x_pr_1[:i+2,part_resamp_1]

        seeds_0_wp[:i+1]=np.copy(seeds_0_wp[:i+1,:,part_resamp_0])
        seeds_1_wp[:i+1]=np.copy(seeds_1_wp[:i+1,:,part_resamp_1])
        
        #x_new=multi_samp_exp(weights,N,x_last,1)[1]
        #print(x_new.shape)
        
        
       #x_new=sr(weights,N,x_pf[i],dim)[1]
    #weights=np.reshape(norm_logweights(log_weights,ax=1),(int(T/d),N,1))
    #pf=np.sum(weights*x_pf,axis=1)
    #Filter
    #spots=np.arange(d_steps,2**l*T+1,d_steps,dtype=int)
    #x_pf=x[spots]
    #weights=norm_logweights(log_weights,ax=1)

    #print(x_pf.shape,weights.shape)
    #suma=np.sum(x_pf[:,:,1]*weights,axis=1)
    return [log_weights_0,log_weights_1,int_Gs_0,int_Gs_1,x_pr_0,x_pr_1,seeds_0_wp,seeds_1_wp]

#%%
# PARTICLE FILTER TEST

# FOR THE OU PROCESS 
"""
N=100000
x0=1.2+np.zeros(N)
l=10
alpha=0
T=10
t0=0
l_d=1
d=2**(l_d)
theta=0.2
sigma=1.2
#sigma_aux=0.2
theta_aux=theta+0.2
sigma_aux=sigma+0.1
#print(theta)
collection_input=[ b_ou_1d,theta,Sig_ou_1d,sigma]
resamp_coef=1
l_max=10
x_true=gen_gen_data_1d(T,x0_sca,l_max,collection_input)
x_reg=cut(T,l,-l_d,x_true)[1:]
# x_reg stands for x regular 
cov=2e0
obs=gen_obs(x_reg,g_normal_1d,cov)
np.random.seed(3)
#print(x_pr)
n_steps=int(2**l*(T-t0))
dt=2**(-l)
x=np.zeros((n_steps+1,N))
x[0]=x0
x_pr=np.random.normal(x0,np.sqrt(sigma**2*(T-t0)),N)
x[-1]=x_pr
int_G=np.zeros(N)
t_test=2.
"""
#print("t0 is->",t0, " x0 is->",x0," T is->",T," theta is->",theta," sigma is->",sigma,\
#" theta_aux is->",theta_aux," sigma_aux is->",sigma_aux," obs is->",obs," cov is->",cov,\
#    " resamp_coef is->",resamp_coef," l is->",l," d is->",d," N is->",N)
#print(obs.shape)
#%%

#(t0,x0,T,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,\
    #sample_funct,sample_pars,obs,log_g_den,g_den_par, aux_trans_den,atdp,\
    #prop_trans_den, resamp_coef, l, d,N,crossed=False):

# FOR THE ALPHA AUXILIARY PROCESS
"""
[log_weights,x_pr,int_Gs]=PF_bridge(t0,x0,T,b_ou_1d,theta,Sig_ou_1d,sigma,b_artificial_1d,0,Sig_alpha,[alpha,sigma,T],\
r_normal,[new_alpha_trans_sd,[alpha,sigma]],H_normal_1d,[new_alpha_trans_sd,[alpha,sigma]],\
sampling_alpha_trans_props, [alpha,sigma],obs,log_g_normal_den,cov,\
aux_trans_den_alpha,[alpha,sigma],aux_trans_den_alpha,\
resamp_coef,l,d, N)
"""
# FOR THE OU AUXILIARY PROCESS
"""
np.random.seed(3)
[log_weights,int_Gs,x_pr]=PF_bridge(t0,x0,T,b_ou_1d,theta,Sig_ou_1d,sigma,b_ou_aux,theta_aux,Sig_ou_aux,sigma_aux,\
r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,[ou_sd,[theta_aux,sigma_aux],theta_aux],\
sampling_ou, [theta_aux,sigma_aux],obs,log_g_normal_den,cov,\
ou_trans_den,[theta_aux,sigma_aux],ou_trans_den,\
resamp_coef,l,d, N)

 #[aux_sd,sigma,mu,t0,x0]=pars
    #aux_trans_den=norm(loc=x0*np.exp(mu*(T-t0)),scale=aux_sd(t0,x0,T,x_pr,[sigma,mu,t0,x0]))

x_mean=np.sum(pff.norm_logweights(log_weights,ax=1)*x_pr,axis=-1)
"""
#print(x_mean.shape)
#print(x_pr.shape)
#%%
# In the following I use the KF filter 
"""
dim=1
dim_o=1
K=np.array([[np.exp(d*theta)]])
G=np.array([[sigma*np.sqrt((np.exp(2*d*theta)-1)/(2*theta))]])
H=np.array([[1]])
D=np.array([[cov]])
x_kf=KF(x0[0],dim,dim_o,K,G,H,D,obs)
times=np.arange(t0,T+1,d)
l_times=np.arange(t0,T,2**(-l))
print(times, l_times)
print(x_mean.shape)
plt.plot(times[1:],x_reg,label="True signal")
plt.plot(times,x_kf[:,0],label="KF")
plt.plot(times[1:], x_mean,label="PF")
plt.plot(times[1:], obs,label="Observations")
#plt.plot(l_times[1:],xs[:-1],lw="0.5")
#frame1 = plt.gca()

#frame1.axes.get_xaxis().set_visible(False)
#frame1.axes.get_yaxis().set_visible(False)
#frame1.axes.get_xaxis().set_ticks([])
#frame1.axes.get_yaxis().set_ticks([])

#plt.savefig("Diffusion2.pdf")
plt.legend()
"""
#%%

# Test: 
# In the following we are going to carry out test for the MSE of the Particle filter, we will decompose this MSE
# into the bias of the time discretization and the variance of the PF. Results should follow the EM weak error for the
# bias and the Monte Carlo variance for the PF. 
"""
x0_sca=1.2
l=5
alpha=0
T=10
l_d=1
d=2**(l_d)
t0=0
theta=0.3
sigma=1.2
#sigma_aux=0.2
theta_aux=theta+0.2
sigma_aux=sigma+0.1
#print(theta)
collection_input=[ b_ou_1d,theta,Sig_ou_1d,sigma]
resamp_coef=1
l_max=10
x_true=gen_gen_data_1d(T,x0_sca,l_max,collection_input)
x_reg=cut(T,l,-l_d,x_true)[1:]
# x_reg stands for x regular 
cov=1e0
obs=gen_obs(x_reg,g_normal_1d,cov)
np.random.seed(3)
p=8
N0=40
eNes=2**np.arange(p)*N0
#print(eNes)
samples=10
x_mean=np.zeros((len(eNes),samples,int(T/d)))
#print(x_mean.shape)

for i in range(len(eNes)):
    for j in range(samples):
        N=int(eNes[i])
        x0=x0_sca+np.zeros(N)
        [log_weights,int_Gs,x_pr]=PF_bridge(t0,x0,T,b_ou_1d,theta,Sig_ou_1d,sigma,b_ou_aux,theta_aux,Sig_ou_aux,sigma_aux,\
        r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,[ou_sd,[theta_aux,sigma_aux],theta_aux],\
        sampling_ou, [theta_aux,sigma_aux],obs,log_g_normal_den,cov,\
        ou_trans_den,[theta_aux,sigma_aux],ou_trans_den,\
        resamp_coef,l,d, N)
        x_mean[i,j]=np.sum(pff.norm_logweights(log_weights,ax=1)*x_pr,axis=-1)
"""
#%%
"""
print(obs.shape)
dim=1
dim_o=1
K=np.array([[np.exp(d*theta)]])
G=np.array([[sigma*np.sqrt((np.exp(2*d*theta)-1)/(2*theta))]])
H=np.array([[1]])
D=np.array([[cov]])
x_kf=KF(x0[0],dim,dim_o,K,G,H,D,obs)
times=np.arange(t0,T+1,d)
l_times=np.arange(t0,T,2**(-l))
print(x_kf.shape)
"""
#%%
"""
comp=-1
print(x_mean.shape )
s_var=np.var(x_mean,axis=1)[:,comp]
MSE=np.mean((x_mean-x_kf[1:,-1])**2,axis=1)[:,comp]
print(s_var.shape)
s_mean=np.mean(x_mean,axis=1)[:,comp]
plt.plot(eNes,s_var,label="Variance")
plt.plot(eNes,MSE,label="MSE")
plt.plot(eNes,eNes[0]*s_var[0]/eNes,label="1/N")
plt.xscale("log")
plt.yscale("log")
plt.legend()
"""


# RESULTS OF THE TEST: 
# THE VARIANCE IS BEHAVING AS EXPECTED, 
#%%
# TEST FOR THE BIAS OF THE PROCESS. 

##
"""
np.random.seed(7)
x0_sca=1.2
N=100000
T=10
l_d=1
d=2**(l_d)
t0=0
theta=0.15
sigma=0.4
#sigma_aux=0.2
theta_aux=theta-0.05
sigma_aux=sigma+0.1
#print(theta)
collection_input=[b_ou_1d,theta,Sig_ou_1d,sigma]
resamp_coef=1
l_max=10
x_true=gen_gen_data_1d(T,x0_sca,l_max,collection_input)
x_reg=cut(T,l_max,-l_d,x_true)[1:]
# x_reg stands for x regular 
cov=5e-1
obs=gen_obs(x_reg,g_normal_1d,cov)
np.random.seed(5)
#print(x_mean.shape)
l=5
x0=x0_sca+np.zeros(N)
[log_weights,int_Gs,x_pr]=PF_bridge(t0,x0,T,b_ou_1d,theta,Sig_ou_1d,sigma,b_ou_aux,theta_aux,Sig_ou_aux,sigma_aux,\
r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,[ou_sd,[theta_aux,sigma_aux],theta_aux],\
sampling_ou, [theta_aux,sigma_aux],obs,log_g_normal_den,cov,\
ou_trans_den,[theta_aux,sigma_aux],ou_trans_den,\
resamp_coef,l,d, N)
x_mean=np.sum(pff.norm_logweights(log_weights,ax=1)*x_pr,axis=-1)
dim=1
dim_o=1
K=np.array([[np.exp(d*theta)]])
G=np.array([[sigma*np.sqrt((np.exp(2*d*theta)-1)/(2*theta))]])
H=np.array([[1]])
D=np.array([[cov]])
x_kf=KF(x0[0],dim,dim_o,K,G,H,D,obs)
times=np.arange(t0,T+1,d)
l_times=np.arange(t0,T,2**(-l_max))
print(x_mean.shape)
plt.plot(times[1:],x_reg,label="True signal")
plt.plot(times,x_kf[:,0],label="KF")
plt.plot(times[1:], x_mean,label="PF")
plt.plot(l_times,x_true[1:] )
plt.plot(times[1:], obs,label="Observations")
plt.legend()
"""
#%%
# cccurrent
"""
np.random.seed(0)
l0=2
L=10
eLes=np.array(range(l0,L+1))
samples=2000
x_mean=np.zeros((len(eLes),samples,int(T/d)))


start=time.time()
for i in range(len(eLes)):
    l=eLes[i]
    print("l is: ",l)   
    for j in range(samples):
        
        x0=x0_sca+np.zeros(N)
        [log_weights,int_Gs,x_pr]=PF_bridge(t0,x0,T,b_ou_1d,theta,Sig_ou_1d,sigma,b_ou_aux,theta_aux,Sig_ou_aux,sigma_aux,\
        r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,[ou_sd,[theta_aux,sigma_aux],theta_aux],\
        sampling_ou, [theta_aux,sigma_aux],obs,log_g_normal_den,cov,\
        ou_trans_den,[theta_aux,sigma_aux],ou_trans_den,\
        resamp_coef,l,d, N)
        x_mean[i,j]=np.sum(pff.norm_logweights(log_weights,ax=1)*x_pr,axis=-1)

end=time.time()
print("the time spent is: ",end-start)
"""
#%%
"""dim=1
dim_o=1
K=np.array([[np.exp(d*theta)]])
G=np.array([[sigma*np.sqrt((np.exp(2*d*theta)-1)/(2*theta))]])
H=np.array([[1]])
D=np.array([[cov]])
x_kf=KF(x0[0],dim,dim_o,K,G,H,D,obs)
times=np.arange(t0,T+1,d)
print(x_kf.shape)
"""
#%%
"""comp=-1
print(x_mean.shape )
s_bias=np.abs(np.mean((x_mean-x_kf[1:,0]),axis=1)[:,comp])
var_bias=np.var((x_mean-x_kf[1:,0]),axis=1)[:,comp]
s_bias_up=s_bias+np.sqrt(var_bias)*1.96/np.sqrt(samples)
s_bias_lb=s_bias-np.sqrt(var_bias)*1.96/np.sqrt(samples)
#MSE=np.mean((x_mean-x_kf[1:,0])**2,axis=1)[:,comp]
plt.plot(eLes,s_bias,label="$bias$")
plt.plot(eLes,s_bias_up,label="ub")
plt.plot(eLes,s_bias_lb,label="lb")
print(s_bias_lb)
plt.plot(eLes,(s_bias[0])*2**(eLes[0])/2**(eLes),label="$2^l$")
#plt.xscale("log")
sm=np.mean(((x_pf[0]-x_pf[1])**2)[:,:,a],axis=1)
sm_lb=sm-np.sqrt(var_sm)*1.96/np.sqrt(samples)


plt.yscale("log")
plt.legend()"""

#%%
#%%
# Plot of the discretization of the discretization of the OU process
# compared to a coarser discretization of the same process.
"""
np.random.seed(3)
T=1
l_1=10
l_2=3
dt_1=1/2**l_1
dt_2=1/2**l_2
n_steps_1=2**l_1
n_steps_2=2**l_2
ws=np.random.normal(0,1,int(2**l_1))*np.sqrt(dt_1)
x_1=np.zeros(int(2**l_1)+1)
x_2=np.zeros(int(2**l_2)+1)
x_1[0]=0
x_2[0]=0
sigma=2
theta=sigma+0.5
count=1
for i in range(int(2**l_1)):
    x_1[i+1]=x_1[i]+theta*x_1[i]*dt_1+sigma*ws[i]
    if i>0 and i%int(2**(l_1-l_2))==0:
        print(i)
        a=i-2**(l_1-l_2)
        b=i
        print(a,b)
        x_2[count]=x_2[count-1]+theta*x_2[count-1]*dt_2+sigma*np.sum(ws[a:b])
        count+=1

x_2[-1]=x_2[-2]+theta*x_2[-2]*dt_2+sigma*np.sum(ws[-2**(l_1-l_2):])

cum_ws=np.cumsum(ws)
plt.plot(np.linspace(0,1,2**l_1+1),x_1)
plt.plot(np.linspace(0,1,2**l_2+1),x_2)
plt.scatter(np.linspace(0,1,2**l_2+1),x_2,color="orange")
frame1 = plt.gca()

frame1.axes.get_xaxis().set_visible(False)
frame1.axes.get_yaxis().set_visible(False)
frame1.axes.get_xaxis().set_ticks([])
frame1.axes.get_yaxis().set_ticks([])

#plt.savefig("Eu.pdf")
#plt.legend()

# In the followign we compute the conditional pf. What do we need? 
# same things as for the regular PF and additionally one path
# what kind of path? we only need the value of the particle at the 
# observation points.

# what is the best way to do this?

"""

#%%

def Cond_PF_bridge(lw_cond,int_Gs_cond,x_cond,seeds_cond,t0,x0,T,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,\
    r,r_pars,H,H_pars,sample_funct,sample_pars,obs,log_g_den,g_den_par, aux_trans_den,atdp,\
    prop_trans_den, resamp_coef, l, d,N,seed,crossed=False):
    #(T,xin,b_ou,A,Sig_ou,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par):

    # ARGUMENTS: 
    # x_cond: the value of the particles at the observation points, it is a T/d dimensional array.
    # lw_cond: the log weights of the particles at the observation points, it is a T/d dimensional array.
    # these are not normalized, as we need them to normalize the new pf.
    # the argument of the Kenel x0 rank 1 dims (dim) 
    # the drift and diffusion are b and Sig, respectively, and they take
    # x(either a (N) dimensional or (N,N) dimensional array) and A as arguments for the drift and x and fi for the diffusion.
    # the level of discretization l, the distance of resampling, the number of
    # particles N.
    # Grad_b is a function that takes (x,A) as argument and computes the gradnient of b wrt the 
    # parameters A, and evaluates it a (x,A).
    # b_til,A_til,Sig_til,fi_til, are the analogous functions for the auxiliar process.
    # a difference is that their arguments are (t,x) for the drift and (t,x,fi_til) for the diffusion.
    # r is the function that computes the gradient of the log of the kernel of the auxiliar process
    # and it takes (t,x,T,x_pr,r_pars) as arguments.
    # H is the function that computes the Hessian of the log of the kernel of the auxiliar process
    # and it takes (t,x,T,x_pr,H_pars) as arguments.
    # crossed is the boolean that indicates if we need the computations for the crossed terms that 
    # are needed for the smoother.

    log_weights=np.zeros((int(T/d),N))
    # In this function x_pr will store the resampled paths
    # this is different from previous functions, where x_pr stored the 
    # paths without resampling (yet).
    x_pr=np.zeros((int(T/d),N))
    #xs=np.zeros((int(T*2**l*d),N))
    int_Gs=np.zeros((int(T/d),N))                      
    x_new=x0
    new_seeds=np.zeros((int(T/d),2,N),dtype=int)
    new_seeds[:,:,0]=seeds_cond
    for i in range(int(T/d)):
        


        tf=t0+(i+1)*d
        ti=t0+(i)*d
        np.random.seed(seed)
        x_pr[i]=sample_funct(x_new,N,d,sample_pars)
        #print(x_pr[i].shape,x_cond[i].shape)
        x_pr[i,0]=x_cond[i]
        # what parameters do we need in order to make the auxiliar density general?
        # x_new,  d, t, x_pr,tf
        # aux_trans_den(t0,x0,T,x_pr,atdp)
        # atdp stands for auxiliar transition density parameters. 
        """"
        int_G=Bridge_1d(ti,x_new,tf,x_pr[i],b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,\
        r,r_pars,H,H_pars,l,d,N,seed,crossed=False,j=seeds_cond[i,1],cond_seed=seeds_cond[i,0])
        seed+=int(2**l*d) -1 # 2**l*d -1 is the number of iterations of the EM 
        """

        int_G=Bridge_1d(ti,x_new,tf,x_pr[i],b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,\
        r,r_pars,H,H_pars,l,d,N,seed+i*int(int(2**l*d-1)),crossed=False,\
        j=seeds_cond[i,1],cond_seed=seeds_cond[i,0])
        new_seeds[i,1,1:]=np.copy(np.arange(N))[1:]
        new_seeds[i,0,1:]=seed+i*int(2**l*d-1)
        int_Gs[i]=int_G
        int_Gs[i,0]=int_Gs_cond[i]
        #print(xi.shape)
        #print(yi,obti-i*d,)
        #print(x_new,xi)
        #print(xi)
        #Things that could be wrong
        # observations, x_new, weights
        #observations seem to be fine

        #xs[2**l*d*i:2**l*d*(i+1)]=x[:-1]

        #print("other parameteres are:",ti,x_new,tf,x_pr[i] )
        #print("atdp is ", atdp)
        #print("object is: ", aux_trans_den(ti,x_new,tf,x_pr[i],atdp))
        
        new_log_weights=int_G+log_g_den(obs[i],x_pr[i],g_den_par,crossed=crossed)\
        +np.log(aux_trans_den(ti,x_new,tf,x_pr[i],atdp,crossed=crossed))-np.log(prop_trans_den(ti,x_new,tf,x_pr[i],sample_pars,crossed=crossed))
        log_weights[i]=log_weights[i]+new_log_weights
        log_weights[i,0]=lw_cond[i]
        weights=pff.norm_logweights(log_weights[i])
        #print(yi,weights)
        #seed_val=i
        #print(weights.shape)
        x_last=x_pr[i]
        
        #ESS=1/np.sum(weights**2)
        #print(ESS,resamp_coef*N)
        
        # This code works only for resamp_coef=1, otherwise 
        # we need to consider some adaptative resampling scheme
        # for the conditional particle filter.
        resamp_coef=1
        
        #print("resampling at time ",i)
        #if True==False:
        #[part0,part1,x0_new,x1_new]=max_coup_sr(w0,w1,N,xi0[-1],xi1[-1],dim)
        #print(x_new.shape)
        [part_resamp, x_new]=multi_samp_exp(weights,N,x_last,1)
        log_weights[:i+1,1:]=log_weights[:i+1,part_resamp[1:]]
        int_Gs[:i+1,1:]=int_Gs[:i+1,part_resamp[1:]]
        x_new[0]=x_cond[i]
        new_seeds[:i+1,:,1:]=new_seeds[:i+1,:,part_resamp[1:]]
        x_pr[:i+1,1:]=x_pr[:i+1,part_resamp[1:]]
        #print(x_new.shape)
        
        #print(i)
    #x_pr[i]=x_new
    #resamp_log_weights[-1]=log_weights[-1]
        
       #x_new=sr(weights,N,x_pf[i],dim)[1]
    #weights=np.reshape(norm_logweights(log_weights,ax=1),(int(T/d),N,1))
    #pf=np.sum(weights*x_pf,axis=1)
    #Filter
    #spots=np.arange(d_steps,2**l*T+1,d_steps,dtype=int)
    #x_pf=x[spots]
    #weights=norm_logweights(log_weights,ax=1)
    #print(x_pf.shape,weights.shape)
    #suma=np.sum(x_pf[:,:,1]*weights,axis=1)
    return [log_weights,int_Gs,x_pr,new_seeds]
    #[log_weights,x_pr,new_lw_cond,new_int_G_cond,new_x_cond,new_seeds_cond]



#%%
def Cond_PF_bridge_new(lw_cond,int_Gs_cond,x_cond,seeds_cond,t0,x0,T,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,\
    r,r_pars,H,H_pars,sample_funct,sample_pars,obs,log_g_den,g_den_par, aux_trans_den,atdp,\
    prop_trans_den, resamp_coef, l, d,N,seed,crossed=False):
    #(T,xin,b_ou,A,Sig_ou,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par):
    # ARGUMENTS: 
    # x_cond: the value of the particles at the observation points, it is a T/d dimensional array.
    # lw_cond: the log weights of the particles at the observation points, it is a T/d dimensional array.
    # these are not normalized, as we need them to normalize the new pf.
    # seeds_cond is an array with rank 2 and dims (int(T/d),2), where seeds_cond[:,0] are the sequence 
    # of seeds and seeds_cond[:,1] is the sequence of corresponding rows.
    # the argument of the Kenel x0 rank 1 dims (dim) 
    # the drift and diffusion are b and Sig, respectively, and they take
    # x(either a (N) dimensional or (N,N) dimensional array) and A as arguments for the drift and x and fi for the diffusion.
    # the level of discretization l, the distance of resampling, the number of
    # particles N.
    # atdp stands for auxiliar transition density parameters. 
    # seed: seeding that will identify the samples of the algorithm so it can be reproducible
    # and more importantly, we can access some samples without the need of storing them

    # Grad_b is a function that takes (x,A) as argument and computes the gradnient of b wrt the 
    # parameters A, and evaluates it a (x,A).
    # b_til,A_til,Sig_til,fi_til, are the analogous functions for the auxiliar process.
    # a difference is that their arguments are (t,x) for the drift and (t,x,fi_til) for the diffusion.
    # r is the function that computes the gradient of the log of the kernel of the auxiliar process
    # and it takes (t,x,T,x_pr,r_pars) as arguments.
    # H is the function that computes the Hessian of the log of the kernel of the auxiliar process
    # and it takes (t,x,T,x_pr,H_pars) as arguments.
    # crossed is the boolean that indicates if we need the computations for the crossed terms that 
    # are needed for the smoother.

    # OUTPUTs: 
    # log_weights: the log weights of the particles at the observation points, it is a T/d dimensional array.
    # As opposed to log_weights in Cond_PF_bridge this weights correspond not to the paths but 
    # each paticle.
    # x_pr: the particles at the observation points, it is a T/d dimensional array.
    # Again, these particles are not the paths but the particles without resampling 
    # new_lw_cond: the log weights of the particles at the observation points, it is a T/d dimensional array.
    # noticed that these quantities are such that correspond to the backward sampled path, meaning that 
    # its lw(x_{t-1},x_t) where x_t is the particle at time t.
    # new_int_G_cond: similartly to new_lw_cond regarding the dependence of the endpoints but 
    # with the integral of G.
    # new_x_cond: the particles sampled backwards at the observation points, it is a T/d dimensional array.
    # new_seeds_cond: the seeds of the particles sampled backwards at the observation points, it is a T/d dimensional array.
    # [log_weights,x_pr,new_lw_cond,new_int_G_cond,new_x_cond,new_seeds_cond]

    log_weights=np.zeros((int(T/d),N))
    new_x_cond=np.zeros(int(T/d)) # This variable stores the x's of the backward sampling
    new_lw_cond=np.zeros(int(T/d)) # This variable stores the log weights corresponding of the 
    # variables of the backward sampling.
    new_int_G_cond=np.zeros(int(T/d)) # This variable stores the log weights corresponding of the 
    # variables of the backward sampling.

    # In this function x_pr will store the resampled paths
    # this is different from previous functions, where x_pr stored the 
    # paths without resampling (yet).
    x_pr=np.zeros((int(T/d),N))
    int_Gs=np.zeros((int(T/d),N))                      
    x_new=x0 # rank 1 with dimensions (dim)
    new_seeds=np.zeros((int(T/d),2,N),dtype=int)
    new_seeds[:,:,0]=seeds_cond
    
    for i in range(int(T/d)):

        tf=t0+(i+1)*d
        ti=t0+(i)*d
        np.random.seed(seed)
        x_pr[i]=sample_funct(x_new,N,d,sample_pars)
        x_pr[i,0]=x_cond[i]
        int_G=Bridge_1d(ti,x_new,tf,x_pr[i],b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,\
        r,r_pars,H,H_pars,l,d,N,seed,crossed=False,j=seeds_cond[i,1],cond_seed=seeds_cond[i,0])
        new_seeds[i,1,1:]=np.copy(np.arange(N))[1:]
        new_seeds[i,0,1:]=seed
        int_Gs[i]=int_G
        #int_Gs[i,0]=int_Gs_cond[i]
        x_last=x_pr[i]
        seed+=int(2**l*d) -1 # 2**l*d -1 is the number of iterations of the EM 
        # method so we can get the approximation in the interval of lenght d 
        # (note that we subtract 1 bcs the last point of the bridge is already determined).
        
        new_log_weights=int_G+log_g_den(obs[i],x_pr[i],g_den_par,crossed=crossed)\
        +np.log(aux_trans_den(ti,x_new,tf,x_pr[i],atdp,crossed=crossed))-np.log(prop_trans_den(ti,x_new,tf,x_pr[i],sample_pars,crossed=crossed))
        log_weights[i]=log_weights[i]+new_log_weights
        #log_weights[i,0]=lw_cond[i]
        weights=pff.norm_logweights(log_weights[i])
        #print(yi,weights)
        #seed_val=i
        #print(weights.shape)
        [part_resamp, x_new]=multi_samp_exp(weights,N,x_last,1)
        x_new[0]=x_cond[i]
        if i<int(T/d)-1:
            log_weights[:i+1,1:]=np.copy(log_weights[:i+1,part_resamp[1:]])
            int_Gs[:i+1,1:]=np.copy(int_Gs[:i+1,part_resamp[1:]])
            #int_Gs[:i+1,0]=np.copy(int_Gs_cond[:i+1])
            new_seeds[:i+1,:,1:]=np.copy(new_seeds[:i+1,:,part_resamp[1:]])
            #new_seeds[:i+1,:,0]=np.copy(seeds_cond[:i+1])
            x_pr[:i+1,1:]=np.copy(x_pr[:i+1,part_resamp[1:]])
            #x_pr[:i+1,0]=np.copy(x_cond[:i+1])

    index=np.random.choice(N,p=weights)
    new_lw_cond=np.copy(log_weights[:,index])
    new_int_G_cond=np.copy(int_Gs[:,index])
    new_x_cond=np.copy(x_pr[:,index])
    new_seeds_cond=np.copy(new_seeds[:,:,index])

    return [log_weights,x_pr,new_lw_cond,new_int_G_cond,new_x_cond,new_seeds_cond]
#%%
# HERE WE PERFORM TEST FOR THE Cond_PF_bridge_new.


#%%


def Cond_PF_bridge_back_samp(lw_cond,int_Gs_cond,x_cond,seeds_cond,t0,\
    x0,T,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,\
    r,r_pars,H,H_pars,sample_funct,sample_pars,obs,log_g_den,g_den_par, aux_trans_den,atdp,\
    prop_trans_den, resamp_coef, l, d,N,seed):
    #(T,xin,b_ou,A,Sig_ou,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par):
    # ARGUMENTS: 
    # x_cond: the value of the particles at the observation points, it is a T/d dimensional array.
    # lw_cond: the log weights of the particles at the observation points, it is a T/d dimensional array.
    # these are not normalized, as we need them to normalize the new pf.
    # seeds_cond is an array with rank 2 and dims (int(T/d),2), where seeds_cond[:,0] are the sequence 
    # of seeds and seeds_cond[:,1] is the sequence of corresponding rows.
    # the argument of the Kenel x0 rank 1 dims (dim) 
    # the drift and diffusion are b and Sig, respectively, and they take
    # x(either a (N) dimensional or (N,N) dimensional array) and A as arguments for the drift and x and fi for the diffusion.
    # the level of discretization l, the distance of resampling, the number of
    # particles N.
    # atdp stands for auxiliar transition density parameters. 
    # seed: seeding that will identify the samples of the algorithm so it can be reproducible
    # and more importantly, we can access some samples without the need of storing them

    # Grad_b is a function that takes (x,A) as argument and computes the gradnient of b wrt the 
    # parameters A, and evaluates it a (x,A).
    # b_til,A_til,Sig_til,fi_til, are the analogous functions for the auxiliar process.
    # a difference is that their arguments are (t,x) for the drift and (t,x,fi_til) for the diffusion.
    # r is the function that computes the gradient of the log of the kernel of the auxiliar process
    # and it takes (t,x,T,x_pr,r_pars) as arguments.
    # H is the function that computes the Hessian of the log of the kernel of the auxiliar process
    # and it takes (t,x,T,x_pr,H_pars) as arguments.
    # crossed is the boolean that indicates if we need the computations for the crossed terms that 
    # are needed for the smoother.

    # OUTPUTs: 
    # log_weights: the log weights of the particles at the observation points, it is a T/d dimensional array.
    # As opposed to log_weights in Cond_PF_bridge this weights correspond not to the paths but 
    # each paticle.
    # x_pr: the particles at the observation points, it is a T/d dimensional array.
    # Again, these particles are not the paths but the particles without resampling 
    # new_lw_cond: the log weights of the particles at the observation points, it is a T/d dimensional array.
    # noticed that these quantities are such that correspond to the backward sampled path, meaning that 
    # its lw(x_{t-1},x_t) where x_t is the particle at time t.
    # new_int_G_cond: similartly to new_lw_cond regarding the dependence of the endpoints but 
    # with the integral of G.
    # new_x_cond: the particles sampled backwards at the observation points, it is a T/d dimensional array.
    # new_seeds_cond: the seeds of the particles sampled backwards at the observation points, it is a T/d dimensional array.
    # [log_weights,x_pr,new_lw_cond,new_int_G_cond,new_x_cond,new_seeds_cond]
    log_weights=np.zeros((int(T/d),N))
    new_x_cond=np.zeros(int(T/d)) # This variable stores the x's of the backward sampling
    new_lw_cond=np.zeros(int(T/d)) # This variable stores the log weights corresponding of the 
    # variables of the backward sampling.
    new_int_G_cond=np.zeros(int(T/d)) # This variable stores the log weights corresponding of the 
    # variables of the backward sampling.

    # In this function x_pr will store the resampled paths
    # this is different from previous functions, where x_pr stored the 
    # paths without resampling (yet).
    x_pr=np.zeros((int(T/d),N))
    int_Gs=np.zeros((int(T/d),N))                      
    x_new=x0 # rank 1 with dimensions (dim)
    for i in range(int(T/d)):
        np.random.seed(seed)
        tf=t0+(i+1)*d
        ti=t0+(i)*d
        x_pr[i]=sample_funct(x_new,N,d,sample_pars)
        x_pr[i,0]=x_cond[i]
        int_G=Bridge_1d(ti,x_new,tf,x_pr[i],b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,\
        r,r_pars,H,H_pars,l,d,N,seed,crossed=False,j=seeds_cond[i,1],cond_seed=seeds_cond[i,0])
        seed+=int(2**l*d) -1 # 2**l*d -1 is the number of iterations of the EM 
        # method so we can get the approximation in the interval of lenght d 
        # (note that we subtract 1 bcs the last point of the bridge is already determined).
        int_Gs[i]=int_G
        #int_Gs[i,0]=int_Gs_cond[i]
        #print(xi.shape)    
        #print(yi,obti-i*d,)
        #print(x_new,xi)
        #print(xi)
        #Things that could be wrong
        # observations, x_new, weights
        #observations seem to be fine
        #xs[2**l*d*i:2**l*d*(i+1)]=x[:-1]
        #print("other parameteres are:",ti,x_new,tf,x_pr[i] )
        #print("atdp is ", atdp)
        #print("object is: ", aux_trans_den(ti,x_new,tf,x_pr[i],atdp))
        
        new_log_weights=int_G+log_g_den(obs[i],x_pr[i],g_den_par)\
        +np.log(aux_trans_den(ti,x_new,tf,x_pr[i],atdp))-np.log(prop_trans_den(ti,x_new,tf,x_pr[i],sample_pars))
        log_weights[i]=log_weights[i]+new_log_weights
        #log_weights[i,0]=lw_cond[i]
        weights=pff.norm_logweights(log_weights[i])
        #print(yi,weights)
        #seed_val=i
        #print(weights.shape)
        x_last=x_pr[i]        
        #ESS=1/np.sum(weights**2)
        ESS=0
        #print(ESS,resamp_coef*N)
        # This code works only for resamp_coef=1, otherwise 
        # we need to consider some adaptative resampling scheme
        # for the conditional particle filter.
        #if ESS<resamp_coef*N and i<int(T/d)-1:
            
            #print("resampling at time ",i)
        #if True==False:
            #[part0,part1,x0_new,x1_new]=max_coup_sr(w0,w1,N,xi0[-1],xi1[-1],dim)
            #print(x_new.shape)
        [part_resamp, x_new]=multi_samp_exp(weights,N,x_last,1)
        x_new[0]=x_cond[i]
        #print(x_new.shape)
        """
        else:    
            #print("time is",i)
            x_new=x_last
            if i< int(T/d)-1:
                log_weights[i+1]=log_weights[i]
        """
        #print(i)
    #x_pr[i]=x_new

    ####################################################################################
    # The following corresponds to the backward samples.  
    # Here we sample from the filtering distribution at time T.
    new_seeds_cond=np.zeros((int(T/d),2),dtype=int) # new_cond_seed[:,0] are the seeds and new_cond_seed[:,1] 
    # are the corresponding rows.
    # what kind of seed should I include here? This seed should already be indexed by the np.random.seed(seed)
    # at the beginning of the previous for loop. 
    seed_counter_2=seed-(int(2**l*d) -1 )*int(T/d)
    np.random.seed(seed_counter_2)
    j=np.random.choice(N,p=weights)
    if j==0:
        new_seeds_cond[-1]=seeds_cond[-1]
        # It's important to note where we need to use new_seeds_cond[-1,1]
        # in this case, i.e., the arguments of Brige_1d below.
    else:
        new_seeds_cond[-1,0]=seed-(2**l*d -1)
        new_seeds_cond[-1,1]=j

    new_x_cond[-1]=x_pr[-1,j]
    #new_lw_cond[-1]=log_weights[-1,j]
    seed_counter=seed-(2**l*d -1) # This variable is defined so 
    # we can keep track of the seed of the (N-1) particles.
    for i in range(int(T/d)-1): 
        t=int(T/d)-i-1 # t is an index, not a time, it goes backward from the last 
        # index int(T/d)-1 until index 1.
        seed=new_seeds_cond[t,0]
        seed_comp=new_seeds_cond[t,1]
        tf= T-i*d  
        ti=T-(i+1)*d  
        #x_pr[i]=sample_funct(x_new,N,d,sample_pars)
        #x_pr[i,0]=x_cond[i]
        x_final=np.zeros(N)+x_pr[t,j] #note here that we make a 
        # difference between j and and seed_comp, these are not necessarely the same,
        # in the case j==0, seed_comp is probably different from zero and correspond 
        # to the component of the original sample (there still a small change it's zero)
        int_G_back=Bridge_1d(ti,x_pr[t-1],tf,x_final,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,\
        r,r_pars,H,H_pars,l,d,N,seed,crossed=False,backward=True,j=seed_comp)      
        degen_log_weights=int_G_back+log_g_den(obs[t],x_final,g_den_par)\
        +np.log(aux_trans_den(ti,x_pr[t-1],tf,x_final,atdp))
        # The following quantity is needed to update the weights of the backward sampling(in order to
        # not compute them again, or at least not to identify them in certain functions)
        degen_log_weights_complete=degen_log_weights\
        -np.log(prop_trans_den(ti,x_pr[t-1],tf,x_final,sample_pars))
        degen_weights=pff.norm_logweights(degen_log_weights+log_weights[t-1])
        seed_counter_2+=int(2**l*d-1)
        np.random.seed(seed_counter_2)
        j=np.random.choice(N,p=degen_weights)
        if j==0:
            #print("seeds_cond[t-1] is: ",seeds_cond[t-1])
            new_seeds_cond[t-1]=seeds_cond[t-1]
        else:
            new_seeds_cond[t-1,0]=seed_counter-int(int(2**l*d-1))
            new_seeds_cond[t-1,1]=j
        seed_counter-=int(2**l*d-1)
        new_x_cond[t-1]=x_pr[t-1,j]
        new_lw_cond[t]=degen_log_weights_complete[j]
        new_int_G_cond[t]=int_G_back[j]
        # The previous asignation corresponds to the int_G with is function of both 
        # (W_{t-1},x_{t-1}) and (W_t,x_t), this quantity is stored to be used in future
        # computations
    # Since the last value for the corresponding lw and int_G of the backward sampling 
    # hasn't been assigned for the last value of the backward sampling, we do it here.
    new_int_G_cond[0]=int_Gs[0,j] # This sample does not depend on the previous sample
    # since x_0 is constant, so it depends but it's the same for everything.
    new_lw_cond[0]=log_weights[0,j]


    return new_func(log_weights, new_x_cond, new_lw_cond, new_int_G_cond, x_pr, new_seeds_cond)
#%%
"""log_weights=np.array(         [-2,45,np.inf, -16.77679532, -16.02682431, -16.60799942,
        -14.31034286, -14.41346517, -16.45366527, -15.15279068,
        -16.17232004, -16.34575267, -16.50447672, -16.65262256,
        -16.38596147, -16.73748419, -15.95163882, -16.4215678 ,
        -16.36592254, -16.33495451, -16.70293677, -16.60680778,
        -16.23237657, -16.82343339, -16.127507  , -15.68746368,
        -16.10064208, -15.19660269, -16.75310794, -16.80931415,
        -16.47325432, -16.33846415, -16.23210315, -15.06468704,
        -15.59191305, -14.44575692, -16.32714221, -15.68038466,
        -15.96671358, -15.72247402, -14.44059612, <-16.643048  ,
        -16.68646474, -15.96552026, -16.37835502, -16.34668516,
        -15.98145204, -16.51446542, -15.4363977 , -15.88327546,
        -16.46986656, -14.39341407])
weights=pff.norm_logweights(log_weights)
#print(weights)
N=50
x_last=np.array(range(len(log_weights)))
print(bdg.multi_samp_exp(weights,N,x_last,1))"""
#%%

def Gen_Cond_PF_bridge_back_samp(in_dist,in_dist_pars,x_cond,seeds_cond,b,A,Sig,\
    fi,b_til,Sig_til,r,H,update_func,sample_funct,sample_pars,\
    obs,obs_times,log_g_den,g_den_par, aux_trans_den, prop_trans_den, resamp_coef, l, d,N,seed):

    #(in_dist,in_dist_pars, b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,\
    #update_func,sample_funct,sample_pars,obs,obs_times,log_g_den,g_den_par, aux_trans_den,atdp,\
    #prop_trans_den, resamp_coef, l, d,N,seed)


    #(T,xin,b_ou,A,Sig_ou,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par):
    # ARGUMENTS: 
    # x_cond: the value of the particles at the observation points, it is a T/d dimensional array.
    # lw_cond: the log weights of the particles at the observation points, it is a T/d dimensional array.
    # these are not normalized, as we need them to normalize the new pf.
    # seeds_cond is an array with rank 2 and dims (int(T/d),2), where seeds_cond[:,0] are the sequence 
    # of seeds and seeds_cond[:,1] is the sequence of corresponding rows.
    # the argument of the Kenel x0 rank 1 dims (dim) 
    # the drift and diffusion are b and Sig, respectively, and they take
    # x(either a (N) dimensional or (N,N) dimensional array) and A as arguments for the drift and x and fi for the diffusion.
    # the level of discretization l, the distance of resampling, the number of
    # particles N.
    # atdp stands for auxiliar transition density parameters. 
    # seed: seeding that will identify the samples of the algorithm so it can be reproducible
    # and more importantly, we can access some samples without the need of storing them

    # Grad_b is a function that takes (x,A) as argument and computes the gradnient of b wrt the 
    # parameters A, and evaluates it a (x,A).
    # b_til,A_til,Sig_til,fi_til, are the analogous functions for the auxiliar process.
    # a difference is that their arguments are (t,x) for the drift and (t,x,fi_til) for the diffusion.
    # r is the function that computes the gradient of the log of the kernel of the auxiliar process
    # and it takes (t,x,T,x_pr,r_pars) as arguments.
    # H is the function that computes the Hessian of the log of the kernel of the auxiliar process
    # and it takes (t,x,T,x_pr,H_pars) as arguments.
    # crossed is the boolean that indicates if we need the computations for the crossed terms that 
    # are needed for the smoother.

    # regarding cond_seed, this vector is of size (int(T/d)-1,2), for the general version
    # we reduce it's size in 1 (int(T/d)-1) since that is the number of samples of brownian 
    # motion

    # OUTPUTs: 
    # log_weights: the log weights of the particles at the observation points, it is a T/d dimensional array.
    # As opposed to log_weights in Cond_PF_bridge this weights correspond not to the paths but 
    # each paticle.
    # x_pr: the particles at the observation points, it is a T/d dimensional array.
    # Again, these particles are not the paths but the particles without resampling 
    # new_lw_cond: the log weights of the particles at the observation points, it is a T/d dimensional array.
    # noticed that these quantities are such that correspond to the backward sampled path, meaning that 
    # its lw(x_{t-1},x_t) where x_t is the particle at time t.
    # new_int_G_cond: similartly to new_lw_cond regarding the dependence of the endpoints but 
    # with the integral of G.
    # new_x_cond: the particles sampled backwards at the observation points, it is a T/d dimensional array.
    # new_seeds_cond: the seeds of the particles sampled backwards at the observation points, it is a T/d dimensional array.
    # [log_weights,x_pr,new_lw_cond,new_int_G_cond,new_x_cond,new_seeds_cond]
    T=len(obs_times)*d
    # The reason T is defined like that is that in the previous version of this function (not generalized function) the
    # number of observations was parametrized by T/d, so in order to not change all varaibles that depend on the number of
    # observation I decided to define T as the number of observations times d.
    log_weights=np.zeros((int(T/d),N))
    new_x_cond=np.zeros(int(T/d)) # This variable stores the x's of the backward sampling
    new_lw_cond=np.zeros(int(T/d)) # This variable stores the log weights corresponding of the 
    # variables of the backward sampling.
    new_int_G_cond=np.zeros(int(T/d)-1) # This variable stores the log weights corresponding of the 
    # variables of the backward sampling.

    # In this function x_pr will store the resampled paths
    # this is different from previous functions, where x_pr stored the 
    # paths without resampling (yet).
    x_pr=np.zeros((int(T/d),N))
    int_Gs=np.zeros((int(T/d)-1,N))                      


    np.random.seed(seed)                   
    x_new=in_dist(in_dist_pars,N)
    x_new[0]=x_cond[0]
    x_pr[0]= copy.deepcopy(x_new)
    log_weights[0]=log_g_den(obs[0],x_pr[0],g_den_par)
    # resampling
    weights=pff.norm_logweights(log_weights[0])
    [part_resamp, x_new]=multi_samp_exp(weights,N,x_new,1)
    x_new[0]=x_cond[0]
    # here we make the first step which is different from the rest.
    Deltas=get_deltas(obs_times)
    intervals=obs_times[1:]-obs_times[:-1]

    for i in range(int(T/d)-1):

        np.random.seed(seed)
        tf=obs_times[i+1]
        ti=obs_times[i]
        x_pr[i+1]=sample_funct(x_new,N,tf-ti,sample_pars)
        x_pr[i+1,0]=x_cond[i+1]
        pars_0=[A,fi]
        [A_til,fi_til,r_pars,H_pars,\
        atdp]=update_func(pars_0,pars_0,ti,x_new,x_new,tf,x_pr[i+1],x_pr[i+1],levels=1)
        
        int_G=Bridge_1d(ti,x_new,tf,x_pr[i+1],b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,\
        r,r_pars,H,H_pars,l,d,N,seed,j=seeds_cond[i,1],cond_seed=seeds_cond[i,0],Dt=Deltas[i]/2**l)
        # regarding cond_seed, this vector is of size (int(T/d)-1,2), for the general version
        # we reduce it's size in 1 (int(T/d)-1) since that is the number of samples of brownian 
        # motion
        seed+=(int(2**l*intervals[i]/Deltas[i]) -1) # 2**l*intervals[i]/Deltas[i]-1 is the number of iterations of the EM 
        # method so we can get the approximation in the interval of lenght d 
        # (note that we subtract 1 bcs the last point of the bridge is already determined).
        int_Gs[i]=int_G
        log_weights[i+1]=int_G+log_g_den(obs[i+1],x_pr[i+1],g_den_par)\
        +np.log(aux_trans_den(ti,x_new,tf,x_pr[i+1],atdp))\
        -np.log(prop_trans_den(ti,x_new,tf,x_pr[i+1],sample_pars))
        
        weights=pff.norm_logweights(log_weights[i+1])
        x_last=copy.deepcopy(x_pr[i+1] )
        
        ESS=0
        
        # This code works only for resamp_coef=1, otherwise 
        # we need to consider some adaptative resampling scheme
        # for the conditional particle filter.
        [part_resamp, x_new]=multi_samp_exp(weights,N,x_last,1)
        x_new[0]=x_cond[i+1]
        
        """
        else:    
            #print("time is",i)
            x_new=x_last
            if i< int(T/d)-1:
                log_weights[i+1]=log_weights[i]
        """
        
    ####################################################################################
    # The following corresponds to the backward samples.  
    # Here we sample from the filtering distribution at time T.
    new_seeds_cond=np.zeros((int(T/d)-1,2),dtype=int) # new_cond_seed[:,0] are the seeds and new_cond_seed[:,1] 
    # are the corresponding rows.+" new_seeds_cond has size (int(T/d)-1,2) instead of (int(T/d),2)
    # bcs in this setting the number of intervals is int(T/d)-1.
    # what kind of seed should I include here? This seed should already be indexed by the np.random.seed(seed)
    # at the beginning of the previous for loop.

    seed_counter_2=seed
    np.random.seed(seed_counter_2)
    j=np.random.choice(N,p=weights)
    #int(2**l*intervals[i]/Deltas[i]) -1 
    #print("Global seed is: ",seed-(int(2**l*intervals[-1]/Deltas[-1]) -1))
    if j==0:
        new_seeds_cond[-1]=seeds_cond[-1]
        # It's important to note where we need to use new_seeds_cond[-1,1]
        # in this case, i.e., the arguments of Brige_1d below.
    else:
        new_seeds_cond[-1,0]=seed-(int(2**l*intervals[-1]/Deltas[-1])-1)
        new_seeds_cond[-1,1]=j

    new_x_cond[-1]=x_pr[-1,j]
    #new_lw_cond[-1]=log_weights[-1,j]
    seed_counter=seed-int(2**l*intervals[-1]/Deltas[-1] -1) # This variable is defined so 
    # we can keep track of the seed of the (N-1) particles.
    for i in range(int(T/d)-1):
        t=int(T/d)-i-1 # t is an index, not a time, it goes backward from the last 
        # index int(T/d)-1 until index 1.
        seed=new_seeds_cond[-(i+1),0]
        seed_comp=new_seeds_cond[-(i+1),1]
        tf= obs_times[-(i+1)]
        ti=obs_times[-(i+2)]
        #x_pr[i]=sample_funct(x_new,N,d,sample_pars)
        #x_pr[i,0]=x_cond[i]
        x_final=np.zeros(N)+x_pr[t,j] #note here that we make a 
        # difference between j and and seed_comp, these are not necessarely the same,
        # in the case j==0, seed_comp is probably different from zero and correspond 
        # to the component of the original sample (there still a small change it's zero)
        pars_0=[A,fi]
        [A_til,fi_til,r_pars,H_pars,\
        atdp]=update_func(pars_0,pars_0,ti,x_pr[t-1],x_pr[t-1],tf,x_final,x_final,levels=1)
        
        #pars_0=[A,fi]
        #[A_til,fi_til,r_pars,H_pars,\
        #atdp]=update_func(pars_0,pars_0,ti,x_pr[t-1],x_pr[t-1],tf,x_pr[t],x_pr[t],levels=1)
        
        
        int_G_back=Bridge_1d(ti,x_pr[t-1],tf,x_final,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,\
        r,r_pars,H,H_pars,l,d,N,seed,crossed=False,backward =True,j=seed_comp,Dt=Deltas[-(i+1)]/2**l)      
        degen_log_weights=int_G_back+log_g_den(obs[t],x_final,g_den_par)\
        +np.log(aux_trans_den(ti,x_pr[t-1],tf,x_final,atdp))
        # The following quantity is needed to update the weights of the backward sampling(in order to
        # not compute them again, or at least not to identify them in certain functions)
        degen_log_weights_complete=degen_log_weights\
        -np.log(prop_trans_den(ti,x_pr[t-1],tf,x_final,sample_pars))
        degen_weights=pff.norm_logweights(degen_log_weights+log_weights[t-1])
        seed_counter_2+=1
        np.random.seed(seed_counter_2)
        j=np.random.choice(N,p=degen_weights)
        if i< int(T/d)-2: 
            if j==0:
                #print("seeds_cond[t-1] is: ",seeds_cond[t-1])
                new_seeds_cond[-(i+2)]=seeds_cond[-(i+2)]
            else:
                new_seeds_cond[-(i+2),0]=seed_counter-int(2**l*intervals[-(i+2)]/Deltas[-(i+2)]-1)
                new_seeds_cond[-(i+2),1]=j
            #print("Global seed is: ",seed_counter-(int(2**l*intervals[-(i+2)]/Deltas[-(i+2)]) -1))
            seed_counter-=int(2**l*intervals[-(i+2)]/Deltas[-(i+2)]-1)
        new_x_cond[t-1]=x_pr[t-1,j]
        new_lw_cond[t]=degen_log_weights_complete[j]
        new_int_G_cond[t-1]=int_G_back[j]
        # The previous asignation corresponds to the int_G with is function of both 
        # (W_{t-1},x_{t-1}) and (W_t,x_t), this quantity is stored to be used in future
        # computations
    # Since the last value for the corresponding lw and int_G of the backward sampling 
    # hasn't been assigned for the last value of the backward sampling, we do it here.
    #new_int_G_cond[0]=int_Gs[0,j] # This sample does not depend on the previous sample
    # since x_0 is constant, so it depends but it's the same for everything.
    new_lw_cond[0]=log_weights[0,j]

    return new_func(log_weights, new_x_cond, new_lw_cond, new_int_G_cond, x_pr, new_seeds_cond)



def new_func(log_weights, new_x_cond, new_lw_cond, new_int_G_cond, x_pr, new_seeds_cond):
    return [log_weights,x_pr,new_lw_cond,new_int_G_cond,new_x_cond,new_seeds_cond]
    

def C_Cond_PF_bridge_back_samp(x_cond_0,x_cond_1,\
    seeds_cond_0,seeds_cond_1,t0,x0,T,b,A_0,A_1,Sig,fi_0,fi_1,b_til,A_til_0,A_til_1,Sig_til,fi_til_0,\
    fi_til_1,r,r_pars_0,r_pars_1,H,H_pars_0,H_pars_1,sample_funct,sample_pars,obs,\
    log_g_den,g_den_par_0,g_den_par_1, aux_trans_den,atdp_0,atdp_1,\
    prop_trans_den, ind_prop_trans_par_0,ind_prop_trans_par_1, l, d,N,seed,crossed=False):
    #(T,xin,b_ou,A,Sig_ou,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par):
    # ARGUMENTS:
    # x_cond: the value of the particles at the observation points, it is a T/d dimensional array.
    # lw_cond: the log weights of the particles at the observation points, it is a T/d dimensional array.
    # these are not normalized, as we need them to normalize the new pf.
    # seeds_cond is an array with rank 2 and dims (int(T/d),2), where seeds_cond[:,0] are the sequence 
    # of seeds and seeds_cond[:,1] is the sequence of corresponding rows.
    # the argument of the Kenel x0 rank 1 dims (dim) 
    # the drift and diffusion are b and Sig, respectively, and they take
    # x(either a (N) dimensional or (N,N) dimensional array) and A as arguments for the drift and x and fi for the diffusion.
    # the level of discretization l, the distance of resampling, the number of
    # particles N.
    # atdp stands for auxiliar transition density parameters. 
    # seed: seeding that will identify the samples of the algorithm so it can be reproducible
    # and more importantly, we can access some samples without the need of storing them

    # Grad_b is a function that takes (x,A) as argument and computes the gradnient of b wrt the 
    # parameters A, and evaluates it a (x,A).
    # b_til,A_til,Sig_til,fi_til, are the analogous functions for the auxiliar process.
    # a difference is that their arguments are (t,x) for the drift and (t,x,fi_til) for the diffusion.
    # r is the function that computes the gradient of the log of the kernel of the auxiliar process
    # and it takes (t,x,T,x_pr,r_pars) as arguments.
    # H is the function that computes the Hessian of the log of the kernel of the auxiliar process
    # and it takes (t,x,T,x_pr,H_pars) as arguments.
    # crossed is the boolean that indicates if we need the computations for the crossed terms that 
    # are needed for the smoother.

    # OUTPUTs: 
    # log_weights: the log weights of the particles at the observation points, it is a T/d dimensional array.
    # As opposed to log_weights in Cond_PF_bridge this weights correspond not to the paths but 
    # each paticle.
    # x_pr: the particles at the observation points, it is a T/d dimensional array.
    # Again, these particles are not the paths but the particles without resampling 
    # new_lw_cond: the log weights of the particles at the observation points, it is a T/d dimensional array.
    # noticed that these quantities are such that correspond to the backward sampled path, meaning that 
    # its lw(x_{t-1},x_t) where x_t is the particle at time t.
    # new_int_G_cond: similartly to new_lw_cond regarding the dependence of the endpoints but 
    # with the integral of G.
    # new_x_cond: the particles sampled backwards at the observation points, it is a T/d dimensional array.
    # new_seeds_cond: the seeds of the particles sampled backwards at the observation points, it is a T/d dimensional array.
    # [log_weights,x_pr,new_lw_cond,new_int_G_cond,new_x_cond,new_seeds_cond]

    log_weights_0=np.zeros((int(T/d),N))
    log_weights_1=np.zeros((int(T/d),N))
    new_x_cond_0=np.zeros(int(T/d)) # This variable stores the x's of the backward sampling
    new_x_cond_1=np.zeros(int(T/d))
    new_lw_cond_0=np.zeros(int(T/d)) # This variable stores the log weights corresponding of the 
    # variables of the backward sampling.
    new_lw_cond_1=np.zeros(int(T/d))
    new_int_G_cond_0=np.zeros(int(T/d)) # This variable stores the log weights corresponding of the 
    # variables of the backward sampling.
    new_int_G_cond_1=np.zeros(int(T/d))
    # In this function x_pr will store the resampled paths
    # this is different from previous functions, where x_pr stored the 
    # paths without resampling (yet).
    x_pr_0=np.zeros((int(T/d),N))
    x_pr_1=np.zeros((int(T/d),N))

    int_Gs_0=np.zeros((int(T/d),N))                      
    int_Gs_1=np.zeros((int(T/d),N))                      
    x_new_0=x0 # rank 1 with dimensions (dim)
    x_new_1=x0 # rank 1 with dimensions (dim)

    for i in range(int(T/d)):
        
        tf=t0+(i+1)*d
        ti=t0+(i)*d
        x_pr_0[i],x_pr_1[i]=sample_funct(x_new_0,x_new_1,N,d,sample_pars)
        x_pr_0[i,0]=x_cond_0[i]
        x_pr_1[i,0]=x_cond_1[i]
        """
        C_Bridge_1d(t0,x0_0,x0_1,T,x_p_0,x_p_1,b,A_0,A_1,Sig,fi_0,fi_1,b_til,A_til_0,A_til_1,\
        Sig_til,fi_til_0,fi_til_1,r,r_pars_0,r_pars_1,H,H_pars_0,H_pars_1,l,d,N,seed\
        ,crossed=False,backward=False,j_0=False,j_1=False,fd=False,N_pf=False,cond_seed_0=False,cond_seed_1=False):
        """
        int_G_0,int_G_1=C_Bridge_1d(ti,x_new_0,x_new_1,tf,x_pr_0[i],x_pr_1[i],b,A_0,A_1,\
        Sig,fi_0,fi_1,b_til,A_til_0,A_til_1,Sig_til,fi_til_0,fi_til_1,\
        r,r_pars_0,r_pars_1,H,H_pars_0,H_pars_1,l,d,N,seed,j_0=seeds_cond_0[i,1],\
        j_1=seeds_cond_1[i,1],cond_seed_0=seeds_cond_0[i,0],cond_seed_1=seeds_cond_1[i,0])
        seed+=int(2**l*d) -1 # 2**l*d -1 is the number of iterations of the EM 
        # method so we can get the approximation in the interval of lenght d 
        # (note that we subtract 1 bcs the last point of the bridge is already determined).
        int_Gs_0[i]=int_G_0
        int_Gs_1[i]=int_G_1
        new_log_weights_0=int_G_0+log_g_den(obs[i],x_pr_0[i],g_den_par_0,crossed=crossed)\
        +np.log(aux_trans_den(ti,x_new_0,tf,x_pr_0[i],atdp_0,crossed=crossed))\
        -np.log(prop_trans_den(ti,x_new_0,tf,x_pr_0[i], ind_prop_trans_par_0,crossed=crossed))
        log_weights_0[i]=log_weights_0[i]+new_log_weights_0

        new_log_weights_1=int_G_1+log_g_den(obs[i],x_pr_1[i],g_den_par_1,crossed=crossed)\
        +np.log(aux_trans_den(ti,x_new_1,tf,x_pr_1[i],atdp_1,crossed=crossed))\
        -np.log(prop_trans_den(ti,x_new_1,tf,x_pr_1[i],ind_prop_trans_par_1,crossed=crossed))
        log_weights_1[i]=log_weights_1[i]+new_log_weights_1
        #log_weights[i,0]=lw_cond[i]
        weights_0=pff.norm_logweights(log_weights_0[i])
        weights_1=pff.norm_logweights(log_weights_1[i])
        #print(yi,weights)
        #seed_val=i
        x_last_0=x_pr_0[i]        
        x_last_1=x_pr_1[i]        
        #ESS=1/np.sum(weights**2)
        ESS=0
        #print(ESS,resamp_coef*N)
        # This code works only for resamp_coef=1, otherwise 
        # we need to consider some adaptative resampling scheme 
        # for the conditional particle filter.
        
            
        #print("resampling at time ",i)
        #if True==False:
        #[part0,part1,x0_new,x1_new]=max_coup_sr(w0,w1,N,xi0[-1],xi1[-1],dim)
        #print(x_new.shape)
        [part_resamp_0,part_resamp_1,x_new_0, x_new_1]=max_coup_multi(weights_0,weights_1,N,x_last_0,x_last_1,1)
        x_new_0[0]=x_cond_0[i]
        x_new_1[0]=x_cond_1[i]
        #print(x_new.shape)
        """
        [part_resamp_0,part_resamp_1, x_new_0,x_new_1]=max_coup_multi\
        (weights_0,weights_1,N,x_last_0,x_last_1,1)
        #print(part_resamp_0,part_resamp_1)
        log_weights_0[:i+1]=log_weights_0[:i+1,part_resamp_0]
        int_Gs_0[:i+1]=int_Gs_0[:i+1,part_resamp_0]
        x_pr_0[:i+1]=x_pr_0[:i+1,part_resamp_0]
        log_weights_1[:i+1]=log_weights_1[:i+1,part_resamp_1]
        int_Gs_1[:i+1]=int_Gs_1[:i+1,part_resamp_1]
        x_pr_1[:i+1]=x_pr_1[:i+1,part_resamp_1]
        """
    #x_pr[i]=x_new

    ####################################################################################
    # The following corresponds to the backward samples.  
    # Here we sample from the filtering distribution at time T.
    new_seeds_cond_0=np.zeros((int(T/d),2),dtype=int) # new_cond_seed[:,0] are the seeds and new_cond_seed[:,1] 
    new_seeds_cond_1=np.zeros((int(T/d),2),dtype=int)

    # are the corresponding rows.
    #j_0=np.random.choice(N,p=weights_0)
    #j_1=np.random.choice(N,p=weights_1)
    # som_0, som_1 is irelevant since we don't use then ahead.
    [j_0,j_1,som_0,som_1]=max_coup_multi_sing_samp(weights_0,weights_1,x_new_0,x_new_1,1)
    j_0=j_0[0]
    j_1=j_1[0]

    if j_0==0:

        new_seeds_cond_0[-1]=seeds_cond_0[-1]
        # It's important to note where we need to use new_seeds_cond[-1,1]
        # in this case, i.e., the arguments of Brige_1d below.

    else:
        new_seeds_cond_0[-1,0]=seed-(int(2**l*d) -1)
        new_seeds_cond_0[-1,1]=j_0


    if j_1==0:
        new_seeds_cond_1[-1]=seeds_cond_1[-1]
        # It's important to note where we need to use new_seeds_cond[-1,1]
        # in this case, i.e., the arguments of Brige_1d below.

    else:
        new_seeds_cond_1[-1,0]=seed-(2**l*d -1)
        new_seeds_cond_1[-1,1]=j_1

    new_x_cond_0[-1]=x_pr_0[-1,j_0]
    new_x_cond_1[-1]=x_pr_1[-1,j_1]
    #new_lw_cond[-1]=log_weights[-1,j]
    seed_counter=seed-(int(2**l*d )-1) # This variable is defined so 
    # we can keep track of the seed of the (N-1) particles.
    for i in range(int(T/d)-1): 
        t=int(T/d)-i-1 # t is an index, not a time, it goes backward from the last 
        # index int(T/d)-1 until index 1.
        seed_0=new_seeds_cond_0[t,0]
        seed_1=new_seeds_cond_1[t,0]
        seed_comp_0=new_seeds_cond_0[t,1]
        seed_comp_1=new_seeds_cond_1[t,1]
        tf= T-i*d  
        ti=T-(i+1)*d
        #x_pr[i]=sample_funct(x_new,N,d,sample_pars)
        #x_pr[i,0]=x_cond[i]
        x_final_0=np.zeros(N)+x_pr_0[t,j_0] #note here that we make a 
        x_final_1=np.zeros(N)+x_pr_1[t,j_1]
        # difference between j and and seed_comp, these are not necessarely the same,
        # in the case j==0, seed_comp is probably different from zero and correspond 
        # to the component of the original sample (there still a small change it's zero)

        """
        C_Bridge_1d(t0,x0_0,x0_1,T,x_p_0,x_p_1,b,A_0,A_1,Sig,fi_0,fi_1,b_til,A_til_0,A_til_1,\
        Sig_til,fi_til_0,fi_til_1,r,r_pars_0,r_pars_1,H,H_pars_0,H_pars_1,l,d,N,seed\
        ,crossed=False,backward=False,j_0=False,j_1=False,fd=False,N_pf=False,cond_seed_0=False,cond_seed_1=False):
        """

        # the parameter seed in the following function is irrelevant since it's not used when backward is true
        int_G_back_0,int_G_back_1=C_Bridge_1d(ti,x_pr_0[t-1],x_pr_1[t-1],tf,x_final_0,x_final_1,b,A_0,A_1,\
        Sig,fi_0,fi_1,b_til,A_til_0,A_til_1,Sig_til,fi_til_0,fi_til_1,\
        r,r_pars_0,r_pars_1,H,H_pars_0,H_pars_1,l,d,N,seed,crossed=False,backward=True,j_0=seed_comp_0,j_1=seed_comp_1,\
        cond_seed_0=seed_0,cond_seed_1=seed_1)      
        degen_log_weights_0=int_G_back_0+log_g_den(obs[t],x_final_0,g_den_par_0,crossed=crossed)\
        +np.log(aux_trans_den(ti,x_pr_0[t-1],tf,x_final_0,atdp_0,crossed=crossed))

        degen_log_weights_1=int_G_back_1+log_g_den(obs[t],x_final_1,g_den_par_1,crossed=crossed)\
        +np.log(aux_trans_den(ti,x_pr_1[t-1],tf,x_final_1,atdp_1,crossed=crossed))
        # The following quantity is needed to update the weights of the backward sampling(in order to
        # not compute them again, or at least not to identify them in certain functions)
        degen_log_weights_complete_0=degen_log_weights_0\
        -np.log(prop_trans_den(ti,x_pr_0[t-1],tf,x_final_0,ind_prop_trans_par_0,crossed=crossed))
        degen_log_weights_complete_1=degen_log_weights_1\
        -np.log(prop_trans_den(ti,x_pr_1[t-1],tf,x_final_1,ind_prop_trans_par_1,crossed=crossed))
        degen_weights_0=pff.norm_logweights(degen_log_weights_0+log_weights_0[t-1])
        degen_weights_1=pff.norm_logweights(degen_log_weights_1+log_weights_1[t-1])
        ir_v_0=np.zeros(N)# irrelenvant vector
        ir_v_1=np.zeros(N)
        [j_0,j_1, ir_v_0,ir_v_1]=max_coup_multi_sing_samp(degen_weights_0,degen_weights_1,ir_v_0,ir_v_1,1)
        #print(j_0,j_1)
        j_0=j_0[0]
        j_1=j_1[0]
        
        if j_0==0:
            #print("seeds_cond[t-1] is: ",seeds_cond[t-1])
            new_seeds_cond_0[t-1]=seeds_cond_0[t-1]
        else:
            new_seeds_cond_0[t-1,0]=seed_counter-int(int(2**l*d-1))
            new_seeds_cond_0[t-1,1]=j_0

        if j_1==0:
            #print("seeds_cond[t-1] is: ",seeds_cond[t-1])
            new_seeds_cond_1[t-1]=seeds_cond_1[t-1]
        else:
            new_seeds_cond_1[t-1,0]=seed_counter-int(int(2**l*d-1))
            new_seeds_cond_1[t-1,1]=j_1
        seed_counter-=int(int(2**l*d-1))
        new_x_cond_0[t-1]=x_pr_0[t-1,j_0]
        new_x_cond_1[t-1]=x_pr_1[t-1,j_1]
        new_lw_cond_0[t]=degen_log_weights_complete_0[j_0]
        new_int_G_cond_0[t]=int_G_back_0[j_0]
        new_lw_cond_1[t]=degen_log_weights_complete_1[j_1]
        new_int_G_cond_1[t]=int_G_back_1[j_1]
        # The previous asignation corresponds to the int_G with is function of both 
        # (W_{t-1},x_{t-1}) and (W_t,x_t), this quantity is stored to be used in future
        # computations
    # Since the last value for the corresponding lw and int_G of the backward sampling 
    # hasn't been assigned for the last value of the backward sampling, we do it here.
    new_int_G_cond_0[0]=int_Gs_0[0,j_0] # This sample does not depend on the previous sample
    # since x_0 is constant, so it depends but it's the same for everything.
    new_lw_cond_0[0]=log_weights_0[0,j_0]

    new_int_G_cond_1[0]=int_Gs_1[0,j_1] # This sample does not depend on the previous sample
    # since x_0 is constant, so it depends but it's the same for everything.
    new_lw_cond_1[0]=log_weights_1[0,j_1]

    return [log_weights_0, log_weights_1, x_pr_0, x_pr_1, new_lw_cond_0,new_lw_cond_1\
    ,new_int_G_cond_0,new_int_G_cond_1,new_x_cond_0,new_x_cond_1,new_seeds_cond_0 ,new_seeds_cond_1]

# Gen_C_PF_bridge(in_dist,in_dist_pars, b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,\
#    update_func,max_sample_funct,sample_pars,obs,obs_times,log_g_den,g_den_par, aux_trans_den,atdp,\
#    prop_trans_den, ind_prop_trans_par,resamp_coef, l, d,N,seed):

#C_Cond_PF_bridge_back_samp(x_cond_0,x_cond_1,\
#    seeds_cond_0,seeds_cond_1,t0,x0,T,b,A_0,A_1,Sig,fi_0,fi_1,b_til,A_til_0,A_til_1,Sig_til,fi_til_0,\
#    fi_til_1,r,r_pars_0,r_pars_1,H,H_pars_0,H_pars_1,sample_funct,sample_pars,obs,\
#    log_g_den,g_den_par_0,g_den_par_1, aux_trans_den,atdp_0,atdp_1,\
#    prop_trans_den, ind_prop_trans_par_0,ind_prop_trans_par_1, l, d,N,seed,crossed=False):


def Gen_C_Cond_PF_bridge_back_samp(in_dist,in_dist_pars,x_cond_0,x_cond_1 ,seeds_cond_0,\
    seeds_cond_1,b,A_0,A_1,Sig,fi_0,fi_1,b_til,Sig_til,\
    r,H,update_func,max_sample_funct,sample_pars,\
    obs,obs_times,log_g_den,g_den_par_0,g_den_par_1, aux_trans_den,\
    prop_trans_den,ind_prop_trans_par_0,ind_prop_trans_par_1, resamp_coef, l, d,N,seed):

    #(in_dist,in_dist_pars, b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,\
    #update_func,sample_funct,sample_pars,obs,obs_times,log_g_den,g_den_par, aux_trans_den,atdp,\
    #prop_trans_den, resamp_coef, l, d,N,seed)


    #(T,xin,b_ou,A,Sig_ou,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par):
    # ARGUMENTS: 
    # x_cond: the value of the particles at the observation points, it is a T/d dimensional array.
    # lw_cond: the log weights of the particles at the observation points, it is a T/d dimensional array.
    # these are not normalized, as we need them to normalize the new pf.
    # seeds_cond is an array with rank 2 and dims (int(T/d),2), where seeds_cond[:,0] are the sequence 
    # of seeds and seeds_cond[:,1] is the sequence of corresponding rows.
    # the argument of the Kenel x0 rank 1 dims (dim) 
    # the drift and diffusion are b and Sig, respectively, and they take
    # x(either a (N) dimensional or (N,N) dimensional array) and A as arguments for the drift and x and fi for the diffusion.
    # the level of discretization l, the distance of resampling, the number of
    # particles N.
    # atdp stands for auxiliar transition density parameters. 
    # seed: seeding that will identify the samples of the algorithm so it can be reproducible
    # and more importantly, we can access some samples without the need of storing them

    # Grad_b is a function that takes (x,A) as argument and computes the gradnient of b wrt the 
    # parameters A, and evaluates it a (x,A).
    # b_til,A_til,Sig_til,fi_til, are the analogous functions for the auxiliar process.
    # a difference is that their arguments are (t,x) for the drift and (t,x,fi_til) for the diffusion.
    # r is the function that computes the gradient of the log of the kernel of the auxiliar process
    # and it takes (t,x,T,x_pr,r_pars) as arguments.
    # H is the function that computes the Hessian of the log of the kernel of the auxiliar process
    # and it takes (t,x,T,x_pr,H_pars) as arguments.
    # crossed is the boolean that indicates if we need the computations for the crossed terms that 
    # are needed for the smoother.

    # regarding cond_seed, this vector is of size (int(T/d)-1,2), for the general version
    # we reduce it's size in 1 (int(T/d)-1) since that is the number of samples of brownian 
    # motion

    # OUTPUTs: 
    # log_weights: the log weights of the particles at the observation points, it is a T/d dimensional array.
    # As opposed to log_weights in Cond_PF_bridge this weights correspond not to the paths but 
    # each paticle.
    # x_pr: the particles at the observation points, it is a T/d dimensional array.
    # Again, these particles are not the paths but the particles without resampling 
    # new_lw_cond: the log weights of the particles at the observation points, it is a T/d dimensional array.
    # noticed that these quantities are such that correspond to the backward sampled path, meaning that 
    # its lw(x_{t-1},x_t) where x_t is the particle at time t.
    # new_int_G_cond: similartly to new_lw_cond regarding the dependence of the endpoints but 
    # with the integral of G.
    # new_x_cond: the particles sampled backwards at the observation points, it is a T/d dimensional array.
    # new_seeds_cond: the seeds of the particles sampled backwards at the observation points, it is a T/d dimensional array.
    # [log_weights,x_pr,new_lw_cond,new_int_G_cond,new_x_cond,new_seeds_cond]
    T=len(obs_times)*d
    log_weights_0=np.zeros((int(T/d),N))
    log_weights_1=np.zeros((int(T/d),N))
    new_x_cond_0=np.zeros(int(T/d)) # This variable stores the x's of the backward sampling
    new_x_cond_1=np.zeros(int(T/d)) # This variable stores the x's of the backward sampling
    new_lw_cond_0=np.zeros(int(T/d)) # This variable stores the log weights corresponding of the 
    new_lw_cond_1=np.zeros(int(T/d)) # This variable stores the log weights corresponding of the 
    # variables of the backward sampling.
    new_int_G_cond_0=np.zeros(int(T/d)-1) # This variable stores the log weights corresponding of the 
    # variables of the backward sampling.
    new_int_G_cond_1=np.zeros(int(T/d)-1)
    # In this function x_pr will store the resampled paths
    # this is different from previous functions, where x_pr stored the 
    # paths without resampling (yet).
    x_pr_0=np.zeros((int(T/d),N),dtype=float)
    x_pr_1=np.zeros((int(T/d),N),dtype=float)
    
    int_Gs_0=np.zeros((int(T/d)-1,N),dtype=float)                      
    int_Gs_1=np.zeros((int(T/d)-1,N),dtype=float)                      

    np.random.seed(seed)
    x_new_0,x_new_1=in_dist(in_dist_pars,N)
    x_new_0[0]=x_cond_0[0]
    x_new_1[0]=x_cond_1[0]
    x_pr_0[0]= copy.deepcopy(x_new_0)
    x_pr_1[0]= copy.deepcopy(x_new_1)
    log_weights_0[0]=log_g_den(obs[0],x_pr_0[0],g_den_par_0)
    log_weights_1[0]=log_g_den(obs[0],x_pr_1[0],g_den_par_1)
    # resampling
    weights_0=pff.norm_logweights(log_weights_0[0])
    weights_1=pff.norm_logweights(log_weights_1[0])
    [part_resamp_0,part_resamp_1, x_new_0,x_new_1]=max_coup_multi\
    (weights_0,weights_1,N,x_new_0,x_new_1,1)
    
    x_new_0[0]=x_cond_0[0]
    x_new_1[0]=x_cond_1[0]

    # here we make the first step which is different from the rest.
    
    Deltas=get_deltas(obs_times)
    intervals=obs_times[1:]-obs_times[:-1]
    #seed+=int(2**l*intervals[0]/Deltas[0]) -1

    for i in range(int(T/d)-1):

        np.random.seed(seed)
        
        tf=obs_times[i+1]
        ti=obs_times[i]

        x_pr_0[i+1],x_pr_1[i+1]=max_sample_funct(x_new_0,x_new_1,N,tf-ti,sample_pars)
        x_pr_0[i+1,0]=x_cond_0[i+1]
        x_pr_1[i+1,0]=x_cond_1[i+1]

        pars_0=[A_0,fi_0]
        pars_1=[A_1,fi_1]

        [A_til_0,A_til_1,fi_til_0,fi_til_1,r_pars_0,r_pars_1,H_pars_0,H_pars_1,\
        atdp_0,atdp_1]=update_func(pars_0,pars_1,ti,x_new_0,x_new_1,tf,x_pr_0[i+1],x_pr_1[i+1],levels=2)

        int_G_0,int_G_1=C_Bridge_1d(ti,x_new_0,x_new_1,tf,x_pr_0[i+1],x_pr_1[i+1],b,A_0,A_1,\
        Sig,fi_0,fi_1,b_til,A_til_0,A_til_1,Sig_til,fi_til_0,fi_til_1,r,r_pars_0,r_pars_1,\
        H,H_pars_0,H_pars_1,l,d,N,seed,j_0=seeds_cond_0[i,1],j_1=seeds_cond_1[i,1],\
        cond_seed_0=seeds_cond_0[i,0],cond_seed_1=seeds_cond_1[i,0],Dt=Deltas[i]/2**l)
        #seed+=int(2**l*intervals[i]/2**l-1)

        """
        int_G_0,int_G_1=C_Bridge_1d(ti,x_new_0,x_new_1,tf,x_pr_0[i],x_pr_1[i],b,A_0,A_1,\
        Sig,fi_0,fi_1,b_til,A_til_0,A_til_1,Sig_til,fi_til_0,fi_til_1,\
        r,r_pars_0,r_pars_1,H,H_pars_0,H_pars_1,l,d,N,seed,j_0=seeds_cond_0[i,1],\
        j_1=seeds_cond_1[i,1],cond_seed_0=seeds_cond_0[i,0],cond_seed_1=seeds_cond_1[i,0])
        """
        #int_G=Bridge_1d(ti,x_new,tf,x_pr[i+1],b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,\
        #r,r_pars,H,H_pars,l,d,N,seed,j=seeds_cond[i,1],cond_seed=seeds_cond[i,0],Dt=Deltas[i]/2**l)
        # regarding cond_seed, this vector is of size (int(T/d)-1,2), for the general version
        # we reduce it's size in 1 (int(T/d)-1) since that is the number of samples of brownian 
        # motion
        #print("Global seed is: ",seed)
        seed+=int(2**l*intervals[i]/Deltas[i]) -1 # 2**l*intervals[i]/Deltas[i]-1 is the number of iterations of the EM 
        # method so we can get the approximation in the interval of lenght d 
        # (note that we subtract 1 bcs the last point of the bridge is already determined).
        int_Gs_0[i]=int_G_0
        int_Gs_1[i]=int_G_1
        

        new_log_weights_0=int_G_0+log_g_den(obs[i+1],x_pr_0[i+1],g_den_par_0)\
        +np.log(aux_trans_den(ti,x_new_0,tf,x_pr_0[i+1],atdp_0))\
        -np.log(prop_trans_den(ti,x_new_0,tf,x_pr_0[i+1],ind_prop_trans_par_0))


        new_log_weights_1=int_G_1+log_g_den(obs[i+1],x_pr_1[i+1],g_den_par_1)\
        +np.log(aux_trans_den(ti,x_new_1,tf,x_pr_1[i+1],atdp_1))\
        -np.log(prop_trans_den(ti,x_new_1,tf,x_pr_1[i+1],ind_prop_trans_par_1))

        log_weights_0[i+1]=new_log_weights_0
        log_weights_1[i+1]=new_log_weights_1
        weights_0=pff.norm_logweights(log_weights_0[i+1])
        weights_1=pff.norm_logweights(log_weights_1[i+1])
        #print(yi,weights)
        #seed_val=i
        #print(weights.shape)
        x_last_0=copy.deepcopy(x_pr_0[i+1])
        x_last_1=copy.deepcopy(x_pr_1[i+1])
        #ESS=1/np.sum(weights**2)
        ESS=0
        #print(ESS,resamp_coef*N)
        # This code works only for resamp_coef=1, otherwise 
        # we need to consider some adaptative resampling scheme
        # for the conditional particle filter.
        #if ESS<resamp_coef*N and i<int(T/d)-1:
            
            #print("resampling at time ",i)
        #if True==False:
            #[part0,part1,x0_new,x1_new]=max_coup_sr(w0,w1,N,xi0[-1],xi1[-1],dim)
            #print(x_new.shape)
        [part_resamp_0,part_resamp_1, x_new_0, x_new_1]=max_coup_multi\
        (weights_0,weights_1,N,x_last_0,x_last_1,1)
        x_new_0[0]=x_cond_0[i+1]
        x_new_1[0]=x_cond_1[i+1]
        #print(x_new.shape)
        """
        else:    
            #print("time is",i)
            x_new=x_last
            if i< int(T/d)-1:
                log_weights[i+1]=log_weights[i]
        """
        #print(i)
    #x_pr[i]=x_new
    ####################################################################################
    # The following corresponds to the backward samples.  
    # Here we sample from the filtering distribution at time T.
    new_seeds_cond_0=np.zeros((int(T/d)-1,2),dtype=int) # new_cond_seed[:,0] are the seeds and new_cond_seed[:,1] 
    # are the corresponding rows.+" new_seeds_cond has size (int(T/d)-1,2) instead of (int(T/d),2)
    # bcs in this setting the number of intervals is int(T/d)-1.
    new_seeds_cond_1=np.zeros((int(T/d)-1,2),dtype=int)
    # what kind of seed should I include here? This seed should already be indexed by the np.random.seed(seed)
    # at the beginning of the previous for loop.
    
    seed_counter_2=seed
    np.random.seed(seed_counter_2)
    [j_0,j_1,som_0,som_1]=max_coup_multi_sing_samp(weights_0,weights_1,x_new_0,x_new_1,1)
    j_0=j_0[0]
    j_1=j_1[0]
    if j_0==0:
        new_seeds_cond_0[-1]=seeds_cond_0[-1]
        # It's important to note where we need to use new_seeds_cond[-1,1]
        # in this case, i.e., the arguments of Brige_1d below.
    else:
        new_seeds_cond_0[-1,0]=seed-(int(2**l*intervals[-1]/Deltas[-1]) -1)
        new_seeds_cond_0[-1,1]=j_0

    if j_1==0:
        new_seeds_cond_1[-1]=seeds_cond_1[-1]
        # It's important to note where we need to use new_seeds_cond[-1,1]
        # in this case, i.e., the arguments of Brige_1d below.
    else:
        new_seeds_cond_1[-1,0]=seed-(int(2**l*intervals[-1]/Deltas[-1]) -1)
        new_seeds_cond_1[-1,1]=j_1
    #print("Global seed is: ",seed-(int(2**l*intervals[-1]/Deltas[-1]) -1))

    new_x_cond_0[-1]=x_pr_0[-1,j_0]
    new_x_cond_1[-1]=x_pr_1[-1,j_1]
    #new_lw_cond[-1]=log_weights[-1,j]
    seed_counter=seed-int(2**l*intervals[-1]/Deltas[-1] -1) # This variable is defined so 
    # we can keep track of the seed of the (N-1) particles.
    for i in range(int(T/d)-1):
        t=int(T/d)-i-1 # t is an index, not a time, it goes backward from the last 
        # index int(T/d)-1 until index 1.
        seed_0=new_seeds_cond_0[-(i+1),0]
        seed_comp_0=new_seeds_cond_0[-(i+1),1]
        
        seed_1=new_seeds_cond_1[-(i+1),0]
        seed_comp_1=new_seeds_cond_1[-(i+1),1]
        
        
        tf= obs_times[-(i+1)]
        ti=obs_times[-(i+2)]
        #x_pr[i]=sample_funct(x_new,N,d,sample_pars)
        #x_pr[i,0]=x_cond[i]
        x_final_0=np.zeros(N)+x_pr_0[t,j_0] #note here that we make a 
        # difference between j and and seed_comp, these are not necessarely the same,
        # in the case j==0, seed_comp is probably different from zero and correspond 
        # to the component of the original sample (there still a small change it's zero)
        x_final_1=np.zeros(N)+x_pr_1[t,j_1]
        pars_0=[A_0,fi_0]
        pars_1=[A_1,fi_1]
        [A_til_0,A_til_1,fi_til_0,fi_til_1,r_pars_0,r_pars_1,H_pars_0,H_pars_1,\
        atdp_0,atdp_1]=update_func(pars_0,pars_1,ti,x_pr_0[t-1],x_pr_1[t-1],tf,x_final_0,x_final_1,levels=2)
        
        int_G_back_0,int_G_back_1=C_Bridge_1d(ti,x_pr_0[t-1],x_pr_1[t-1],tf,x_final_0,x_final_1,b,A_0,A_1,\
        Sig,fi_0,fi_1,b_til,A_til_0,A_til_1,Sig_til,fi_til_0,fi_til_1,\
        r,r_pars_0,r_pars_1,H,H_pars_0,H_pars_1,l,d,N,seed,crossed=False,backward =True,j_0=seed_comp_0,\
        j_1=seed_comp_1, cond_seed_0=seed_0,cond_seed_1=seed_1, Dt=Deltas[-(i+1)]/2**l) 

        degen_log_weights_0=int_G_back_0+log_g_den(obs[t],x_final_0,g_den_par_0)\
        +np.log(aux_trans_den(ti,x_pr_0[t-1],tf,x_final_0,atdp_0))
        
        degen_log_weights_1=int_G_back_1+log_g_den(obs[t],x_final_1,g_den_par_1)\
        +np.log(aux_trans_den(ti,x_pr_1[t-1],tf,x_final_1,atdp_1))
        
        # The following quantity is needed to update the weights of the backward sampling(in order to
        # not compute them again, or at least not to identify them in certain functions)
        degen_log_weights_complete_0=degen_log_weights_0\
        -np.log(prop_trans_den(ti,x_pr_0[t-1],tf,x_final_0,ind_prop_trans_par_0))

        degen_log_weights_complete_1=degen_log_weights_1\
        -np.log(prop_trans_den(ti,x_pr_1[t-1],tf,x_final_1,ind_prop_trans_par_1))

        degen_weights_0=pff.norm_logweights(degen_log_weights_0+log_weights_0[t-1])

        degen_weights_1=pff.norm_logweights(degen_log_weights_1+log_weights_1[t-1])
        seed_counter_2+=1
        np.random.seed(seed_counter_2)

        ir_v_0=np.zeros(N)# irrelenvant vector
        ir_v_1=np.zeros(N)

        [j_0,j_1,ir_v_0,ir_v_1]=max_coup_multi_sing_samp(degen_weights_0,degen_weights_1,ir_v_0,ir_v_1,1)
        j_0=j_0[0]
        j_1=j_1[0]
        if i< int(T/d)-2: 
            if j_0==0:
                #print("seeds_cond[t-1] is: ",seeds_cond[t-1])
                new_seeds_cond_0[-(i+2)]=seeds_cond_0[-(i+2)]
            else:
                new_seeds_cond_0[-(i+2),0]=seed_counter-int(2**l*intervals[-(i+2)]/Deltas[-(i+2)]-1)
                new_seeds_cond_0[-(i+2),1]=j_0
            
            if j_1==0:
                #print("seeds_cond[t-1] is: ",seeds_cond[t-1])
                new_seeds_cond_1[-(i+2)]=seeds_cond_1[-(i+2)]
            else:
                new_seeds_cond_1[-(i+2),0]=seed_counter-int(2**l*intervals[-(i+2)]/Deltas[-(i+2)]-1)
                new_seeds_cond_1[-(i+2),1]=j_1

            #print("Global seed is: ",seed_counter-int(2**l*intervals[-(i+2)]/Deltas[-(i+2)]-1))
            seed_counter-=int(2**l*intervals[-(i+2)]/Deltas[-(i+2)]-1)
        new_x_cond_0[t-1]=x_pr_0[t-1,j_0]
        new_lw_cond_0[t]=degen_log_weights_complete_0[j_0]
        new_int_G_cond_0[t-1]=int_G_back_0[j_0]

        new_x_cond_1[t-1]=x_pr_1[t-1,j_1]
        new_lw_cond_1[t]=degen_log_weights_complete_1[j_1]
        new_int_G_cond_1[t-1]=int_G_back_1[j_1]

        # The previous asignation corresponds to the int_G with is function of both 
        # (W_{t-1},x_{t-1}) and (W_t,x_t), this quantity is stored to be used in future
        # computations
    # Since the last value for the corresponding lw and int_G of the backward sampling 
    # hasn't been assigned for the last value of the backward sampling, we do it here.
    #new_int_G_cond_0[0]=int_Gs_0[0,j_0] # This sample does not depend on the previous sample
    # since x_0 is constant, so it depends but it's the same for everything.
    #new_int_G_cond_1[0]=int_Gs_1[0,j_1]

    new_lw_cond_0[0]=log_weights_0[0,j_0]

    new_lw_cond_1[0]=log_weights_1[0,j_1]

    return [log_weights_0, log_weights_1, x_pr_0, x_pr_1, new_lw_cond_0,new_lw_cond_1\
    ,new_int_G_cond_0,new_int_G_cond_1,new_x_cond_0,new_x_cond_1,new_seeds_cond_0 ,new_seeds_cond_1]


#%%

#%%
# CONDITIONAL PARTICLE FILTER WITH ADDITIONAL BACKWARD SAMPLING TEST
# In the following we form a chain of conditional particle filters, we initialize with
# with the regular PF
# FOR THE OU PROCESS 

"""
N=10
x0_sca=1.2
x0=x0_sca+np.zeros(N)
l=6
alpha=0
T=10
t0=0
l_d=0
d=2**(l_d)
theta=-0.3
sigma=1.2
sd=0.55
#sigma_aux=0.2
theta_aux=theta-0.2
sigma_aux=sigma
#print(theta)
np.random.seed(1)
collection_input=[ b_ou_1d,theta,Sig_ou_1d,sigma]
resamp_coef=1
l_max=10
x_true=gen_gen_data_1d(T,x0_sca,l_max,collection_input)
x_reg=cut(T,l_max,-l_d,x_true)[1:]
# x_reg stands for x regular
obs=gen_obs(x_reg,g_normal_1d,sd)
np.random.seed(3)
#print(x_pr)
n_steps=int(2**l*(T-t0))
dt=2**(-l)
x=np.zeros((n_steps+1,N))
x[0]=x0
x_pr=np.random.normal(x0,np.sqrt(sigma**2*(T-t0)),N)
x[-1]=x_pr
int_G=np.zeros(N)
t_test=2.
"""
#print("t0 is->",t0, " x0 is->",x0," T is->",T," theta is->",theta," sigma is->",sigma,\
#" theta_aux is->",theta_aux," sigma_aux is->",sigma_aux," obs is->",obs," cov is->",cov,\
#    " resamp_coef is->",resamp_coef," l is->",l," d is->",d," N is->",N)
#print(obs.shape)
#%%
"""
x0_sca=1.2
x0=x0_sca
l=10
T=2
t0=0
l_d=0
d=2**(l_d)
theta_true=-0.3
sigma_true=1.2
np.random.seed(7)
collection_input=[b_ou_1d,theta_true,Sig_ou_1d,sigma_true]
resamp_coef=1
l_max=10
x_true= gen_gen_data_1d(T,x0_sca,l_max,collection_input)
x_reg=cut(T,l_max,-l_d,x_true)[1:]
times=np.array(range(t0,int(T/d)+1))*d
l_times=np.arange(t0,T,2**(-l))
l_max_times=np.arange(t0,T,2**(-l_max))
sd_true=0.55
np.random.seed(3)
d_times=np.array(range(t0+d,int(T/d)+1))*d
obs=gen_obs(x_reg,g_normal_1d,sd_true)

#obs=np.array([])
resamp_coef=1
N=50
start=time.time()
mcmc_links=100
fd=1e-4
theta_in=-0.6
sigma_in=1.6
sd_in=sd_true
#obs=np.array([theta_in*x0_sca+np.sqrt(1/2)+1])
theta_in_fd=theta_in+fd

sigma_in_fd=sigma_in+fd
sigma_in_aux=sigma_in
theta_in_aux=theta_in+0.2
sigma_in_aux_fd=sigma_in_aux+fd

gamma=0.2
alpha=0.5
seed=2393+5*10
x0=x0_sca+np.zeros(N)
inputs=[]
samples=40*10
CL=0.10381823
CL0=3.31636298
CP0=1.78431146
CP=0.00287355
s0=2**3
pmax=7
l0=4
lmax=10
beta_l=1
beta_p=1
"""

#%%

#(t0,x0,T,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,\
    #sample_funct,sample_pars,obs,log_g_den,g_den_par, aux_trans_den,atdp,\
    #prop_trans_den, resamp_coef, l, d,N,crossed=False):

# FOR THE ALPHA AUXILIARY PROCESS
"""
[log_weights,x_pr,int_Gs]=PF_bridge(t0,x0,T,b_ou_1d,theta,Sig_ou_1d,sigma,b_artificial_1d,0,Sig_alpha,[alpha,sigma,T],\
r_normal,[new_alpha_trans_sd,[alpha,sigma]],H_normal_1d,[new_alpha_trans_sd,[alpha,sigma]],\
sampling_alpha_trans_props, [alpha,sigma],obs,log_g_normal_den,cov,\
aux_trans_den_alpha,[alpha,sigma],aux_trans_den_alpha,\
resamp_coef,l,d, N)
"""
# FOR THE OU AUXILIARY PROCESS
#B0=100
#Bes_l=7
#eBes=B0*2**np.arange(Bes_l)
#print(eBes)
"""
start=time.time()
B=2
samples=50
seed=0
mcmc_mean=np.zeros((samples,2,int(T/d)))
resamp_coef=1
l=4
for i in range(samples):

    np.random.seed(i)
    #print("Seed feeded to PF_bridge is: ",seed)
    [log_weights,int_Gs,x_pr]=PF_bridge(t0,x0,T,b_ou_1d,theta,Sig_ou_1d,sigma,b_ou_aux,theta_aux,Sig_ou_aux,sigma_aux,\
    r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,[ou_sd,[theta_aux,sigma_aux],theta_aux],\
    sampling_ou, [theta_aux,sigma_aux],obs,log_g_normal_den,sd,\
    ou_trans_den,[theta_aux,sigma_aux],ou_trans_den,\
    resamp_coef,l,d, N,seed)
    #x_mean=np.sum(pff.norm_logweights(log_weights,ax=1)*x_pr,axis=-1)
    weights=pff.norm_logweights(log_weights[-1])
    #print(weights.shape)
    index=np.random.choice(np.array(range(N)),p=weights)
    cond_path=x_pr[:,index]
    cond_log_weights=log_weights[:,index]
    clwn=np.copy(cond_log_weights)
    cond_int_G=int_Gs[:,index]
    cign=np.copy(cond_int_G)
    seeds_cond=np.zeros((int(T/d),2),dtype=int)
    seeds_cond[:,0]=seed+np.array(range(int(T/d)))*int(int(2**l*d-1))
    seeds_cond[:,1]=index*np.ones(int(T/d))
    scn=np.copy(seeds_cond)
    ch_paths=np.zeros((B,int(T/d)))
    ch_weights=np.zeros((B,int(T/d)))

    ch_whole_paths=np.zeros((B,int(T/d)))
    ch_whole_weights=np.zeros((B,int(T/d)))

    seed+=(int(T/d))*int(int(2**l*d-1))
    cond_whole_path=cond_path
    cwpn=cond_path
    cond_whole_log_weights=cond_log_weights
    
    for b in range(B):
        print("sample iteration: ",i," chain iteration: ",b)

        seed+=int((int(T/d))*int(int(2**l*d-1)))
        np.random.seed(seed)
        print("seed is: ",seed)
        [resamp_log_weights,x_pr,clwn,cign,cwpn,scn]=Cond_PF_bridge_new(clwn,cign,cwpn,\
        scn,t0,x0,T,b_ou_1d,theta,Sig_ou_1d,sigma,b_ou_aux,theta_aux,Sig_ou_aux,sigma_aux,\
        r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,[ou_sd,[theta_aux,sigma_aux],theta_aux],\
        sampling_ou, [theta_aux,sigma_aux],obs,log_g_normal_den,sd,\
        ou_trans_den,[theta_aux,sigma_aux],ou_trans_den,\
        resamp_coef,l,d, N,seed)
    
        weights=pff.norm_logweights(resamp_log_weights[-1])
        index=np.random.choice(np.array(range(N)))
        #print("x_pr's shape: ",x_pr.shape)   
        cond_whole_path=x_pr[:,index]
        #print("cond_path's shape: ",cond_path.shape)
        cond_whole_log_weights=resamp_log_weights[:,index]
        ch_whole_paths[b]=cwpn
        #ch_whole_weights[b]=clwn
        
        #seed=b*int((int(T/d))*int(int(2**l*d-1)))
        np.random.seed(seed)
        [log_weights,x_pr,cond_log_weights,int_Gs_cond,cond_path,seeds_cond]=\
        Cond_PF_bridge_back_samp(cond_log_weights,cond_int_G,cond_path,seeds_cond,t0,x0,\
        T,b_ou_1d,theta,Sig_ou_1d,sigma,b_ou_aux,theta_aux,Sig_ou_aux,sigma_aux,\
        r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,\
        [ou_sd,[theta_aux,sigma_aux],theta_aux],\
        sampling_ou, [theta_aux,sigma_aux],obs,log_g_normal_den,sd,\
        ou_trans_den,[theta_aux,sigma_aux],ou_trans_den,\
        resamp_coef,l,d, N,seed,crossed=False)

        #seed=int((int(T/d))*int(int(2**l*d-1)))
        ch_paths[b]=cond_path
        ch_weights[b]=cond_log_weights
        #print("seed conditionals are:",seeds_cond)

    mcmc_mean[i,0]=np.mean(ch_paths,axis=0)
    mcmc_mean[i,1]=np.mean(ch_whole_paths,axis=0)

end=time.time()

"""
#%%
# In the following I use the KF filter 
# the following x_mean corresponds to samples from the MCMC method

"""
l_dis=l
dim=1
dim_o=1
K=np.array([[(1+theta/2**l_dis)**(2**l_dis*d)]])
#G=np.array([[sigma*np.sqrt((np.exp(2*d*theta)-1)/(2*theta))]])
G=np.array([[sigma*np.sqrt(((1+theta/2**l_dis)**(2*2**l_dis*d)-1)/(2*theta+theta**2/2**l_dis))]])
H=np.array([[1]])
D=np.array([[sd]])
#print(K,G**2,H,D)
Grad_S=np.array([[[0]],[[0]],[[1]]],dtype=float)
#Grad_R_sigma_s=(np.exp(2*theta*d)-1)/(2*theta)
Grad_R_sigma_s=G[0,0]**2/sigma**2
#Grad_R_theta=(sigma**2/(2*theta**2))*(1-np.exp(2*theta*d)+2*d*theta*np.exp(2*theta*d))
Grad_R_theta=-G[0,0]**2*(2+2*theta/2**l_dis)/(2*theta+theta**2/2**l_dis)\
+(sigma**2/(2*theta+theta**2/2**l_dis))*(1+theta/2**l_dis)**(2*2**l_dis*d-1)*2*d
Grad_R=np.zeros((3,1,1),dtype=float)
Grad_R[0,0,0]=Grad_R_theta
Grad_R[1,0,0]=Grad_R_sigma_s
Grad_K=np.array([[[d*(1+theta/2**l_dis)**(2**l_dis*d-1)]],[[0]],[[0]]],dtype=float)
#print(Grad_K,Grad_R,Grad_S)
x_kf,x_kf_smooth,Grad_log_lik=KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
#Grads_dis[j,i]=Grad_log_lik[:,0,0]
times=np.arange(t0,T+1,d)
l_times=np.arange(t0,T,2**(-l))
#print(times, l_times)

#plt.plot(times[1:],x_reg,label="True signal")
#plt.plot(times[1:],x_kf_smooth[1:,0],label="KF smooth")
#plt.plot(times,x_kf[:,0],label="KF")
#plt.plot(times[1:], obs,label="Observations")
plt.title("N=2, and chain of length 50, with 50 samples")
MSE=np.mean((x_kf_smooth[1:,0]-mcmc_mean)**2,axis=0)
print(MSE)
plt.plot(times[1:],MSE[0],label="backward sampling")
plt.plot(times[1:],MSE[1],label="Multinomial path sampling")
plt.yscale("log")
plt.ylabel("MSE")
plt.xlabel("t")
#print(MSE)
print(end-start)
plt.legend()
"""
#plt.savefig("MSE_backward_sampling_2.pdf")    
 #%%

# RESULTS: We can see that the MSE is constant in time for the backward sampling method
# as opposed to the multinomial path sampling method. The multinomials sample makes the 
# error grow exponentially in time until it reaches a point where it stabilizes.

#%%
"""
x_mean=np.mean(ch_paths,axis=0)
print(x_mean.shape)
plt.plot(times[1:], x_mean,label="PF_backward_sampling")
x_whole_mean=np.mean(ch_whole_paths,axis=0)
plt.plot(times[1:], x_whole_mean,label="PF_whole")
#plt.plot(times[1:],x_reg,label="True signal")
plt.plot(times[1:],x_kf_smooth[1:,0],label="KF smooth")
plt.plot(times,x_kf[:,0],label="KF")
#plt.plot(times[1:], obs,label="Observations")
#plt.plot(l_times[1:],xs[:-1],lw="0.5")
#frame1 = plt.gca()
#frame1.axes.get_xaxis().set_visible(False)
#frame1.axes.get_yaxis().set_visible(False)
#frame1.axes.get_xaxis().set_ticks([])
#frame1.axes.get_yaxis().set_ticks([])
plt.xlabel("t")
plt.legend()
"""
#plt.savefig("Backward_multinomial_comparation.pdf")
#%%
#%%
# CONDITIONAL PARTICLE FILTER TEST
# In the following we form a chain of conditional particle filters, we initialize with
# with the regular PF
# FOR THE OU PROCESS 
"""
N=100
x0=1.2+np.zeros(N)
l=10
alpha=0
T=5
t0=0
l_d=0
d=2**(l_d)
theta=-0.2
sigma=1.2
#sigma_aux=0.2
theta_aux=theta-0.2
sigma_aux=sigma
#print(theta)
collection_input=[ b_ou_1d,theta,Sig_ou_1d,sigma]
resamp_coef=1
l_max=10
x_true=gen_gen_data_1d(T,x0_sca,l_max,collection_input)
x_reg=cut(T,l_max,-l_d,x_true)[1:]
# x_reg stands for x regular
cov=2e0
obs=gen_obs(x_reg,g_normal_1d,cov)
np.random.seed(3)
#print(x_pr)
n_steps=int(2**l*(T-t0))
dt=2**(-l)
x=np.zeros((n_steps+1,N))
x[0]=x0
x_pr=np.random.normal(x0,np.sqrt(sigma**2*(T-t0)),N)
x[-1]=x_pr
int_G=np.zeros(N)
t_test=2.
"""
#print("t0 is->",t0, " x0 is->",x0," T is->",T," theta is->",theta," sigma is->",sigma,\
#" theta_aux is->",theta_aux," sigma_aux is->",sigma_aux," obs is->",obs," cov is->",cov,\
#    " resamp_coef is->",resamp_coef," l is->",l," d is->",d," N is->",N)
#print(obs.shape)
#%%

#(t0,x0,T,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,\
    #sample_funct,sample_pars,obs,log_g_den,g_den_par, aux_trans_den,atdp,\
    #prop_trans_den, resamp_coef, l, d,N,crossed=False):

# FOR THE ALPHA AUXILIARY PROCESS
"""
[log_weights,x_pr,int_Gs,xs]=PF_bridge(t0,x0,T,b_ou_1d,theta,Sig_ou_1d,sigma,b_artificial_1d,0,Sig_alpha,[alpha,sigma,T],\
r_normal,[new_alpha_trans_sd,[alpha,sigma]],H_normal_1d,[new_alpha_trans_sd,[alpha,sigma]],\
sampling_alpha_trans_props, [alpha,sigma],obs,log_g_normal_den,cov,\
aux_trans_den_alpha,[alpha,sigma],aux_trans_den_alpha,\
resamp_coef,l,d, N)
"""
# FOR THE OU AUXILIARY PROCESS
"""
np.random.seed(0)
seed=1
[log_weights,int_Gs,x_pr]=PF_bridge(t0,x0,T,b_ou_1d,theta,Sig_ou_1d,sigma,b_ou_aux,theta_aux,Sig_ou_aux,sigma_aux,\
r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,[ou_sd,[theta_aux,sigma_aux],theta_aux],\
sampling_ou, [theta_aux,sigma_aux],obs,log_g_normal_den,cov,\
ou_trans_den,[theta_aux,sigma_aux],ou_trans_den,\
resamp_coef,l,d, N,seed)

 #[aux_sd,sigma,mu,t0,x0]=pars
    #aux_trans_den=norm(loc=x0*np.exp(mu*(T-t0)),scale=aux_sd(t0,x0,T,x_pr,[sigma,mu,t0,x0]))

x_mean=np.sum(pff.norm_logweights(log_weights,ax=1)*x_pr,axis=-1)
#print(x_mean.shape)
#print(x_pr.shape)

# here we sample the first path
weights=pff.norm_logweights(log_weights[-1])
print(weights.shape)
index=np.random.choice(np.array(range(N)),p=weights)
cond_path=x_pr[:,index]
cond_log_weights=log_weights[:,index]
cond_int_G=int_Gs[:,index]  

B=100
ch_paths=np.zeros((B,int(T/d)))
ch_weights=np.zeros((B,int(T/d)))
for b in range(B):
    print(b)
    np.random.seed(b)

    [resamp_log_weights,int_Gs,x_pr]=Cond_PF_bridge(cond_log_weights,cond_int_G,cond_path,t0,x0,T,b_ou_1d,theta,Sig_ou_1d,sigma,b_ou_aux,theta_aux,Sig_ou_aux,sigma_aux,\
    r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,[ou_sd,[theta_aux,sigma_aux],theta_aux],\
    sampling_ou, [theta_aux,sigma_aux],obs,log_g_normal_den,cov,\
    ou_trans_den,[theta_aux,sigma_aux],ou_trans_den,\
    resamp_coef,l,d, N,seed)
    seed+=(int(T/d))*int(int(2**l*d-1))

    #if b==4:
    #    print("cond_log_weihgts : ",cond_log_weights)
    #    print("resamp_log_weights : ",resamp_log_weights)
    weights=pff.norm_logweights(resamp_log_weights[-1])
    index=np.random.choice(np.array(range(N)),p=weights)
    #print("x_pr's shape: ",x_pr.shape)   
    cond_path=x_pr[:,index]
    cond_int_G=int_Gs[:,index]
    #print("cond_path's shape: ",cond_path.shape)
    cond_log_weights=resamp_log_weights[:,index]
    ch_paths[b]=cond_path
    ch_weights[b]=cond_log_weights
    
"""
#%%
# In the following I use the KF filter 
# the following x_mean corresponds to samples from the MCMC method
"""
dim=1
dim_o=1
K=np.array([[np.exp(d*theta)]])
G=np.array([[sigma*np.sqrt((np.exp(2*d*theta)-1)/(2*theta))]])
H=np.array([[1]])
D=np.array([[cov]])
x_kf,x_kf_smooth=KF(x0[0],dim,dim_o,K,G,H,D,obs)
times=np.arange(t0,T+1,d)
l_times=np.arange(t0,T,2**(-l))
print(times, l_times)
x_mean=np.mean(ch_paths,axis=0)
print(x_mean.shape)
plt.plot(times[1:],x_reg,label="True signal")
plt.plot(times,x_kf_smooth[:,0],label="KF smooth")
plt.plot(times,x_kf[:,0],label="KF")
plt.plot(times[1:], obs,label="Observations")
plt.plot(times[1:], x_mean,label="PF")
#plt.plot(l_times[1:],xs[:-1],lw="0.5")
#frame1 = plt.gca()
#frame1.axes.get_xaxis().set_visible(False)
#frame1.axes.get_yaxis().set_visible(False)
#frame1.axes.get_xaxis().set_ticks([])
#frame1.axes.get_yaxis().set_ticks([])

#plt.savefig("Diffusion2.pdf")
plt.legend()
"""
#%%


def Grad_Cond_PF_bridge_back_samp_an(lw_cond,int_Gs_cond,x_cond,seeds_cond,t0,x0,T,b,A,A_fd,\
    Sig,fi,fi_fd,b_til,A_til,Sig_til,fi_til,fi_til_fd,r,r_pars,r_pars_fd,H,H_pars,H_pars_fd,\
    sample_funct,sample_pars,obs,log_g_den,g_den_par, aux_trans_den,atdp,\
    Grad_log_aux_trans,prop_trans_den, Grad_log_G,resamp_coef, l, d,N,seed,fd_rate):
    # NEW ARGUMENTS:
    # A_fd,Sig_fd,fi_fd, fi_til_fd,r_pars_fd,H_pars_fd are parameter used to obtain the finite difference 
    # derivative. They have the same format as their counterparts without the _fd.

    # Grad_log_G if a function that computes the gradient of the observation likelihood,
    # it has the same parameters as the observation likelihood.
    # Grad_log_aux is the gradient of the transition density of the auxiliary process.
    # It has the same parameters as the transition density of the auxiliary process.

    # NEW OUTPUTS:

    # Grads: is a 3 dimensional array with the gradient of the score function with respect to the
    # parameters A and fi (this is a bit simplified since the parameters A and fi might include more parameters
    # thank just the ones we get the derivative w.r.t.) and the gradient of the observation likelihood.
    [log_weights,x_pr,cond_log_weights,cond_int_G,cond_path,seeds_cond]\
    =Cond_PF_bridge_back_samp(lw_cond,int_Gs_cond,x_cond,seeds_cond,t0,x0,T,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,\
    r,r_pars,H,H_pars,sample_funct,sample_pars,obs,log_g_den,g_den_par, aux_trans_den,atdp,\
    prop_trans_den, resamp_coef, l, d,N,seed)
    Grad_log_Gs=0
    Grad_log_aux_transition=np.zeros(2)
    Grad_the_int_psi=0
    Grad_sig_int_psi=0
    sum_x_the_mid=0
    sum_x_sig_mid=0
    Grads=np.zeros(3)

    for j in range(int(T/d)):
            t_in=t0+j*d
            t_fin=t0+(j+1)*d
            Grad_log_Gs+=Grad_log_G(cond_path[j],obs[j],g_den_par)
            if j==0:
                x_in=x0[0]
            else:
                x_in=cond_path[j-1]
            Grad_log_aux_transition+=\
            Grad_log_aux_trans(t_in,x_in,t_fin,cond_path[j],atdp)
            # IMPORTANT: Notice that here the argument is atdp, which is related to the auxiliar 
            # arguments, careful consideration is necessary in the definition of the funciton 
            # Grad_log_aux_trans.
            intG, int_psi_the,int_psi_sig,x_the_mid,x_sig_mid =Bridge_1d(t_in,x_in,t_fin,cond_path[j],b,A_fd,Sig,fi,b_til,\
            A_til,Sig_til,fi_til,r,r_pars,H,\
            H_pars,l,d,1,seeds_cond[j,0],\
            j=seeds_cond[j,1],fd=True,N_pf=N,an=True)
            sum_x_the_mid+=x_the_mid
            sum_x_sig_mid+=x_sig_mid
            Grad_the_int_psi+=int_psi_the
            Grad_sig_int_psi+=int_psi_sig
            
    Grads[0]=Grad_the_int_psi#+Grad_log_aux_transition[0]
    # The comment in the previous line is because we assume the auxiliar drift parameter does not 
    # depend on the original drift parameter.

    Grads[1]=Grad_sig_int_psi+Grad_log_aux_transition[1]
    
    Grads[2]=Grad_log_Gs
    return [log_weights,x_pr, cond_log_weights,cond_int_G,cond_path,seeds_cond,Grads]

# IN THE FOLLOWING WE COMPUTE THE GRADIENT FUNCTION OF THE SCORE FUNCTION
def Grad_Cond_PF_bridge_back_samp(lw_cond,int_Gs_cond,x_cond,seeds_cond,t0,x0,T,b,\
    A,A_fd,Sig,fi,fi_fd,b_til,A_til,Sig_til,fi_til,\
    fi_til_fd,r,r_pars,r_pars_fd,H,H_pars,H_pars_fd,sample_funct,sample_pars,obs,log_g_den,g_den_par, aux_trans_den,atdp,\
    Grad_log_aux_trans,prop_trans_den, Grad_log_G,resamp_coef, l, d,N,seed,fd_rate):
    # NEW ARGUMENTS:
    # A_fd,Sig_fd,fi_fd, fi_til_fd,r_pars_fd,H_pars_fd are parameter used to obtain the finite difference 
    # derivative. They have the same format as their counterparts without the _fd.
    # Grad_log_G if a function that computes the gradient of the observation likelihood,
    # it has the same parameters as the observation likelihood.
    # Grad_log_aux is the gradient of the transition density of the auxiliary process.
    # It has the same parameters as the transition density of the auxiliary process.

    # NEW OUTPUTS:

    # Grads: is a 3 dimensional array with the gradient of the score function with respect to the
    # parameters A and fi (this is a bit simplified since the parameters A and fi might include more parameters
    # thank just the ones we get the derivative w.r.t.) and the gradient of the observation likelihood.
    [log_weights,x_pr,cond_log_weights,cond_int_G,cond_path,seeds_cond]\
    =Cond_PF_bridge_back_samp(lw_cond,int_Gs_cond,x_cond,seeds_cond,t0,\
    x0,T,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,\
    r,r_pars,H,H_pars,sample_funct,sample_pars,obs,log_g_den,g_den_par, aux_trans_den,atdp,\
    prop_trans_den, resamp_coef, l, d,N,seed)
    Grad_log_Gs=0
    Grad_log_aux_transition=np.zeros(2)
    Grad_A_intGs=0
    Grad_fi_intGs=0
    Grads=np.zeros(3)

    for j in range(int(T/d)):
            t_in=t0+j*d
            t_fin=t0+(j+1)*d
            Grad_log_Gs+=Grad_log_G(cond_path[j],obs[j],g_den_par)
            if j==0:
                x_in=x0[0]
            else:
                x_in=cond_path[j-1]
            Grad_log_aux_transition+=\
            Grad_log_aux_trans(t_in,x_in,t_fin,cond_path[j],atdp)
            # IMPORTANT: Notice that here the argument is atdp, which is related to the auxiliar 
            # arguments, careful consideration is necessary in the definition of the funciton 
            # Grad_log_aux_trans.
            intG_mod_A=Bridge_1d(t_in,x_in,t_fin,cond_path[j],b,A_fd,Sig,fi,b_til,\
            A_til,Sig_til,fi_til,r,r_pars,H,\
            H_pars,l,d,1,seeds_cond[j,0],\
            j=seeds_cond[j,1],fd=True,N_pf=N)

            intG_mod_fi=Bridge_1d(t_in,x_in,t_fin,cond_path[j],b,A,Sig,fi_fd,b_til,\
            A_til,Sig_til,fi_til_fd,r,r_pars_fd,H,\
            H_pars_fd,l,d,1,seeds_cond[j,0],\
            j=seeds_cond[j,1],fd=True,N_pf=N)
            # are these terms representative? in the case of the 
            # Gradient w.r.t. theta, it is everything we have 
            Grad_A_intGs-=((cond_int_G[j]-intG_mod_A)/(fd_rate))
            Grad_fi_intGs-=((cond_int_G[j]-intG_mod_fi)/(fd_rate))
            
    Grads[0]=Grad_A_intGs[0]#+Grad_log_aux_transition[0]
    # The comment in the previous line is because we assume the auxiliar drift parameter does not 
    # depend on the original drift parameter.

    Grads[1]=Grad_fi_intGs[0]+Grad_log_aux_transition[1]
    
    Grads[2]=Grad_log_Gs
    return [log_weights,x_pr,cond_log_weights,cond_int_G,cond_path,seeds_cond,Grads]



def Gen_Grad_Cond_PF_bridge_back_samp(in_dist,in_dist_pars,Grad_log_in_dist,\
    x_cond,seeds_cond,b,A,A_fd,Sig,fi,fi_fd,b_til,Sig_til,r,\
    H,update_func,sample_funct,sample_pars,obs,obs_times,log_g_den,g_den_par,\
    aux_trans_den,Grad_log_aux_trans,prop_trans_den, Grad_log_G,resamp_coef, l, d,N,seed,\
    fd_rate):
    # NEW ARGUMENTS:
    # A_fd,Sig_fd,fi_fd, fi_til_fd,r_pars_fd,H_pars_fd are parameter used to obtain the finite difference 
    # derivative. They have the same format as their counterparts without the _fd.
    # A_fd, for this specific case A_fd is 3 times bigger than A, this is bcs we need different sets of parameters
    # for each of the parameters we are differentiating.
    # Grad_log_G if a function that computes the gradient of the observation likelihood,
    # it has the same parameters as the observation likelihood.
    # Grad_log_aux is the gradient of the transition density of the auxiliary process.
    # It has the same parameters as the transition density of the auxiliary process.

    # NEW OUTPUTS:
    # Grads: is a 3 dimensional array with the gradient of the score function with respect to the
    # parameters A and fi (this is a bit simplified since the parameters A and fi might include more parameters
    # thank just the ones we get the derivative w.r.t.) and the gradient of the observation likelihood.

    [log_weights,x_pr,cond_log_weights,cond_int_G,cond_path,seeds_cond]\
    =Gen_Cond_PF_bridge_back_samp(in_dist,in_dist_pars,x_cond,seeds_cond,b,A,Sig,\
    fi,b_til,Sig_til,r,H,update_func,sample_funct,sample_pars,\
    obs,obs_times,log_g_den,g_den_par, aux_trans_den, prop_trans_den, resamp_coef, l, d,N,seed)
    
    Grads=np.zeros(4)
    Grads[:3]=Grad_log_in_dist(cond_path[0],in_dist_pars)
    Grad_log_Gs=Grad_log_G(cond_path[0],obs[0],g_den_par)
    Grad_log_aux_transition=np.zeros(3)

    Grad_the1_intGs=0
    Grad_the2_intGs=0
    Grad_the3_intGs=0

    T=int(len(obs_times)*d)
    Deltas=get_deltas(obs_times)

    for j in range(int(T/d)-1):
            t_in=obs_times[j]
            t_fin=obs_times[j+1]
            Grad_log_Gs+=Grad_log_G(cond_path[j+1],obs[j+1],g_den_par)
            x_in=cond_path[j]
            x_fin=cond_path[j+1]
            pars_0=[A,fi]
            [A_til,fi_til,r_pars,H_pars,\
            atdp]=update_func(pars_0,pars_0,t_in,x_in,x_in,t_fin,x_fin,x_fin,levels=1)
            Grad_log_aux_transition[2]=Grad_log_aux_transition[2]+\
            Grad_log_aux_trans(t_in,x_in,t_fin,cond_path[j+1],atdp)
            # IMPORTANT: Notice that here the argument is atdp, which is related to the auxiliar 
            # arguments, careful consideration is necessary in the definition of the funciton 
            # Grad_log_aux_trans.

            # Since we need a different A_fd for the finite difference of the 3 parameters 
            # we choose A_fd to have 3 additional dimensions, one for each parameter.
            # We choose b_til and Sig_til not to depend on the parameters we are differentiating.
            # This can be done since the auxiliary distribution is an arbitrary distribution.

            intG_mod_the1=Bridge_1d(t_in,x_in,t_fin,cond_path[j+1],b,A_fd[0],Sig,fi,b_til,\
            A_til,Sig_til,fi_til,r,r_pars,H,\
            H_pars,l,d,1,seeds_cond[j,0],\
            j=seeds_cond[j,1],fd=True,N_pf=N,Dt=Deltas[j]/2**l)

            intG_mod_the2=Bridge_1d(t_in,x_in,t_fin,cond_path[j+1],b,A_fd[1],Sig,fi,b_til,\
            A_til,Sig_til,fi_til,r,r_pars,H,\
            H_pars,l,d,1,seeds_cond[j,0],\
            j=seeds_cond[j,1],fd=True,N_pf=N,Dt=Deltas[j]/2**l)
            pars_0_fd=[A_fd[2],fi_fd]
            [A_til_fd,fi_til_fd,r_pars_fd,H_pars_fd,\
            atdp_fd]=update_func(pars_0_fd,pars_0_fd,t_in,x_in,x_in,t_fin,x_fin,x_fin,levels=1)

            intG_mod_the3=Bridge_1d(t_in,x_in,t_fin,cond_path[j+1],b,A_fd[2],Sig,fi_fd,b_til,\
            A_til_fd,Sig_til,fi_til_fd,r,r_pars_fd,H,\
            H_pars_fd,l,d,1,seeds_cond[j,0],\
            j=seeds_cond[j,1],fd=True,N_pf=N,Dt=Deltas[j]/2**l)

            """intG_mod_fi=Bridge_1d(t_in,x_in,t_fin,cond_path[j],b,A,Sig,fi_fd,b_til,\
            A_til,Sig_til,fi_til_fd,r,r_pars_fd,H,\
            H_pars_fd,l,d,1,seeds_cond[j,0],\
            j=seeds_cond[j,1],fd=True,N_pf=N,Dt=Deltas[j])
            """
            # are these terms representative? in the case of the 
            # Gradient w.r.t. theta, it is everything we have 
            Grad_the1_intGs-=(cond_int_G[j]-intG_mod_the1)/fd_rate
            Grad_the2_intGs-=(cond_int_G[j]-intG_mod_the2)/fd_rate
            Grad_the3_intGs-=(cond_int_G[j]-intG_mod_the3)/fd_rate
             
    Grads[0]+=Grad_the1_intGs#+Grad_log_aux_transition[0]
    Grads[1]+=Grad_the2_intGs
    Grads[2]+=Grad_the3_intGs+Grad_log_aux_transition[2]
    Grads[3]=Grad_log_Gs
    # The comment in the previous line is because we assume the auxiliar drift parameter does not 
    # depend on the original drift parameter.
    return [log_weights,x_pr, cond_log_weights,cond_int_G,cond_path,seeds_cond,Grads]

def Gen_C_Grad_Cond_PF_bridge_back_samp(in_dist,in_dist_pars,Grad_log_in_dist,\
    x_cond_0,x_cond_1,seeds_cond_0,seeds_cond_1,b,A_0,A_1,A_fd_0,A_fd_1,Sig,\
    fi_0,fi_1,fi_fd_0,fi_fd_1,b_til,Sig_til,r,\
    H,update_func,max_sample_funct,sample_pars,obs,obs_times,log_g_den,g_den_par_0,g_den_par_1,\
    aux_trans_den,Grad_log_aux_trans,prop_trans_den,ind_prop_trans_par_0,ind_prop_trans_par_1,\
    Grad_log_G,resamp_coef, l, d,N,seed,fd_rate):
    # NEW ARGUMENTS:
    # A_fd,Sig_fd,fi_fd, fi_til_fd,r_pars_fd,H_pars_fd are parameter used to obtain the finite difference 
    # derivative. They have the same format as their counterparts without the _fd.
    # A_fd, for this specific case A_fd is 3 times bigger than A, this is bcs we need different sets of parameters
    # for each of the parameters we are differentiating.
    # Grad_log_G if a function that computes the gradient of the observation likelihood,
    # it has the same parameters as the observation likelihood.
    # Grad_log_aux is the gradient of the transition density of the auxiliary process.
    # It has the same parameters as the transition density of the auxiliary process.

    # NEW OUTPUTS:

    # Grads: is a 3 dimensional array with the gradient of the score function with respect to the
    # parameters A and fi (this is a bit simplified since the parameters A and fi might include more parameters
    # thank just the ones we get the derivative w.r.t.) and the gradient of the observation likelihood.

    #[log_weights_0, log_weights_1, x_pr_0, x_pr_1, new_lw_cond_0,new_lw_cond_1\
    #,new_int_G_cond_0,new_int_G_cond_1,new_x_cond_0,new_x_cond_1,new_seeds_cond_0 ,new_seeds_cond_1]
    
    [log_weights_0,log_weights_1,x_pr_0,x_pr_1,cond_log_weights_0,cond_log_weights_1,\
    cond_int_G_0,cond_int_G_1,cond_path_0,cond_path_1,seeds_cond_0,seeds_cond_1]\
    =Gen_C_Cond_PF_bridge_back_samp(in_dist,in_dist_pars,x_cond_0,x_cond_1 ,seeds_cond_0,\
    seeds_cond_1,b,A_0,A_1,Sig,fi_0,fi_1,b_til,Sig_til,\
    r,H,update_func,max_sample_funct,sample_pars,\
    obs,obs_times,log_g_den,g_den_par_0,g_den_par_1, aux_trans_den,\
    prop_trans_den,ind_prop_trans_par_0,ind_prop_trans_par_1, resamp_coef, l, d,N,seed)
    
    Grads_0=np.zeros(4)
    Grads_1=np.zeros(4)
    Grads_0[:3]=Grad_log_in_dist(cond_path_0[0],in_dist_pars[:3])
    # The argument ind_dist_pars is a list of size 6, where the first three correspond
    # to the initial parameters at level 0.
    Grads_1[:3]=Grad_log_in_dist(cond_path_1[0],in_dist_pars[3:])


    Grad_log_Gs_0=Grad_log_G(cond_path_0[0],obs[0],g_den_par_0)
    Grad_log_Gs_1=Grad_log_G(cond_path_1[0],obs[0],g_den_par_1)
    Grad_log_aux_transition_0=np.zeros(3)
    Grad_log_aux_transition_1=np.zeros(3)

    Grad_the1_intGs_0=0
    Grad_the2_intGs_0=0
    Grad_the3_intGs_0=0

    Grad_the1_intGs_1=0
    Grad_the2_intGs_1=0
    Grad_the3_intGs_1=0

    T=int(len(obs_times)*d)
    Deltas=get_deltas(obs_times)


    for j in range(int(T/d)-1):
            t_in=obs_times[j]
            t_fin=obs_times[j+1]
            Grad_log_Gs_0+=Grad_log_G(cond_path_0[j+1],obs[j+1],g_den_par_0)
            Grad_log_Gs_1+=Grad_log_G(cond_path_1[j+1],obs[j+1],g_den_par_1)
            x_in_0=cond_path_0[j]
            x_in_1=cond_path_1[j]
            x_fin_0=cond_path_0[j+1]
            x_fin_1=cond_path_1[j+1]
            pars_0=[A_0,fi_0]
            pars_1=[A_1,fi_1]
            [A_til_0,A_til_1,fi_til_0,fi_til_1,r_pars_0,\
            r_pars_1,H_pars_0,H_pars_1,atdp_0,atdp_1]=update_func(pars_0,\
            pars_1,t_in,x_in_0,x_in_1,t_fin,x_fin_0,x_fin_1,levels=2)

            Grad_log_aux_transition_0[2]=Grad_log_aux_transition_0[2]+\
            Grad_log_aux_trans(t_in,x_in_0,t_fin,cond_path_0[j+1],atdp_0)
            Grad_log_aux_transition_1[2]=Grad_log_aux_transition_1[2]+\
            Grad_log_aux_trans(t_in,x_in_1,t_fin,cond_path_1[j+1],atdp_1)
            
            # IMPORTANT: Notice that here the argument is atdp, which is related to the auxiliar 
            # arguments, careful consideration is necessary in the definition of the funciton 
            # Grad_log_aux_trans.

            # Since we need a different A_fd for the finite difference of the 3 parameters 
            # we choose A_fd to have 3 additional dimensions, one for each parameter.
            # We choose b_til and Sig_til not to depend on the parameters we are differentiating.
            # This can be done since the auxiliary distribution is an arbitrary distribution.

            #C_Bridge_1d(t_in,x_in_0,x_in_1,t_fin,cond_path_0[k],cond_path_1[k],\
            #b,A_fd_0,A_fd_1,Sig,fi_0,fi_1,b_til,A_til_0,A_til_1,Sig_til,fi_til_0,fi_til_1,\
            #r,r_pars_0,r_pars_1,H,H_pars_0,H_pars_1,l,d,N,2,j_0=seeds_cond_0[k,1],j_1=seeds_cond_1[k,1],fd=True,N_pf=N,
            #cond_seed_0=seeds_cond_0[k,0],cond_seed_1=seeds_cond_1[k,0])

            intG_mod_the1_0,intG_mod_the1_1=C_Bridge_1d(t_in,x_in_0,x_in_1,t_fin,\
            cond_path_0[j+1],cond_path_1[j+1], b,A_fd_0[0],A_fd_1[0],Sig,fi_0,fi_1,\
            b_til,A_til_0,A_til_1,Sig_til,fi_til_0,fi_til_1,\
            r,r_pars_0,r_pars_1,H,H_pars_0,H_pars_1,l,d,N,2,j_0=seeds_cond_0[j,1],\
            j_1=seeds_cond_1[j,1],fd=True,N_pf=N,
            cond_seed_0=seeds_cond_0[j,0],cond_seed_1=seeds_cond_1[j,0],Dt=Deltas[j]/2**l)

            intG_mod_the2_0,intG_mod_the2_1=C_Bridge_1d(t_in,x_in_0,x_in_1,t_fin,\
            cond_path_0[j+1],cond_path_1[j+1], b,A_fd_0[1],A_fd_1[1],Sig,fi_0,fi_1,\
            b_til,A_til_0,A_til_1,Sig_til,fi_til_0,fi_til_1,\
            r,r_pars_0,r_pars_1,H,H_pars_0,H_pars_1,l,d,N,2,j_0=seeds_cond_0[j,1],\
            j_1=seeds_cond_1[j,1],fd=True,N_pf=N,
            cond_seed_0=seeds_cond_0[j,0],cond_seed_1=seeds_cond_1[j,0],Dt=Deltas[j]/2**l)

            pars_fd_0=[A_fd_0[2],fi_fd_0]
            pars_fd_1=[A_fd_1[2],fi_fd_1]

            [A_til_fd_0,A_til_fd_1,fi_til_fd_0,fi_til_fd_1,r_pars_fd_0,\
            r_pars_fd_1,H_pars_fd_0,H_pars_fd_1,atdp_fd_0,atdp_fd_1]=update_func(pars_fd_0,\
            pars_fd_1,t_in,x_in_0,x_in_1,t_fin,x_fin_0,x_fin_1,levels=2)
            
            
            intG_mod_the3_0,intG_mod_the3_1=C_Bridge_1d(t_in,x_in_0,x_in_1,t_fin,cond_path_0[j+1],\
            cond_path_1[j+1], b,A_fd_0[2],A_fd_1[2],Sig,fi_fd_0,fi_fd_1,b_til,\
            A_til_fd_0,A_til_fd_1,Sig_til,fi_til_fd_0,fi_til_fd_1,r,r_pars_fd_0,r_pars_fd_1,H,\
            H_pars_fd_0,H_pars_fd_1,l,d,N,2,j_0=seeds_cond_0[j,1],j_1=seeds_cond_1[j,1],\
            fd=True,N_pf=N,\
            cond_seed_0=seeds_cond_0[j,0],cond_seed_1=seeds_cond_1[j,0],\
            Dt=Deltas[j]/2**l)

            """intG_mod_fi=Bridge_1d(t_in,x_in,t_fin,cond_path[j],b,A,Sig,fi_fd,b_til,\
            A_til,Sig_til,fi_til_fd,r,r_pars_fd,H,\
            H_pars_fd,l,d,1,seeds_cond[j,0],\
            j=seeds_cond[j,1],fd=True,N_pf=N,Dt=Deltas[j])
            """
            # are these terms representative? in the case of the 
            # Gradient w.r.t. theta, it is everything we have 
            Grad_the1_intGs_0-=(cond_int_G_0[j]-intG_mod_the1_0)/fd_rate
            Grad_the1_intGs_1-=(cond_int_G_1[j]-intG_mod_the1_1)/(0.5*fd_rate)
            
            Grad_the2_intGs_0-=(cond_int_G_0[j]-intG_mod_the2_0)/fd_rate
            Grad_the2_intGs_1-=(cond_int_G_1[j]-intG_mod_the2_1)/(0.5*fd_rate)
            
            Grad_the3_intGs_0-=(cond_int_G_0[j]-intG_mod_the3_0)/fd_rate
            Grad_the3_intGs_1-=(cond_int_G_1[j]-intG_mod_the3_1)/(0.5*fd_rate)
             
    
    Grads_0[0]+=Grad_the1_intGs_0[0]#+Grad_log_aux_transition[0]
    Grads_1[0]+=Grad_the1_intGs_1[0]

    Grads_0[1]+=Grad_the2_intGs_0[0]
    Grads_1[1]+=Grad_the2_intGs_1[0]

    Grads_0[2]+=Grad_the3_intGs_0[0]+Grad_log_aux_transition_0[2]
    Grads_1[2]+=Grad_the3_intGs_1[0]+Grad_log_aux_transition_1[2]


    Grads_0[3]=Grad_log_Gs_0
    Grads_1[3]=Grad_log_Gs_1
    # The comment in the previous line is because we assume the auxiliar drift parameter does not 
    # depend on the original drift parameter.

    return [log_weights_0,log_weights_1,x_pr_0,x_pr_1, \
    cond_log_weights_0,cond_log_weights_1,cond_int_G_0,cond_int_G_1,\
    cond_path_0,cond_path_1,seeds_cond_0,seeds_cond_1,Grads_0,Grads_1]


def Grad_Cond_PF_bridge_new(lw_cond,int_Gs_cond,x_cond,seeds_cond,t0,x0,T,b,A,A_fd,Sig,fi,fi_fd,b_til,A_til,Sig_til,fi_til,\
    fi_til_fd,r,r_pars,r_pars_fd,H,H_pars,H_pars_fd,sample_funct,sample_pars,obs,log_g_den,g_den_par, aux_trans_den,atdp,\
    Grad_log_aux_trans,prop_trans_den, Grad_log_G,resamp_coef, l, d,N,seed,fd_rate,crossed=False):
    # NEW ARGUMENTS:
    # A_fd,Sig_fd,fi_fd, fi_til_fd,r_pars_fd,H_pars_fd are parameter used to obtain the finite difference 
    # derivative. They have the same format as their counterparts without the _fd.

    # Grad_log_G if a function that computes the gradient of the observation likelihood,
    # it has the same parameters as the observation likelihood.
    # Grad_log_aux is the gradient of the transition density of the auxiliary process.
    # It has the same parameters as the transition density of the auxiliary process.

    # NEW OUTPUTS:

    # Grads: is a 3 dimensional array with the gradient of the score function with respect to the
    # parameters A and fi (this is a bit simplified since the parameters A and fi might include more parameters
    # thank just the ones we get the derivative w.r.t.) and the gradient of the observation likelihood.

    #[log_weights,x_pr,new_lw_cond,new_int_G_cond,new_x_cond,new_seeds_cond]
    [log_weights,x_pr,cond_log_weights,cond_int_G,cond_path,seeds_cond]\
    =Cond_PF_bridge_new(lw_cond,int_Gs_cond,x_cond,seeds_cond,t0,x0,T,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,\
    r,r_pars,H,H_pars,sample_funct,sample_pars,obs,laog_g_den,g_den_par, aux_trans_den,atdp,\
    prop_trans_den, resamp_coef, l, d,N,seed,crossed=False)
    Grad_log_Gs=0
    Grad_log_aux_transition=np.zeros(2)
    Grad_A_intGs=0
    Grad_fi_intGs=0
    Grads=np.zeros(3)

    #int_G_sub1=0
    #int_G_sub2=0
    for j in range(int(T/d)):
            t_in=t0+j*d
            t_fin=t0+(j+1)*d
            Grad_log_Gs+=Grad_log_G(cond_path[j],obs[j],g_den_par)

            if j==0:
                x_in=x0[0]
            else:
                x_in=cond_path[j-1]
            Grad_log_aux_transition+=\
            Grad_log_aux_trans(t_in,x_in,t_fin,cond_path[j],atdp)
            # IMPORTANT: Notice that here the argument is atdp, which is related to the auxiliar 
            # arguments, careful consideration is necessary in the definition of the funciton 
            # Grad_log_aux_trans.

            intG_mod_A=Bridge_1d(t_in,x_in,t_fin,cond_path[j],b,A_fd,Sig,fi,b_til,\
            A_til,Sig_til,fi_til,r,r_pars,H,\
            H_pars,l,d,1,seeds_cond[j,0],\
            j=seeds_cond[j,1],fd=True,N_pf=N)
 
            intG_mod_fi=Bridge_1d(t_in,x_in,t_fin,cond_path[j],b,A,Sig,fi_fd,b_til,\
            A_til,Sig_til,fi_til_fd,r,r_pars_fd,H,\
            H_pars_fd,l,d,1,seeds_cond[j,0],\
            j=seeds_cond[j,1],fd=True,N_pf=N)
            
            Grad_A_intGs-=((cond_int_G[j]-intG_mod_A)/(fd_rate))
            Grad_fi_intGs-=((cond_int_G[j]-intG_mod_fi)/(fd_rate))

            
    Grads[0]=Grad_A_intGs[0]#+Grad_log_aux_transition[0]
    # The comment in the previous line is because we assume the auxiliar drift parameter does not 
    # depend on the original drift parameter.
    Grads[1]=Grad_fi_intGs[0]+Grad_log_aux_transition[1]
    
    Grads[2]=Grad_log_Gs
    return [log_weights,x_pr,cond_log_weights,cond_int_G,cond_path,seeds_cond,Grads]


def Grad_Cond_PF_bridge(lw_cond,int_Gs_cond,x_cond,seeds_cond,t0,x0,T,b,A,A_fd,Sig,fi,fi_fd,b_til,A_til,Sig_til,fi_til,\
    fi_til_fd,r,r_pars,r_pars_fd,H,H_pars,H_pars_fd,sample_funct,sample_pars,obs,log_g_den,g_den_par, aux_trans_den,atdp,\
    Grad_log_aux_trans,prop_trans_den, Grad_log_G,resamp_coef, l, d,N,seed,fd_rate,crossed=False):

    # NEW ARGUMENTS:
    # A_fd,Sig_fd,fi_fd, fi_til_fd,r_pars_fd,H_pars_fd are parameter used to obtain the finite difference 
    # derivative. They have the same format as their counterparts without the _fd.

    # Grad_log_G if a function that computes the gradient of the observation likelihood,
    # it has the same parameters as the observation likelihood.
    # Grad_log_aux is the gradient of the transition density of the auxiliary process.
    # It has the same parameters as the transition density of the auxiliary process.

    # NEW OUTPUTS:

    # Grads: is a 3 dimensional array with the gradient of the score function with respect to the
    # parameters A and fi (this is a bit simplified since the parameters A and fi might include more parameters
    # thank just the ones we get the derivative w.r.t.) and the gradient of the observation likelihood.

    #(lw_cond,int_Gs_cond,x_cond,t0,x0,T,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,\
    #r,r_pars,H,H_pars,sample_funct,sample_pars,obs,log_g_den,g_den_par, aux_trans_den,atdp,\
    #prop_trans_den, resamp_coef, l, d,N,seed,crossed=False)
    
    #[log_weights,x_pr,cond_log_weights,cond_int_G,cond_path,seeds_cond,Grads]
    #[[log_weights,int_Gs,x_pr,seeds_cond]
    [log_weights,int_Gs,x_pr,seeds_cond]\
    =Cond_PF_bridge(lw_cond,int_Gs_cond,x_cond,seeds_cond,t0,x0,T,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,\
    r,r_pars,H,H_pars,sample_funct,sample_pars,obs,log_g_den,g_den_par, aux_trans_den,atdp,\
    prop_trans_den, resamp_coef, l, d,N,seed,crossed=False)

    index=np.random.choice(np.array(range(N)))
    cond_log_weights=log_weights[:,index]
    #print("x_pr's shape: ",x_pr.shape)   
    cond_path=x_pr[:,index]
    cond_int_G=int_Gs[:,index]
    seeds_cond=seeds_cond[:,:,index]    
    Grad_log_Gs=0
    Grad_log_aux_transition=np.zeros(2)
    Grad_A_intGs=0
    Grad_fi_intGs=0
    Grads=np.zeros(3)
    #int_G_sub1=0
    #int_G_sub2=0
    for j in range(int(T/d)):
            
            t_in=t0+j*d
            t_fin=t0+(j+1)*d
            Grad_log_Gs+=Grad_log_G(cond_path[j],obs[j],g_den_par)

            if j==0:
                x_in=x0[0]
            else:
                x_in=cond_path[j-1]
            Grad_log_aux_transition+=\
            Grad_log_aux_trans(t_in,x_in,t_fin,cond_path[j],atdp)
            # IMPORTANT: Notice that here the argument is atdp, which is related to the auxiliar 
            # arguments, careful consideration is necessary in the definition of the funciton 
            # Grad_log_aux_trans.

            intG_mod_A=Bridge_1d(t_in,x_in,t_fin,cond_path[j],b,A_fd,Sig,fi,b_til,\
            A_til,Sig_til,fi_til,r,r_pars,H,\
            H_pars,l,d,1,seeds_cond[j,0],\
            j=seeds_cond[j,1],fd=True,N_pf=N)
 
            intG_mod_fi=Bridge_1d(t_in,x_in,t_fin,cond_path[j],b,A,Sig,fi_fd,b_til,\
            A_til,Sig_til,fi_til_fd,r,r_pars_fd,H,\
            H_pars_fd,l,d,1,seeds_cond[j,0],\
            j=seeds_cond[j,1],fd=True,N_pf=N)
            
            Grad_A_intGs-=((cond_int_G[j]-intG_mod_A)/(fd_rate))
            Grad_fi_intGs-=((cond_int_G[j]-intG_mod_fi)/(fd_rate))
    Grads[0]=Grad_A_intGs[0]#+Grad_log_aux_transition[0]
    # The comment in the previous line is because we assume the auxiliar drift parameter does not 
    # depend on the original drift parameter.
    Grads[1]=Grad_fi_intGs[0]+Grad_log_aux_transition[1]
    Grads[2]=Grad_log_Gs
    return [log_weights,x_pr,cond_log_weights,cond_int_G,cond_path,seeds_cond,Grads]

# REMINDERS: The function Grad_log_G is in terms of the same parameters as the function
# log_g_normal_den, g_den_par, but the function that we designed Grad_log_G is not exactly in 
# terms of the same parameters. Check that. 
#%%

def C_Grad_Cond_PF_bridge_back_samp(x_cond_0,x_cond_1,\
    seeds_cond_0,seeds_cond_1,t0,x0,T,b,A_0,A_1,A_fd_0,A_fd_1,Sig,fi_0,fi_1,fi_fd_0,fi_fd_1,b_til,\
    A_til_0,A_til_1,Sig_til,fi_til_0,fi_til_1,fi_til_fd_0,fi_til_fd_1,r,r_pars_0,r_pars_1,r_pars_fd_0,\
    r_pars_fd_1,H,H_pars_0,H_pars_1,H_pars_fd_0,H_pars_fd_1,sample_funct,sample_pars,obs,log_g_den,\
    g_den_par_0,g_den_par_1, aux_trans_den,atdp_0,atdp_1,\
    Grad_log_aux_trans,prop_trans_den,ind_prop_trans_par_0,ind_prop_trans_par_1,Grad_log_G,l,d,N,seed,fd_rate,crossed=False):
    # NEW ARGUMENTS:
    # A_fd,Sig_fd,fi_fd, fi_til_fd,r_pars_fd,H_pars_fd are parameter used to obtain the finite difference 
    # derivative. They have the same format as their counterparts without the _fd.
    # Comments: For the computation of the finite differeences we compute the terms intG_mod_A_0, intG_mod_A_1,
    # intG_mod_fi_0 and intG_mod_fi_1. Notice that in the case intG_mod_fi_0 and intG_mod_fi_1 we also modify the 
    # auxiliar variables for the diffusion term, i.e., we have fi_til_fd_0,fi_til_fd_1 but not A_til_fd_0,A_til_fd_1 in 
    # intG_mod_A_0, intG_mod_A_1. This is done bcs one of the conditions of the bridge is that the diffusion term of the 
    # auxiliar process and the original process are the same. 

    # For the same reason as above the parameters r_pars_fd_0, r_pars_fd_1, H_pars_fd_0, H_pars_fd_1 are exclusively used
    # in the computation of the modified int_G terms for the diffusion terms. 

    # Grad_log_G if a function that computes the gradient of the observation likelihood,
    # it has the same parameters as the observation likelihood.
    # Grad_log_aux is the gradient of the transition density of the auxiliary process.
    # It has the same parameters as the transition density of the auxiliary process.

    # NEW OUTPUTS:

    # Grads: is a 3 dimensional array with the gradient of the score function with respect to the
    # parameters A and fi (this is a bit simplified since the parameters A and fi might include more parameters
    # thank just the ones we get the derivative w.r.t.) and the gradient of the observation likelihood.


    """
    [log_weights_0, log_weights_1, x_pr_0, x_pr_1, new_lw_cond_0,new_lw_cond_1\
    ,new_int_G_cond_0,new_int_G_cond_1,new_x_cond_0,new_x_cond_1,new_seeds_cond_0 ,new_seeds_cond_1]

    """
    [log_weights_0,log_weights_1,x_pr_0,x_pr_1,cond_log_weights_0,cond_log_weights_1\
    ,cond_int_G_0,cond_int_G_1,cond_path_0,cond_path_1,seeds_cond_0 ,seeds_cond_1]\
    =C_Cond_PF_bridge_back_samp(x_cond_0,x_cond_1,\
    seeds_cond_0,seeds_cond_1,t0,x0,T,b,A_0,A_1,Sig,fi_0,fi_1,b_til,A_til_0,A_til_1,Sig_til,fi_til_0,\
    fi_til_1,r,r_pars_0,r_pars_1,H,H_pars_0,H_pars_1,sample_funct,sample_pars,obs,log_g_den,\
    g_den_par_0,g_den_par_1, aux_trans_den,atdp_0,atdp_1,\
    prop_trans_den,ind_prop_trans_par_0,ind_prop_trans_par_1, l, d,N,seed,crossed=False)
    """
    C_Cond_PF_bridge_back_samp(x_cond_0,x_cond_1,\
    seeds_cond_0,seeds_cond_1,t0,x0,T,b,A_0,A_1,Sig,fi_0,fi_1,b_til,A_til_0,A_til_1,Sig_til,fi_til_0,\
    fi_til_1,r,r_pars_0,r_pars_1,H,H_pars_0,H_pars_1,sample_funct,sample_pars,obs,\
    log_g_den,g_den_par_0,g_den_par_1, aux_trans_den,atdp_0,atdp_1,\
    prop_trans_den, ind_prop_trans_par_0,ind_prop_trans_par_1, l, d,N,seed,crossed=False):
    
    """
    Grad_log_Gs_0=0
    Grad_log_Gs_1=0
    Grad_log_aux_transition_0=np.zeros(2)
    Grad_log_aux_transition_1=np.zeros(2)
    Grad_A_intGs_0=0
    Grad_A_intGs_1=0
    Grad_fi_intGs_0=0
    Grad_fi_intGs_1=0
    Grads_1=np.zeros(3)
    Grads_0=np.zeros(3)
    #int_G_sub1=0
    #int_G_sub2=0
    for k in range(int(T/d)):
            t_in=t0+k*d
            t_fin=t0+(k+1)*d
            Grad_log_Gs_0+=Grad_log_G(cond_path_0[k],obs[k],g_den_par_0)
            Grad_log_Gs_1+=Grad_log_G(cond_path_1[k],obs[k],g_den_par_1)
            if k==0:
                x_in_0=x0[0]
                x_in_1=x0[0]
            else:
                x_in_0=cond_path_0[k-1]
                x_in_1=cond_path_1[k-1]

            Grad_log_aux_transition_0+=\
            Grad_log_aux_trans(t_in,x_in_0,t_fin,cond_path_0[k],atdp_0)
            Grad_log_aux_transition_1+=\
            Grad_log_aux_trans(t_in,x_in_1,t_fin,cond_path_1[k],atdp_1)
            # IMPORTANT: Notice that here the argument is atdp, which is related to the auxiliar 
            # arguments, careful consideration is necessary in the definition of the funciton 
            # Grad_log_aux_trans.

            """
            intG_mod_A_0=Bridge_1d(t_in,x_in_0,t_fin,cond_path_0[k],b,A_fd_0,Sig,fi_0,b_til,\
            A_til_0,Sig_til,fi_til_0,r,r_pars_0,H,\
            H_pars_0,l-1,d,1,seeds_cond_0[k,0],\
            j=seeds_cond_0[k,1],fd=True,N_pf=N)

            intG_mod_A_1=Bridge_1d(t_in,x_in_1,t_fin,cond_path_1[k],b,A_fd_1,Sig,fi_1,b_til,\
            A_til_1,Sig_til,fi_til_1,r,r_pars_1,H,\
            H_pars_1,l,d,1,seeds_cond_1[k,0],\
            j=seeds_cond_1[k,1],fd=True,N_pf=N)
            """
            """
            C_Bridge_1d(t0,x0_0,x0_1,T,x_p_0,x_p_1,b,A_0,A_1,Sig,fi_0,fi_1,b_til,A_til_0,A_til_1,\
            Sig_til,fi_til_0,fi_til_1,r,r_pars_0,r_pars_1,H,H_pars_0,H_pars_1,l,d,N,seed\
            ,crossed=False,backward=False,j_0=False,j_1=False,fd=False,N_pf=False,cond_seed_0=False,cond_seed_1=False):
    
            """
            intG_mod_A_0,intG_mod_A_1=C_Bridge_1d(t_in,x_in_0,x_in_1,t_fin,cond_path_0[k],cond_path_1[k],\
            b,A_fd_0,A_fd_1,Sig,fi_0,fi_1,b_til,A_til_0,A_til_1,Sig_til,fi_til_0,fi_til_1,\
            r,r_pars_0,r_pars_1,H,H_pars_0,H_pars_1,l,d,N,2,j_0=seeds_cond_0[k,1],j_1=seeds_cond_1[k,1],fd=True,N_pf=N,
            cond_seed_0=seeds_cond_0[k,0],cond_seed_1=seeds_cond_1[k,0])

            """
            intG_mod_fi_0=Bridge_1d(t_in,x_in_0,t_fin,cond_path_0[k],b,A_0,Sig,fi_fd_0,b_til,\
            A_til_0,Sig_til,fi_til_fd_0,r,r_pars_fd_0,H,\
            H_pars_fd_0,l-1,d,1,seeds_cond_0[k,0],\
            j=seeds_cond_0[k,1],fd=True,N_pf=N)
            
            intG_mod_fi_1=Bridge_1d(t_in,x_in_1,t_fin,cond_path_1[k],b,A_1,Sig,fi_fd_1,b_til,\
            A_til_1,Sig_til,fi_til_fd_1,r,r_pars_fd_1,H,\
            H_pars_fd_1,l,d,1,seeds_cond_1[k,0],\
            j=seeds_cond_1[k,1],fd=True,N_pf=N)
            """
            """
            C_Bridge_1d(t0,x0_0,x0_1,T,x_p_0,x_p_1,b,A_0,A_1,Sig,fi_0,fi_1,b_til,A_til_0,A_til_1,\
            Sig_til,fi_til_0,fi_til_1,r,r_pars_0,r_pars_1,H,H_pars_0,H_pars_1,l,d,N,seed\
            ,crossed=False,backward=False,j_0=False,j_1=False,fd=False,N_pf=False,cond_seed_0=False,cond_seed_1=False):
    
            """
            intG_mod_fi_0,intG_mod_fi_1=C_Bridge_1d(t_in,x_in_0,x_in_1,t_fin,cond_path_0[k],cond_path_1[k],\
            b,A_0,A_1,Sig,fi_fd_0,fi_fd_1,b_til,A_til_0,A_til_1,Sig_til,fi_til_fd_0,fi_til_fd_1,\
            r,r_pars_fd_0,r_pars_fd_1,H,H_pars_fd_0,H_pars_fd_1,l,d,N,2,j_0=seeds_cond_0[k,1],\
            j_1=seeds_cond_1[k,1],fd=True,N_pf=N,cond_seed_0=seeds_cond_0[k,0],cond_seed_1=seeds_cond_1[k,0])


            """
            print("intG_mod_A_0 is: ",intG_mod_A_0)
            print("cond_int_G_0[j] is: ",cond_int_G_0[k])
            print("intG_mod_A_1 is: ",intG_mod_A_1)
            print("cond_int_G_1[j] is: ",cond_int_G_1[k])
            print("intG_mod_fi_0 is: ",intG_mod_fi_0)
            print("cond_int_G_0[j] is: ",cond_int_G_0[k])
            print("intG_mod_fi_1 is: ",intG_mod_fi_1)
            print("cond_int_G_1[j] is: ",cond_int_G_1[k])
            """
            
            Grad_A_intGs_0-=((cond_int_G_0[k]-intG_mod_A_0)/fd_rate)
            Grad_A_intGs_1-=((cond_int_G_1[k]-intG_mod_A_1)/(fd_rate*0.5))
            Grad_fi_intGs_0-=((cond_int_G_0[k]-intG_mod_fi_0)/fd_rate)
            Grad_fi_intGs_1-=((cond_int_G_1[k]-intG_mod_fi_1)/(fd_rate*0.5))
            
    Grads_0[0]=Grad_A_intGs_0[0]#+Grad_log_aux_transition[0]
    # The comment in the previous line is because we assume the auxiliar drift parameter does not 
    # depend on the original drift parameter.
    Grads_1[0]=Grad_A_intGs_1[0]
    Grads_0[1]=Grad_fi_intGs_0[0]+Grad_log_aux_transition_0[1]
    Grads_1[1]=Grad_fi_intGs_1[0]+Grad_log_aux_transition_1[1]
    Grads_0[2]=Grad_log_Gs_0
    Grads_1[2]=Grad_log_Gs_1
    return [log_weights_0,log_weights_1,x_pr_0,x_pr_1,cond_log_weights_0,cond_log_weights_1,\
    cond_int_G_0,cond_int_G_1,cond_path_0,cond_path_1,seeds_cond_0,seeds_cond_1,Grads_0,Grads_1]
#%%

#%%
# In the following we design a way to compute the gradient of the score function. 
# First we compute this with finite diffrerences of the weights. This can be done 
# by computing analitically some parts of the gradient( which is a sum of different terms)
# and them computing the int_G with the perturbed weights.

# what are the parameters we are changing in this scenario? 
# Same old story, drift parameter, diffusion parameters and observations parameter (probably
# the covariance of the observations).

#%%
"""
N=40
x0_sca=1.2
x0=x0_sca+np.zeros(N)
l=4
T=1
t0=0
l_d=0
d=2**(l_d)
theta=-0.8
sigma=0.9
#sigma_aux=0.2
theta_aux=theta+0.2
sigma_aux=sigma
#print(theta)
np.random.seed(7)
collection_input=[b_ou_1d,theta,Sig_ou_1d,sigma]
resamp_coef=1
l_max=10    
x_true= gen_gen_data_1d(T,x0_sca,l_max,collection_input)
x_reg=cut(T,l_max,-l_d,x_true)[1:]
times=np.arange(t0,T+1,d)
l_times=np.arange(t0,T,2**(-l))
l_max_times=np.arange(t0,T,2**(-l_max))
#print(times, l_times)
plt.plot(times[1:],x_reg,label="True signal")
plt.plot(l_max_times,x_true[:-1],label="True complete signal")
# x_reg stands for x regular
sd=5e-1
np.random.seed(3)
obs=gen_obs(x_reg,g_normal_1d,sd)
plt.plot(times[1:], obs,label="Observations")
print(obs,x_reg)
"""
#%%
"""
fd_rate=1e-4
[theta_fd,sigma_fd,sd_fd]=np.array([theta,sigma,sd])+fd_rate*np.array([1,1,1])
print(theta_fd,sigma_fd,sd_fd)

"""
#%%
"""
start=time.time()
B=2
samples=1
seed=0
mcmc_mean=np.zeros((samples,2,int(T/d))) # This varible was originally designed 
# to store the mean of both processes, the one with multinomial sampling and the one with
# backward sampling.
resamp_coef=1
Grads=np.zeros((samples,B,3))
for i in range(samples):


    np.random.seed(i)
    [log_weights,int_Gs,x_pr]=PF_bridge(t0,x0,T,b_ou_1d,theta,Sig_ou_1d,sigma,b_ou_aux,theta_aux,Sig_ou_aux,sigma_aux,\
    r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,[ou_sd,[theta_aux,sigma_aux],theta_aux],\
    sampling_ou, [theta_aux,sigma_aux],obs,log_g_normal_den,sd,\
    ou_trans_den,[theta_aux,sigma_aux],ou_trans_den,\
    resamp_coef,l,d, N,seed)

    #x_mean=np.sum(pff.norm_logweights(log_weights,ax=1)*x_pr,axis=-1)
    weights=pff.norm_logweights(log_weights[-1])
    #print(weights.shape)
    index=np.random.choice(np.array(range(N)),p=weights)
    cond_path=x_pr[:,index]
    cond_log_weights=log_weights[:,index]
    seeds_cond=np.zeros((int(T/d),2),dtype=int)
    seeds_cond[:,0]=seed+np.array(range(int(T/d)))*int(int(2**l*d-1))
    seeds_cond[:,1]=index*np.ones(int(T/d))
    cond_int_G=int_Gs[:,index]
    ch_paths=np.zeros((B,int(T/d)))
    ch_weights=np.zeros((B,int(T/d)))
    #cov
    print("Sample iterations is: ",i)
    #print("The starting seed is: ",seed)
    #print("The conditional seed is: ",seeds_cond)

    #print("The condtional path is:",cond_path)    
    #cond_log_weights_test,cond_int_G_test,cond_path_test,seeds_cond_test=\
    #cond_log_weights.copy() ,cond_int_G.copy(),cond_path.copy(),seeds_cond.copy()

    for b in range(B):

        # the varaible int_Gs is meant to have the record of int_G of the 
        # backward sampled path.
        print("mcmc iteration is:", b)
        #Sig_ou_1d
        seed+=int((int(T/d))*int(int(2**l*d-1)))
        np.random.seed(b)
        [log_weights,x_pr,cond_log_weights,cond_int_G,cond_path,seeds_cond]=\
        Cond_PF_bridge_back_samp(cond_log_weights,cond_int_G,cond_path,seeds_cond,t0,x0,\
        T,b_ou_1d,theta,Sig_ou_1d,sigma,b_ou_aux,theta_aux,Sig_ou_aux,sigma_aux,\
        r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,\
        [ou_sd,[theta_aux,sigma_aux],theta_aux],\
        sampling_ou, [theta_aux,sigma_aux],obs,log_g_normal_den,sd,\
        ou_trans_den,[theta_aux,sigma_aux],ou_trans_den,\
        resamp_coef,l,d, N,seed,crossed=False)
        #print("The other condtional path is:",cond_path)
        ch_paths[b]=cond_path
        ch_weights[b]=cond_log_weights
        Grad_log_Gs=0
        Grad_log_aux_trans=np.zeros(2)
        Grad_theta_intGs=0
        Grad_sigma_intGs=0
        for j in range(int(T/d)):
            # here we compute the gradients of the observation likelihood
            # and the transition density of the auxiliary process.
            # Grad_G(x,y,pars)

            t_in=t0+j*d
            t_fin=t0+(j+1)*d    

            Grad_log_Gs+=Grad_log_G(cond_path[j],obs[j],sd**2)
            #print("Grad_log_Gs is: ",Grad_log_Gs)
            if j==0:
                x_in=x0[0]
            else:
                x_in=cond_path[j-1]
            Grad_log_aux_trans+=\
            Grad_log_aux_trans_ou(t_in,x_in,t_fin,cond_path[j],np.array([theta_aux,sigma**2]))

            intG_mod_theta=Bridge_1d(t_in,x_in,t_fin,cond_path[j],b_ou_1d,theta_fd,Sig_ou_1d,sigma,b_ou_aux,\
            theta_aux,Sig_ou_aux,sigma_aux,r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,\
            [ou_sd,[theta_aux,sigma_aux],theta_aux],l,d,1,seeds_cond[j,0],\
            j=seeds_cond[j,1],fd=True, N_pf=N)


            sigma_aux=sigma_fd
            intG_mod_sigma_s=Bridge_1d(t_in,x_in,t_fin,cond_path[j],b_ou_1d,theta,Sig_ou_1d,sigma_fd,b_ou_aux,\
            theta_aux,Sig_ou_aux,sigma_aux,r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,\
            [ou_sd,[theta_aux,sigma_aux],theta_aux],l,d,1,seeds_cond[j,0],\
            j=seeds_cond[j,1],fd=True, N_pf=N)
            sigma_aux=sigma
            Grad_theta_intGs-=((cond_int_G[j]-intG_mod_theta)/fd_rate)
            #Grad_theta_intGs+=((intG_mod_theta_0-intG_mod_theta)/fd_rate)
            Grad_sigma_intGs-=((cond_int_G[j]-intG_mod_sigma_s)/fd_rate)
            #Grad_sigma_intGs+=((intG_mod_sigma_s_0-intG_mod_sigma_s)/fd_rate)
        #print("Grad_theta_intGs has shape: ",Grad_theta_intGs.shape)    
        #print("Grad_theta_intGs has shape: ",Grad_theta_intGs.shape)
        Grads[i,b,0]=Grad_theta_intGs[0]#+Grad_log_aux_trans[0]
        Grads[i,b,1]=Grad_sigma_intGs[0]+Grad_log_aux_trans[1]*2*sigma
        #print("The GRad of the auxiliar transition is ",Grad_log_aux_trans[1]*2*sigma)
        Grads[i,b,2]=Grad_log_Gs
            #Bridge_1d(t0,x0,T,x_p,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,l,d,N,seed\
            #,crossed=False,backward=False,j=False,fd=False,N_pf=False)
        #print("seed conditionals are:",seeds_cond)
        #Bridge_1d(t0,x0,T,x_p,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,l,d,N,seed\
        #,crossed=False,backward=False,j=False,fd=False,N_pf=False)
        
        


    mcmc_mean[i,0]=np.mean(ch_paths,axis=0)
print(np.mean(Grads,axis=1))
"""
#%%
"""
start=time.time()
B=50
samples=50
seed=0
mcmc_mean=np.zeros((samples,2,int(T/d))) # This varible was originally designed 
# to store the mean of both processes, the one with multinomial sampling and the one with
# backward sampling.
resamp_coef=1
Grads_test=np.zeros((samples,B,3))
for i in range(samples):
    np.random.seed(i)
    [log_weights,int_Gs,x_pr]=PF_bridge(t0,x0,T,b_ou_1d,theta,Sig_ou_1d,sigma,b_ou_aux,theta_aux,Sig_ou_aux,sigma_aux,\
    r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,[ou_sd,[theta_aux,sigma_aux],theta_aux],\
    sampling_ou, [theta_aux,sigma_aux],obs,log_g_normal_den,sd,\
    ou_trans_den,[theta_aux,sigma_aux],ou_trans_den,\
    resamp_coef,l,d, N,seed)
    #x_mean=np.sum(pff.norm_logweights(log_weights,ax=1)*x_pr,axis=-1)
    weights=pff.norm_logweights(log_weights[-1])
    #print(weights.shape)
    index=np.random.choice(np.array(range(N)),p=weights)
    cond_path=x_pr[:,index]
    cond_log_weights=log_weights[:,index]
    seeds_cond=np.zeros((int(T/d),2),dtype=int)
    seeds_cond[:,0]=seed+np.array(range(int(T/d)))*int(int(2**l*d-1))
    seeds_cond[:,1]=index*np.ones(int(T/d))
    cond_int_G=int_Gs[:,index]
    ch_paths=np.zeros((B,int(T/d)))
    ch_weights=np.zeros((B,int(T/d)))
    #cov
    print("Sample iterations is: ",i)
    #print("The starting seed is: ",seed)
    #print("The conditional seed is: ",seeds_cond)
    #print("The condtional path is:",cond_path)    
    cond_log_weights_test,cond_int_G_test,cond_path_test,seeds_cond_test=\
    cond_log_weights.copy() ,cond_int_G.copy(),cond_path.copy(),seeds_cond.copy()
    for b in range(B):
        # the varaible int_Gs is meant to have the record of int_G of the 
        # backward sampled path.
        print("mcmc iteration is:", b)
        seed+=int((int(T/d))*int(int(2**l*d-1)))
        np.random.seed(b)
        [log_weights_test,x_pr_test,cond_log_weights_test,cond_int_G_test,cond_path_test,seeds_cond_test,Grads_t]=\
        Grad_Cond_PF_bridge_back_samp(cond_log_weights_test,cond_int_G_test,cond_path_test,seeds_cond_test,t0,x0,T,b_ou_1d,\
        theta,theta_fd,Sig_ou_1d,\
        sigma,sigma_fd,b_ou_aux,theta_aux,Sig_ou_aux,sigma_aux,sigma_fd,r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],\
        [ou_sd,[theta_aux,sigma_fd]],H_quasi_normal,[ou_sd,[theta_aux,sigma_aux],theta_aux],[ou_sd,[theta_aux,sigma_fd],theta_aux],\
        sampling_ou, [theta_aux,sigma_aux],obs,log_g_normal_den,sd, ou_trans_den,[theta_aux,sigma_aux],\
        Grad_log_aux_trans_ou_new,ou_trans_den, Grad_log_G_new,resamp_coef, l, d,N,seed,fd_rate,crossed=False)

        Grads_test[i,b]=Grads_t
        ch_paths[b]=cond_path_test
    mcmc_mean[i,0]=np.mean(ch_paths,axis=0)
end=time.time()
print("The time spend is with ",N, " is ",end-start)
"""

#%%
"""
l_dis=l
dim=1
dim_o=1
K=np.array([[(1+theta/2**l_dis)**(2**l_dis*d)]])
#G=np.array([[sigma*np.sqrt((np.exp(2*d*theta)-1)/(2*theta))]])
G=np.array([[sigma*np.sqrt(((1+theta/2**l_dis)**(2*2**l_dis*d)-1)/(2*theta+theta**2/2**l_dis))]])
H=np.array([[1]])
D=np.array([[sd]])
#print(K,G**2,H,D)
Grad_S=np.array([[[0]],[[0]],[[1]]],dtype=float)
#Grad_R_sigma_s=(np.exp(2*theta*d)-1)/(2*theta)
Grad_R_sigma_s=G[0,0]**2/sigma**2
#Grad_R_theta=(sigma**2/(2*theta**2))*(1-np.exp(2*theta*d)+2*d*theta*np.exp(2*theta*d))
Grad_R_theta=-G[0,0]**2*(2+2*theta/2**l_dis)/(2*theta+theta**2/2**l_dis)\
+(sigma**2/(2*theta+theta**2/2**l_dis))*(1+theta/2**l_dis)**(2*2**l_dis*d-1)*2*d
Grad_R=np.zeros((3,1,1),dtype=float)
Grad_R[0,0,0]=Grad_R_theta
Grad_R[1,0,0]=Grad_R_sigma_s
Grad_K=np.array([[[d*(1+theta/2**l_dis)**(2**l_dis*d-1)]],[[0]],[[0]]],dtype=float)
#print(Grad_K,Grad_R,Grad_S)
x_kf,x_kf_smooth,Grad_log_lik=KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
#Grads_dis[j,i]=Grad_log_lik[:,0,0]
times=np.arange(t0,T+1,d)
l_times=np.arange(t0,T,2**(-l))
l_max_times=np.arange(t0,T,2**(-l_max))
#print(times, l_times)
x_mean=np.mean(mcmc_mean[:,0],axis=0)
#print(x_mean.shape)
#plt.plot(times[1:],x_reg,label="True signal")
#plt.plot(l_max_times,x_true[:-1],label="True complete signal")
plt.plot(times,x_kf_smooth[:,0],label="KF smooth")
#plt.plot(times,x_kf[:,0],label="KF")
#plt.plot(times[1:], obs,label="Observations")
plt.plot(times[1:], x_mean,label="PGibbs")
print(x_mean)
plt.legend()
print(Grad_log_lik)
print(np.mean(Grads_test,axis=(0,1)))
"""
#%%

#print(Grad_log_lik)
#%%
# IN THIS SECTION I RUN THE SGD ALGORITHM. I WILL CHECK THE PARAMETERS SO I CAN RUN THE ALGORITHM
# Parameter choice, and data.
"""
N=50
x0_sca=1.2
x0=x0_sca+np.zeros(N)
l=8
T=2
t0=0
l_d=-4
d=2**(l_d)
theta_true=-0.3
sigma_true=1.2
#sigma_aux=0.2
#print(theta)
np.random.seed(7)
collection_input=[b_ou_1d,theta_true,Sig_ou_1d,sigma_true]
resamp_coef=1
l_max=10    
x_true= gen_gen_data_1d(T,x0_sca,l_max,collection_input)
x_reg=cut(T,l_max,-l_d,x_true)[1:]
print(x_reg.shape)
times=np.array(range(t0,int(T/d)+1))*d
print(times)
#times=np.arange(t0,T+1,d)
#print(times)
l_times=np.arange(t0,T,2**(-l))
l_max_times=np.arange(t0,T,2**(-l_max))
#print(times, l_times)
plt.plot(times[1:],x_reg,label="True signal")
plt.plot(l_max_times,x_true[:-1],label="True complete signal")
# x_reg stands for x regular
sd_true=5e-1
np.random.seed(3)
obs=gen_obs(x_reg,g_normal_1d,sd_true)
plt.plot(times[1:], obs,label="Observations")
print(obs,x_reg)
fd_rate=1e-4
[theta_fd,sigma_fd,sd_fd]=np.array([theta_true,sigma_true,sd_true])+fd_rate*np.array([1,1,1])
print(theta_fd,sigma_fd,sd_fd)
"""
#%%
"""
start=time.time()
Grad_mcmc_links=1000
SGD_steps=1
B=Grad_mcmc_links*SGD_steps
gamma=0.1
alpha=0.5
samples=1
seed=1
mcmc_mean=np.zeros((samples,2,int(T/d))) # This varible was originally designed 
# to store the mean of both processes, the one with multinomial sampling and the one with
# backward sampling.
resamp_coef=1
pars=np.zeros((samples,SGD_steps+1,3))
theta_0=0.2
sigma_0=1.6
sd=sd_true
pars[:,0]=np.array([theta_0,sigma_0,sd])
Grads_test=np.zeros((samples,B,3))
for i in range(samples):
    np.random.seed(i)
    print("theta_0 is: ",theta_0)   
    print("sigma_0 is: ",sigma_0)
    theta=theta_0
    sigma=sigma_0
    theta_fd=theta+fd_rate
    sigma_fd=sigma+fd_rate
    theta_aux=theta+0.2
    sigma_aux=sigma
    [log_weights,int_Gs,x_pr]=PF_bridge(t0,x0,T,b_ou_1d,theta,Sig_ou_1d,sigma,b_ou_aux,theta_aux,Sig_ou_aux,sigma_aux,\
    r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,[ou_sd,[theta_aux,sigma_aux],theta_aux],\
    sampling_ou, [theta_aux,sigma_aux],obs,log_g_normal_den,sd,\
    ou_trans_den,[theta_aux,sigma_aux],ou_trans_den,\
    resamp_coef,l,d, N,seed)
    #x_mean=np.sum(pff.norm_logweights(log_weights,ax=1)*x_pr,axis=-1)
    weights=pff.norm_logweights(log_weights[-1])
    #print(weights.shape)
    index=np.random.choice(np.array(range(N)),p=weights)
    cond_path=x_pr[:,index]
    cond_log_weights=log_weights[:,index]
    seeds_cond=np.zeros((int(T/d),2),dtype=int)
    seeds_cond[:,0]=seed+np.array(range(int(T/d)))*int(int(2**l*d-1))
    seeds_cond[:,1]=index*np.ones(int(T/d))
    cond_int_G=int_Gs[:,index]
    ch_paths=np.zeros((B,int(T/d)))
    ch_weights=np.zeros((B,int(T/d)))
    #cov
    print("Sample iterations is: ",i)
    #print("The starting seed is: ",seed)
    #print("The conditional seed is: ",seeds_cond)
    #print("The condtional path is:",cond_path) 
    n=1   
    cond_log_weights_test,cond_int_G_test,cond_path_test,seeds_cond_test=\
    cond_log_weights.copy() ,cond_int_G.copy(),cond_path.copy(),seeds_cond.copy()
    for b in range(B):
        # the varaible int_Gs is meant to have the record of int_G of the 
        # backward sampled path.
        print("mcmc iteration is:", b)
        seed+=int((int(T/d))*int(int(2**l*d-1)))
        np.random.seed(b)
        [log_weights_test,x_pr_test,cond_log_weights_test,cond_int_G_test,cond_path_test,seeds_cond_test,Grads_t]=\
        Grad_Cond_PF_bridge_back_samp(cond_log_weights_test,cond_int_G_test,cond_path_test,seeds_cond_test,t0,x0,T,b_ou_1d,\
        theta,theta_fd,Sig_ou_1d,\
        sigma,sigma_fd,b_ou_aux,theta_aux,Sig_ou_aux,sigma_aux,sigma_fd,r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],\
        [ou_sd,[theta_aux,sigma_fd]],H_quasi_normal,[ou_sd,[theta_aux,sigma_aux],theta_aux],[ou_sd,[theta_aux,sigma_fd],theta_aux],\
        sampling_ou, [theta_aux,sigma_aux],obs,log_g_normal_den,sd, ou_trans_den,[theta_aux,sigma_aux],\
        Grad_log_aux_trans_ou_new,ou_trans_den, Grad_log_G_new,resamp_coef, l, d,N,seed,fd_rate,crossed=False)
        Grads_test[i,b]=Grads_t
        ch_paths[b]=cond_path_test
        if (b+1)%Grad_mcmc_links==0:
            Grad_mcmc=np.mean(Grads_test[i,b+1-Grad_mcmc_links:b+1],axis=1)
            theta+=gamma*Grad_mcmc[0]/n**(0.5+alpha)
            sigma+=gamma*Grad_mcmc[1]/n**(0.5+alpha)
            #sd+=gamma*Grad_mcmc[2]/n**(0.5+alpha)
            pars[i,n]=np.array([theta,sigma,sd])
            theta_fd=theta+fd_rate
            sigma_fd=sigma+fd_rate
        
            theta_aux=theta+0.2
            sigma_aux=sigma
            n+=1
            print("The new parameters are: ",theta,sigma,sd)
    mcmc_mean[i,0]=np.mean(ch_paths,axis=0)
end=time.time()
print("The time spend is with ",N, " is ",end-start)
"""
#%%
"""
print("The original parameters are: ",theta_0,sigma_0)
print("The final parameters are: ",-0.4,0.9)
plt.plot(pars[:,:,0].T,pars[:,:,1].T)
print(pars)
plt.xlabel("Theta")
plt.ylabel("Sigma")
plt.title("SGD")
"""
#%%
## Analytical gradient of the loglikelihood
"""
theta=theta_true
sigma=sigma_true
theta_aux=theta+0.2
sigma_aux=sigma
print(theta_aux,sigma_aux)
sd=sd_true
fd_rate=1e-5
[theta_fd,sigma_fd,sd_fd]=np.array([theta,sigma,sd])+fd_rate*np.array([1,1,1])
print(theta_fd,sigma_fd,sd_fd)
"""
#%%
"""
Grid_p=10
thetas=np.linspace(-1,1,Grid_p)*0.5+theta_true    
sigmas=np.linspace(-1,1,Grid_p)*0.5+sigma_true
theta_aux=thetas+0.2
sigma_aux=sigmas
sds=np.linspace(-1,1,Grid_p)*0.5+ sd_true
fd_rate=1e-4
[theta_fd,sigma_fd,sd_fd]=np.array([thetas,sigmas,sds])+fd_rate*(np.zeros((3,Grid_p))+1)
print(thetas,sigmas)
"""
#%%
# IN 2d
"""
Grads=np.zeros((Grid_p,Grid_p,3))
dim=1
dim_o=1
for i in range(len(thetas)):
    theta=thetas[i]
    for j in range(len(sigmas)):
        sigma=sigmas[j]
        print(theta,sigma)
        K=np.array([[np.exp(d*theta)]])
        G=np.array([[sigma*np.sqrt((np.exp(2*d*theta)-1)/(2*theta))]])
        H=np.array([[1]])
        D=np.array([[sd]])
        #print(K,G**2,H,D)
        Grad_S=np.array([[[0]],[[0]],[[1]]],dtype=float)
        Grad_R_sigma_s=(np.exp(2*theta*d)-1)/(2*theta)
        Grad_R_theta=(sigma**2/(2*theta**2))*(1-np.exp(2*theta*d)+2*d*theta*np.exp(2*theta*d))
        Grad_R=np.zeros((3,1,1),dtype=float)
        Grad_R[0,0,0]=Grad_R_theta
        Grad_R[1,0,0]=Grad_R_sigma_s
        Grad_K=np.array([[[d*np.exp(d*theta)]],[[0]],[[0]]],dtype=float)
        #print(Grad_K,Grad_R,Grad_S)
        x_kf,x_kf_smooth,Grad_log_lik=KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
        Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
        Grads[j,i]=Grad_log_lik[:,0,0]
"""
#%%
# for 2d
#print("sd is: ",sds[int(Grid_p/2)]-1)
"""

print("sd is: ",sd_true)
thetas_Grid,sigmas_Grid=np.meshgrid(thetas,sigmas)
plt.quiver(thetas_Grid,sigmas_Grid,Grads[:,:,0],Grads[:,:,1])
max=np.max(Grads[:,:,0]**2+Grads[:,:,1]**2)
min=np.min(Grads[:,:,0]**2+Grads[:,:,1]**2)
print("The maximum gradient is: ",np.sqrt(max), "The minimum gradient is: ",np.sqrt(min))

#print("The starting guesses are: ",theta_0,sigma_0)
#print("The actual parameters are: ",theta_true,sigma_true)
plt.plot(pars[:,:,0].T,pars[:,:,1].T)
plt.xlabel("Theta")
plt.ylabel("Sigma")
plt.title("SGD")
"""
#%%

# Definition of the function that computes the SGD algorithm.


def SGD_bridge(t0,x0,T,b,A_0,A_fd_0,Sig,fi_0,fi_fd_0,b_til,A_til_0,Sig_til,fi_til_0,\
    fi_til_fd_0,r,r_pars,r_pars_fd,H,H_pars,H_pars_fd,sample_funct,sample_pars,\
    obs,log_g_den,g_den_par_0, aux_trans_den,atdp,\
    Grad_log_aux_trans,prop_trans_den, Grad_log_G,resamp_coef, l, d,N,seed,fd_rate,\
    mcmc_links,SGD_steps,gamma, alpha, update_pars):

    # new parameters:
    # A_0, fi_0, g_par_0: initial parameters for the SGD algorithm.
    # mcmc_links, SGD_steps: number of mcmc links and number of SGD steps.
    # gamma, alpha


    B=mcmc_links*SGD_steps
    mcmc_mean=np.zeros((int(T/d)))
    #gamma=0.1
    #alpha=0.25
    resamp_coef=1
    pars=np.zeros((SGD_steps+1,3))
    #A_0=0.2
    #fi_0=1.6
    #g_par_0=1
    pars[0]=np.array([A_0,fi_0,g_den_par_0])
    Grads_test=np.zeros((B,3))
    A=A_0
    A_til=A_til_0
    A_fd=A_fd_0
    fi=fi_0
    fi_fd=fi_fd_0
    fi_til=fi_til_0
    fi_til_fd=fi_til_fd_0
    g_den_par=g_den_par_0
    #g_par
    # The next part might depend on the specific dynamics of the example
    # since we have to define the finite difference for the parameters and 
    # A, fi, g_par might be a list parameters, each. 

    
    # Similarly for the auxiliar parameters, these might be related to 
    # the parameters of the dynamics.

    """
    [log_weights,int_Gs,x_pr]=PF_bridge(t0,x0,T,b_ou_1d,theta,Sig_ou_1d,sigma,b_ou_aux,theta_aux,Sig_ou_aux,sigma_aux,\
    r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,[ou_sd,[theta_aux,sigma_aux],theta_aux],\
    sampling_ou, [theta_aux,sigma_aux],obs,log_g_normal_den,sd,\
    ou_trans_den,[theta_aux,sigma_aux],ou_trans_den,\
    resamp_coef,l,d, N,seed)
    """
    np.random.seed(seed)
    [log_weights,int_Gs,x_pr]=PF_bridge(t0,x0,T,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,\
    r,r_pars,H,H_pars,\
    sample_funct,sample_pars,obs,log_g_den,g_den_par,\
    aux_trans_den,atdp,prop_trans_den,\
    resamp_coef,l,d, N,seed)
    #x_mean=np.sum(pff.norm_logweights(log_weights,ax=1)*x_pr,axis=-1)
    weights=pff.norm_logweights(log_weights[-1])
    #print(weights.shape)
    index=np.random.choice(np.array(range(N)))
    cond_path=x_pr[:,index]
    cond_log_weights=log_weights[:,index]
    seeds_cond=np.zeros((int(T/d),2),dtype=int)
    seeds_cond[:,0]=seed+np.array(range(int(T/d)))*int(int(2**l*d-1))
    seeds_cond[:,1]=index*np.ones(int(T/d))
    cond_int_G=int_Gs[:,index]
    Grads_mcmc=np.zeros((SGD_steps,3))
    ch_paths=np.zeros((B,int(T/d)))
    ch_weights=np.zeros((B,int(T/d)))
    
    #cov
    #print("The starting seed is: ",seed)
    #print("The conditional seed is: ",seeds_cond)
    #print("The condtional path is:",cond_path) 
    n=1
    cond_log_weights_test,cond_int_G_test,cond_path_test,seeds_cond_test=\
    cond_log_weights.copy() ,cond_int_G.copy(),cond_path.copy(),seeds_cond.copy()

    [A_fd,fi_fd,A_til,fi_til,fi_til_fd,\
    r_pars,r_pars_fd,H_pars,H_pars_fd,sample_pars,g_den_par,\
    atdp]=update_pars([A_0,fi_0,g_den_par_0],[A_0,fi_0,g_den_par_0],fd_rate,levels=1)

    for b_ind in range(B):
        # the varaible int_Gs is meant to have the record of int_G of the 
        # backward sampled path.
        #print("mcmc iteration is:", b)
        seed+=int((int(T/d))*int(int(2**l*d-1)))
        np.random.seed(b_ind)


        [log_weights,x_pr,cond_log_weights,cond_int_G,cond_path,seeds_cond,Grads_t]=\
        Grad_Cond_PF_bridge_back_samp(cond_log_weights,cond_int_G,cond_path,seeds_cond,t0,x0,T,b,\
        A,A_fd,Sig,fi,fi_fd,b_til,A_til,Sig_til,fi_til,fi_til_fd,r,r_pars,\
        r_pars_fd,H,H_pars,H_pars_fd,\
        sample_funct, sample_pars,obs,log_g_den,g_den_par, aux_trans_den,atdp,\
        Grad_log_aux_trans,prop_trans_den, Grad_log_G,resamp_coef, l, d,N,seed,fd_rate)
        Grads_test[b_ind]=Grads_t
        ch_paths[b_ind]=cond_path
        if (b_ind+1)%mcmc_links==0:
            Grad_mcmc=np.mean(Grads_test[b_ind+1-mcmc_links:b_ind+1],axis=0)
            Grads_mcmc[n-1]=Grad_mcmc
            A+=gamma*Grad_mcmc[0]/n**(0.5+alpha)
            #fi+=gamma*Grad_mcmc[1]/n**(0.5+alpha)
            fi*=np.exp(gamma*Grad_mcmc[1]*fi/n**(0.5+alpha))
            #g_den_par+=gamma*Grad_mcmc[2]/n**(0.5+alpha)
            g_den_par*=np.exp(gamma*Grad_mcmc[2]*g_den_par/n**(0.5+alpha))
            
            pars[n]=np.array([A,fi,g_den_par])
            
            [A_fd,fi_fd,A_til,fi_til,fi_til_fd,\
            r_pars,r_pars_fd,H_pars,H_pars_fd,sample_pars,g_den_par,\
            atdp]=update_pars(pars[n],pars[n],fd_rate,levels=1)

            n+=1
            #print("The new parameters are: ",A,fi,g_den_par)
    #mcmc_mean[i]=np.mean(ch_paths,axis=0)

    return ch_paths,pars 


def Gen_SGD_bridge(in_dist,in_dist_pars,Grad_log_in_dist,b,A,Sig,fi,b_til,Sig_til,\
    r,H,update_func,sample_funct,sample_pars,\
    obs,obs_times,log_g_den,g_den_par, aux_trans_den,\
    Grad_log_aux_trans,prop_trans_den, Grad_log_G,resamp_coef, l, d,N,seed,fd_rate,\
    mcmc_links,SGD_steps,gamma, gammas,alpha,ast, update_pars):
    # new parameters:
    # A_0, fi_0, g_par_0: initial parameters for the SGD algorithm.
    # mcmc_links, SGD_steps: number of mcmc links and number of SGD steps.
    # gamma, alpha
    B=mcmc_links*SGD_steps
    T=int(len(obs_times)*d)
    K=2*(2**ast-1)
    mcmc_mean=np.zeros((int(T/d)))
    resamp_coef=1
    pars=np.zeros((SGD_steps+1+ast,4))
    
    pars[0,:3]=in_dist_pars
    pars[0,3]=g_den_par
    Grads_test=np.zeros((B+K,4))
    # The following parameters are set to 1 since they are not actually used in the function
    # PF_bridge(the only instance where they are called), they are just vestigial parameters from
    # previous implementations.
    A_til=1
    fi_til=1
    r_pars=1
    H_pars=1
    atdp=1
    #g_par
    # The next part might depend on the specific dynamics of the example
    # since we have to define the finite difference for the parameters and 
    # A, fi, g_par might be a list parameters, each. 
    
    # Similarly for the auxiliar parameters, these might be related to 
    # the parameters of the dynamics.
    np.random.seed(seed)
    [log_weights,int_Gs,x_pr,indices]=Gen_PF_bridge(in_dist,in_dist_pars, b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,\
    update_func,sample_funct,sample_pars,obs,obs_times,log_g_den,g_den_par, aux_trans_den,atdp,\
    prop_trans_den, resamp_coef, l, d,N,seed)
    index=np.random.choice(np.array(range(N)))
    
    cond_path=x_pr[:,index].copy()
    cond_log_weights=log_weights[:,index].copy()
    seeds_cond=np.zeros((int(T/d)-1,2),dtype=int)
    Deltas=get_deltas(obs_times)
    intervals=obs_times[1:]-obs_times[:-1]
    seeds_cond[0,0]=seed
    for i in range(int(T/d)-2):
        seeds_cond[i+1,0]=seeds_cond[i,0]+int(2**l*intervals[i]/Deltas[i]) -1

    seeds_cond[:,1]=indices[:,index].copy()
    
    Grads_mcmc=np.zeros((SGD_steps+ast,4))
    ch_paths=np.zeros((B+K,int(T/d)))
    
    n=1
    the1=A[0]
    the2=A[1]
    the3=A[2]
    the4=g_den_par

    mcmc_step=2**ast
    count=0

    [in_dist_pars,A_fd,fi_fd,sample_pars]=update_pars([A,g_den_par],[A,g_den_par],fd_rate,levels=1)

    for b_ind in range(B+K):
        # the varaible int_Gs is meant to have the record of int_G of the 
        # backward sampled path.
        #print("mcmc iteration is:", b)
        seed+=int((int(T/d))*int(int(2**(l+1)-1)))
        np.random.seed(b_ind)

        [log_weights,x_pr,cond_log_weights,cond_int_G,cond_path,seeds_cond,Grads_t]=\
        Gen_Grad_Cond_PF_bridge_back_samp(in_dist,in_dist_pars,Grad_log_in_dist,\
        cond_path,seeds_cond,b,A,A_fd,Sig,fi,fi_fd,b_til,Sig_til,r,\
        H,update_func,sample_funct,sample_pars,obs,obs_times,log_g_den,g_den_par,\
        aux_trans_den,Grad_log_aux_trans,prop_trans_den, Grad_log_G,resamp_coef, l, d,N,seed,\
        fd_rate)


        Grads_test[b_ind]=Grads_t
        ch_paths[b_ind]=cond_path

        if n<=ast:
            if b_ind+1==count+int(mcmc_step*2/2**n):
                count+=int(mcmc_step*2/2**n)
        

                Grad_mcmc=np.mean(Grads_test[b_ind+1-int(mcmc_step*2/2**n):b_ind+1],axis=0)
                Grads_mcmc[n-1]=Grad_mcmc

                the1*=np.exp(gamma*gammas[0]*Grad_mcmc[0]*the1/n**(0.5+alpha))
                #fi+=gamma*Grad_mcmc[1]/n**(0.5+alpha)
                the2*=np.exp(gamma*gammas[1]*Grad_mcmc[1]*the2/n**(0.5+alpha))
                #g_den_par+=gamma*Grad_mcmc[2]/n**(0.5+alpha)
                the3*=np.exp(gamma*gammas[2]*Grad_mcmc[2]*the3/n**(0.5+alpha))
                the4*=np.exp(gamma*gammas[3]*Grad_mcmc[3]*the4/n**(0.5+alpha))


                pars[n]=np.array([the1,the2,the3,the4]) 
                A=pars[n,:3].copy()
                fi=pars[n,2]
                g_den_par=pars[n,3]

                the1s=pars[n,0]
                the2s=pars[n,1]
                the3s=pars[n,2]
                the4s=pars[n,3]
                w=the3s**2/2+the1s
                xi=the2s
                sigma=the3s
                alpha_pars=2*w/sigma**2-1
                theta=sigma**2/(2*xi)
                #print(" The pars are: ",the1s,the2s,the3s,the4s)    
                #print("The mean is: ",alpha_pars*theta)  
                #print("The gradeint is:",Grad_mcmc)
                #print("The value of n is: ",n)  


                pars_0=A,g_den_par
                [in_dist_pars,A_fd,fi_fd,sample_pars]=update_pars(pars_0,pars_0,fd_rate,levels=1)

                n+=1

        else:
            if (b_ind+1-count)%mcmc_links==0:
                Grad_mcmc=np.mean(Grads_test[b_ind+1-mcmc_links:b_ind+1],axis=0)
                Grads_mcmc[n-1]=Grad_mcmc

                the1*=np.exp(gamma*gammas[0]*Grad_mcmc[0]*the1/n**(0.5+alpha))
                #fi+=gamma*Grad_mcmc[1]/n**(0.5+alpha)
                the2*=np.exp(gamma*gammas[1]*Grad_mcmc[1]*the2/n**(0.5+alpha))
                #g_den_par+=gamma*Grad_mcmc[2]/n**(0.5+alpha)
                the3*=np.exp(gamma*gammas[2]*Grad_mcmc[2]*the3/n**(0.5+alpha))
                the4*=np.exp(gamma*gammas[3]*Grad_mcmc[3]*the4/n**(0.5+alpha))


                pars[n]=np.array([the1,the2,the3,the4]) 
                A=pars[n,:3].copy()
                fi=pars[n,2]
                g_den_par=pars[n,3]

                the1s=pars[n,0]
                the2s=pars[n,1]
                the3s=pars[n,2]
                the4s=pars[n,3]
                w=the3s**2/2+the1s
                xi=the2s
                sigma=the3s
                alpha_pars=2*w/sigma**2-1
                theta=sigma**2/(2*xi)
                #print(" The pars are: ",the1s,the2s,the3s,the4s)    
                #print("The mean is: ",alpha_pars*theta)  
                #print("The gradeint is:",Grad_mcmc)
                #print("The value of n is: ",n)  


                pars_0=A,g_den_par
                [in_dist_pars,A_fd,fi_fd,sample_pars]=update_pars(pars_0,pars_0,fd_rate,levels=1)

                n+=1
            
        
    return ch_paths,pars,Grads_mcmc 



def Gen_C_SGD_bridge(in_dist,in_dist_pars,Grad_log_in_dist,b,A,Sig,fi,b_til,Sig_til,\
    r,H,update_func,max_sample_funct,sample_pars,\
    obs,obs_times,log_g_den,g_den_par, aux_trans_den,\
    Grad_log_aux_trans,prop_trans_den,ind_prop_trans_par, Grad_log_G,resamp_coef, l, d,N,seed,fd_rate,\
    mcmc_links,SGD_steps,gamma,gammas, alpha,ast, update_pars):

    # new parameters:
    # A_0, fi_0, g_par_0: initial parameters for the SGD algorithm.
    # mcmc_links, SGD_steps: number of mcmc links and number of SGD steps.
    # gamma, alpha
    B=mcmc_links*SGD_steps
    K=2*(2**ast-1)
    T=int(len(obs_times)*d)
    mcmc_mean=np.zeros((2,int(T/d)))
    resamp_coef=1
    pars_0=np.zeros((SGD_steps+1+ast,4))
    pars_1=np.zeros((SGD_steps+1+ast,4))
    
    pars_0[0,:3]=A
    pars_1[0,:3]=A
    pars_0[0,3]=g_den_par
    pars_1[0,3]=g_den_par
    Grads_test_0=np.zeros((B+K,4))
    Grads_test_1=np.zeros((B+K,4))
    
    # The following parameters are set to 1 since they are not actually used in the function
    # PF_bridge(the only instance where they are called), they are just vestigial parameters from
    # previous implementations.
    A_til=1
    fi_til=1
    r_pars=1
    H_pars=1
    atdp=1
    #g_par
    # The next part might depend on the specific dynamics of the example
    # since we have to define the finite difference for the parameters and 
    # A, fi, g_par might be a list parameters, each. 
    
    # Similarly for the auxiliar parameters, these might be related to 
    # the parameters of the dynamics.
    np.random.seed(seed)
    [log_weights_0,log_weights_1,int_Gs_0,int_Gs_1,x_pr_0,x_pr_1,seeds_0_wp,seeds_1_wp]=\
    Gen_C_PF_bridge(in_dist,in_dist_pars, b,A,Sig,fi,b_til,A_til,\
    Sig_til,fi_til,r,r_pars,H,H_pars,\
    update_func,max_sample_funct,sample_pars,obs,obs_times,log_g_den,g_den_par, aux_trans_den,atdp,\
    prop_trans_den, ind_prop_trans_par,resamp_coef, l, d,N,seed)


    index=np.random.choice(np.array(range(N)))
    
    cond_path_0=x_pr_0[:,index].copy()
    cond_path_1=x_pr_1[:,index].copy()

    #cond_log_weights_0=log_weights_0[:,index].copy()
    #cond_log_weights_1=log_weights_1[:,index].copy()
    Deltas=get_deltas(obs_times)
    intervals=obs_times[1:]-obs_times[:-1]
    seeds_cond_0=seeds_0_wp[:,:,index].copy()
    seeds_cond_1=seeds_1_wp[:,:,index].copy()

    Grads_mcmc_0=np.zeros((SGD_steps+ast,4))
    Grads_mcmc_1=np.zeros((SGD_steps+ast,4))
    ch_paths_0=np.zeros((B+K,int(T/d)))
    ch_paths_1=np.zeros((B+K,int(T/d)))
    
    n=1
    the1_0=A[0]
    the2_0=A[1]
    the3_0=A[2]
    the4_0=g_den_par
    A_0=A
    fi_0=fi
    g_den_par_0=g_den_par

    the1_1=A[0]
    the2_1=A[1]
    the3_1=A[2]
    the4_1=g_den_par
    A_1=A
    fi_1=fi
    g_den_par_1=g_den_par

    mcmc_step=2**ast
    count=0


    [in_dist_pars,A_fd_0,A_fd_1,fi_fd_0,fi_fd_1,sample_pars,\
        ind_prop_trans_par_0,ind_prop_trans_par_1]=update_pars([A,g_den_par],[A,g_den_par],fd_rate,levels=2)
    gamma_1,gamma_2,gamma_3,gamma_4=gammas

    for b_ind in range(B+K):
        # the varaible int_Gs is meant to have the record of int_G of the 
        # backward sampled path.
        #print("mcmc iteration is:", b)
        seed+=int((int(T/d))*int(int(2**(l+1)-1)))
        np.random.seed(b_ind)

        [log_weights_0,log_weights_1,x_pr_0,x_pr_1, \
        cond_log_weights_0,cond_log_weights_1,cond_int_G_0,cond_int_G_1,\
        cond_path_0,cond_path_1,seeds_cond_0,seeds_cond_1,Grads_t_0,Grads_t_1]=\
        Gen_C_Grad_Cond_PF_bridge_back_samp(in_dist,in_dist_pars,Grad_log_in_dist,\
        cond_path_0,cond_path_1,seeds_cond_0,seeds_cond_1,b,A_0,A_1,A_fd_0,A_fd_1,Sig,\
        fi_0,fi_1,fi_fd_0,fi_fd_1,b_til,Sig_til,r,\
        H,update_func,max_sample_funct,sample_pars,obs,obs_times,log_g_den,g_den_par_0,g_den_par_1,\
        aux_trans_den,Grad_log_aux_trans,prop_trans_den,ind_prop_trans_par_0,ind_prop_trans_par_1,\
        Grad_log_G,resamp_coef, l, d,N,seed,fd_rate)


        Grads_test_0[b_ind]=Grads_t_0
        Grads_test_1[b_ind]=Grads_t_1
        ch_paths_0[b_ind]=cond_path_0
        ch_paths_1[b_ind]=cond_path_1


        #if (b_ind+1)%mcmc_links==0:
        if n<=ast:
            if b_ind+1==count+int(mcmc_step*2/2**n):
                count+=int(mcmc_step*2/2**n)
        
                Grad_mcmc_0=np.mean(Grads_test_0[b_ind+1-int(mcmc_step*2/2**n):b_ind+1],axis=0)
                Grad_mcmc_1=np.mean(Grads_test_1[b_ind+1-int(mcmc_step*2/2**n):b_ind+1],axis=0)

                Grads_mcmc_0[n-1]=Grad_mcmc_0
                Grads_mcmc_1[n-1]=Grad_mcmc_1

                the1_0*=np.exp(gamma_1*gamma*Grad_mcmc_0[0]*the1_0/n**(0.5+alpha))
                the1_1*=np.exp(gamma_1*gamma*Grad_mcmc_1[0]*the1_1/n**(0.5+alpha))
                #fi+=gamma*Grad_mcmc[1]/n**(0.5+alpha)
                the2_0*=np.exp(gamma_2*gamma*Grad_mcmc_0[1]*the2_0/n**(0.5+alpha))
                the2_1*=np.exp(gamma_2*gamma*Grad_mcmc_1[1]*the2_1/n**(0.5+alpha))
                #g_den_par+=gamma*Grad_mcmc[2]/n**(0.5+alpha)
                the3_0*=np.exp(gamma_3*gamma*Grad_mcmc_0[2]*the3_0/n**(0.5+alpha))
                the3_1*=np.exp(gamma_3*gamma*Grad_mcmc_1[2]*the3_1/n**(0.5+alpha))

                the4_0*=np.exp(gamma_4*gamma*Grad_mcmc_0[3]*the4_0/n**(0.5+alpha))
                the4_1*=np.exp(gamma_4*gamma*Grad_mcmc_1[3]*the4_1/n**(0.5+alpha))



                pars_0[n]=np.array([the1_0,the2_0,the3_0,the4_0]) 
                A_0=pars_0[n,:3].copy()
                fi_0=pars_0[n,2]
                g_den_par_0=pars_0[n,3]

                pars_1[n]=np.array([the1_1,the2_1,the3_1,the4_1])
                A_1=pars_1[n,:3].copy()
                fi_1=pars_1[n,2]
                g_den_par_1=pars_1[n,3]

                the1s_0=pars_0[n,0]
                the2s_0=pars_0[n,1]
                the3s_0=pars_0[n,2]
                the4s_0=pars_0[n,3]
                w_0=the3s_0**2/2+the1s_0
                xi_0=the2s_0
                sigma_0=the3s_0
                alpha_pars_0=2*w_0/sigma_0**2-1
                theta_0=sigma_0**2/(2*xi_0)

                #print(" The pars for 0 are: ",the1s_0,the2s_0,the3s_0,the4s_0)    
                #print("The mean for 0 is: ",alpha_pars_0*theta_0)  
                #print("The gradeint for 0 is:",Grad_mcmc_0)


                the1s_1=pars_1[n,0]
                the2s_1=pars_1[n,1]
                the3s_1=pars_1[n,2]
                the4s_1=pars_1[n,3]
                w_1=the3s_1**2/2+the1s_1
                xi_1=the2s_1
                sigma_1=the3s_1
                alpha_pars_1=2*w_1/sigma_1**2-1
                theta_1=sigma_1**2/(2*xi_1)
                #print(" The pars for 1 are: ",the1s_1,the2s_1,the3s_1,the4s_1)
                #print("The mean for 1 is: ",alpha_pars_1*theta_1)
                #print("The gradeint for 1 is:",Grad_mcmc_1)
                #print("The value of n for 1 is: ",n)





                [in_dist_pars,A_fd_0,A_fd_1,fi_fd_0,fi_fd_1,sample_pars,\
                ind_prop_trans_par_0,ind_prop_trans_par_1]=\
                update_pars([A_0,g_den_par_0],[A_1,g_den_par_1],fd_rate,levels=2)


                n+=1

        else:
            
            if (b_ind+1-count)%mcmc_links==0:
                
                Grad_mcmc_0=np.mean(Grads_test_0[b_ind+1-mcmc_links:b_ind+1],axis=0)
                Grad_mcmc_1=np.mean(Grads_test_1[b_ind+1-mcmc_links:b_ind+1],axis=0)
    
                Grads_mcmc_0[n-1]=Grad_mcmc_0
                Grads_mcmc_1[n-1]=Grad_mcmc_1
                print(np.array(gammas)*gamma/n**(0.5+alpha))
                print(Grad_mcmc_1)
                the1_0*=np.exp(gamma_1*gamma*Grad_mcmc_0[0]*the1_0/n**(0.5+alpha))
                the1_1*=np.exp(gamma_1*gamma*Grad_mcmc_1[0]*the1_1/n**(0.5+alpha))
                #fi+=gamma*Grad_mcmc[1]/n**(0.5+alpha)
                the2_0*=np.exp(gamma_2*gamma*Grad_mcmc_0[1]*the2_0/n**(0.5+alpha))
                the2_1*=np.exp(gamma_2*gamma*Grad_mcmc_1[1]*the2_1/n**(0.5+alpha))
                #g_den_par+=gamma*Grad_mcmc[2]/n**(0.5+alpha)
                the3_0*=np.exp(gamma_3*gamma*Grad_mcmc_0[2]*the3_0/n**(0.5+alpha))
                the3_1*=np.exp(gamma_3*gamma*Grad_mcmc_1[2]*the3_1/n**(0.5+alpha))
                
                the4_0*=np.exp(gamma_4*gamma*Grad_mcmc_0[3]*the4_0/n**(0.5+alpha))
                the4_1*=np.exp(gamma_4*gamma*Grad_mcmc_1[3]*the4_1/n**(0.5+alpha))
    
                
                
                pars_0[n]=np.array([the1_0,the2_0,the3_0,the4_0]) 
                A_0=pars_0[n,:3].copy()
                fi_0=pars_0[n,2]
                g_den_par_0=pars_0[n,3]
    
                pars_1[n]=np.array([the1_1,the2_1,the3_1,the4_1])
                A_1=pars_1[n,:3].copy()
                fi_1=pars_1[n,2]
                g_den_par_1=pars_1[n,3]
    
                the1s_0=pars_0[n,0]
                the2s_0=pars_0[n,1]
                the3s_0=pars_0[n,2]
                the4s_0=pars_0[n,3]
                w_0=the3s_0**2/2+the1s_0
                xi_0=the2s_0
                sigma_0=the3s_0
                alpha_pars_0=2*w_0/sigma_0**2-1
                theta_0=sigma_0**2/(2*xi_0)
                
                #print(" The pars for 0 are: ",the1s_0,the2s_0,the3s_0,the4s_0)    
                #print("The mean for 0 is: ",alpha_pars_0*theta_0)  
                #print("The gradeint for 0 is:",Grad_mcmc_0)
    
    
                the1s_1=pars_1[n,0]
                the2s_1=pars_1[n,1]
                the3s_1=pars_1[n,2]
                the4s_1=pars_1[n,3]
                w_1=the3s_1**2/2+the1s_1
                xi_1=the2s_1
                sigma_1=the3s_1
                alpha_pars_1=2*w_1/sigma_1**2-1
                theta_1=sigma_1**2/(2*xi_1)
                #print(" The pars for 1 are: ",the1s_1,the2s_1,the3s_1,the4s_1)
                #print("The mean for 1 is: ",alpha_pars_1*theta_1)
                #print("The gradeint for 1 is:",Grad_mcmc_1)
                #print("The value of n for 1 is: ",n)
                
                [in_dist_pars,A_fd_0,A_fd_1,fi_fd_0,fi_fd_1,sample_pars,\
                ind_prop_trans_par_0,ind_prop_trans_par_1]=\
                update_pars([A_0,g_den_par_0],[A_1,g_den_par_1],fd_rate,levels=2)
        
    
                n+=1
            
        
    return ch_paths_0,ch_paths_1,pars_0,pars_1 



def SGD_bridge_vanilla(t0,x0,T,b,A_0,A_fd_0,Sig,fi_0,fi_fd_0,b_til,A_til_0,Sig_til,fi_til_0,\
    fi_til_fd_0,r,r_pars,r_pars_fd,H,H_pars,H_pars_fd,sample_funct,sample_pars,\
    obs,log_g_den,g_den_par_0, aux_trans_den,atdp,\
    Grad_log_aux_trans,prop_trans_den, Grad_log_G,resamp_coef, l, d,N,seed,fd_rate,\
    mcmc_links,SGD_steps,gamma, alpha, update_pars):

    # new parameters:
    # A_0, fi_0, g_par_0: initial parameters for the SGD algorithm.
    # mcmc_links, SGD_steps: number of mcmc links and number of SGD steps.
    # gamma, alpha
    
    B=mcmc_links*SGD_steps
    mcmc_mean=np.zeros((int(T/d)))
    #gamma=0.1
    #alpha=0.25
    resamp_coef=1
    pars=np.zeros((SGD_steps+1,3))
    #A_0=0.2
    #fi_0=1.6
    #g_par_0=1
    pars[0]=np.array([A_0,fi_0,g_den_par_0])
    Grads_test=np.zeros((B,3))
    A=A_0
    A_til=A_til_0
    A_fd=A_fd_0
    fi=fi_0
    fi_fd=fi_fd_0
    fi_til=fi_til_0
    fi_til_fd=fi_til_fd_0
    g_den_par=g_den_par_0
    #g_par
    # The next part might depend on the specific dynamics of the example
    # since we have to define the finite difference for the parameters and 
    # A, fi, g_par might be a list parameters, each. 

    
    # Similarly for the auxiliar parameters, these might be related to 
    # the parameters of the dynamics.

    """
    [log_weights,int_Gs,x_pr]=PF_bridge(t0,x0,T,b_ou_1d,theta,Sig_ou_1d,sigma,b_ou_aux,theta_aux,Sig_ou_aux,sigma_aux,\
    r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,[ou_sd,[theta_aux,sigma_aux],theta_aux],\
    sampling_ou, [theta_aux,sigma_aux],obs,log_g_normal_den,sd,\
    ou_trans_den,[theta_aux,sigma_aux],ou_trans_den,\
    resamp_coef,l,d, N,seed)
    """
    np.random.seed(seed)
    [log_weights,int_Gs,x_pr]=PF_bridge(t0,x0,T,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,\
    r,r_pars,H,H_pars,\
    sample_funct,sample_pars,obs,log_g_den,g_den_par,\
    aux_trans_den,atdp,prop_trans_den,\
    resamp_coef,l,d, N,seed)
    #x_mean=np.sum(pff.norm_logweights(log_weights,ax=1)*x_pr,axis=-1)
    weights=pff.norm_logweights(log_weights[-1])
    #print(weights.shape)
    index=np.random.choice(np.array(range(N)))
    cond_path=x_pr[:,index]
    cond_log_weights=log_weights[:,index]
    seeds_cond=np.zeros((int(T/d),2),dtype=int)
    seeds_cond[:,0]=seed+np.array(range(int(T/d)))*int(int(2**l*d-1))
    seeds_cond[:,1]=index*np.ones(int(T/d))
    cond_int_G=int_Gs[:,index]
    ch_paths=np.zeros((B,int(T/d)))
    Grads_mcmc=np.zeros((SGD_steps,3))
    ch_weights=np.zeros((B,int(T/d)))
    #cov
    #print("The starting seed is: ",seed)
    #print("The conditional seed is: ",seeds_cond)
    #print("The condtional path is:",cond_path) 
    n=1
    cond_log_weights_test,cond_int_G_test,cond_path_test,seeds_cond_test=\
    cond_log_weights.copy() ,cond_int_G.copy(),cond_path.copy(),seeds_cond.copy()

    [A_fd,fi_fd,A_til,fi_til,fi_til_fd,\
    r_pars,r_pars_fd,H_pars,H_pars_fd,sample_pars,g_den_par,\
    atdp]=update_pars([A_0,fi_0,g_den_par_0],[A_0,fi_0,g_den_par_0],fd_rate,levels=1)

    for b_ind in range(B):
        # the varaible int_Gs is meant to have the record of int_G of the 
        # backward sampled path.
        #print("mcmc iteration is:", b)
        seed+=int((int(T/d))*int(int(2**l*d-1)))
        np.random.seed(b_ind)

        """
        (lw_cond,int_Gs_cond,x_cond,seeds_cond,t0,x0,T,b,A,A_fd,Sig,fi,fi_fd,b_til,A_til,Sig_til,fi_til,\
        fi_til_fd,r,r_pars,r_pars_fd,H,H_pars,H_pars_fd,sample_funct,sample_pars,obs,log_g_den,g_den_par, aux_trans_den,atdp,\
        Grad_log_aux_trans,prop_trans_den, Grad_log_G,resamp_coef, l, d,N,seed,fd_rate,crossed=False)
        """

        [log_weights_test,x_pr_test,cond_log_weights_test,cond_int_G_test,cond_path_test,seeds_cond_test,Grads_t]=\
        Grad_Cond_PF_bridge_new(cond_log_weights_test,cond_int_G_test,cond_path_test,seeds_cond_test,t0,x0,T,b,\
        A,A_fd,Sig,fi,fi_fd,b_til,A_til,Sig_til,fi_til,fi_til_fd,r,r_pars,\
        r_pars_fd,H,H_pars,H_pars_fd,\
        sample_funct, sample_pars,obs,log_g_den,g_den_par, aux_trans_den,atdp,\
        Grad_log_aux_trans,prop_trans_den, Grad_log_G,resamp_coef, l, d,N,seed,fd_rate,crossed=False)
        Grads_test[b_ind]=Grads_t
        ch_paths[b_ind]=cond_path_test
        if (b_ind+1)%mcmc_links==0:
            Grad_mcmc=np.mean(Grads_test[b_ind+1-mcmc_links:b_ind+1],axis=0)
            Grads_mcmc[n-1]=Grad_mcmc
            A+=gamma*Grad_mcmc[0]/n**(0.5+alpha)
            #fi+=gamma*Grad_mcmc[1]/n**(0.5+alpha)
            fi*=np.exp(gamma*Grad_mcmc[1]*fi/n**(0.5+alpha))
            #g_den_par+=gamma*Grad_mcmc[2]/n**(0.5+alpha)
            g_den_par*=np.exp(gamma*Grad_mcmc[2]*g_den_par/n**(0.5+alpha))
            
            pars[n]=np.array([A,fi,g_den_par])
            
            [A_fd,fi_fd,A_til,fi_til,fi_til_fd,\
            r_pars,r_pars_fd,H_pars,H_pars_fd,sample_pars,g_den_par,\
            atdp]=update_pars(pars[n],pars[n],fd_rate,levels=1)

            n+=1
            #print("The new parameters are: ",A,fi,g_den_par)
    #mcmc_mean[i]=np.mean(ch_paths,axis=0)

    return ch_paths,pars,Grads_mcmc

#%%


# what is the problem? we don't the result we are seeking, i.e. second moment of
# the difference between the two levels of the mcmc samples don't 
# follow the particle filter rates.

# possible causes: The filter almost follow this rates, there is a change the 
# the sample doesn't, why? doesn't make sense bcs the pf filter has these 
# rates and the backward sample resembles running the particle filter for 
# time 2T

# the coupling of the PF or the backward sample is not correct

# do we need more samples? I don't think so.

def C_SGD_bridge(t0,x0,T,b,A_in,A_fd_in,Sig,fi_in,fi_fd_in,b_til,A_til_in,Sig_til,fi_til_in,\
    fi_til_fd_in,r,r_pars,r_pars_fd,H,H_pars,H_pars_fd,max_sample_funct,sample_pars,\
    obs,log_g_den,g_den_par_in, aux_trans_den,atdp,\
    Grad_log_aux_trans,prop_trans_den,ind_prop_trans_par, Grad_log_G,update_pars,resamp_coef, l, d,N,seed,fd_rate,\
    mcmc_links,SGD_steps,gamma, alpha):


    # new parameters:
    # A_in, fi_in, g_par_0: initial parameters for the SGD algorithm.
    # mcmc_links, SGD_steps: number of mcmc links and number of SGD steps.
    # gamma, alpha
    B=mcmc_links*SGD_steps
    mcmc_mean=np.zeros((2,int(T/d)))
    #gamma=0.1
    #alpha=0.25
    resamp_coef=1
    pars_0=np.zeros((SGD_steps+1,3))
    pars_1=np.zeros((SGD_steps+1,3))
    #A_in=0.2
    #fi_in=1.6
    #g_par_0=1
    pars_0[0,:]=np.array([A_in,fi_in,g_den_par_in])
    pars_1[0,:]=np.array([A_in,fi_in,g_den_par_in])
    Grads_test_0=np.zeros((B,3))
    Grads_test_1=np.zeros((B,3))
    A=A_in
    A_til=A_til_in
    A_fd=A_fd_in
    fi=fi_in
    fi_fd=fi_fd_in
    fi_til=fi_til_in
    fi_til_fd=fi_til_fd_in
    g_den_par=g_den_par_in
    
    A_0=A_in
    fi_0=fi_in
    g_den_par_0=g_den_par_in
    A_1=A_in
    fi_1=fi_in
    g_den_par_1=g_den_par_in


    [A_fd_0,A_fd_1,fi_fd_0,fi_fd_1,A_til_0,A_til_1,fi_til_0,fi_til_1,fi_til_fd_0,\
    fi_til_fd_1,r_pars_0,r_pars_1,\
    r_pars_fd_0,r_pars_fd_1,H_pars_0,H_pars_1,H_pars_fd_0,H_pars_fd_1,\
    sample_pars,g_den_par_0,g_den_par_1,atdp_0,atdp_1,\
    ind_prop_trans_par_0,ind_prop_trans_par_1]=\
    update_pars([A_0,fi_0,g_den_par_0],[A_1,fi_1,g_den_par_1],fd_rate,levels=2)
    
    #"""
    """
    A_til_0=A_til_in
    A_fd_0=A_fd_in
    fi_fd_0=fi_fd_in
    fi_til_0=fi_til_in
    fi_til_fd_0=fi_til_fd_in
    g_den_par_0=g_den_par_in
    r_pars_0=r_pars
    H_pars_0=H_pars
    r_pars_fd_0=r_pars_fd
    H_pars_fd_0=H_pars_fd
    atdp_0=atdp
    ind_prop_trans_par_0=ind_prop_trans_par
    
    
    A_til_1=A_til_in
    A_fd_1=A_fd_in
    fi_fd_1=fi_fd_in
    fi_til_1=fi_til_in
    fi_til_fd_1=fi_til_fd_in
    g_den_par_1=g_den_par_in
    r_pars_1=r_pars
    H_pars_1=H_pars
    r_pars_fd_1=r_pars_fd
    H_pars_fd_1=H_pars_fd   
    atdp_1=atdp
    ind_prop_trans_par_1=ind_prop_trans_par
    """
    
    #g_par
    # The next part might depend on the specific dynamics of the example
    # since we have to define the finite difference for the parameters and 
    # A, fi, g_par might be a list parameters, each. 

    
    # Similarly for the auxiliar parameters, these might be related to 
    # the parameters of the dynamics.

    """
    C_PF_bridge(t0,x0,T,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,\
    max_sample_funct,sample_pars,obs,log_g_den,g_den_par, aux_trans_den,atdp,\
    prop_trans_den,ind_prop_trans_par, resamp_coef, l, d,N,seed,crossed=False):
    """

    """
    [log_weights_0,log_weights_1,int_Gs_0,int_Gs_1,x_pr_0,x_pr_1]
    """
    np.random.seed(seed)
    [log_weights_0,log_weights_1,int_Gs_0,int_Gs_1,x_pr_0,x_pr_1,seeds_0_wp,seeds_1_wp]=\
    C_PF_bridge(t0,x0,T,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,\
    r,r_pars,H,H_pars,\
    max_sample_funct,sample_pars,obs,log_g_den,g_den_par,\
    aux_trans_den,atdp,prop_trans_den,ind_prop_trans_par,\
    resamp_coef,l,d, N,seed)
    #x_mean=np.sum(pff.norm_logweights(log_weights,ax=1)*x_pr,axis=-1)
    
    weights_0=pff.norm_logweights(log_weights_0[-1])
    weights_1=pff.norm_logweights(log_weights_1[-1])
    #print(weights.shape)
    index=np.random.choice(np.array(range(N)))
    #index_1=np.random.choice(np.array(range(N)),p=weights_1)
    cond_path_0=np.copy(x_pr_0[:,index])
    cond_path_1=np.copy(x_pr_1[:,index])
    cond_log_weights_0=np.copy(log_weights_0[:,index])
    cond_log_weights_1=np.copy(log_weights_1[:,index])
    seeds_cond_0=np.copy(seeds_0_wp[:,:,index])
    seeds_cond_1=np.copy(seeds_1_wp[:,:,index])

    ch_paths_0=np.zeros((B,int(T/d))) 
    ch_weights_0=np.zeros((B,int(T/d)))
    ch_paths_1=np.zeros((B,int(T/d)))
    ch_weights_1=np.zeros((B,int(T/d)))
    """
    cond_int_G_0=int_Gs_0[:,index_0]
    cond_int_G_1=int_Gs_1[:,index_1]
    """
    
    #cov
    #print("The starting seed is: ",seed)
    #print("The conditional seed is: ",seeds_cond)
    #print("The condtional path is:",cond_path) 
    """
    n=1
    cond_log_weights_test_0,cond_int_G_test_0,cond_path_test_0,seeds_cond_test_0=\
    cond_log_weights_0.copy() ,cond_int_G_0.copy(),cond_path_0.copy(),seeds_cond_0.copy()
    
    cond_log_weights_test_1,cond_int_G_test_1,cond_path_test_1,seeds_cond_test_1=\
    cond_log_weights_1.copy() ,cond_int_G_1.copy(),cond_path_1.copy(),seeds_cond_1.copy()
    """
    n=1

    for b_ind in range(B):
        # the varaible int_Gs is meant to have the record of int_G of the 
        # backward sampled path.
        #print("mcmc iteration is:", b)
        seed+=int((int(T/d))*int(int(2**l*d-1)))
        np.random.seed(b_ind)
 
        """
        C_Grad_Cond_PF_bridge_back_samp(x_cond_0,x_cond_1,\
        seeds_cond_0,seeds_cond_1,t0,x0,T,b,A_0,A_1,A_fd_0,A_fd_1,Sig,fi_0,fi_1,fi_fd_0,fi_fd_1,b_til,\
        A_til_0,A_til_1,Sig_til,fi_til_0,fi_til_1,fi_til_fd_0,fi_til_fd_1,r,r_pars_0,r_pars_1,r_pars_fd_0,\
        r_pars_fd_1,H,H_pars_0,H_pars_1,H_pars_fd_0,H_pars_fd_1,sample_funct,sample_pars,obs,log_g_den,\
        g_den_par_0,g_den_par_1, aux_trans_den,atdp_0,atdp_1,\
        Grad_log_aux_trans,prop_trans_den,ind_prop_trans_par_0,ind_prop_trans_par_1,Grad_log_G,l,d,N,seed,fd_rate,crossed=False):
        """
        [log_weights_test_0,log_weights_test_1,x_pr_test_0,x_pr_test_1,cond_log_weights_test_0,cond_log_weights_test_1,\
        cond_int_G_test_0,cond_int_G_test_1,cond_path_0,cond_path_1,seeds_cond_0,seeds_cond_1,Grads_t_0,Grads_t_1]=\
        C_Grad_Cond_PF_bridge_back_samp(cond_path_0,cond_path_1,seeds_cond_0,seeds_cond_1,t0,x0,T,b,\
        A_0,A_1,A_fd_0,A_fd_1,Sig,fi_0,fi_1,fi_fd_0,fi_fd_1,b_til,A_til_0,A_til_1,Sig_til,fi_til_0,fi_til_1,fi_til_fd_0,\
        fi_til_fd_1,r,r_pars_0,r_pars_1,\
        r_pars_fd_0,r_pars_fd_1,H,H_pars_0,H_pars_1,H_pars_fd_0,H_pars_fd_1,\
        max_sample_funct, sample_pars,obs,log_g_den,g_den_par_0,g_den_par_1, aux_trans_den,atdp_0,atdp_1,\
        Grad_log_aux_trans,prop_trans_den, ind_prop_trans_par_0,ind_prop_trans_par_1,Grad_log_G, l, d,N,seed,fd_rate,crossed=False)
        Grads_test_0[b_ind]=Grads_t_0
        Grads_test_1[b_ind]=Grads_t_1
        ch_paths_0[b_ind]=cond_path_0
        ch_paths_1[b_ind]=cond_path_1
        if (b_ind+1)%mcmc_links==0:
            #print("SGD step is:",n)
            Grad_mcmc_0=np.mean(Grads_test_0[b_ind+1-mcmc_links:b_ind+1],axis=0)
            Grad_mcmc_1=np.mean(Grads_test_1[b_ind+1-mcmc_links:b_ind+1],axis=0)
            A_0+=gamma*Grad_mcmc_0[0]/n**(0.5+alpha)
            A_1+=gamma*Grad_mcmc_1[0]/n**(0.5+alpha)
            # The following lines are also SGD applied to the log of the parameters
            # which translate in the following SGD equations
            fi_0*=np.exp(gamma*Grad_mcmc_0[1]*fi_0/n**(0.5+alpha))
            fi_1*=np.exp(gamma*Grad_mcmc_1[1]*fi_1/n**(0.5+alpha))
            g_den_par_0*=np.exp(gamma*Grad_mcmc_0[2]*g_den_par_0/n**(0.5+alpha))
            g_den_par_1*=np.exp(gamma*Grad_mcmc_1[2]*g_den_par_1/n**(0.5+alpha))
            #fi_0+=gamma*Grad_mcmc_0[1]/n**(0.5+alpha)
            #fi_1+=gamma*Grad_mcmc_1[1]/n**(0.5+alpha)
            #g_den_par_0+=gamma*Grad_mcmc_0[2]/n**(0.5+alpha)*0
            #g_den_par_1+=gamma*Grad_mcmc_1[2]/n**(0.5+alpha)*0
            pars_0[n]=np.array([A_0,fi_0,g_den_par_0])
            pars_1[n]=np.array([A_1,fi_1,g_den_par_1])

            """
            [A_fd_0,A_fd_1,fi_fd_0,fi_fd_1,A_til_0,A_til_1,fi_til_0,fi_til_1,fi_til_fd_0,\
            fi_til_fd_1,r_pars_0,r_pars_1,\
            r_pars_fd_0,r_pars_fd_1,H_pars_0,H_pars_1,H_pars_fd_0,H_pars_fd_1,\
            sample_pars,g_den_par_0,g_den_par_1,atdp_0,atdp_1,\
            ind_prop_trans_par_0,ind_prop_trans_par_1]
            """

            [A_fd_0,A_fd_1,fi_fd_0,fi_fd_1,A_til_0,A_til_1,fi_til_0,fi_til_1,fi_til_fd_0,\
            fi_til_fd_1,r_pars_0,r_pars_1,\
            r_pars_fd_0,r_pars_fd_1,H_pars_0,H_pars_1,H_pars_fd_0,H_pars_fd_1,\
            sample_pars,g_den_par_0,g_den_par_1,atdp_0,atdp_1,\
            ind_prop_trans_par_0,ind_prop_trans_par_1]=update_pars(pars_0[n],pars_1[n],fd_rate,levels=2)
            
            n+=1
            #print("The new parameters are: ",A,fi,g_den_par)
    #mcmc_mean[i]=np.mean(ch_paths,axis=0)

    return ch_paths_0,ch_paths_1,pars_0 ,pars_1, Grads_test_0,Grads_test_1

#%%

#"""

def Unbiased(t0,x0,T,b,A_in,A_fd_in,Sig,fi_in,fi_fd_in,b_til,A_til_in,Sig_til,fi_til_in,\
    fi_til_fd_in,r,r_pars,r_pars_fd,H,H_pars,H_pars_fd,sing_sample_funct, sing_sample_pars,max_sample_funct,sample_pars,\
    obs,log_g_den,g_den_par_in, aux_trans_den,atdp,\
    Grad_log_aux_trans,prop_trans_den,ind_prop_trans_par, Grad_log_G,update_pars,resamp_coef, d,N,seed,fd_rate,\
    mcmc_links,gamma, alpha, CL,CL0,CP,CP0,s0,pmax,l0,lmax,beta_l, beta_p,\
    crossed=False):

    # new parameters:
    # l0: Base level of the grid.
    # lmax: Maximum level of the grid.
    # pmax: Maximum level of SGD steps.
    # beta: coupling rate of the time discretization
    # CL,CL0,CP,CP0: parameters of the coupling.
    # s0: number of SGD steps at the base level.¿
    np.random.seed(seed+1)
    # cumulative for the time discretization
    eLes=np.arange(l0,lmax+1)
    beta=beta_l
    q=4
    P0=(1+CL*np.sum((q+eLes[1:]-l0)*np.log(q+eLes[1:]-l0)**2/2**(beta*eLes[1:]))/\
    (CL0*(q+1)*np.log(q+1)**2))**(-1)
    
    l_cumu=P_l_cumu_gen(P0,lmax-l0+1,beta,l0)
    l=eLes[sampling(l_cumu)]
    # cumulative for the number of SGD steps
    beta=beta_p
    ePes=np.arange(0,pmax+1)
    eSes=s0*2**ePes
    P0=(1+CP*np.sum((ePes[1:]+q)*np.log(ePes[1:]+q)**2/eSes[1:]**(beta))\
    /(CP0*(q+1)*np.log(1+q)**2))**(-1)
    
    p_cumu=P_p_cumu_gen(P0,pmax,beta,s0)
    p=sampling(p_cumu)
    #print("P0p, p_cumu, p",P0,p_cumu,p)
    #print(CP0/P0,CP/(p_cumu[1]-p_cumu[0])) 
    pars=np.zeros((2,2,3))
    #print("l and p are: ",l,p)
    #print("The parameters are: ",l,p)
    # The first dimensions of pars are for the two levels of time discretization. 
    
    if l==l0:
        SGD_steps=eSes[p]
        """
        SGD_bridge(t0,x0,T,b,A_0,A_fd_0,Sig,fi_0,fi_fd_0,b_til,A_til_0,Sig_til,fi_til_0,\
        fi_til_fd_0,r,r_pars,r_pars_fd,H,H_pars,H_pars_fd,sample_funct,sample_pars,\
        obs,log_g_den,g_den_par_0, aux_trans_den,atdp,\
        Grad_log_aux_trans,prop_trans_den, Grad_log_G,resamp_coef, l, d,N,seed,fd_rate,\
        mcmc_links,SGD_steps,gamma, alpha, \
        crossed=False)
        """


        parameters=SGD_bridge(t0,x0,T,b,A_in,A_fd_in,Sig,fi_in,fi_fd_in,b_til,A_til_in,Sig_til,fi_til_in,\
        fi_til_fd_in,r,r_pars,r_pars_fd,H,H_pars,H_pars_fd,sing_sample_funct,sing_sample_pars,\
        obs,log_g_den,g_den_par_in, aux_trans_den,atdp,\
        Grad_log_aux_trans,prop_trans_den, Grad_log_G,resamp_coef, l, d,N,seed,fd_rate,\
        mcmc_links,SGD_steps,gamma, alpha,update_pars)[1]
        pars[1,1]=parameters[eSes[p]]
        if p!=0:
            pars[1,0]=parameters[eSes[p-1]]
    else:
        SGD_steps=eSes[p]
        parameters=C_SGD_bridge(t0,x0,T,b,A_in,A_fd_in,Sig,fi_in,fi_fd_in,b_til,A_til_in,Sig_til,fi_til_in,\
        fi_til_fd_in,r,r_pars,r_pars_fd,H,H_pars,H_pars_fd,max_sample_funct,sample_pars,\
        obs,log_g_den,g_den_par_in, aux_trans_den,atdp,\
        Grad_log_aux_trans,prop_trans_den,ind_prop_trans_par,\
        Grad_log_G,update_pars,resamp_coef, l, d,N,seed,fd_rate/2**(l-l0-1),\
        mcmc_links,SGD_steps,gamma, alpha)
        pars[1,1]=parameters[3][eSes[p]]
        pars[0,1]=parameters[2][eSes[p]]
        if p!=0:
            pars[1,0]=parameters[3][eSes[p-1]]
            pars[0,0]=parameters[2][eSes[p-1]]
    
    return pars,(l,p)


def Gen_Unbiased(sing_in_dist,sing_in_dist_pars,in_dist,in_dist_pars,Grad_log_in_dist,b,A,Sig,fi,b_til,Sig_til,\
    r,H,update_func,  sing_sample_funct, sing_sample_pars,max_sample_funct,sample_pars,\
    obs,obs_times,log_g_den,g_den_par, aux_trans_den,\
    Grad_log_aux_trans,prop_trans_den,ind_prop_trans_par, Grad_log_G,resamp_coef, d,N,seed,fd_rate,\
    mcmc_links,gamma,gammas, alpha,ast,update_pars, CL,CL0,CP,CP0,s0,pmax,l0,lmax,beta_l, beta_p,\
    crossed=False):

    """
    Gen_C_SGD_bridge(in_dist,in_dist_pars,Grad_log_in_dist,b,A,Sig,fi,b_til,Sig_til,\
        r,H,update_func,max_sample_funct,sample_pars,\
        obs,obs_times,log_g_den,g_den_par, aux_trans_den,\
        Grad_log_aux_trans,prop_trans_den,ind_prop_trans_par, Grad_log_G,resamp_coef, l, d,N,seed,fd_rate,\
        mcmc_links,SGD_steps,gamma,gammas, alpha, update_pars)   
    
    """

    """
        Gen_SGD_bridge(in_dist,in_dist_pars,Grad_log_in_dist,b,A,Sig,fi,b_til,Sig_til,\
        r,H,update_func,sample_funct,sample_pars,\
        obs,obs_times,log_g_den,g_den_par, aux_trans_den,\
        Grad_log_aux_trans,prop_trans_den, Grad_log_G,resamp_coef, l, d,N,seed,fd_rate,\
        mcmc_links,SGD_steps,gamma, gammas,alpha, update_pars)
        """



    # new parameters:
    # l0: Base level of the grid.
    # lmax: Maximum level of the grid.
    # pmax: Maximum level of SGD steps.
    # beta: coupling rate of the time discretization
    # CL,CL0,CP,CP0: parameters of the coupling.
    # s0: number of SGD steps at the base level.
    np.random.seed(seed+1)
    # cumulative for the time discretization
    eLes=np.arange(l0,lmax+1)
    beta=beta_l
    q=4
    P0=(1+CL*np.sum((q+eLes[1:]-l0)*np.log(q+eLes[1:]-l0)**2/2**(beta*eLes[1:]))/\
    (CL0*(q+1)*np.log(q+1)**2))**(-1)
    
    l_cumu=P_l_cumu_gen(P0,lmax-l0+1,beta,l0)
    l=eLes[sampling(l_cumu)]
    # cumulative for the number of SGD steps
    beta=beta_p
    ePes=np.arange(0,pmax+1)
    eSes=s0*2**ePes
    P0=(1+CP*np.sum((ePes[1:]+q)*np.log(ePes[1:]+q)**2/eSes[1:]**(beta))\
    /(CP0*(q+1)*np.log(1+q)**2))**(-1)
    
    p_cumu=P_p_cumu_gen(P0,pmax,beta,s0)
    p=sampling(p_cumu)
    #print("P0p, p_cumu, p",P0,p_cumu,p)
    #print(CP0/P0,CP/(p_cumu[1]-p_cumu[0])) 
    pars=np.zeros((2,2,3))
    #print("l and p are: ",l,p)
    #print("The parameters are: ",l,p)
    # The first dimensions of pars are for the two levels of time discretization. 
    
    if l==l0:
        SGD_steps=eSes[p]

        """
        Gen_SGD_bridge(in_dist,in_dist_pars,Grad_log_in_dist,b,A,Sig,fi,b_til,Sig_til,\
        r,H,update_func,sample_funct,sample_pars,\
        obs,obs_times,log_g_den,g_den_par, aux_trans_den,\
        Grad_log_aux_trans,prop_trans_den, Grad_log_G,resamp_coef, l, d,N,seed,fd_rate,\
        mcmc_links,SGD_steps,gamma, gammas,alpha, update_pars)
        """

        


        parameters=Gen_SGD_bridge(sing_in_dist,sing_in_dist_pars,Grad_log_in_dist,b,A,Sig,fi,b_til,Sig_til,\
        r,H,update_func,sing_sample_funct,sing_sample_pars,\
        obs,obs_times,log_g_den,g_den_par, aux_trans_den,\
        Grad_log_aux_trans,prop_trans_den, Grad_log_G,resamp_coef, l, d,N,seed,fd_rate,\
        mcmc_links,SGD_steps,gamma, gammas,alpha, ast,update_pars)[1]
        pars[1,1]=parameters[eSes[p]]
        if p!=0:
            pars[1,0]=parameters[eSes[p-1]]
    else:
        """
        Gen_C_SGD_bridge(in_dist,in_dist_pars,Grad_log_in_dist,b,A,Sig,fi,b_til,Sig_til,\
        r,H,update_func,max_sample_funct,sample_pars,\
        obs,obs_times,log_g_den,g_den_par, aux_trans_den,\
        Grad_log_aux_trans,prop_trans_den,ind_prop_trans_par, Grad_log_G,resamp_coef, l, d,N,seed,fd_rate,\
        mcmc_links,SGD_steps,gamma,gammas, alpha, update_pars)   
        """

        SGD_steps=eSes[p]
        parameters=Gen_C_SGD_bridge(in_dist,in_dist_pars,Grad_log_in_dist,b,A,Sig,fi,b_til,Sig_til,\
        r,H,update_func,max_sample_funct,sample_pars,\
        obs,obs_times,log_g_den,g_den_par, aux_trans_den,\
        Grad_log_aux_trans,prop_trans_den,ind_prop_trans_par, Grad_log_G,resamp_coef, l, d,N,seed,fd_rate,\
        mcmc_links,SGD_steps,gamma,gammas, alpha, ast,update_pars)   
        pars[1,1]=parameters[3][eSes[p]]
        pars[0,1]=parameters[2][eSes[p]]
        if p!=0:
            pars[1,0]=parameters[3][eSes[p-1]]
            pars[0,0]=parameters[2][eSes[p-1]]
            
    return pars,np.array([l,p])

#%%
# First test of the unbiased function
"""
x0_sca=1.2
x0=x0_sca
l=10
T=5
t0=0
l_d=0
d=2**(l_d)
theta_true=-0.3
sigma_true=1.2
np.random.seed(7)
collection_input=[b_ou_1d,theta_true,Sig_ou_1d,sigma_true]
resamp_coef=1
l_max=10
x_true= gen_gen_data_1d(T,x0_sca,l_max,collection_input)
x_reg=cut(T,l_max,-l_d,x_true)[1:]
times=np.array(range(t0,int(T/d)+1))*d
l_times=np.arange(t0,T,2**(-l))
l_max_times=np.arange(t0,T,2**(-l_max))
sd_true=0.55
np.random.seed(3)
d_times=np.array(range(t0+d,int(T/d)+1))*d
obs=gen_obs(x_reg,g_normal_1d,sd_true)
resamp_coef=1
N=50
start=time.time()
mcmc_links=100
SGD_steps=4
B=mcmc_links*SGD_steps
fd=1e-4
theta_in=-0.67
theta_in_fd=theta_in+fd
sigma_in=0.7
sigma_in_fd=sigma_in+fd
sigma_in_aux=sigma_in
theta_in_aux=theta_in+0.2
sigma_in_aux_fd=sigma_in_aux+fd
sd_in=0.7
samples=100
gamma=0.5
alpha=0.5
seed=2393
x0=x0_sca+np.zeros(N)
"""
#%%
"""
(t0,x0,T,b,A_in,A_fd_in,Sig,fi_in,fi_fd_in,b_til,A_til_in,Sig_til,fi_til_in,\
    fi_til_fd_in,r,r_pars,r_pars_fd,H,H_pars,H_pars_fd,sing_sample_funct, sing_sample_pars,max_sample_funct,sample_pars,\
    obs,log_g_den,g_den_par_in, aux_trans_den,atdp,\
    Grad_log_aux_trans,prop_trans_den,ind_prop_trans_par, Grad_log_G,update_pars,resamp_coef, l, d,N,seed,fd_rate,\
    mcmc_links,SGD_steps,gamma, alpha, CL,CL0,CP,CP0,s0,pmax,l0,lmax,beta,\
    crossed=False)  
"""
"""
CL=0.07550483659872
CL0=1.57486813
CP=2.7747103882175255
CP0=8.02153147
beta_l=1
beta_p=1
s0=4
pmax=3
l0=0
lmax=6
pars=Unbiased(t0,x0,T,b_ou_1d,theta_in,theta_in_fd,Sig_ou_1d,sigma_in,sigma_in_fd,\
            b_ou_aux,theta_in_aux,Sig_ou_aux,sigma_in_aux,sigma_in_aux_fd,\
            r_quasi_normal_1d,[ou_sd,[theta_in_aux,sigma_in_aux]],[ou_sd,[theta_in_aux,sigma_in_aux_fd]],\
            H_quasi_normal,\
            [ou_sd,[theta_in_aux,sigma_in_aux],theta_in_aux],[ou_sd,[theta_in_aux,sigma_in_aux_fd],theta_in_aux],\
            sampling_ou,[theta_in_aux,sigma_in_aux],\
            rej_max_coup_ou, [theta_in_aux,sigma_in_aux,theta_in_aux,sigma_in_aux],obs,log_g_normal_den,sd_in,\
            ou_trans_den,[theta_in_aux,sigma_in_aux],Grad_log_aux_trans_ou_new,\
            ou_trans_den,[theta_in_aux,sigma_in_aux],Grad_log_G_new,update_pars_ou,resamp_coef,d, N,seed,fd,mcmc_links,gamma,\
            alpha,CL,CL0,CP,CP0,s0,pmax,l0,lmax,beta_l,beta_p)
"""
#"""
#%%
#%%
#TEST FOR THE SGD_BRIDGE FUNCTION. 
"""
N=100
x0_sca=1.2
x0=x0_sca+np.zeros(N)
l=8
T=10
t0=0
l_d=0
d=2**(l_d)
theta_true=-0.2
sigma_true=1.1
#sigma_aux=0.2
#print(theta)

np.random.seed(7)
collection_input=[b_ou_1d,theta_true,Sig_ou_1d,sigma_true]
resamp_coef=1
l_max=10    
x_true= gen_gen_data_1d(T,x0_sca,l_max,collection_input)
x_reg=cut(T,l_max,-l_d,x_true)[1:]
times=np.arange(t0,T+1,d)
l_times=np.arange(t0,T,2**(-l))
l_max_times=np.arange(t0,T,2**(-l_max))
#print(times, l_times)

plt.plot(times[1:],x_reg,label="True signal")
plt.plot(l_max_times,x_true[:-1],label="True complete signal")
# x_reg stands for x regular
sd_true=5e-1
np.random.seed(3)
obs=gen_obs(x_reg,g_normal_1d,sd_true)
plt.plot(times[1:], obs,label="Observations")
print(obs,x_reg)
fd_rate=1e-4
[theta_fd,sigma_fd,sd_fd]=np.array([theta_true,sigma_true,sd_true])+fd_rate*np.array([1,1,1])
print(theta_fd,sigma_fd,sd_fd)
"""
#%%
"""
start=time.time()
mcmc_links=10
SGD_steps=2
B=mcmc_links*SGD_steps
gamma=0.1
alpha=0.25
seed=1
#mcmc_mean=np.zeros((samples,2,int(T/d))) # This varible was originally designed 
# to store the mean of both processes, the one with multinomial sampling and the one with
# backward sampling.
resamp_coef=1
pars=np.zeros((SGD_steps+1,3))
theta_0=0.2
sigma_0=1.6
theta_0_fd=theta_0+fd_rate
sigma_0_fd=sigma_0+fd_rate
theta_0_aux=theta_0+0.2
sigma_0_aux=sigma_0
sigma_0_aux_fd=sigma_0_aux+fd_rate
sd_0=sd_true
[ch_paths,pars]=SGD_bridge(t0,x0,T,b_ou_1d,theta_0,theta_0_fd,Sig_ou_1d,sigma_0,sigma_0_fd,b_ou_aux,theta_0_aux,Sig_ou_aux,sigma_0_aux,\
sigma_0_aux_fd,r_quasi_normal_1d,[ou_sd,[theta_0_aux,sigma_0_aux]],[ou_sd,[theta_0_aux,sigma_0_aux_fd]],\
H_quasi_normal,[ou_sd,[theta_0_aux,sigma_0_aux],theta_0_aux],[ou_sd,[theta_0_aux,sigma_0_aux_fd],theta_0_aux],\
sampling_ou,[theta_0_aux,sigma_0_aux],\
obs,log_g_normal_den,sd_0, ou_trans_den,[theta_0_aux,sigma_0_aux],\
Grad_log_aux_trans_ou_new,ou_trans_den, Grad_log_G_new,resamp_coef, l, d,N,seed,fd_rate,\
mcmc_links,SGD_steps,gamma, alpha, \
crossed=False)
"""
"""
(t0,x0,T,b,A_0,A_fd_0,Sig,fi_0,fi_fd_0,b_til,A_til_0,Sig_til,fi_til_0,\
    fi_til_fd_0,r,r_pars,r_pars_fd,H,H_pars,H_pars_fd,sample_funct,sample_pars,\
    obs,log_g_den,g_den_par_0, aux_trans_den,atdp,\
    Grad_log_aux_trans,prop_trans_den, Grad_log_G,resamp_coef, l, d,N,seed,fd_rate,\
    mcmc_links,SGD_steps,gamma, alpha, \
    crossed=False)
"""
#####################################################################
#%%
"""
Grads=np.zeros((Grid_p,Grid_p,Grid_p,3))
dim=1
dim_o=1
for i in range(len(thetas)):
    theta=thetas[i]
    
    for j in range(len(sigmas)):
        sigma=sigmas[j]
        
        for k in range(len(sds)):
            sd=sds[k]
            print(theta,sigma)
            K=np.array([[np.exp(d*theta)]])
            G=np.array([[sigma*np.sqrt((np.exp(2*d*theta)-1)/(2*theta))]])
            H=np.array([[1]])
            D=np.array([[sd]])
            #print(K,G**2,H,D)
            Grad_S=np.array([[[0]],[[0]],[[1]]],dtype=float)
            Grad_R_sigma_s=(np.exp(2*theta*d)-1)/(2*theta)
            Grad_R_theta=(sigma**2/(2*theta**2))*(1-np.exp(2*theta*d)+2*d*theta*np.exp(2*theta*d))
            Grad_R=np.zeros((3,1,1),dtype=float)
            Grad_R[0,0,0]=Grad_R_theta
            Grad_R[1,0,0]=Grad_R_sigma_s
            Grad_K=np.array([[[d*np.exp(d*theta)]],[[0]],[[0]]],dtype=float)
            #print(Grad_K,Grad_R,Grad_S)
            x_kf,x_kf_smooth,Grad_log_lik=KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
            Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
            Grads[j,i,k]=Grad_log_lik[:,0,0]
"""
#%%
# for 2d
"""
print("sd is: ",sds[int(Grid_p/2)]-1)
thetas_Grid,sigmas_Grid=np.meshgrid(thetas,sigmas)
plt.quiver(thetas_Grid,sigmas_Grid,Grads[:,:,int(Grid_p/2),0],Grads[:,:,int(Grid_p/2),1])
print("The original parameters are: ",theta_0,sigma_0)
print("The final parameters are: ",-0.4,0.9)
#plt.plot(pars[:,:,0].T,pars[:,:,1].T)
plt.xlabel("Theta")
plt.ylabel("Sigma")
plt.title("SGD")
#plt.xlabel("Theta")
#plt.ylabel("Sigma")
#plt.title("Gradient of the loglikelihood")
plt.show()
"""
#%%
########################################################################
########################################################################
########################################################################
########################################################################
"""
start=time.time()
B=1000
samples=3
seed=0
mcmc_mean=np.zeros((samples,2,int(T/d))) # This varible was originally designed 
# to store the mean of both processes, the one with multinomial sampling and the one with
# backward sampling.
resamp_coef=1
Grads=np.zeros((samples,B,3))

for i in range(samples):
    np.random.seed(i)
    [log_weights,int_Gs,x_pr]=PF_bridge(t0,x0,T,b_ou_1d,theta,Sig_ou_1d,sigma,b_ou_aux,theta_aux,Sig_ou_aux,sigma_aux,\
    r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,[ou_sd,[theta_aux,sigma_aux],theta_aux],\
    sampling_ou, [theta_aux,sigma_aux],obs,log_g_normal_den,sd,\
    ou_trans_den,[theta_aux,sigma_aux],ou_trans_den,\
    resamp_coef,l,d, N,seed)
    #x_mean=np.sum(pff.norm_logweights(log_weights,ax=1)*x_pr,axis=-1)
    weights=pff.norm_logweights(log_weights[-1])
    #print(weights.shape)
    index=np.random.choice(np.array(range(N)),p=weights)
    cond_path=x_pr[:,index]
    cond_log_weights=log_weights[:,index]
    seeds_cond=np.zeros((int(T/d),2),dtype=int)
    seeds_cond[:,0]=seed+np.array(range(int(T/d)))*int(int(2**l*d-1))
    seeds_cond[:,1]=index*np.ones(int(T/d))
    cond_int_G=int_Gs[:,index]
    ch_paths=np.zeros((B,int(T/d)))
    ch_weights=np.zeros((B,int(T/d)))
    
    #cov
    print("Sample iterations is: ",i)
   
    #print("The condtional path is:",cond_path)    
    for b in range(B):
        # the varaible int_Gs is meant to have the record of int_G of the 
        # backward sampled path.
        
        print("mcmc iteration is:", b)
        seed+=int((int(T/d))*int(int(2**l*d-1)))
        np.random.seed(b)
        [log_weights,x_pr,cond_log_weights,cond_int_G,cond_path,seeds_cond]=\
        Cond_PF_bridge_back_samp(cond_log_weights,cond_int_G,cond_path,seeds_cond,t0,x0,\
        T,b_ou_1d,theta,Sig_ou_1d,sigma,b_ou_aux,theta_aux,Sig_ou_aux,sigma_aux,\
        r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,\
        [ou_sd,[theta_aux,sigma_aux],theta_aux],\
        sampling_ou, [theta_aux,sigma_aux],obs,log_g_normal_den,sd,\
        ou_trans_den,[theta_aux,sigma_aux],ou_trans_den,\
        resamp_coef,l,d, N,seed,crossed=False)
        #print("The other condtional path is:",cond_path)
        ch_paths[b]=cond_path
        ch_weights[b]=cond_log_weights
        Grad_log_Gs=0
        Grad_log_aux_trans=np.zeros(2)
        Grad_theta_intGs=0
        Grad_sigma_intGs=0

        for j in range(int(T/d)):
            # here we compute the gradients of the observation likelihood
            # and the transition density of the auxiliary process.
            # Grad_G(x,y,pars)
            t_in=t0+j*d
            t_fin=t0+(j+1)*d    

            Grad_log_Gs+=Grad_log_G(cond_path[j],obs[j],sd**2)
            if j==0:
                x_in=x0[0]
            else:
                x_in=cond_path[j-1]
            Grad_log_aux_trans+=\
            Grad_log_aux_trans_ou(t_in,x_in,t_fin,cond_path[j],np.array([theta_aux,sigma**2]))
            
            #intG_mod_theta_0=Bridge_1d(t_in,x_in,t_fin,cond_path[j],b_ou_1d,theta,Sig_ou_1d,sigma,b_ou_aux,\
            #theta_aux,Sig_ou_aux,sigma_aux,r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,\
            #[ou_sd,[theta_aux,sigma_aux],theta_aux],l,d,1,seeds_cond[j,0],\
            #j=seeds_cond[j,1],fd=True, N_pf=N)
            


            #intG_mod_theta=Bridge_1d(t_in,x_in,t_fin,cond_path[j],b_ou_1d,theta_fd,Sig_ou_1d,sigma,b_ou_aux,\
            theta_aux,Sig_ou_aux,sigma_aux,r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,\
            [ou_sd,[theta_aux,sigma_aux],theta_aux],l,d,1,seeds_cond[j,0],\
            j=seeds_cond[j,1],fd=True, N_pf=N)

            
            #intG_mod_sigma_s_0=Bridge_1d(t_in,x_in,t_fin,cond_path[j],b_ou_1d,theta,Sig_ou_1d,sigma,b_ou_aux,\
            #theta_aux,Sig_ou_aux,sigma_aux,r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,\
            #[ou_sd,[theta_aux,sigma_aux],theta_aux],l,d,1,seeds_cond[j,0],\
            #j=seeds_cond[j,1],fd=True, N_pf=N)
            
            sigma_aux=sigma_fd
            intG_mod_sigma_s=Bridge_1d(t_in,x_in,t_fin,cond_path[j],b_ou_1d,theta,Sig_ou_1d,sigma_fd,b_ou_aux,\
            theta_aux,Sig_ou_aux,sigma_aux,r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,\
            [ou_sd,[theta_aux,sigma_aux],theta_aux],l,d,1,seeds_cond[j,0],\
            j=seeds_cond[j,1],fd=True, N_pf=N)
            sigma_aux=sigma
            Grad_theta_intGs-=((cond_int_G[j]-intG_mod_theta)/fd_rate)
            #Grad_theta_intGs+=((intG_mod_theta_0-intG_mod_theta)/fd_rate)
            Grad_sigma_intGs-=((cond_int_G[j]-intG_mod_sigma_s)/fd_rate)
            #Grad_sigma_intGs+=((intG_mod_sigma_s_0-intG_mod_sigma_s)/fd_rate)
        #print("Grad_theta_intGs has shape: ",Grad_theta_intGs.shape)    
        #print("Grad_theta_intGs has shape: ",Grad_theta_intGs.shape)
        Grads[i,b,0]=Grad_theta_intGs[0]#+Grad_log_aux_trans[0]
        Grads[i,b,1]=Grad_sigma_intGs[0]+Grad_log_aux_trans[1]*2*sigma
        Grads[i,b,2]=Grad_log_Gs

            #Bridge_1d(t0,x0,T,x_p,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,l,d,N,seed\
            #,crossed=False,backward=False,j=False,fd=False,N_pf=False)
        #print("seed conditionals are:",seeds_cond)
        #Bridge_1d(t0,x0,T,x_p,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,l,d,N,seed\
        #,crossed=False,backward=False,j=False,fd=False,N_pf=False)
    mcmc_mean[i,0]=np.mean(ch_paths,axis=0)
end=time.time()

"""
#%%
#%%
#  TEST FOR THE DIFFERENT GRADIENT FUNCTION

#%%
# RESULTS: Whenever we set the parameters of the PF with the same parameters of the realization
# the gradient approximates 0. It is not exactly 0 since the likelihood we have is not necessarely 
# maximized at the true parameters.
#%%
"""
dim=1
dim_o=1
K=np.array([[np.exp(d*theta)]])
G=np.array([[sigma*np.sqrt((np.exp(2*d*theta)-1)/(2*theta))]])
H=np.array([[1]])
D=np.array([[sd]])
x_kf,x_kf_smooth=KF(x0[0],dim,dim_o,K,G,H,D,obs)
times=np.arange(t0,T+1,d)
l_times=np.arange(t0,T,2**(-l))
l_max_times=np.arange(t0,T,2**(-l_max))
#print(times, l_times)
x_mean=np.mean(mcmc_mean[:,0],axis=0)
print(x_mean.shape)
plt.plot(times[1:],x_reg,label="True signal")
plt.plot(l_max_times,x_true[:-1],label="True complete signal")
plt.plot(times,x_kf_smooth[:,0],label="KF smooth")
plt.plot(times,x_kf[:,0],label="KF")
plt.plot(times[1:], obs,label="Observations")
plt.plot(times[1:], x_mean,label="PGibbs")
plt.legend()
"""
#%%

# IN THE FOLLOWING WE TEST THE GRADIENT OF THE LOGLIKELIHOOD FOR DIFFERENT PARAMETERS 
# TO THE ONES OF THE REALIZATION.
"""
N=100
x0_sca=1.2
x0=x0_sca+np.zeros(N)
l=10
T=100
t0=0
l_d=0
d=2**(l_d)
theta_true=-0.9
sigma_true=0.5
#sigma_aux=0.2
#print(theta)
np.random.seed(7)
collection_input=[b_ou_1d,theta_true,Sig_ou_1d,sigma_true]
resamp_coef=1
l_max=10
x_true= gen_gen_data_1d(T,x0_sca,l_max,collection_input)
x_reg=cut(T,l_max,-l_d,x_true)[1:]
times=np.arange(t0,T+1,d)
l_times=np.arange(t0,T,2**(-l))
l_max_times=np.arange(t0,T,2**(-l_max))
#print(times, l_times)
plt.plot(times[1:],x_reg,label="True signal")
plt.plot(l_max_times,x_true[:-1],label="True complete signal")
# x_reg stands for x regular
sd_true=1e-1
np.random.seed(3)
obs=gen_obs(x_reg,g_normal_1d,sd_true)

plt.plot(times[1:], obs,label="Observations")
print(obs,x_reg)
"""
#%%

"""
theta=theta_true
sigma=sigma_true
theta_aux=theta+0.2
sigma_aux=sigma
print(theta_aux,sigma_aux)
sd=sd_true
fd_rate=1e-5
[theta_fd,sigma_fd,sd_fd]=np.array([theta,sigma,sd])+fd_rate*np.array([1,1,1])
print(theta_fd,sigma_fd,sd_fd)
"""
#%%
"""
start=time.time()
B=10
samples=3
seed=0
mcmc_mean=np.zeros((samples,2,int(T/d))) # This varible was originally designed 
# to store the mean of both processes, the one with multinomial sampling and the one with
# backward sampling.
resamp_coef=1
Grads=np.zeros((samples,B,3))

for i in range(samples):
    np.random.seed(i)
    [log_weights,int_Gs,x_pr]=PF_bridge(t0,x0,T,b_ou_1d,theta,Sig_ou_1d,sigma,b_ou_aux,theta_aux,Sig_ou_aux,sigma_aux,\
    r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,[ou_sd,[theta_aux,sigma_aux],theta_aux],\
    sampling_ou, [theta_aux,sigma_aux],obs,log_g_normal_den,sd,\
    ou_trans_den,[theta_aux,sigma_aux],ou_trans_den,\
    resamp_coef,l,d, N,seed)
    #x_mean=np.sum(pff.norm_logweights(log_weights,ax=1)*x_pr,axis=-1)
    weights=pff.norm_logweights(log_weights[-1])
    #print(weights.shape)
    index=np.random.choice(np.array(range(N)),p=weights)
    cond_path=x_pr[:,index]
    cond_log_weights=log_weights[:,index]
    seeds_cond=np.zeros((int(T/d),2),dtype=int)
    seeds_cond[:,0]=seed+np.array(range(int(T/d)))*int(int(2**l*d-1))
    seeds_cond[:,1]=index*np.ones(int(T/d))
    cond_int_G=int_Gs[:,index]
    ch_paths=np.zeros((B,int(T/d)))
    ch_weights=np.zeros((B,int(T/d)))
    
    #cov
    print("Sample iterations is: ",i)
   
    #print("The condtional path is:",cond_path)    
    for b in range(B):
        # the varaible int_Gs is meant to have the record of int_G of the 
        # backward sampled path.


        
        print("mcmc iteration is:", b)
        seed+=int((int(T/d))*int(int(2**l*d-1)))
        np.random.seed(b)
        [log_weights,x_pr,cond_log_weights,cond_int_G,cond_path,seeds_cond]=\
        Cond_PF_bridge_back_samp(cond_log_weights,cond_int_G,cond_path,seeds_cond,t0,x0,\
        T,b_ou_1d,theta,Sig_ou_1d,sigma,b_ou_aux,theta_aux,Sig_ou_aux,sigma_aux,\
        r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,\
        [ou_sd,[theta_aux,sigma_aux],theta_aux],\
        sampling_ou, [theta_aux,sigma_aux],obs,log_g_normal_den,sd,\
        ou_trans_den,[theta_aux,sigma_aux],ou_trans_den,\
        resamp_coef,l,d, N,seed,crossed=False)
        #print("The other condtional path is:",cond_path)
        ch_paths[b]=cond_path
        ch_weights[b]=cond_log_weights
        Grad_log_Gs=0
        Grad_log_aux_trans=np.zeros(2)
        Grad_theta_intGs=0
        Grad_sigma_intGs=0

        for j in range(int(T/d)):
            # here we compute the gradients of the observation likelihood
            # and the transition density of the auxiliary process.
            # Grad_G(x,y,pars)
            t_in=t0+j*d
            t_fin=t0+(j+1)*d    

            Grad_log_Gs+=Grad_log_G(cond_path[j],obs[j],sd**2)
            if j==0:
                x_in=x0[0]
            else:
                x_in=cond_path[j-1]
            Grad_log_aux_trans+=\
            Grad_log_aux_trans_ou(t_in,x_in,t_fin,cond_path[j],np.array([theta_aux,sigma**2]))
            
            #intG_mod_theta_0=Bridge_1d(t_in,x_in,t_fin,cond_path[j],b_ou_1d,theta,Sig_ou_1d,sigma,b_ou_aux,\
            #theta_aux,Sig_ou_aux,sigma_aux,r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,\
            #[ou_sd,[theta_aux,sigma_aux],theta_aux],l,d,1,seeds_cond[j,0],\
            #j=seeds_cond[j,1],fd=True, N_pf=N)
            


            intG_mod_theta=Bridge_1d(t_in,x_in,t_fin,cond_path[j],b_ou_1d,theta_fd,Sig_ou_1d,sigma,b_ou_aux,\
            theta_aux,Sig_ou_aux,sigma_aux,r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,\
            [ou_sd,[theta_aux,sigma_aux],theta_aux],l,d,1,seeds_cond[j,0],\
            j=seeds_cond[j,1],fd=True, N_pf=N)

            
            #intG_mod_sigma_s_0=Bridge_1d(t_in,x_in,t_fin,cond_path[j],b_ou_1d,theta,Sig_ou_1d,sigma,b_ou_aux,\
            #theta_aux,Sig_ou_aux,sigma_aux,r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,\
            #[ou_sd,[theta_aux,sigma_aux],theta_aux],l,d,1,seeds_cond[j,0],\
            #j=seeds_cond[j,1],fd=True, N_pf=N)
            
            sigma_aux=sigma_fd
            intG_mod_sigma_s=Bridge_1d(t_in,x_in,t_fin,cond_path[j],b_ou_1d,theta,Sig_ou_1d,sigma_fd,b_ou_aux,\
            theta_aux,Sig_ou_aux,sigma_aux,r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,\
            [ou_sd,[theta_aux,sigma_aux],theta_aux],l,d,1,seeds_cond[j,0],\
            j=seeds_cond[j,1],fd=True, N_pf=N)
            sigma_aux=sigma
            Grad_theta_intGs-=((cond_int_G[j]-intG_mod_theta)/fd_rate)
            #Grad_theta_intGs+=((intG_mod_theta_0-intG_mod_theta)/fd_rate)
            Grad_sigma_intGs-=((cond_int_G[j]-intG_mod_sigma_s)/fd_rate)
            #Grad_sigma_intGs+=((intG_mod_sigma_s_0-intG_mod_sigma_s)/fd_rate)
        #print("Grad_theta_intGs has shape: ",Grad_theta_intGs.shape)    
        #print("Grad_theta_intGs has shape: ",Grad_theta_intGs.shape)
        Grads[i,b,0]=Grad_theta_intGs[0]#+Grad_log_aux_trans[0]
        Grads[i,b,1]=Grad_sigma_intGs[0]+Grad_log_aux_trans[1]*2*sigma
        Grads[i,b,2]=Grad_log_Gs

            #Bridge_1d(t0,x0,T,x_p,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,l,d,N,seed\
            #,crossed=False,backward=False,j=False,fd=False,N_pf=False)
        #print("seed conditionals are:",seeds_cond)
        #Bridge_1d(t0,x0,T,x_p,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,l,d,N,seed\
        #,crossed=False,backward=False,j=False,fd=False,N_pf=False)
    mcmc_mean[i,0]=np.mean(ch_paths,axis=0)
end=time.time()"""
#%%
# In this run of the algorithm I used some parameters that decrease 
# the variance of the gradient estimator as to see if the estimation converges 
# to the actual value of the gradient. 
"""
print("The time it took was: ",end-start)
print(np.mean(Grads,axis=1))
print(np.sqrt(np.var(Grads,axis=1)/B))
"""
#%%
# neest test
#KF_Grad(xin,dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S,theta, sigma, sigma_obs):
"""
dim=1
dim_o=1
K=np.array([[np.exp(d*theta)]])
G=np.array([[sigma*np.sqrt((np.exp(2*d*theta)-1)/(2*theta))]])
H=np.array([[1]])
D=np.array([[sd]])
#print(K,G**2,H,D)
Grad_S=np.array([[[0]],[[0]],[[1]]],dtype=float)
Grad_R_sigma_s=(np.exp(2*theta*d)-1)/(2*theta)
Grad_R_theta=(sigma**2/(2*theta**2))*(1-np.exp(2*theta*d)+2*d*theta*np.exp(2*theta*d))
Grad_R=np.zeros((3,1,1),dtype=float)
Grad_R[0,0,0]=Grad_R_theta
Grad_R[1,0,0]=Grad_R_sigma_s
Grad_K=np.array([[[d*np.exp(d*theta)]],[[0]],[[0]]],dtype=float)
#print(Grad_K,Grad_R,Grad_S)
x_kf,x_kf_smooth,Grad_log_lik=KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
times=np.arange(t0,T+1,d)
l_times=np.arange(t0,T,2**(-l))
l_max_times=np.arange(t0,T,2**(-l_max))
#print(times, l_times)
#x_mean=np.mean(mcmc_mean[:,0],axis=0)
#print(x_mean.shape)
plt.plot(times[1:],x_reg,label="True signal")
plt.plot(l_max_times,x_true[:-1],label="True complete signal")
plt.plot(times,x_kf_smooth[:,0],label="KF smooth")
plt.plot(times,x_kf[:,0],label="KF")
plt.plot(times[1:], obs,label="Observations")
#plt.plot(times[1:], x_mean,label="PGibbs")
plt.legend()
"""
#%%
"""
print(np.mean(Grads,axis=1))
print(1.96*np.sqrt(np.var(Grads,axis=1)/B))
print(Grad_log_lik)
"""
#iii
#%%

# In here I will use the analytical gradient of the loglikelihood 
# in order to undertand the behaviour of the gradient of the loglikelihood
#print(theta_true,sigma_true)

#%%
"""
Grid_p=5
thetas= np.linspace(-1,1,Grid_p)*0.2+theta_true
sigmas= np.linspace(-1,1,Grid_p)*0.5+sigma_true
theta_aux=thetas+0.2
sigma_aux=sigmas
sds=np.linspace(-1,1,Grid_p)*0.5+ sd_true
fd_rate=1e-4
[theta_fd,sigma_fd,sd_fd]=np.array([thetas,sigmas,sds])+fd_rate*(np.zeros((3,Grid_p))+1)
print(thetas,sigmas)
"""
#%%
"""
Grads=np.zeros((Grid_p,Grid_p,Grid_p,3))
dim=1
dim_o=1
for i in range(len(thetas)):
    theta=thetas[i]
    
    for j in range(len(sigmas)):
        sigma=sigmas[j]
        
        for k in range(len(sds)):

            sd=sds[k]
            print(theta,sigma)
            K=np.array([[np.exp(d*theta)]])
            G=np.array([[sigma*np.sqrt((np.exp(2*d*theta)-1)/(2*theta))]])
            H=np.array([[1]])
            D=np.array([[sd]])
            #print(K,G**2,H,D)
            Grad_S=np.array([[[0]],[[0]],[[1]]],dtype=float)
            Grad_R_sigma_s=(np.exp(2*theta*d)-1)/(2*theta)
            Grad_R_theta=(sigma**2/(2*theta**2))*(1-np.exp(2*theta*d)+2*d*theta*np.exp(2*theta*d))
            Grad_R=np.zeros((3,1,1),dtype=float)
            Grad_R[0,0,0]=Grad_R_theta
            Grad_R[1,0,0]=Grad_R_sigma_s
            Grad_K=np.array([[[d*np.exp(d*theta)]],[[0]],[[0]]],dtype=float)
            #print(Grad_K,Grad_R,Grad_S)
            x_kf,x_kf_smooth,Grad_log_lik=KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
            Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
            Grads[j,i,k]=Grad_log_lik[:,0,0]
"""
#%%
# for 2d
"""
thetas_Grid,sigmas_Grid=np.meshgrid(thetas,sigmas)
plt.quiver(thetas_Grid,sigmas_Grid,Grads[:,:,0],Grads[:,:,1])
plt.xlabel("Theta")
plt.ylabel("Sigma")
plt.title("Gradient of the loglikelihood")
"""
#%%
# for 3d
"""
ax = plt.figure().add_subplot(projection='3d')




# Make the grid
thetas_Grid, sigmas_Grid, sds_Grid = np.meshgrid(thetas,sigmas,sds)
ax.quiver(thetas_Grid, sigmas_Grid, sds_Grid, Grads[:,:,:,0], Grads[:,:,:,1], Grads[:,:,:,2],\
length=0.05,normalize=True)
plt.show()
"""
#%%
#%%
"""
print(Grads[-1,:,1])
times=np.arange(t0,T+1,d)
l_times=np.arange(t0,T,2**(-l))
l_max_times=np.arange(t0,T,2**(-l_max))
"""
#%%





#print(times, l_times)
#x_mean=np.mean(mcmc_mean[:,0],axis=0)
#print(x_mean.shape)
#plt.plot(times[1:],x_reg,label="True signal")
#plt.plot(l_max_times,x_true[:-1],label="True complete signal")
#plt.plot(times,x_kf_smooth[:,0],label="KF smooth")
#plt.plot(times,x_kf[:,0],label="KF")
#plt.plot(times[1:], obs,label="Observations")
#plt.plot(times[1:], x_mean,label="PGibbs")
#plt.legend()

#print(Grad_log_lik)






#%%
"""
A=np.array([K[0,0]])
R=np.array([G[0,0]**2])
S=np.array(sd**2)
DA=np.array([Grad_K[0,0,0],0,0])
DR=Grad_R[:,0,0]
DS=np.array([0,0,1],dtype=float)
#print("A,R,S,DA,DR,DS are:",A,R,S,DA,DR,DS)
Grad_log=-(1/2)*(-(1/(R+S)**2)*(obs[0]-A*x0_sca)**2*(DR+DS)+2*(obs[0]-A*x0_sca)*(-DA*x0_sca)/(R+S)+(DR+DS)/(R+S))
print(Grad_log)
"""

#%%
# the following is the definition of the function that computes the gradient. 

"""
def Grad_Gibbs_back(cond_log_weights,cond_int_G,cond_path,seeds_cond,t0,x0,\
    T,b_ou_1d,theta,Sig_ou_1d,sigma,b_ou_aux,theta_aux,Sig_ou_aux,sigma_aux,\
    r_quasi_normal_1d,[ou_sd,[theta_aux,sigma_aux]],H_quasi_normal,\
    [ou_sd,[theta_aux,sigma_aux],theta_aux],\
    sampling_ou, [theta_aux,sigma_aux],obs,log_g_normal_den,sd,\
    ou_trans_den,[theta_aux,sigma_aux],ou_trans_den,\
    resamp_coef,l,d, N,seed,crossed=False):

    # the varaible int_Gs is meant to have the record of int_G of the"""





##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
#%%

# Smoothing function for the 1D case.

# The following function is going to be used for the smoothing of the 1D case.
# It is specific for additive functions so the computation can be made online.
# Although it will be used to the computation of the likelihood, this version 
# is more general. 

def Sm_bridge(t0,x0,T,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,\
    sample_funct,sample_pars,obs,log_g_den,g_den_par, aux_trans_den,atdp,\
    prop_trans_den, resamp_coef, l, d,N,crossed=False):
    #(T,xin,b_ou,A,Sig_ou,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par):
    # ARGUMENTS: the argument of the Kenel x0 rank 1 dims (dim) 
    # the drift and diffusion are b and Sig, respectively, and they take
    # x(either a (N) dimensional or (N,N) dimensional array) and A as arguments for the drift and x and fi for the diffusion.
    # the level of discretization l, the distance of resampling, the number of
    # particles N.
    # Grad_b is a function that takes (x,A) as argument and computes the gradnient of b wrt the 
    # parameters A, and evaluates it a (x,A).
    # b_til,A_til,Sig_til,fi_til, are the analogous functions for the auxiliar process.
    # a difference is that their arguments are (t,x) for the drift and (t,x,fi_til) for the diffusion.
    # r is the function that computes the gradient of the log of the kernel of the auxiliar process
    # and it takes (t,x,T,x_pr,r_pars) as arguments.
    # H is the function that computes the Hessian of the log of the kernel of the auxiliar process
    # and it takes (t,x,T,x_pr,H_pars) as arguments.
    # crossed is the boolean that indicates if we need the computations for the crossed terms that 
    # are needed for the smoother.


    log_weights=np.zeros((int(T/d),N,N))
    x_pr=np.zeros((int(T/d),N))
    #xs=np.zeros((int(T*2**l*d),N))
    int_Gs=np.zeros((int(T/d),N,N))                      
    x_new=x0
    ind_resamp=np.zeros(int(T/d),N)+np.arange(N)

    for i in range(int(T/d)):
        tf=t0+(i+1)*d
        ti=t0+(i)*d
        x_pr[i]=sample_funct(x_new,N,d,sample_pars)
        # what parameters do we need in order to make the auxiliar density general?
        # x_new,  d, t, x_pr, tf.
        # aux_trans_den(t0,x0,T,x_pr,atdp)
        # atdp stands for auxiliar transition density parameters. 
        int_G=Bridge_cheap(ti,x_new,tf,x_pr[i],b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,\
        r,r_pars,H,H_pars,l,d,N,crossed=True)
        int_Gs[i]=int_G
        #print(xi.shape)
        #print(yi,obti-i*d,)
        #print(x_new,xi)
        #print(xi)
        #Things that could be wrong
        # observations, x_new, weights
        #observations seem to be fine
        #print("other parameteres are:",ti,x_new,tf,x_pr[i] )
        #print("atdp is ", atdp)
        #print("object is: ", aux_trans_den(ti,x_new,tf,x_pr[i],atdp))

        log_weights[i]=log_weights[i]+int_G+log_g_den(obs[i],x_pr,g_den_par)\
        +np.log(aux_trans_den(ti,x_new,tf,x_pr[i],atdp))-np.log(prop_trans_den(ti,x_new,tf,x_pr[i],sample_pars))
        weights=pff.norm_logweights(log_weights[i].diagonal())
        #print(yi,weights)
        #seed_val=i
        #print(weights.shape)
        x_last=x_pr[i]
        
        ESS=1/np.sum(weights**2)
        #print(ESS,resamp_coef*N)
        if ESS<resamp_coef*N:
            #print("resampling at time ",i)
        #if True==False:
            #[part0,part1,x0_new,x1_new]=max_coup_sr(w0,w1,N,xi0[-1],xi1[-1],dim)
            #print(x_new.shape)
            ind_resamp[i],x_new=pff.multi_samp(weights,N,x_last,1)
            #print(x_new.shape)
        else:            
            #print("time is",i)
            x_new=x_last
            if i< int(T/d)-1:
                log_weights[i+1]=log_weights[i]
        #print(i)
        
       #x_new=sr(weights,N,x_pf[i],dim)[1]
    #weights=np.reshape(norm_logweights(log_weights,ax=1),(int(T/d),N,1))
    #pf=np.sum(weights*x_pf,axis=1)
    #Filter
    #spots=np.arange(d_steps,2**l*T+1,d_steps,dtype=int)
    #x_pf=x[spots]
    #weights=norm_logweights(log_weights,ax=1)
    #print(x_pf.shape,weights.shape)
    #suma=np.sum(x_pf[:,:,1]*weights,axis=1)
    return [log_weights,x_pr,int_Gs,xs]


#%%
"""np.random.seed(1)
T=10
dim=1
dim_o=dim
xin=np.zeros(dim)+1
l=8
collection_input=[]
I=identity(dim).toarray()
#comp_matrix = ortho_group.rvs(dim)
comp_matrix=np.array([[1]])
inv_mat=la.inv(comp_matrix)
#S=diags(np.random.normal(1,0.1,dim),0).toarray()
S=diags(np.random.normal(0.99,0.1,dim),0).toarray()
#S=np.array([[1.]])
fi=inv_mat@S@comp_matrix
#B=diags(np.random.normal(-1,0.1,dim),0).toarray()
B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
B=inv_mat@B@comp_matrix
#B=np.array([[-1.]])
#print(B)
#print(B)
#B=comp_matrix-comp_matrix.T  +B 
collection_input=[dim, b_ou,B,Sig_ou,fi]
cov=I*1e-1
g_pars=cov
g_par=cov
x_true=gen_gen_data(T,xin,l,collection_input)
"""
#%%

"""x_reg=cut(T,l,0,x_true)[1:]
# x_reg stands for x regular 
obs=gen_obs(x_reg,g_normal,g_pars)
    
times=2**(-l)*np.array(range(int(T*2**l+1)))
times_reg=cut(T,l,0,times)[1:]

plt.plot(times,x_true,label="True signal")
plt.plot(times_reg,obs,label="Observations")
"""
#%%
"""
import numpy as np
B=np.array([0,2,4])
B_2=np.array([4,5,6])
print(B[:,np.newaxis]+B_2[np.newaxis])
A = np.array([[1, 2], [3, 4]])
sig=A
n_pars_s=3
print(A[np.newaxis].shape)

x=np.array([[[1,2],[1,3]],[[10,3],[5,5]]])
sigma_pars=[sig,n_pars_s]
grad_t=Grad_x_Sig_ou(x,sig)
#print(grad_t.shape)
"""

#%%

#%%

def normal_stride(x,stride_par,d):
    # This is the function for the stride distribution of the auxiliar process
    # The normal variable taht this function represent is one dimensional
    # ARGUMENTS:
    # ** x: is a rank 2 array, with rank (N,dim)
    # ** stride_par: is a rank 2 array with rank (dim,dim)
    # ** d: scalar, it is the inteval in which we compute the stride distribution
    # OUTPUTS:
    # ** x_p is a rank 2 array (N,dim)

    x_p=np.random.normal(x,np.sqrt(stride_par[0]**2*(d)))

    return x_p

# TEST FOR normal_stride

#x=np.array([[0],[2],[5]])
#stride_par=np.array([[2]])
#d=3
#print(normal_stride(x,stride_par,d))




#%%
"""
def MSB(T,dim,x_in,x_p,l,N,d,resamp_coef,drift_f,drift_par,drift_n,diff_f,diff_par,\
    diff_n,stride_f,stride_par,\
    stride_n,drift_aux_f,drift_aux_par,drift_aux_n,diff_aux_f,diff_aux_par,diff_aux_n,\
    obs_lik_f,obs_lik_par,obs_lik_n):
    # ! Fill description fo the function once it's complete
    # What do I want from this function? 
    # I want to compute an array of rank 
    # Outline of the function:
    # 1. Define the parameters and the zero arrays
    # 2. 
    
    # We start by finding the stride x_p

    x_p=stride_f(x_in,stride_par,d)
    steps=int((2**(l))*d)
    dt=1./2**l
    # x transition, it is made so we can strore the values of the transition
    x_trans=np.zeros((2,N,N,dim))
    x[0]=x_in
    I=identity(dim).toarray()
    
    t=0
    
    dW=np.random.multivariate_normal(np.zeros(dim),I,N)*np.sqrt(dt) #(dim,N)

    for t in range(steps-1): #we subtract bcs the last step is just the stride
        # Here we just compute Psi, the rest of the terms of Phi don't need to be computed in
        # the iteration.
        
        drift= pff.b_ou(x[i],-theta)+pff.Sig_ou(x[i],sigma)[:,0]**(2)\
        *(x_pr-x[i])/(sigma[0]**2*(T-t))

    #d [0]=dW
    #print("dW_fist is ",dW)
    #print("dw is ",dW)
    # Uncomment the following two lines for the GBM and comment
    # the third line
    Sigma=Sig(x[t],fi)
    #print("Sigma is ",Sigma.shape)
    diff=np.einsum("nd,njd->nj",dW,Sigma)
    Sigma_nr=Sig(x_nr,fi)
    Sigma_inv_nr=np.linalg.inv(Sigma_nr)
    #print(Sigma_inv_nr.shape,(Sig(x[t],fi)).shape)
    diff_2_nr=np.einsum("nd,ndj->nj",dW,Sigma_inv_nr)
    #term0=term0+np.einsum("nj,ni->nji",diff_2_nr,x_nr)
    Gradient_b=Grad_b(x[t],A)
    term0=term0+np.einsum("nj,nij->ni",diff_2_nr,Gradient_b)
    #print("x[t] is",x[t])
    #print("diff shape is ",diff.shape)
    #print("x[t] shape is",x[t].shape)
    #print("A is",A)
    #print("b is", b(x[t],A).shape)
    #Uncomment the following line for the GBM 
    x[t+1]=x[t]+b(x[t],A)*dt+ diff




def M_smooth_W(x0,x_nr,b,A,Sig,fi,Grad_b,b_numb_par,l,d,N,dim):
    # This is the function for the transition Kernel M(x,du): R^{d_x}->P(E_l)
    # This function incorporates modification so it is helpful computing the smoother.
    # ARGUMENTS: the argument of the Kenel x0 \in R^{d_x} (rank 1 dimesion dim=d_x array)
    # the drift and diffusion b, and Sig respectively (rank 1 and 2 numpy arrays of dim=d_x respectively)
    # the level of discretization l, the distance of resampling, the number of
    # particles N, and the dimension of the problem dim=d_x
    # x_nr: the particles not-resampled particles. 
    # OUTCOMES: x, an array of rank 3  2**l*d,N,dim that represents the path simuled
    # along the discretized time for a number of particles N.
    # term0: an array of rank 3  N,dim,dim that represent a term in the computation of Lambda for 
    # the smoother for the first time step
    # term1: an array of rank 3  N,dim,dim that represent a term in the computation of Lambda for
    # the smoother for the rest of the time steps

    steps=int((2**(l))*d)
    dt=1./2**l
    x=np.zeros((steps+1,N,dim))
    x[0]=x0
    I=identity(dim).toarray()
    term=np.zeros((N,b_numb_par))
    term0=np.zeros((N,b_numb_par))
    #dWs=np.zeros((steps,N,dim))
    # Here we compute the first term of the gradient
    t=0
    dW=np.random.multivariate_normal(np.zeros(dim),I,N)*np.sqrt(dt)
    #dWs[0]=dW
    #print("dW_fist is ",dW)
    #print("dw is ",dW)
    # Uncomment the following two lines for the GBM and comment
    # the third line
    Sigma=Sig(x[t],fi)
    #print("Sigma is ",Sigma.shape)
    diff=np.einsum("nd,njd->nj",dW,Sigma)
    Sigma_nr=Sig(x_nr,fi)
    Sigma_inv_nr=np.linalg.inv(Sigma_nr)
    #print(Sigma_inv_nr.shape,(Sig(x[t],fi)).shape)
    diff_2_nr=np.einsum("nd,ndj->nj",dW,Sigma_inv_nr)
    #term0=term0+np.einsum("nj,ni->nji",diff_2_nr,x_nr)
    Gradient_b=Grad_b(x[t],A)
    term0=term0+np.einsum("nj,nij->ni",diff_2_nr,Gradient_b)
    #print("x[t] is",x[t])
    #print("diff shape is ",diff.shape)
    #print("x[t] shape is",x[t].shape)
    #print("A is",A)
    #print("b is", b(x[t],A).shape)
    #Uncomment the following line for the GBM 
    x[t+1]=x[t]+b(x[t],A)*dt+ diff
    #print("the shapes are", b(x[t],A).shape,(Sig(x[t],fi).T).shape,(dW@(Sig(x[t],fi).T)).shape)
    #x[t+1]=x[t]+b(x[t],A)*dt+ dW@(Sig(x[t],fi).T)
    for t in np.array(range(steps))[1:]:
        dW=np.random.multivariate_normal(np.zeros(dim),I,N)*np.sqrt(dt)
        #dWs[t]=dW
        #print("dW_secodn is ",dW)
        # Uncomment the following two lines for the GBM and comment
        # the third line
        Sigma=Sig(x[t],fi)
        Sigma_inv=np.linalg.inv(Sigma)
        diff=np.einsum("nd,njd->nj",dW,Sigma)
        diff_2=np.einsum("nd,ndj->nj",dW,Sigma_inv)
        Gradient_b=Grad_b(x[t],A)
        #if t==1:
            #print("x is ",x[t])
            #print("Gradient b is ",Gradient_b)
        term=term+np.einsum("nj,nij->ni",diff_2,Gradient_b)
        #term=term+np.einsum("nj,ni->nji",diff_2,x[t])
        #print("x[t] is",x[t])
        x[t+1]=x[t]+b(x[t],A)*dt+diff
        # For the OU process comment the previous two lines and uncomment 
        # the following line
        #print(b(x[t],A).shape,Sig(x[t],fi).shape)
        #x[t+1]=x[t]+b(x[t],A)*dt+ dW@(Sig(x[t],fi).T)
        # Uncomment the following lines for the nldt process
        #x[t+1]=x[t]+b(x[t],A)*dt+ dW*(Sig(x[t],fi))
    return x,term0,term #,dWs    
"""
#%%
"""
def SDA(T,dim,x_in,l,N,d,resamp_coef,drift_f,drift_par,drift_n,diff_f,diff_par,diff_n,stride_f,stride_par,\
    stride_n,drift_aux_f,drift_aux_par,drift_aux_n,diff_aux_f,diff_aux_par,diff_aux_n,\
    obs_lik_f,obs_lik_par,obs_lik_n):
    #(T,x_in,b_ou,A,Sig_ou,fi,Grad_b,b_numb_par,obs,obs_time,l,d,N,dim,resamp_coef,g_den,Grad_Log_g_den,g_par,Lambda,Grad_Lambda,Grad_Log_Lambda,\
    #Lamb_par,Lamb_par_numb,step_0,beta):
    # This function computes the diffusion bridge sampling algorithm for a 
    # given diffusion process.
    # ARGUMENTS: 
    # ** x0: initial condition of the process, rank 1 array of dimension dim
    # ** T: final time of the propagation, T>0 and preferably integer
    # ** t0: initial time of the propagation, t0<T and preferably integer
    # ** drift_par: initial parameter of the drift of the process
    # ** diff_par: initial parameter of the diffusion of the process
    # ** stride_pars: Parameters of the stride distribution that samples the process from x0 at time t0 to x' at time T in an importance sampling fashion.
    # ** drift_aux_par: Initial parameter of the drift of the auxiliary process.
    # ** diff_aux_par: Initial parameter of the diffusion of the auxiliary process. (Consider if this parameter is necessary.)
    # ** obs_lik_par: Initial parameter of the observation likelihood function.
    # ** drift_n: Number of parameters of the drift function.
    # ** diff_n: Number of parameters of the diffusion function.
    # ** stride_n: Number of parameters of the stride distribution.
    # ** drift_aux_n: Number of parameters of the drift of the auxiliary process.
    # ** diff_aux_n: Number of parameters of the diffusion of the auxiliary process.
    # ** obs_lik_n: Number of parameters of the observation likelihood function.
    # ** l: Level of discretization of the observations.
    # ** N: Number of samples to be generated.

   

    # DEFINITION OF PARAMETERS AND ZERO ARRAYS
    # log of the particle filter weights at d-spaced 
    # times
    log_weights=np.zeros((int(T/d),N))
    # Particles(not resampled) at d-spaced times                         
    x_pf=np.zeros((int(T/d),N,dim))
    # x_new is the initial condition for each iteration, resampled particles  
    x_new=x_in
    x_last=np.zeros((N,dim))+x_in 

    d_steps=int(d*2**l)
    # Arrays to store the parameters of the gradient descent througt the iterations
    drift_pars=np.zeros((int(T/d)+1,drift_n))
    drift_pars[0]=drift_par
    # the definition of the dims of the array might change depending on the 
    # process considered
    diff_pars=np.zeros((int(T/d)+1,diff_n))
    diff_pars[0]=diff_par
    stride_pars=np.zeros((int(T/d)+1,stride_n))
    stride_pars[0]=stride_par
    # depending on the values of the parameters we might need to change the
    # values of the auxiliar parameters, a clear example is the diffusion 
    # of the auxiliar process and the hidden diffusion, which must be equal at
    # the end of the stride.

    #drift_aux_pars=np.zeros((int(T/d)+1,drift_aux_n))
    #drift_aux_pars[0]=drift_aux_par
    #diff_aux_pars=np.zeros((int(T/d)+1,diff_aux_n))
    #diff_aux_pars[0]=diff_aux_par
    obs_lik_pars=np.zeros((int(T/d)+1,obs_lik_n))
    obs_lik_pars[0]=obs_lik_par
    
    # In the following three lines we define the variables that will store the gradient,
    # these structures might change and overlap depending on the inclusion of the parameters 
    # in the b, lambda, and g_den function.
    F_drift=np.zeros((int(T/d)+1,N,drift_n)) #Function F corresponding to the parameters
    # of drift, the drift of the process
    F_diff=np.zeros((int(T/d)+1,N,diff_n)) # Function F corresponding to the parameters
    # of diffusion, the diffusion of the process
    F_obs_lik=np.zeros((int(T/d)+1,N,obs_lik_n)) # Function F corresponding to the parameters
    # of the observation likelihood function
    # Variable that stores the value of F temporarily
    f_drift=F_drift[0]
    f_diff=F_diff[0]
    f_obs_lik=F_obs_lik[0]
    # Variables that store the gradient of the parameters
    Grads_drift=np.zeros((int(T/d)+1,drift_n))
    Grads_diff=np.zeros((int(T/d)+1,diff_n))
    Grads_obs_lik=np.zeros((int(T/d)+1,obs_lik_n))

    # Array that stores the indices of the resampled particles
    samp_par=np.array(range(N))
    # Variable that stores the log of the previous weights, why is this necessary?
    # I don't remember 
    log_w_prev=log_weights[0]

    # MAIN LOOP
    for i in range(int(T/d)):
        # in the following two lines we get the observations and observation times for the interval
        # (i*d,(i+1)*d)
        drift_par=drift_pars[i]
        diff_par=diff_pars[i]
        stride_par=stride_pars[i]
        #drift_aux_par=drift_aux_pars[i]
        #diff_aux_par=diff_aux_pars[i]
        obs_lik_par=obs_lik_pars[i]
        # WE NEED TO DEFINE THE FUNCTIONS M_SMOOTH HERE

        # we propagate the particles and obtain the term2 function (the term of F that involves the brownian motions)
        xi,term2A,term2B=M_smooth_W(x_new,x_last,b_ou,A,Sig_ou,fi,Grad_b,b_numb_par,l,d,N,dim)
        term1A=-np.sum(Grad_Lambda(x_last[np.newaxis,:,:],Lamb_par,Lamb_par_numb,ax1=-1),axis=0)/2**l
        term1B=-np.sum(Grad_Lambda(xi[1:-1],Lamb_par,Lamb_par_numb,ax1=-1),axis=0)/2**l
        Log_trans=log_trans(x_last,xi[1],b_ou,A,Sig_ou,fi,l)
        #print(Log_trans.shape)
        #print("log_trans is",Log_trans)
        log_w ,log_w_init, term3_Lambd,term3_g_den=Gox_SM_W(yi,obti-i*d,xi,x_last,samp_par, Lambda\
        ,Grad_Log_Lambda,Lamb_par,Lamb_par_numb,l,N,dim,g_den,Grad_Log_g_den,g_par)




"""


def Cox_PE(T,x_in,b_ou,A,Sig_ou,fi,Grad_b,b_numb_par,obs,obs_time,l,d,N,dim,resamp_coef,g_den,Grad_Log_g_den,g_par,Lambda,Grad_Lambda,Grad_Log_Lambda,\
    Lamb_par,Lamb_par_numb,step_0,beta):
    # This version takes Cox_SM_W and introduces the changes in the parameters 
    # so we can adaptatively estimate and improve the estimation
    #Memory friendly version where instead of 3 ouputs we just output the pf.
    # (T,z,lmax,x0,b_ou,A,Sig_ou,fi,ht,H,l,d,N,dim,resamp_coef,para=True):
    # The particle filter function is inspired in 
    # Bain, A., Crisan, D.: Fundamentals of Stochastic Filtering. Springer,
    # New York (2009).
    # ARGUMENTS: T: final time of the propagation, T>0 and preferably integer
    # z: observation process, its a rank two array with discretized observation
    # at the intervals 2^(-lmax)i, i \in {0,1,...,T2^lmax}. with dimension
    # (T2^{lmax}+1) X dim
    # lmax: level of discretization of the observations
    # x0: initial condition of the particle filter, rank 1 array of dimension
    # dim
    # b_out: function that represents the drift of the process (its specifications
    # is already in the document. A is the arguments taht takes
    # Sig_out: function that represents the diffusion of the process (its specifications
    # is already in the document. Its arguments are included in fi.
    # ht: function in the observation process (its specifications
    # is already in the document). Its arguments are included in H.
    # d: time span in which the resampling is computed. d must be a divisor of T.
    # N: number of particles, N \in naturals greater than 1
    # dim: dimension of the problem
    # para: key to wheter compute the paralelization or not.
    # OUTPUTS: x: is the rank 3 array with the resampled particles at times 
    # 2^{-l}*i, i \in {0,1,..., T*2^l}, its dimentions are (2**l*T+1,N,dim)
    # log_weights: logarithm of the weights at times i*d, for i \in {0,1,...,T/d}.
    # it is a rank 2 array with dimensions (int(T/d),N)
    # suma: its the computation of the particle filter for each dimension of the problem
    # its a rank 2 array with dimensions (int(T/d),dim)
    #x=np.zeros((2**l*T+1,N,dim))
    log_weights=np.zeros((int(T/d),N)) # log of the particle filter weights at d-spaced 
    # times
    x_pf=np.zeros((int(T/d),N,dim))  # Particles(not resampled) at d-spaced times                        
    x_new=x_in
    x_last=np.zeros((N,dim))+xin # x_new is the initial condition for each iteration, resampled particles
    #x[0]=xin
    d_steps=int(d*2**l)
    c_indices=np.digitize(obs_time,d*np.array(range(int(T/d)+1)),right=True)-1
    # recover the indices for the partinent observation in a certain interval 
    # partitioned by d.
    # Values of the gradient descent parameters
    Lambda_pars=np.zeros((int(T/d)+1,Lamb_par_numb))
    Lambda_pars[0]=Lamb_par
    g_pars=np.zeros((int(T/d)+1,dim,dim))
    g_pars[0]=g_par
    As=np.zeros((int(T/d)+1,b_numb_par))
    As[0]=A
    # In the following three lines we define the variables that will store the gradient,
    # these structures might change and overlap depending on the inclusion of the parameters 
    # in the b, lambda, and g_den function.
    F_Lambda=np.zeros((int(T/d)+1,N,Lamb_par_numb)) #Function F corresponding to the parameters 
    # of Lambda, the intensity funciton of the Cox process
    F_g_den=np.zeros((int(T/d)+1,N,dim,dim)) # Function F corresponding to the parameters 
    # of observation likelihood function g(y|x;g_par)
    F_b=np.zeros((int(T/d)+1,N,b_numb_par)) # Function F correspoding to the parameters of the 
    # drift, in this case, A. 
    f_Lambda=F_Lambda[0]
    f_g_den=F_g_den[0]
    f_b=F_b[0]
    Grads_Lambda=np.zeros((int(T/d)+1,Lamb_par_numb)) 
    Grads_g_den=np.zeros((int(T/d)+1,dim,dim))
    Grads_b=np.zeros((int(T/d)+1,b_numb_par))
    samp_par=np.array(range(N))
    log_w_prev=log_weights[0]
    for i in range(int(T/d)):
        # in the following two lines we get the observations and observation times for the interval
        # (i*d,(i+1)*d)
        A=As[i]
        # Uncomment the following line when computing the OU configuration.
        #A=A[:,np.newaxis]
        g_par=g_pars[i]
        Lamb_par=Lambda_pars[i]
        obti=obs_time[np.nonzero(c_indices==i)]
        yi=obs[np.nonzero(c_indices==i)]
        # we propagate the particles and obtain the term2 function (the term of F that involves the brownian motions)
        xi,term2A,term2B=M_smooth_W(x_new,x_last,b_ou,A,Sig_ou,fi,Grad_b,b_numb_par,l,d,N,dim)
        term1A=-np.sum(Grad_Lambda(x_last[np.newaxis,:,:],Lamb_par,Lamb_par_numb,ax1=-1),axis=0)/2**l
        term1B=-np.sum(Grad_Lambda(xi[1:-1],Lamb_par,Lamb_par_numb,ax1=-1),axis=0)/2**l
        Log_trans=log_trans(x_last,xi[1],b_ou,A,Sig_ou,fi,l)
        #print(Log_trans.shape)
        #print("log_trans is",Log_trans)
        log_w ,log_w_init, term3_Lambd,term3_g_den=Gox_SM_W(yi,obti-i*d,xi,x_last,samp_par, Lambda\
        ,Grad_Log_Lambda,Lamb_par,Lamb_par_numb,l,N,dim,g_den,Grad_Log_g_den,g_par)
        if log_w_init.ndim==1:
            #Lambda parameters 
            #mini_w=norm_logweights(Log_trans+log_w_init[:,np.newaxis],ax=0)
            #print(log_w_prev[:,np.newaxis].shape)
            mini_w=norm_logweights(Log_trans+log_w_init[:,np.newaxis]+log_w_prev[:,np.newaxis],ax=0)
            #print("mini_w is ",mini_w)
            f_LambdaA=np.einsum("ji,jp->ip",mini_w,f_Lambda+term1A)
            f_LambdaB=term1B +term3_Lambd
            f_Lambda= f_LambdaA+f_LambdaB
            
            F_Lambda[i+1]=f_Lambda
            #print("f_Lambda is ",f_Lambda)
            #print("dims are",mini_w.shape,f_b.shape,term2A.shape)
            f_bA=np.einsum("ji,jp->ip",mini_w,f_b+term2A)
            #print("mini_w shape is",mini_w.shape)
            #print("f_b shape is",f_b.shape)
            #print("term2A shape is",term2A.shape)
            #print("term2B shape is",term2B.shape)   
            f_bB=term2B

            f_b=f_bA+f_bB
            #print("f_b shape is ",f_b.shape)
            F_b[i+1]=f_b
            #print("f_b is ",f_b)
            f_g_denA=np.einsum("ji,jpq->ipq",mini_w,f_g_den)
            f_g_denB=term3_g_den
            f_g_den=f_g_denA+f_g_denB
            F_g_den[i+1]=f_g_den
            #print("f_g_den is ",f_g_den)

        if log_w_init.ndim==2:
            #print("log_w_init",log_w_init)
            mini_w=norm_logweights(Log_trans+log_w_init+log_w_prev[:,np.newaxis],ax=0)
            #print("mini_w is ",mini_w)
            f_Lambda=np.einsum("ji,jip->ip",mini_w,\
            (f_Lambda+term1A)[:,np.newaxis]+term1B+term3_Lambd)
            #print( "vec is",(term1A)[:,np.newaxis]+term1B)
            
            F_Lambda[i+1]=f_Lambda
            #print("f_Lambda is ",f_Lambda)
            f_bA=np.einsum("ji,jp->ip",mini_w,f_b+term2A)
            f_bB=term2B
            f_b=f_bA+f_bB
            F_b[i+1]=f_b
            #print("f_b is ",f_b)
            f_g_den=np.einsum("ji,jipq->ipq",mini_w,\
            (f_g_den[samp_par])+term3_g_den)
            #print("term3_g_den is ",term3_g_den)
            #print("f_g_den is ",f_g_den)
            F_g_den[i+1]=f_g_den
        #print("the size of f_b is: ",f_b.shape)
        #print(f_Lambda.shape,norm_logweights(log_w)[:,np.newaxis].shape)
        Grads_Lambda[i+1]=np.sum(norm_logweights(log_w)[:,np.newaxis]*f_Lambda,axis=0)
        Grads_g_den[i+1]=np.sum(norm_logweights(log_w)[:,np.newaxis,np.newaxis]*f_g_den,axis=0)
        Grads_b[i+1]=np.sum(norm_logweights(log_w)[:,np.newaxis]*f_b,axis=0)
        #print("Grads_b is ",Grads_b[i+1])
        step_size=step_0*(i+1)**(-beta)/d
        
        #print("step_size is ",step_size)
        Lambda_pars[i+1]=Lambda_pars[i]+step_size*(Grads_Lambda[i+1]-Grads_Lambda[i])*Lamb_step
        g_pars[i+1]=g_pars[i]+(Grads_g_den[i+1]-Grads_g_den[i])*g_step*step_size
        As[i+1]=As[i]+(Grads_b[i+1]-Grads_b[i])*b_step*step_size
        log_weights[i]=log_weights[i]+log_w
        #print("log_w_init is ",log_w_init)
        weights=norm_logweights(log_weights[i],ax=0)
        x_last=xi[-1]
        x_pf[i]=xi[-1]
        ESS=1/np.sum(weights**2)
        if ESS<resamp_coef*N:
            [samp_par,x_new]=multi_samp(weights,N,x_last,dim)
        else:
            x_new=x_last
            if i< int(T/d)-1:
                log_weights[i+1]=log_weights[i]
            samp_par=np.array(range(N))
        log_w_prev=log_weights[i]
    return [log_weights,x_pf,Lambda_pars,g_pars,As]


#%%

# Test for the noninformative particle filter

def log_g_den_nonin(obs,x_pr,g_den_par,crossed=False):

    return 0.1

def Grad_log_g_nonin(x, y, the4):
    return 0


def constant_sampling(pars,N):
    return np.zeros(N)+1

#%%
"""
N=100000
the1=2
the2=1
the3=0.5
dist_params=[the1,the2,the3]
in_dist_pars=dist_params
t0=1.2
T=3.2
A_til=1
fi_til=the3
r_pars=1
resamp_coef=1
l=1
d=1
H_pars=1
seed=6
obs=np.array([1.4,2,3,4.7,5.3,6.5])
obs_times=np.array([1.2,2,3.5,4.7,5.3,6.5])/3


#(in_dist,in_dist_pars, b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,\
    #update_func,sample_funct,sample_pars,obs,obs_times,log_g_den,g_den_par, aux_trans_den,atdp,\
    #prop_trans_den, resamp_coef, l, d,N,seed)
start=time.time()
l0=4
Lmax=4
eLes=np.array(range(l0,Lmax+1))
samples=1
log_weightss=np.zeros((len(eLes),samples,len(obs),N))
int_Gss=np.zeros((len(eLes),samples,len(obs),N))    
x_prs=np.zeros((len(eLes),samples,len(obs),N))

for sample in range(samples):
    seed+=1
    print("The sample is",sample)
    for i in range(len(eLes)):
        print("The level is",eLes[i])
        l=eLes[i]
        print(in_dist_pars)
        [log_weights,int_Gs,x_pr]=Gen_PF_bridge(gamma_sampling,in_dist_pars, bdg.b_log,dist_params,\
        bdg.Sig_gbm_1d,the3,bdg.b_log_aux,A_til,Sig_aux_gbm_1d,fi_til,r_log_normal,\
        r_pars,bdg.H_log_normal,H_pars,update_log_functs,\
        sampling_CIR,dist_params,obs,obs_times,log_g_den_nonin,\
        1, bdg.trans_log_normal,1, trans_noncentral_chisquare, resamp_coef, l, d,N,seed)
        log_weightss[i,sample]=log_weights
        int_Gss[i,sample]=int_Gs
        x_prs[i,sample]=x_pr
end=time.time()
print("The time is",end-start)
#%%
print(obs_times[1:]-obs_times[:-1])
print(get_deltas(obs_times))

#%%
the1,the2,the3= dist_params
w=the3**2/2+the1
xi=the2
sigma=the3
alpha=2*w/sigma**2-1
theta=sigma**2/(2*xi)
print(alpha,theta)
counts, bin_edges, _ = plt.hist(x_pr[-1], bins=100, density=True,alpha=0.5, label="Sampled histogram")
#3. Evaluate the PDF on a grid of points covering the sample range
x_grid = np.linspace(bin_edges[0], bin_edges[-1], 300)  # e.g. 300 points
#print(t0,x0,)
pdf_vals = gamma.pdf(x_grid,alpha,scale=theta)
plt.plot(x_grid, pdf_vals, 'r-', linewidth=2, label="Theoretical PDF")
#%%
weightss=pff.norm_logweights(log_weightss,ax=3)
x_means=np.sum(x_prs*weightss,axis=(3))
x_sm=np.sum(x_prs**2*weightss,axis=(3))
#%%
x_means_means=np.mean(x_means,axis=(1))
var_x_means=np.var(x_means,axis=(1))
x_sm_means=np.mean(x_sm,axis=(1))
var_x_sm=np.var(x_sm,axis=(1))
#%%
bias=np.abs(x_means_means-theta*alpha)
#print(bias)
sm_bias=np.abs(x_sm_means-(theta**2*alpha+(theta*alpha)**2))
sm_bias_ub=sm_bias+np.sqrt(var_x_sm)*1.96/np.sqrt(samples)
bias_ub=bias+np.sqrt(var_x_means)*1.96/np.sqrt(samples)
bias_lb=bias-np.sqrt(var_x_means)*1.96/np.sqrt(samples)
sm_bias_lb=sm_bias-np.sqrt(var_x_sm)*1.96/np.sqrt(samples)
plt.plot(eLes,bias[:,-1],label="Bias")
plt.plot(eLes,bias_ub[:,-1],label="Bias UB")
plt.plot(eLes,bias_lb[:,-1],label="Bias LB")
plt.plot(eLes,sm_bias[:,-1],label="sm bias")
plt.plot(eLes,sm_bias_ub[:,-1],label="sm bias UB")
plt.plot(eLes,sm_bias_lb[:,-1],label="sm bias LB")
plt.plot(eLes,bias[0,-1]*2**eLes[0]/2**eLes)
plt.yscale("log")
plt.legend()
print(bias[:,-1])
# %%
"""
#%%
"""
N=100000
the1=2
the2=1
the3=0.5
dist_params=[the1,the2,the3]
in_dist_pars=dist_params
t0=1.2
T=3.2
A_til=1
fi_til=the3
r_pars=1 
resamp_coef=1
l=1
d=1
H_pars=1
seed=6
obs=np.array([1.4,2,3,4.7,5.3,6.5])
obs_times=np.array([1.2,2,3.5,4.7,5.3,6.5])/3


#(in_dist,in_dist_pars, b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,\
    #update_func,sample_funct,sample_pars,obs,obs_times,log_g_den,g_den_par, aux_trans_den,atdp,\
    #prop_trans_den, resamp_coef, l, d,N,seed)
start=time.time()
l0=1
Lmax=7
eLes=np.array(range(l0,Lmax+1))
samples=140
log_weightss=np.zeros((2,len(eLes),samples,len(obs),N))
int_Gss=np.zeros((2,len(eLes),samples,len(obs),N))    
x_prs=np.zeros((2,len(eLes),samples,len(obs),N))
rej_dist_params=np.array([the1,the2,the3,the1,the2,the3])
for sample in range(samples):
    seed+=1
    print("The sample is",sample)
    for i in range(len(eLes)):
        print("The level is",eLes[i])
        l=eLes[i]
        print(in_dist_pars)
       
        [log_weights_0,log_weights_1,int_Gs_0,int_Gs_1,x_pr_0,x_pr_1,seeds_0_wp,seeds_1_wp]=\
        Gen_C_PF_bridge(gamma_sampling,in_dist_pars, b_log,dist_params,\
        Sig_gbm_1d,the3,b_log_aux,A_til,Sig_aux_gbm_1d,fi_til,r_log_normal,\
        r_pars,H_log_normal,H_pars,update_log_functs,\
        rej_max_coup_CIR,rej_dist_params,obs,obs_times,log_g_den_nonin,\
        1, trans_log_normal,1, trans_noncentral_chisquare,dist_params, resamp_coef, l, d,N,seed)
        log_weightss[:,i,sample]=[log_weights_0,log_weights_1]
        int_Gss[:,i,sample]=[int_Gs_0,int_Gs_1]
        x_prs[:,i,sample]=[x_pr_0,x_pr_1]
end=time.time()
print("The time is",end-start)
#%%
print(obs_times[1:]-obs_times[:-1])
print(get_deltas(obs_times))

#%%
the1,the2,the3= dist_params
w=the3**2/2+the1
xi=the2
sigma=the3
alpha=2*w/sigma**2-1
theta=sigma**2/(2*xi)
print(alpha,theta)
counts, bin_edges, _ = plt.hist(x_pr_0[-1], bins=100, density=True,alpha=0.5, label="Sampled histogram")
#3. Evaluate the PDF on a grid of points covering the sample range
x_grid = np.linspace(bin_edges[0], bin_edges[-1], 300)  # e.g. 300 points
#print(t0,x0,)
pdf_vals = gamma.pdf(x_grid,alpha,scale=theta)
plt.plot(x_grid, pdf_vals,'r-', linewidth=2, label="Theoretical PDF")
#%%
weightss=pff.norm_logweights(log_weightss,ax=4)
x_means=np.sum(x_prs*weightss,axis=(4))
x_sm=np.sum(x_prs**2*weightss,axis=(4))
#%%
# x_prs=np.zeros((2,len(eLes),samples,len(obs),N))
x_means_means=np.mean((x_means[0]-x_means[1])**2,axis=(1))
var_x_means=np.var((x_means[0]-x_means[1])**2,axis=(1))
#print(((x_means[0]-x_means[1])**2)[:,:2])
#print(var_x_means)
x_sm_means=np.mean((x_sm[0]-x_sm[1])**2,axis=(1))
var_x_sm=np.var((x_sm[0]-x_sm[1])**2,axis=(1))
#%%
x_means_lb=x_means_means-np.sqrt(var_x_means)*1.96/np.sqrt(samples)
x_means_ub=x_means_means+np.sqrt(var_x_means)*1.96/np.sqrt(samples)
x_sm_lb=x_sm_means-np.sqrt(var_x_sm)*1.96/np.sqrt(samples)
x_sm_ub=x_sm_means+np.sqrt(var_x_sm)*1.96/np.sqrt(samples)
plt.plot(eLes,x_means_means[:,-1],label="Mean squared difference")
plt.plot(eLes,x_means_ub[:,-1],label="Mean squared difference UB")
plt.plot(eLes,x_means_lb[:,-1],label="Mean squared difference LB")
plt.plot(eLes,x_sm_means[:,-1],label="sm msd")
plt.plot(eLes,x_sm_ub[:,-1],label="UB")
plt.plot(eLes,x_sm_lb[:,-1],label="LB")
plt.plot(eLes,x_means_means[0,-1]*2**eLes[0]/2**eLes)
plt.yscale("log")
plt.legend()
#%%
level=0
x_means_means=np.mean(x_means,axis=(2))[level]
var_x_means=np.var(x_means,axis=(2))[level]
x_sm_means=np.mean(x_sm,axis=(2))[level]
var_x_sm=np.var(x_sm,axis=(2))[level]

bias=np.abs(x_means_means-theta*alpha)
#print(bias)
sm_bias=np.abs(x_sm_means-(theta**2*alpha+(theta*alpha)**2))
sm_bias_ub=sm_bias+np.sqrt(var_x_sm)*1.96/np.sqrt(samples)
bias_ub=bias+np.sqrt(var_x_means)*1.96/np.sqrt(samples)
bias_lb=bias-np.sqrt(var_x_means)*1.96/np.sqrt(samples)
sm_bias_lb=sm_bias-np.sqrt(var_x_sm)*1.96/np.sqrt(samples)
plt.plot(eLes,bias[:,-1],label="Bias")
plt.plot(eLes,bias_ub[:,-1],label="Bias UB")
plt.plot(eLes,bias_lb[:,-1],label="Bias LB")
plt.plot(eLes,sm_bias[:,-1],label="sm bias")
plt.plot(eLes,sm_bias_ub[:,-1],label="sm bias UB")
plt.plot(eLes,sm_bias_lb[:,-1],label="sm bias LB")
plt.plot(eLes,bias[0,-1]*2**eLes[0]/2**eLes)
plt.yscale("log")
plt.legend()
print(bias[:,-1])
"""
# %%
