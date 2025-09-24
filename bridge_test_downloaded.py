# File to test the functions in the module bridge.py
#%%

#from Un_cox_PF_functions_def import *
import PF_functions_def as pff
import bridge as bdg
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


np.random.seed(2)
# %%
l_0=6
dl_0=1/2**(l_0)
theta_true=-0.1
sigma_true=0.5
l_1=8
dl_1=1/2**(l_1)
x0=1
real_0=np.zeros(int(1/dl_0)+1)
real_1=np.zeros(int(1/dl_1)+1)
real_0[0]=x0
real_1[0]=x0
dWs=np.zeros(int(dl_0/dl_1))
x_0=x0
x_1=x0
for i in range(int(1/dl_0)):
    dWs=np.random.normal(0,1,int(dl_0/dl_1))
    x_0=x_0+theta_true*(x_0)*dl_0+sigma_true*np.sum(dWs)*np.sqrt(dl_0)
    real_0[i+1]=x_0
    for j in range(int(dl_0/dl_1)):
        x_1+=theta_true*(x_1)*dl_1+sigma_true*dWs[j]*np.sqrt(dl_1)
        real_1[i*int(dl_0/dl_1) + j+1]=x_1
l_0_times= np.arange(0,1+dl_0,dl_0)
l_1_times= np.arange(0,1+dl_1,dl_1)
plt.plot(l_0_times,real_0,label="True path 0")
plt.plot(l_1_times,real_1,label="True path 1")
frame1 = plt.gca()
frame1.axes.get_xaxis().set_visible(False)
frame1.axes.get_yaxis().set_visible(False)
frame1.axes.get_xaxis().set_ticks([])
frame1.axes.get_yaxis().set_ticks([])
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = False
#plt.legend()
plt.savefig("True_paths.pdf")   

#%%
first_array = np.array([[1, 2], [3, 4], [5, 6]])
second_array = np.array([[-1, -2]])

# Assign second_array to the first row of first_array
first_array[0] = second_array # Direct assignment of values

print(first_array)  # Output: [[-1, -2], [3, 4], [5, 6]]

#second_ar=np.array([3,4,5])
#new_arr=second_ar
#print(test_ar)
#print(new_arr)
#%% 
first_array = np.array([[1, 2], [3, 4]])
slice_view=1+np.array([[1, 2], [3, 4]])
# Create a slice (a view) of the first two rows
first_array[0:2]=slice_view 

# Modify the slice
first_array[0, 0] = 100  # Changes both slice_view and first_array
print(slice_view)   # Output: [[100, 2], [3, 4]]
print(first_array)  # Output: [[100, 2], [3, 4]]  (also modified)

# An array has an identificator, when an array is composed of multidimentional 
# arrays, do the smaller arrays have an identificator? 

# %%


# In the following we test speficially the funciton Cond_PF_bridge_back_samp with 
# d different from 1, and check with KF.

N=50
x0_sca=1.2
x0=x0_sca+np.zeros(N)
l=6
T=10
t0=0
l_d=-1
d=2**(l_d)
theta_true=-0.3
sigma_true=1.2
#sigma_aux=0.2
#print(theta)
np.random.seed(7)
collection_input=[bdg.b_ou_1d,theta_true,bdg.Sig_ou_1d,sigma_true]
resamp_coef=1
l_max=17   
x_true= bdg.gen_gen_data_1d(T,x0_sca,l_max,collection_input)
x_reg=bdg.cut(T,l_max,-l_d,x_true)[1:]
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
obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)
plt.plot(times[1:], obs,label="Observations")
print(obs,x_reg)
fd_rate=1e-4
[theta_fd,sigma_fd,sd_fd]=np.array([theta_true,sigma_true,sd_true])+fd_rate*np.array([1,1,1])
print(theta_fd,sigma_fd,sd_fd)

#%%

"""
Cond_PF_bridge_back_samp(lw_cond,int_Gs_cond,x_cond,seeds_cond,t0,x0,T,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,\
    r,r_pars,H,H_pars,sample_funct,sample_pars,obs,log_g_den,g_den_par, aux_trans_den,atdp,\
    prop_trans_den, resamp_coef, l, d,N,seed,crossed=False)
"""
theta=theta_true
theta_aux=theta+0.2
sigma=sigma_true
sigma_aux=sigma
start=time.time()
B=5
samples=10
seed=0
mcmc_mean=np.zeros((samples,2,int(T/d))) # This varible was originally designed 
# to store the mean of both processes, the one with multinomial sampling and the one with
# backward sampling.
lws=np.zeros((samples,int(T/d),N))
x_prs=np.zeros((samples,int(T/d),N))
resamp_coef=1
Grads_test=np.zeros((samples,B,3))
for i in range(samples):
    np.random.seed(i)
    [log_weights,int_Gs,x_pr]=bdg.PF_bridge(t0,x0,T,bdg.b_ou_1d,theta,bdg.Sig_ou_1d,sigma,bdg.b_ou_aux,theta_aux,bdg.Sig_ou_aux,sigma_aux,\
    bdg.r_quasi_normal_1d,[bdg.ou_sd,[theta_aux,sigma_aux]],bdg.H_quasi_normal,[bdg.ou_sd,[theta_aux,sigma_aux],theta_aux],\
    bdg.sampling_ou, [theta_aux,sigma_aux],obs,bdg.log_g_normal_den,sd_true,\
    bdg.ou_trans_den,[theta_aux,sigma_aux],bdg.ou_trans_den,\
    resamp_coef,l,d, N,seed)
    lws[i]=log_weights
    x_prs[i]=x_pr
    #x_mean=np.sum(pff.norm_logweights(log_weights,ax=1)*x_pr,axis=-1)
#%%
sd=sd_true
dim=1
dim_o=1
print(theta, sigma)
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
x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
d_times=np.array(range(t0,int(T/d)+1))*d
weights=pff.norm_logweights(lws,ax=2)
PF_mean=np.mean(np.sum(x_prs*weights,axis=-1),axis=0)
l_times=np.arange(t0,T,2**(-l))
l_max_times=np.arange(t0,T,2**(-l_max))
#print(times, l_times)
#x_mean=np.mean(mcmc_mean[:,0],axis=0)
#print(x_mean.shape)
plt.plot(d_times[1:],PF_mean,label="PF")
#plt.plot(d_times[1:],x_reg,label="True signal")
#plt.plot(l_max_times,x_true[:-1],label="True complete signal")
#plt.plot(d_times,x_kf_smooth[:,0],label="KF smooth")
plt.plot(d_times,x_kf[:,0],label="KF")
#plt.plot(d_times[1:], obs,label="Observations")
#plt.plot(d_times[1:], x_mean,label="PGibbs")
plt.legend()    
print(PF_mean-x_kf[1:,0])
# RESULTS OF THE TEST:
# The kalman filter coincides with the particle filter, at least visually. 
 
#%%
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################

# In the following we test the Grad_PF_bridge_back_samp function with d different from 1, and check with the
# analytical gradient.

N=100
x0_sca=1.2
x0=x0_sca+np.zeros(N)
l=12
T=3
t0=0
l_d=-6
d=2**(l_d)
theta_true=-0.3
sigma_true=1.2
#sigma_aux=0.2
#print(theta)
np.random.seed(7)
collection_input=[bdg.b_ou_1d,theta_true,bdg.Sig_ou_1d,sigma_true]
resamp_coef=1
l_max=10    
x_true= bdg.gen_gen_data_1d(T,x0_sca,l_max,collection_input)
x_reg=bdg.cut(T,l_max,-l_d,x_true)[1:]
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
obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)
plt.plot(times[1:], obs,label="Observations")
print(obs,x_reg)
fd_rate=1e-4
[theta_fd,sigma_fd,sd_fd]=np.array([theta_true,sigma_true,sd_true])+fd_rate*np.array([1,1,1])
print(theta_fd,sigma_fd,sd_fd)


#%%

theta=theta_true
theta_aux=theta+0.2
sigma=sigma_true
sigma_aux=sigma
start=time.time()
B=10
samples=2
seed=0
mcmc_mean=np.zeros((samples,2,int(T/d))) # This varible was originally designed 
# to store the mean of both processes, the one with multinomial sampling and the one with
# backward sampling.
resamp_coef=1
Grads=np.zeros((samples,B,3))
lws=np.zeros((samples,B,int(T/d),N))
x_prs=np.zeros((samples,B,int(T/d),N))


for i in range(samples):
    np.random.seed(i)
    [log_weights,int_Gs,x_pr]=bdg.PF_bridge(t0,x0,T,bdg.b_ou_1d,theta,bdg.Sig_ou_1d,sigma,bdg.b_ou_aux,theta_aux,bdg.Sig_ou_aux,sigma_aux,\
    bdg.r_quasi_normal_1d,[bdg.ou_sd,[theta_aux,sigma_aux]],bdg.H_quasi_normal,[bdg.ou_sd,[theta_aux,sigma_aux],theta_aux],\
    bdg.sampling_ou, [theta_aux,sigma_aux],obs,bdg.log_g_normal_den,sd,\
    bdg.ou_trans_den,[theta_aux,sigma_aux],bdg.ou_trans_den,\
    resamp_coef,l,d, N,seed)
    #x_mean=np.sum(pff.norm_logweights(log_weights,ax=1)*x_pr,axis=-1)
    weights=pff.norm_logweights(log_weights[-1])
    #print(weights.shape)
    index=np.random.choice(np.array(range(N)),p=weights)
    cond_path=x_pr[:,index]
    cond_log_weights=log_weights[:,index]
    seeds_cond=np.zeros((int(T/d),2),dtype=int)
    seeds_cond[:,0]=seed+np.array(range(int(T/d)))*(int(2**l*d)-1)
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
        seed+=int((int(T/d))*(int(2**l*d)-1))
        np.random.seed(b)
        [log_weights,x_pr,cond_log_weights,cond_int_G,cond_path,seeds_cond]=\
        bdg.Cond_PF_bridge_back_samp(cond_log_weights,cond_int_G,cond_path,seeds_cond,t0,x0,\
        T,bdg.b_ou_1d,theta,bdg.Sig_ou_1d,sigma,bdg.b_ou_aux,theta_aux,bdg.Sig_ou_aux,sigma_aux,\
        bdg.r_quasi_normal_1d,[bdg.ou_sd,[theta_aux,sigma_aux]],bdg.H_quasi_normal,\
        [bdg.ou_sd,[theta_aux,sigma_aux],theta_aux],\
        bdg.sampling_ou, [theta_aux,sigma_aux],obs,bdg.log_g_normal_den,sd,\
        bdg.ou_trans_den,[theta_aux,sigma_aux],bdg.ou_trans_den,\
        resamp_coef,l,d, N,seed,crossed=False)
        lws[i,b]=log_weights
        x_prs[i,b]=x_pr 
        #print("The other condtional path is:",cond_path)
        
end=time.time()
#%%
sd=sd_true
dim=1
dim_o=1
print(theta, sigma)
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
x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
d_times=np.array(range(t0,int(T/d)+1))*d
weights=pff.norm_logweights(lws,ax=3)
PF_mean=np.mean(np.sum(x_prs*weights,axis=-1),axis=(0,1))
l_times=np.arange(t0,T,2**(-l))
l_max_times=np.arange(t0,T,2**(-l_max))
#print(times, l_times)
#x_mean=np.mean(mcmc_mean[:,0],axis=0)
#print(x_mean.shape)
plt.plot(d_times[1:],PF_mean,label="PF")
#plt.plot(d_times[1:],x_reg,label="True signal")
#plt.plot(l_max_times,x_true[:-1],label="True complete signal")
#plt.plot(d_times,x_kf_smooth[:,0],label="KF smooth")
plt.plot(d_times,x_kf[:,0],label="KF")
#plt.plot(d_times[1:], obs,label="Observations")
#plt.plot(d_times[1:], x_mean,label="PGibbs")
plt.legend()    
print(PF_mean-x_kf[1:,0])
# RESULTS OF THE TEST:
# The kalman filter coincides with the particle filter, at least visually. 

#%%

# RESULTS OF THE TEST: 
# The resutls are satisfactory, visually the difference between the particle filter of the function "Cond_PF_bridge_back_samp"
# and the Kalman filter is almost imperceptible.

#%%

N=10
x0_sca=1.2
x0=x0_sca+np.zeros(N)
l=8
T=3
t0=0
l_d=-2
d=2**(l_d)
theta_true=-0.3
sigma_true=1.2
#sigma_aux=0.2
#print(theta)
np.random.seed(7)
collection_input=[bdg.b_ou_1d,theta_true,bdg.Sig_ou_1d,sigma_true]
resamp_coef=1
l_max=10    
x_true= bdg.gen_gen_data_1d(T,x0_sca,l_max,collection_input)
x_reg=bdg.cut(T,l_max,-l_d,x_true)[1:]
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
obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)
plt.plot(times[1:], obs,label="Observations")
print(obs,x_reg)
fd_rate=1e-4
[theta_fd,sigma_fd,sd_fd]=np.array([theta_true,sigma_true,sd_true])+fd_rate*np.array([1,1,1])
print(theta_fd,sigma_fd,sd_fd)
#%%

theta=theta_true
theta_aux=theta+0.2
sigma=sigma_true
sigma_aux=sigma
sd=sd_true
start=time.time()
B=50
samples=2
seed=0
mcmc_mean=np.zeros((samples,2,int(T/d))) # This varible was originally designed 
# to store the mean of both processes, the one with multinomial sampling and the one with
# backward sampling.
resamp_coef=1
Grads_test=np.zeros((samples,B,3))
for i in range(samples):
    np.random.seed(i)
    [log_weights,int_Gs,x_pr]=bdg.PF_bridge(t0,x0,T,bdg.b_ou_1d,theta,bdg.Sig_ou_1d,sigma,bdg.b_ou_aux,theta_aux,bdg.Sig_ou_aux,sigma_aux,\
    bdg.r_quasi_normal_1d,[bdg.ou_sd,[theta_aux,sigma_aux]],bdg.H_quasi_normal,[bdg.ou_sd,[theta_aux,sigma_aux],theta_aux],\
    bdg.sampling_ou, [theta_aux,sigma_aux],obs,bdg.log_g_normal_den,sd,\
    bdg.ou_trans_den,[theta_aux,sigma_aux],bdg.ou_trans_den,\
    resamp_coef,l,d, N,seed)
    
    weights=pff.norm_logweights(log_weights[-1])
    index=np.random.choice(np.array(range(N)),p=weights)
    cond_path=x_pr[:,index]
    cond_log_weights=log_weights[:,index]
    seeds_cond=np.zeros((int(T/d),2),dtype=int)
    seeds_cond[:,0]=seed+np.array(range(int(T/d)))*(int(2**l*d)-1)
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
        seed+=int((int(T/d))*(int(2**l*d)-1))
        np.random.seed(b)
        [log_weights_test,x_pr_test,cond_log_weights_test,cond_int_G_test,cond_path_test,seeds_cond_test,Grads_t,int_G_sub1,int_G_sub2]=\
        bdg.Grad_Cond_PF_bridge_back_samp(cond_log_weights_test,cond_int_G_test,cond_path_test,seeds_cond_test,t0,x0,T,bdg.b_ou_1d,\
        theta,theta_fd,bdg.Sig_ou_1d,\
        sigma,sigma_fd,bdg.b_ou_aux,theta_aux, bdg.Sig_ou_aux,sigma_aux,sigma_fd,bdg.r_quasi_normal_1d,[bdg.ou_sd,[theta_aux,sigma_aux]],\
        [bdg.ou_sd,[theta_aux,sigma_fd]],bdg.H_quasi_normal,[bdg.ou_sd,[theta_aux,sigma_aux],theta_aux],[bdg.ou_sd,[theta_aux,sigma_fd],theta_aux],\
        bdg.sampling_ou, [theta_aux,sigma_aux],obs,bdg.log_g_normal_den,sd, bdg.ou_trans_den,[theta_aux,sigma_aux],\
        bdg.Grad_log_aux_trans_ou_new,bdg.ou_trans_den, bdg.Grad_log_G_new,resamp_coef, l, d,N,seed,fd_rate,crossed=False)
        print("int_G_sub is: ",int_G_sub1,int_G_sub2)   
        Grads_test[i,b]=Grads_t
        ch_paths[b]=cond_path_test
    mcmc_mean[i,0]=np.mean(ch_paths,axis=0)
end=time.time()


print("The time spend is with ",N, " is ",end-start)
#%%

sd=sd_true
dim=1
dim_o=1
print(theta, sigma)
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
x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
d_times=np.array(range(t0,int(T/d)+1))*d
weights=pff.norm_logweights(lws,ax=3)
PF_mean=np.mean(np.sum(x_prs*weights,axis=-1),axis=(0,1))
l_times=np.arange(t0,T,2**(-l))
l_max_times=np.arange(t0,T,2**(-l_max))
#print(times, l_times)
x_mean=np.mean(mcmc_mean[:,0],axis=0)
#print(x_mean.shape)
#plt.plot(d_times[1:],PF_mean,label="PF")
#plt.plot(d_times[1:],x_reg,label="True signal")
#plt.plot(l_max_times,x_true[:-1],label="True complete signal")
plt.plot(d_times,x_kf_smooth[:,0],label="KF smooth")
#plt.plot(d_times,x_kf[:,0],label="KF")
#plt.plot(d_times[1:], obs,label="Observations")
plt.plot(d_times[1:], x_mean,label="PGibbs")
plt.legend()    
print(PF_mean-x_kf[1:,0])
#%%
print(np.mean(Grads_test,axis=1))
print(np.sqrt(np.var(Grads_test,axis=1)/B)*1.96)
print(Grad_log_lik[:,0,0])

#%%

# IN THIS SECTIOTN WE ANALYZE THE RESULTS OF THE PARALLLELIZED TEST.

# RESULTS OF THE TEST:
N=100
x0_sca=1.2
x0=x0_sca+np.zeros(N)
l=9
T=10
t0=0
l_d=0
d=2**(l_d)
theta_true=-0.3
sigma_true=1.2
np.random.seed(7)
collection_input=[bdg.b_ou_1d,theta_true,bdg.Sig_ou_1d,sigma_true]
resamp_coef=1
l_max=10
x_true= bdg.gen_gen_data_1d(T,x0_sca,l_max,collection_input)
x_reg=bdg.cut(T,l_max,-l_d,x_true)[1:]
times=np.array(range(t0,int(T/d)+1))*d
l_times=np.arange(t0,T,2**(-l))
l_max_times=np.arange(t0,T,2**(-l_max))
#plt.plot(times[1:],x_reg,label="True signal")
#plt.plot(l_max_times,x_true[:-1],label="True complete signal")
sd_true=5e-1
np.random.seed(3)
obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)
#plt.plot(times[1:], obs,label="Observations")
#print(obs,x_reg)
fd_rate=1e-4
[theta_fd,sigma_fd,sd_fd]=np.array([theta_true,sigma_true,sd_true])+fd_rate*np.array([1,1,1])
#%%
samples=10
start=time.time()
mcmc_links=10*200
SGD_steps=1
B=mcmc_links*SGD_steps
gamma=0.05
alpha=0.01
seed=1
#%%
theta=theta_true+0.4
theta_aux=theta+0.2
sigma=sigma_true-0.3
sigma_aux=sigma
sd=sd_true
dim=1
dim_o=1
print(theta, sigma)
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
x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
d_times=np.array(range(t0,int(T/d)+1))*d
weights=pff.norm_logweights(lws,ax=3)
#PF_mean=np.mean(np.sum(x_prs*weights,axis=-1),axis=(0,1))
l_times=np.arange(t0,T,2**(-l))
l_max_times=np.arange(t0,T,2**(-l_max))
#print(times, l_times)
x_mean=np.mean(mcmc_mean[:,0],axis=0)
#print(x_mean.shape)
#plt.plot(d_times[1:],PF_mean,label="PF")
#plt.plot(d_times[1:],x_reg,label="True signal")
#plt.plot(l_max_times,x_true[:-1],label="True complete signal")
plt.plot(d_times,x_kf_smooth[:,0],label="KF smooth")
#plt.plot(d_times,x_kf[:,0],label="KF")
#plt.plot(d_times[1:], obs,label="Observations")
#plt.plot(d_times[1:], x_mean,label="PGibbs")
plt.legend()    
print(PF_mean-x_kf[1:,0])
#%%

ch_paths=np.reshape(np.loadtxt("Observations&data/Prl_SGD_bridge_ch_paths_vtest6.txt",dtype=float),(samples,B,int(T/d)))
Grads=np.reshape(np.loadtxt("Observations&data/Prl_SGD_bridge_Grads_vtest6.txt",dtype=float),(samples,B,3))
#%%
print(np.mean(Grads,axis=(0,1)))
print(np.sqrt(np.var(Grads,axis=(0,1))/(samples*B))*1.96)
print(Grad_log_lik[:,0,0])

#%%
# RESULTS OF THE TEST: The analytical gradient is within the confidence interval of the empirical gradient.

#%%

N=100
x0_sca=1.2
x0=x0_sca+np.zeros(N)
l=8
T=10
t0=0
l_d=-3
d=2**(l_d)
theta_true=-0.3
sigma_true=1.2
np.random.seed(7)
collection_input=[bdg.b_ou_1d,theta_true,bdg.Sig_ou_1d,sigma_true]
resamp_coef=1
l_max=10    
x_true= bdg.gen_gen_data_1d(T,x0_sca,l_max,collection_input)
x_reg=bdg.cut(T,l_max,-l_d,x_true)[1:]
times=np.array(range(t0,int(T/d)+1))*d
l_times=np.arange(t0,T,2**(-l))
l_max_times=np.arange(t0,T,2**(-l_max))
plt.plot(times[1:],x_reg,label="True signal")
plt.plot(l_max_times,x_true[:-1],label="True complete signal")
sd_true=5e-1
np.random.seed(3)
obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)
plt.plot(times[1:], obs,label="Observations")
#print(obs,x_reg)
fd_rate=1e-4
[theta_fd,sigma_fd,sd_fd]=np.array([theta_true,sigma_true,sd_true])+fd_rate*np.array([1,1,1])
samples=20
start=time.time()
mcmc_links=10
SGD_steps=10
B=mcmc_links*SGD_steps
gamma=0.0
alpha=0.25
seed=1
#mcmc_mean=np.zeros((samples,2,int(T/d))) # This varible was originally designed 
# to store the mean of both processes, the one with multinomial sampling and the one with
# backward sampling.
resamp_coef=1
pars=np.zeros((SGD_steps+1,3))
theta_0=theta_true+0.4
sigma_0=sigma_true-0.3
theta_0_fd=theta_0+fd_rate
sigma_0_fd=sigma_0+fd_rate
theta_0_aux=theta_0+0.2
sigma_0_aux=sigma_0
sigma_0_aux_fd=sigma_0_aux+fd_rate
sd_0=sd_true

#%%
Grid_p=20
thetas=np.linspace(-1,1,Grid_p)*0.75+theta_true
sigmas=np.linspace(-1,1,Grid_p)*0.75+1.5
theta_aux=thetas+0.2
sigma_aux=sigmas
sds=np.linspace(-1,1,Grid_p)*0.5+ sd_true
fd_rate=1e-4
[theta_fd,sigma_fd,sd_fd]=np.array([thetas,sigmas,sds])+fd_rate*(np.zeros((3,Grid_p))+1)
print(thetas,sigmas)
#%%
# IN 2d
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
        x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
        Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
        Grads[j,i]=Grad_log_lik[:,0,0]

# %%
# In teh following I will perform an SGD with the analytical gradient of the Kalman filter
# in order to find the best hyperparameters for the model.
[theta_0,sigma_0,sd_0]=np.array([theta_true,sigma_true,sd_true])+np.array([0.4,-0.3,0])
SGD_steps=100 
pars=np.zeros((SGD_steps+1,3))
Grads_test=np.zeros((SGD_steps+1,3))
alpha=0.0001
gamma=0.05
theta=theta_0+1.2
sigma=sigma_0
sd=sd_0
pars[0,:]=np.array([theta,sigma,sd])

for b_ind in range(SGD_steps):
    
    #sigma=sigmas[j]
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
    x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
    Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
    Grads_test[b_ind]=Grad_log_lik[:,0,0]
    print(Grads_test[b_ind,:2])
    theta+=gamma*Grads_test[b_ind,0]/(b_ind+1)**(0.5+alpha)
    sigma+=gamma*Grads_test[b_ind,1]/(b_ind+1)**(0.5+alpha)
    #sd+=gamma*Grads_test[b_ind,2]/(b_ind+1)**(0.5+alpha)
    pars[b_ind+1]=np.array([theta,sigma,sd])

#%%
print(pars)
print("sd is: ",sd_true)
plt.plot(pars[:,0].T,pars[:,1].T)
thetas_Grid,sigmas_Grid=np.meshgrid(thetas,sigmas)
plt.quiver(thetas_Grid,sigmas_Grid,Grads[:,:,0],Grads[:,:,1])
print("The starting guesses are: ",theta_0,sigma_0)
print("The actual parameters are: ",theta_true,sigma_true)
max=np.max(Grads[:,:,0]**2+Grads[:,:,1]**2)
min=np.min(Grads[:,:,0]**2+Grads[:,:,1]**2)
print("The maximum gradient is: ",np.sqrt(max), "The minimum gradient is: ",np.sqrt(min))
plt.xlabel("Theta")
plt.ylabel("Sigma")
plt.title("SGD")
#plt.savefig("Gradiend_flow_&_SGD.pdf")
plt.show()
    
#%%
print(pars)
print(Grads_test)
# %%
# In the following cells we observe the particle sgd, this results were parallelized

N=100
x0_sca=1.2
x0=x0_sca+np.zeros(N)
l=9
T=10
t0=0
l_d=0
d=2**(l_d)
theta_true=-0.3
sigma_true=1.2
np.random.seed(7)
collection_input=[bdg.b_ou_1d,theta_true,bdg.Sig_ou_1d,sigma_true]
resamp_coef=1
l_max=10
x_true= bdg.gen_gen_data_1d(T,x0_sca,l_max,collection_input)
x_reg=bdg.cut(T,l_max,-l_d,x_true)[1:]
times=np.array(range(t0,int(T/d)+1))*d
l_times=np.arange(t0,T,2**(-l))
l_max_times=np.arange(t0,T,2**(-l_max))
plt.plot(times[1:],x_reg,label="True signal")
plt.plot(l_max_times,x_true[:-1],label="True complete signal")
sd_true=5e-1
np.random.seed(3)
obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)
#plt.plot(times[1:], obs,label="Observations")
#print(obs,x_reg)
fd_rate=1e-4
[theta_fd,sigma_fd,sd_fd]=np.array([theta_true,sigma_true,sd_true])+fd_rate*np.array([1,1,1])
samples=20
start=time.time()
mcmc_links=10*20
SGD_steps=80
B=mcmc_links*SGD_steps
gamma=0.05
alpha=0.01
seed=1
#mcmc_mean=np.zeros((samples,2,int(T/d))) # This varible was originally designed 
# to store the mean of both processes, the one with multinomial sampling and the one with
# backward sampling.
resamp_coef=1
pars=np.zeros((SGD_steps+1,3))
theta_0=0.1
sigma_0=1.
theta_0_fd=theta_0+fd_rate
sigma_0_fd=sigma_0+fd_rate
theta_0_aux=theta_0+0.2
sigma_0_aux=sigma_0
sigma_0_aux_fd=sigma_0_aux+fd_rate
sd_0=sd_true
inputs=[]
seed=0


# %%
ch_paths=np.reshape(np.loadtxt("Observations&data/Prl_SGD_bridge_ch_paths_vtest26.txt",dtype=float),(samples,B,int(T/d)))
Grads=np.reshape(np.loadtxt("Observations&data/Prl_SGD_bridge_Grads_vtest26.txt",dtype=float),(samples,B,3))
pars=np.reshape(np.loadtxt("Observations&data/Prl_SGD_bridge_pars_vtest26.txt",dtype=float),(samples,SGD_steps+1,3))
#print(pars)
new_Grads=np.zeros((samples,SGD_steps,3))
for i in range(SGD_steps):
    new_Grads[:,i]=np.mean(Grads[:,i*mcmc_links:(i+1)*mcmc_links],axis=1)
print(pars[:,-1,:])
#print(new_Grads[0,-1])
#print(new_Grads[0])
pars[0]

#%%

s_sample=0
i=0
[theta,sigma,sd]=pars[s_sample,i]
print(theta,sigma,sd)
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
x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
d_times=np.array(range(t0,int(T/d)+1))*d
#weights=pff.norm_logweights(lws,ax=3)
#PF_mean=np.mean(np.sum(x_prs*weights,axis=-1),axis=(0,1))
#l_times=np.arange(t0,T,2**(-l))
#l_max_times=np.arange(t0,T,2**(-l_max))
#print(times, l_times)
x_mean=np.mean(ch_paths[s_sample,i*mcmc_links:(i+1)*mcmc_links],axis=0)
#print(x_mean.shape)
#plt.plot(d_times[1:],PF_mean,label="PF")
#plt.plot(d_times[1:],x_reg,label="True signal")
#plt.plot(l_max_times,x_true[:-1],label="True complete signal")
plt.plot(d_times,x_kf_smooth[:,0],label="KF smooth")
#print(x_kf_smooth[:,0])
#plt.plot(d_times,x_kf[:,0],label="KF")
#plt.plot(d_times[1:], obs,label="Observations")

plt.plot(d_times[1:], x_mean,label="PGibbs")
plt.legend()  
print(np.mean(new_Grads,axis=0))  
print(Grad_log_lik)
# %%
Grid_p=20
thetas=np.linspace(-1,1,Grid_p)*0.5-0.4
sigmas=np.linspace(-1,1,Grid_p)*0.5+sigma_true
theta_aux=thetas+0.2
sigma_aux=sigmas
sds=np.linspace(-1,1,Grid_p)*0.5+ sd_true
fd_rate=1e-4
[theta_fd,sigma_fd,sd_fd]=np.array([thetas,sigmas,sds])+fd_rate*(np.zeros((3,Grid_p))+1)
print(thetas,sigmas)
#%%
# IN 2d
Grads=np.zeros((Grid_p,Grid_p,3))
dim=1
sd=0.45
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
        #print(Grad_R_sigma_s)
        Grad_R_theta=(sigma**2/(2*theta**2))*(1-np.exp(2*theta*d)+2*d*theta*np.exp(2*theta*d))
        Grad_R=np.zeros((3,1,1),dtype=float)
        Grad_R[0,0,0]=Grad_R_theta
        Grad_R[1,0,0]=Grad_R_sigma_s
        Grad_K=np.array([[[d*np.exp(d*theta)]],[[0]],[[0]]],dtype=float)
        #print(Grad_R)
        x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
        Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
        Grads[j,i]=Grad_log_lik[:,0,0]
sd=sd_true
 
#%%
print(pars)
print("sd true is: ",sd_true)
plt.plot(pars[:,:,0].T,pars[:,:,1].T)
thetas_Grid,sigmas_Grid=np.meshgrid(thetas,sigmas)
plt.quiver(thetas_Grid,sigmas_Grid,Grads[:,:,0],Grads[:,:,1])
print("The starting guesses are: ",theta_0,sigma_0)
print("The actual parameters are: ",theta_true,sigma_true)
max=np.max(Grads[:,:,0]**2+Grads[:,:,1]**2)
min=np.min(Grads[:,:,0]**2+Grads[:,:,1]**2)
print("The maximum gradient is: ",np.sqrt(max), "The minimum gradient is: ",np.sqrt(min))
plt.xlabel("Theta")
plt.ylabel("Sigma")
plt.title("SGD")
#plt.savefig("Gradiend_flow_&_SGD_1.pdf")
plt.show()
#%%

errors=(np.mean(pars[:,-1],axis=0))**2



#%%
print(((pars[:,1].T)).shape)
#%%

# In the following we test the coupled version em sampling.
N=10
x0_sca=1.2
x0=x0_sca+np.zeros(N)
l=9
T=2
t0=0
l_d=1
d=2**(l_d)
theta_true=-0.3
sigma_true=1.2
np.random.seed(7)
collection_input=[bdg.b_ou_1d,theta_true,bdg.Sig_ou_1d,sigma_true]
resamp_coef=1
l_max=10
x_true= bdg.gen_gen_data_1d(T,x0_sca,l_max,collection_input)
x_reg=bdg.cut(T,l_max,-l_d,x_true)[1:]
times=np.array(range(t0,int(T/d)+1))*d
l_times=np.arange(t0,T,2**(-l))
l_max_times=np.arange(t0,T,2**(-l_max))
plt.plot(times[1:],x_reg,label="True signal")
plt.plot(l_max_times,x_true[:-1],label="True complete signal")
sd_true=5e-1
np.random.seed(3)
obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)

x0_0=x0_sca
x0_1=x0_sca
x_p_0=np.random.normal(x0,2)
x_p_1=x_p_0.copy()+np.random.normal(0,0.2)
A_0=theta_true
A_1=A_0
fi_0=sigma_true
fi_1=fi_0
A_til_0=theta_true+0.2
A_til_1=A_til_0
fi_til_0=sigma_true
fi_til_1=fi_til_0
seed=0
print(l)
int_G_0,int_G_1,x_0,x_1=bdg.C_Bridge_1d(t0,x0_0,x0_1,T,x_p_0,x_p_1,bdg.b_ou_1d,A_0,A_1,\
bdg.Sig_ou_1d,fi_0,fi_1,bdg.b_ou_aux,A_til_0,A_til_1,\
bdg.Sig_ou_aux,fi_til_0,fi_til_1,bdg.r_quasi_normal_1d,[bdg.ou_sd,[A_til_0,fi_til_0]],\
[bdg.ou_sd,[A_til_1,fi_til_1]],bdg.H_quasi_normal,[bdg.ou_sd,[A_til_0,fi_til_0],A_til_0],[bdg.ou_sd,[A_til_1,fi_til_1],A_til_1],l,d,N,seed\
,crossed=False,backward=False,j_0=False,j_1=False,fd=False,N_pf=False,cond_seed_0=False,cond_seed_1=False)
"""

r_quasi_normal_1d,[bdg.ou_sd,[theta_0_aux,sigma_0_aux]],[bdg.ou_sd,[theta_0_aux,sigma_0_aux_fd]],\
        bdg.H_quasi_normal,[bdg.ou_sd,[theta_0_aux,sigma_0_aux],theta_0_aux],[bdg.ou_sd,[theta_0_aux,sigma_0_aux_fd],theta_0_aux],\
        bdg.sampling_ou,[theta_0_aux,sigma_0_aux],\
        obs,bdg.log_g_normal_den,sd_0, bdg.ou_trans_den,[theta_0_aux,sigma_0_aux],\
        bdg.Grad_log_aux_trans_ou_new,bdg.ou_trans_den, bdg.Grad_log_G_new,resamp_coef, l, d,N,seed,fd_rate,\
        mcmc_links,SGD_steps,gamma, alpha]
"""
#%%
print(x_1.shape)
print(x_p_0)
#%%
l_times=np.arange(t0,T,2**(-l))
#%%
l0_times=np.arange(t0,T,2**(-l+1))
print(l)
plt.plot(l0_times,x_0[:-1],label="Coupled path 0")
plt.plot(l_times,x_1[:-1],label="Coupled path 1")
#plt.legend()
print(x_1[-3:,0])
#%%

# TEST FOR THE FUNCTION C_PF_BRIDGE


N=5000
x0_sca=1.2
x0=x0_sca+np.zeros(N)
l=9
T=10
t0=0
l_d=1
d=2**(l_d)
theta_true=-0.3
sigma_true=1.2
np.random.seed(7)
collection_input=[bdg.b_ou_1d,theta_true,bdg.Sig_ou_1d,sigma_true]
resamp_coef=1
l_max=10
x_true= bdg.gen_gen_data_1d(T,x0_sca,l_max,collection_input)
x_reg=bdg.cut(T,l_max,-l_d,x_true)[1:]
times=np.array(range(t0,int(T/d)+1))*d
l_times=np.arange(t0,T,2**(-l))
l_max_times=np.arange(t0,T,2**(-l_max))
plt.plot(times[1:],x_reg,label="True signal")
plt.plot(l_max_times,x_true[:-1],label="True complete signal")
sd_true=5e-1
np.random.seed(3)
obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)

x0_0=x0_sca
x0_1=x0_sca
x_p_0=np.random.normal(x0,2)
x_p_1=x_p_0.copy()+np.random.normal(0,0.2)
A_0=theta_true
A_1=A_0
fi_0=sigma_true
fi_1=fi_0
A_til_0=theta_true+0.2
A_til_1=A_til_0
fi_til_0=sigma_true
fi_til_1=fi_til_0
seed=0
print(A_til_0,fi_til_0 )
print(l)
np.random.seed(22)
[log_weights_0,log_weights_1,int_Gs_0,int_Gs_1,x_pr_0,x_pr_1]=bdg.C_PF_bridge(t0,x0,T,bdg.b_ou_1d,A_0,\
bdg.Sig_ou_1d,fi_0,bdg.b_ou_aux,A_til_0,\
bdg.Sig_ou_aux,fi_til_0,bdg.r_quasi_normal_1d,[bdg.ou_sd,[A_til_0,fi_til_0]],\
bdg.H_quasi_normal,[bdg.ou_sd,[A_til_0,fi_til_0],A_til_0],bdg.rej_max_coup_ou,[A_til_0,fi_til_0,A_til_0,fi_til_0],\
obs,bdg.log_g_normal_den,sd_true,bdg.ou_trans_den,[A_til_0,fi_til_0],bdg.ou_trans_den,[A_til_0,fi_til_0], resamp_coef,l,d,N,seed)

#%%
s_sample=0
i=0
[theta,sigma,sd]=[theta_true,sigma_true,sd_true]
print(theta,sigma,sd)
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
x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
d_times=np.array(range(t0,int(T/d)+1))*d
#weights=pff.norm_logweights(lws,ax=3)
#PF_mean=np.mean(np.sum(x_prs*weights,axis=-1),axis=(0,1))
#l_times=np.arange(t0,T,2**(-l))
#l_max_times=np.arange(t0,T,2**(-l_max))
#print(times, l_times)
#x_mean=np.mean(ch_paths[s_sample,i*mcmc_links:(i+1)*mcmc_links],axis=0)
x_mean_0=np.mean(x_pr_0,axis=-1)
x_mean_1=np.mean(x_pr_1,axis=-1)
#print(x_mean.shape)
#plt.plot(d_times[1:],PF_mean,label="PF")
#plt.plot(d_times[1:],x_reg,label="True signal")
#plt.plot(l_max_times,x_true[:-1],label="True complete signal")
plt.plot(d_times,x_kf_smooth[:,0],label="KF smooth")
#print(x_kf_smooth[:,0])
#plt.plot(d_times,x_kf[:,0],label="KF")
#plt.plot(d_times[1:], obs,label="Observations")

plt.plot(d_times[1:], x_mean_0,label="PF_0")
plt.plot(d_times[1:], x_mean_1,label="PF_1")
plt.legend()  
#print(np.mean(new_Grads,axis=0))  
#print(Grad_log_lik)

#print(log_weights_0[-1],log_weights_1[-1])
print(x_mean_0[-5:],x_mean_1[-5:])
print(x_pr_0[-1,-5:],x_pr_1[-1,-5:])
"""
(t0,x0,T,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,\
    max_sample_funct,sample_pars,obs,log_g_den,g_den_par, aux_trans_den,atdp,\
    prop_trans_den, resamp_coef, l, d,N,seed,crossed=False)
    sd_true,\
    bdg.ou_trans_den,[theta_aux,sigma_aux],bdg.ou_trans_den,\
    resamp_coef,l,d, N,seed)
"""
#%%
L_max=12
L_min=2
eLes=np.array(range(L_min,L_max+1))
print(eLes)
samples=50
x_means=np.zeros((samples,len(eLes),2,int(T/d)))
for sample in range(samples):
    print("the sample is: ",sample)
    for i in range(len(eLes)):
        l=eLes[i]
        print("l is: ",l)
        [log_weights_0,log_weights_1,int_Gs_0,int_Gs_1,x_pr_0,x_pr_1]=bdg.C_PF_bridge(t0,x0,T,bdg.b_ou_1d,A_0,\
        bdg.Sig_ou_1d,fi_0,bdg.b_ou_aux,A_til_0,\
        bdg.Sig_ou_aux,fi_til_0,bdg.r_quasi_normal_1d,[bdg.ou_sd,[A_til_0,fi_til_0]],\
        bdg.H_quasi_normal,[bdg.ou_sd,[A_til_0,fi_til_0],A_til_0],bdg.rej_max_coup_ou,[A_til_0,fi_til_0,A_til_0,fi_til_0],\
        obs,bdg.log_g_normal_den,sd_true,bdg.ou_trans_den,[A_til_0,fi_til_0],bdg.ou_trans_den,[A_til_0,fi_til_0], resamp_coef,l,d,N,seed)

        x_means[sample,i,0]=np.mean(x_pr_0,axis=-1)
        x_means[sample,i,1]=np.mean(x_pr_1,axis=-1)


#%%
sms=np.mean((x_means[:,:,0,-1]-x_means[:,:,1,-1])**2,axis=0)       
plt.plot(eLes,sms)
refa=1/2**(eLes/2)
refb=1/2**(eLes)
plt.plot(eLes,refb*sms[0]/refb[0],label="$\Delta_l$")
plt.plot(eLes,refa*sms[0]/refa[0],label="$\Delta_l^{1/2}$")
plt.yscale("log")
plt.xlabel("l")
plt.ylabel("Strong error")
plt.legend()
plt.savefig("strong_error.pdf")

# %%
#%%
##########################################################################################
##########################################################################################
##########################################################################################

# TEST FOR THE FUNCTION C_COND_PF_BRIDGE_back_samp

x0_sca=1.2
x0=x0_sca
l=10
T=3
t0=0
l_d=0
d=2**(l_d)
theta_true=-0.3
sigma_true=1.2
np.random.seed(7)
collection_input=[bdg.b_ou_1d,theta_true,bdg.Sig_ou_1d,sigma_true]
resamp_coef=1
l_max=10
x_true= bdg.gen_gen_data_1d(T,x0_sca,l_max,collection_input)
x_reg=bdg.cut(T,l_max,-l_d,x_true)[1:]
times=np.array(range(t0,int(T/d)+1))*d
l_times=np.arange(t0,T,2**(-l))
l_max_times=np.arange(t0,T,2**(-l_max))
plt.plot(times[1:],x_reg,label="True signal")
plt.plot(l_max_times,x_true[:-1],label="True complete signal")
sd_true=1
np.random.seed(3)
d_times=np.array(range(t0+d,int(T/d)+1))*d
obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)
plt.plot(d_times, obs,label="Observations")
print(obs)
#%%
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
#B0=100
#Bes_l=7
#eBes=B0*2**np.arange(Bes_l)
#print(eBes)
theta=theta_true
sigma=sigma_true
theta_aux=theta+0.2
sigma_aux=sigma
sd=sd_true
start=time.time()
B=5000
samples=20
# interactive 1 samples=100
N=50
x0=x0_sca+np.zeros(N)
seed=2
l0=2
L_max=8
eLes=np.array(range(l0,L_max+1))
mcmc_mean=np.zeros((len(eLes),samples,2,int(T/d)))
pf_diffs=np.zeros((len(eLes),samples,int(T/d)))
pf_l=np.zeros((len(eLes),samples,2,int(T/d)))
ori_pf_l=np.zeros((len(eLes),samples,int(T/d)))
resamp_coef=1
for k in range(len(eLes)):
    l=eLes[k] 
    for i in range(samples):
        np.random.seed(i+10000)
        #print("Seed feeded to PF_bridge is: ",seed)
        [log_weights,int_Gs,x_pr]=bdg.PF_bridge(t0,x0,T,bdg.b_ou_1d,theta,bdg.Sig_ou_1d,sigma,bdg.b_ou_aux,theta_aux,\
        bdg.Sig_ou_aux,sigma_aux,bdg.r_quasi_normal_1d,[bdg.ou_sd,[theta_aux,sigma_aux]],bdg.H_quasi_normal,\
        [bdg.ou_sd,[theta_aux,sigma_aux],theta_aux],bdg.sampling_ou, [theta_aux,sigma_aux],obs,bdg.log_g_normal_den,sd,\
        bdg.ou_trans_den,[theta_aux,sigma_aux],bdg.ou_trans_den,resamp_coef,l,d, N,seed)
        #x_mean=np.sum(pff.norm_logweights(log_weights,ax=1)*x_pr,axis=-1)
        weights=pff.norm_logweights(log_weights[-1,:])
        #ori_pf_l[k,i]=np.sum(((pff.norm_logweights(log_weights,ax=1))*x_pr),axis=1)
        ori_pf_l[k,i]=np.mean((x_pr),axis=1)
        #print(weights.shape)
        index=np.random.choice(np.array(range(N)))
        cond_path=x_pr[:,index]
        cond_path_0=cond_path
        cond_path_1=cond_path
        cond_log_weights=log_weights[:,index]
        cond_int_G=int_Gs[:,index]
        seeds_cond=np.zeros((int(T/d),2),dtype=int)
        seeds_cond[:,0]=seed+np.array(range(int(T/d)))*int(int(2**l*d-1))
        seeds_cond[:,1]=index*np.ones(int(T/d))
        seeds_cond_0=seeds_cond
        seeds_cond_1=seeds_cond

        ch_paths=np.zeros((B,2,int(T/d)))
        comp_pf_diffs=np.zeros((B,int(T/d)))
        comp_pf_l=np.zeros((B,2,int(T/d)))
        ch_weights=np.zeros((B,2,int(T/d)))

        ch_whole_paths=np.zeros((B,2,int(T/d)))
        ch_whole_weights=np.zeros((B,2,int(T/d)))

        seed+=(int(T/d))*int(int(2**l*d-1))
        cond_whole_path=cond_path
        cond_whole_log_weights=cond_log_weights
        for b in range(B):
            print("The sample is: ",i," The batch is: ",b)
            print("The level is: ",l)
            """
            C_Cond_PF_bridge_back_samp(x_cond_0,x_cond_1,\
            seeds_cond_0,seeds_cond_1,t0,x0,T,b,A_0,A_1,Sig,fi_0,fi_1,b_til,A_til_0,A_til_1,Sig_til,fi_til_0,\
            fi_til_1,r,r_pars_0,r_pars_1,H,H_pars_0,H_pars_1,sample_funct,sample_pars,obs,\
            log_g_den,g_den_par_0,g_den_par_1, aux_trans_den,atdp_0,atdp_1,\
            prop_trans_den, ind_prop_trans_par_0,ind_prop_trans_par_1, l, d,N,seed,crossed=False):
            """
            #[log_weights_0,log_weights_1,x_pr_0,x_pr_1,new_lw_cond_0,new_lw_cond_1\
            #,new_int_G_cond_0,new_int_G_cond_1,new_x_cond_0,new_x_cond_1,new_seeds_cond_0 ,new_seeds_cond_1]
            # The first 4 argument of the C_cond_pf... function are irrelevant, they are a 
            # vestige of the previous version of the function.
            [log_weights_0,log_weights_1,x_pr_0,x_pr_1,cond_log_weights_0,cond_log_weights_1,\
            cond_int_G_0,cond_int_G_1,cond_path_0,cond_path_1,seeds_cond_0,seeds_cond_1]=\
            bdg.C_Cond_PF_bridge_back_samp(\
            cond_path_0,cond_path_1,seeds_cond_0,seeds_cond_1,t0,x0,\
            T,bdg.b_ou_1d,theta,theta,bdg.Sig_ou_1d,sigma,sigma,bdg.b_ou_aux,theta_aux,theta_aux,\
            bdg.Sig_ou_aux,sigma_aux,sigma_aux,bdg.r_quasi_normal_1d,[bdg.ou_sd,[theta_aux,sigma_aux]],\
            [bdg.ou_sd,[theta_aux,sigma_aux]],bdg.H_quasi_normal,\
            [bdg.ou_sd,[theta_aux,sigma_aux],theta_aux],[bdg.ou_sd,[theta_aux,sigma_aux],theta_aux],\
            bdg.rej_max_coup_ou, [theta_aux,sigma_aux,theta_aux,sigma_aux],obs,bdg.log_g_normal_den,sd,sd,\
            bdg.ou_trans_den,[theta_aux,sigma_aux],[theta_aux,sigma_aux],bdg.ou_trans_den,\
            [theta_aux,sigma_aux],[theta_aux,sigma_aux],l,d, N,seed,crossed=False)

            seed+=int((int(T/d))*int(int(2**l*d-1)))
            ch_paths[b]=[cond_path_0,cond_path_1]
            comp_pf_diffs[b]=np.sum((x_pr_0*pff.norm_logweights(log_weights_0,ax=1)\
            -x_pr_1*pff.norm_logweights(log_weights_1,ax=1)),axis=1)
            #comp_pf_diffs[b]=np.mean((x_pr_0\
            #-x_pr_1)**2,axis=1)
            comp_pf_l[b,1]=np.sum((pff.norm_logweights(log_weights_1,ax=1)*x_pr_1),axis=1)
            comp_pf_l[b,0]=np.sum((pff.norm_logweights(log_weights_0,ax=1)*x_pr_0),axis=1)           
    
        mcmc_mean[k,i]=np.mean(ch_paths,axis=0)
        pf_diffs[k,i]=np.mean(comp_pf_diffs**2,axis=0)
        pf_l[k,i]=np.mean(comp_pf_l,axis=0)
    #mcmc_mean[i,1]=np.mean(ch_whole_paths,axis=0)
end=time.time()
print(end-start)#%%
# what should I code now? 
# I want to get these results with a large number of samples so I can justify the 
# choice of parameters. 
# I want a comparation with multinomial sampling rather than backward sampling.

#%%

pf_l=   np.reshape(np.loadtxt("Observations&data/pf_l.txt"),(len(eLes),samples,2,int(T/d)))
ori_pf_l=np.reshape(np.loadtxt("Observations&data/ori_pf_l.txt"),(len(eLes),samples,int(T/d)))
pf_diffs=np.reshape(np.loadtxt("Observations&data/pf_diffs.txt"),(len(eLes),samples,int(T/d)))
mcmc_mean=np.reshape(np.loadtxt("Observations&data/mcmc_mean.txt"),(len(eLes),samples,2,int(T/d)))
dim=1
dim_o=1
print(theta, sigma,sd)
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
x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
#KF(xin,dim,dim_o,K,G,H,D,obs)
x_kf_2=bdg.KF(x0[0],dim,dim_o,K,G,H,D,obs)[0]
Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
d_times=np.array(range(t0,int(T/d)+1))*d
#weights=pff.norm_logweights(lws,ax=2)
#PF_mean=np.mean(np.sum(x_prs*weights,axis=-1),axis=0)
l=L_max
l_times=np.arange(t0,T,2**(-l))
l_max_times=np.arange(t0,T,2**(-l_max))
#print(times, l_times)
x_mean=np.mean(mcmc_mean,axis=1)[-1,1]
#print(x_mean.shape)
#plt.plot(d_times[1:],PF_mean,label="PF")
plt.plot(d_times[1:],x_reg,label="True signal")
#plt.plot(l_max_times,x_true[:-1],label="True complete signal")
plt.plot(d_times,x_kf_smooth[:,0],label="KF smooth")
plt.plot(d_times,x_kf[:,0],label="KF")
plt.plot(d_times[1:], obs,label="Observations")
plt.plot(d_times[1:], x_mean.T,label="PGibbs")
print(obs)
plt.legend()    

"""np.savetxt("Observations&data/pf_l.txt",(pf_l).flatten())
np.savetxt("Observations&data/ori_pf_l.txt",(ori_pf_l).flatten())
np.savetxt("Observations&data/pf_diffs.txt",(pf_diffs).flatten())
np.savetxt("Observations&data/mcmc_mean.txt",(mcmc_mean).flatten())"""
#%%
#%%
print("T is: ",T)
print("eLes is: ",eLes)
print("samples, B and N are: ",samples,B,N) 
print(pf_l.shape)
print((np.mean(pf_l,axis=1)[-1]))
r_bias_eles=np.abs(np.mean(pf_l[:,:,1]-pf_l[:,:,0],axis=1))[:,-1]
bias_eles=np.abs(np.mean(pf_l[:,:,0],axis=1)[:,-1]-x_kf[-1,0])
bias_1_eles=np.abs(np.mean(pf_l[:,:,1],axis=1)[:,-1]-x_kf[-1,0])
var_eles=np.var(pf_l[:,:,1]-x_kf[1:,0],axis=1)[:,-1]
bias_eles_up=bias_eles+np.sqrt(var_eles)*1.96/np.sqrt(samples)
bias_eles_lb=bias_eles-np.sqrt(var_eles)*1.96/np.sqrt(samples)
ori_bias_eles=np.abs(np.mean(ori_pf_l[:,:],axis=1)[:,-1]-x_kf[-1,0])
var_ori_bias_eles=np.var(ori_pf_l[:,:]-x_kf[1:,0],axis=1)[:,-1]
bias_eles_up=bias_eles+np.sqrt(var_eles)*1.96/np.sqrt(samples)
bias_eles_lb=bias_eles-np.sqrt(var_eles)*1.96/np.sqrt(samples)
print(bias_eles_lb)
plt.plot(eLes,ori_bias_eles,label="Original PF bias")
#plt.plot(eLes,bias_eles_2,label="Second PF bias")
plt.plot(eLes,bias_eles_up)
plt.plot(eLes,bias_eles_lb)
plt.plot(eLes,r_bias_eles,label="Richardson bias")
print(x_kf)
plt.plot(eLes,bias_eles+bias_1_eles,label="PF bias")
plt.plot(eLes,2**(eLes[0])/2**(eLes)*r_bias_eles[0],label="$\Delta_l$")
plt.plot(eLes,2**(eLes[0])/2**(eLes)*bias_eles[0],label="$\Delta_l$")
plt.yscale("log")
plt.legend()
#print(PF_mean-x_kf[1:,0])
# RESULTS OF THE TEST:
# The kalman filter coincides with the particle filter, at least visually. 
#%%
i=0
s_R_bias=np.abs(np.mean(mcmc_mean[:,:,0]-mcmc_mean[:,:,1],axis=1))[:,i]  #smoothing richardson bias
sum_s_R_bias=np.sum(np.abs(np.mean(mcmc_mean[:,:,0]-mcmc_mean[:,:,1],axis=1)),axis=-1)
s_bias=np.abs(np.mean(mcmc_mean[:,:,1]-x_kf_smooth[1:,0],axis=1))[:,i]
plt.plot(eLes,s_R_bias,label="Smoothing Richardson bias")
plt.plot(eLes,sum_s_R_bias,label="Sum smoothing Richardson bias")
plt.plot(eLes,s_bias,label="Smoothing bias")
plt.plot(eLes,2**(eLes[0])/2**(eLes)*sum_s_R_bias[0],label="$\Delta_l$")
plt.plot(eLes,2**(eLes[0]/2)/2**(eLes/2)*sum_s_R_bias[0],label="$\Delta_l^{1/2}$")
plt.legend()
plt.yscale("log")
#%%
"""
error_wpf=np.mean(pf_diffs,axis=1)[:,-1] # wpf stands for whole particle filter(not just the chosen conditional sample)
error_wpf_sum=np.sum(np.mean(pf_diffs,axis=1),axis=-1)
plt.plot(eLes,error_wpf,label="Error_wpf")
plt.plot(eLes,error_wpf_sum,label="Error_wpf_sum")
plt.plot(eLes, 2**(eLes[0])/2**(eLes)*error_wpf[0],label="$\Delta_l$")
plt.plot(eLes, 2**(eLes[0])/2**(eLes)*error_wpf_sum[0],label="$\Delta_l$")
#plt.plot(eLes, 2**(eLes[0]/2)/2**(eLes/2)*error_wpf[0],label="$\Delta_l^{1/2}$")
#plt.plot(eLes, 2**(eLes[0]/2)/2**(eLes/2)*error_wpf_sum[0],label="$\Delta_l^{1/2}$")
"""
#"""
error=np.mean((mcmc_mean[:,:,0,-1]-mcmc_mean[:,:,1,-1])**2,axis=1)
error_sum=np.sum(np.mean((mcmc_mean[:,:,0,:]-mcmc_mean[:,:,1,:])**2,axis=1),axis= -1)
var_error=np.var((mcmc_mean[:,:,0,-1]-mcmc_mean[:,:,1,-1])**2,axis=1)
var_error_sum=np.var(np.sum((mcmc_mean[:,:,0]-mcmc_mean[:,:,1])**2,axis=-1),axis=1)
error_up=error+np.sqrt(var_error)*1.96/np.sqrt(samples)  
error_lb=error-np.sqrt(var_error)*1.96/np.sqrt(samples)
error_sum_up=error_sum+np.sqrt(var_error_sum)*1.96/np.sqrt(samples)
error_sum_lb=error_sum-np.sqrt(var_error_sum)*1.96/np.sqrt(samples)
#MSE=np.mean((x_mean-x_kf[1:,0])**2,axis=1)[:,comp]
plt.plot(eLes, 2**(eLes[0]/2)/2**(eLes/2)*error[0],label="$\Delta_l^{1/2}$")
plt.plot(eLes, 2**(eLes[0])/2**(eLes)*error[0],label="$\Delta_l$")
plt.plot(eLes, 2**(eLes[0])/2**(eLes)*error_sum[0],label="$\Delta_l$")
plt.plot(eLes,error_sum,label="Error sum")
plt.plot(eLes,error_up,label="Error up")
plt.plot(eLes,error_sum_up,label="Error sum up")
plt.plot(eLes,error_sum_lb,label="Error sum lb")
plt.plot(eLes,error_lb,label="Error lb")
plt.plot(eLes,error,label="Error")
print(error)
#"""
"""
sm_r_bias=np.abs(np.mean((mcmc_mean[:,:,0,-1]-mcmc_mean[:,:,1,-1]),axis=1))
var_sm_r_bias=np.var((mcmc_mean[:,:,0,-1]-mcmc_mean[:,:,1,-1]),axis=1)
var_sm_r_bias=np.var(np.sum((mcmc_mean[:,:,0]-mcmc_mean[:,:,1]),axis=-1),axis=1)
sm_r_bias_up=sm_r_bias+np.sqrt(var_sm_r_bias)*1.96/np.sqrt(samples)  
sm_r_bias_lb=sm_r_bias-np.sqrt(var_sm_r_bias)*1.96/np.sqrt(samples)
plt.plot(eLes, 2**(eLes[0]/2)/2**(eLes/2)*sm_r_bias[0],label="$\Delta_l^{1/2}$")
plt.plot(eLes,sm_r_bias,label="sm_r_bias")
plt.plot(eLes,sm_r_bias_up,label="sm_r_bias ub")
plt.plot(eLes,sm_r_bias_lb,label="sm_r_bias lb")
"""
plt.yscale("log")
plt.legend()
#%%
#ndc
# TEST FOR THE FUNCTION Grad_Cond_PF_bridge_back_samp
if True:
    
    x0_sca=1.2
    x0=x0_sca
    l=10
    T=10
    t0=0
    l_d=0
    d=2**(l_d)
    theta_true=-0.3
    sigma_true=0.8
    sd_true=0.8
    np.random.seed(7)
    collection_input=[bdg.b_ou_1d,theta_true,bdg.Sig_ou_1d,sigma_true]
    resamp_coef=1
    l_max=10
    x_true= bdg.gen_gen_data_1d(T,x0_sca,l_max,collection_input)
    x_reg=bdg.cut(T,l_max,-l_d,x_true)[1:]
    times=np.array(range(t0,int(T/d)+1))*d
    l_times=np.arange(t0,T,2**(-l))
    l_max_times=np.arange(t0,T,2**(-l_max))
    np.random.seed(1007)
    d_times=np.array(range(t0+d,int(T/d)+1))*d
    obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)
    resamp_coef=1
    N=50
    start=time.time()
    mcmc_links=600*10
    #mcmc_links=10
    SGD_steps=1
    B=mcmc_links*SGD_steps
    fd=1e-8
    theta_in=-0.8
    sigma_in=1
    sd_in=1
    theta_in_fd=theta_in+fd
    sigma_in_fd=sigma_in+fd
    sigma_in_aux=sigma_in
    theta_in_aux=theta_in+0.2
    sigma_in_aux_fd=sigma_in_aux+fd
    
    #arg_cm=int(sys.argv[1])
    #arg_cm=32
    samples=40
    seed=4253#+samples*(arg_cm-1)
    #samples=2
    gamma=0.1
    alpha=0.5
    seed=2393
    x0=x0_sca+np.zeros(N)
    l0=3
    L_max=10
    # come here
    eLes=np.array(range(l0,L_max+1))    
#%%
for k in range(len(eLes)):
    l=eLes[k] 
    for i in range(samples):
        np.random.seed(i+10000)
        [log_weights,int_Gs,x_pr]=bdg.PF_bridge(t0,x0,T,bdg.b_ou_1d,theta,bdg.Sig_ou_1d,sigma,bdg.b_ou_aux,theta_aux,\
        bdg.Sig_ou_aux,sigma_aux,bdg.r_quasi_normal_1d,[bdg.ou_sd,[theta_aux,sigma_aux]],bdg.H_quasi_normal,\
        [bdg.ou_sd,[theta_aux,sigma_aux],theta_aux],bdg.sampling_ou, [theta_aux,sigma_aux],obs,bdg.log_g_normal_den,sd,\
        bdg.ou_trans_den,[theta_aux,sigma_aux],bdg.ou_trans_den,resamp_coef,l,d, N,seed)
        #x_mean=np.sum(pff.norm_logweights(log_weights,ax=1)*x_pr,axis=-1)
        weights=pff.norm_logweights(log_weights[-1,:])
        #ori_pf_l[k,i]=np.sum(((pff.norm_logweights(log_weights,ax=1))*x_pr),axis=1)
        ori_pf_l[k,i]=np.mean((x_pr),axis=1)
        #print(weights.shape)
        index=np.random.choice(np.array(range(N)))
        cond_path=x_pr[:,index]
        cond_log_weights=log_weights[:,index]
        cond_int_G=int_Gs[:,index]
        seeds_cond=np.zeros((int(T/d),2),dtype=int)
        seeds_cond[:,0]=seed+np.array(range(int(T/d)))*int(int(2**l*d-1))
        seeds_cond[:,1]=index*np.ones(int(T/d))
        lw_cond=log_weights[:,index]
        int_Gs_cond=int_Gs[:,index]

        ch_paths=np.zeros((B,int(T/d)))
        comp_pf_diffs=np.zeros((B,int(T/d)))
        comp_pf_l=np.zeros((B,2,int(T/d)))
        ch_weights=np.zeros((B,2,int(T/d)))
        grads=np.zeros((B,3))
        ch_whole_paths=np.zeros((B,2,int(T/d)))
        ch_whole_weights=np.zeros((B,2,int(T/d)))

        seed+=(int(T/d))*int(int(2**l*d-1))
        cond_whole_path=cond_path
        cond_whole_log_weights=cond_log_weights
        for b in range(B):
            
            print("The sample is: ",i," The batch is: ",b)
            print("The level is: ",l)
            [log_weights,x_pr,cond_log_weights,\
            cond_int_G,cond_path,seeds_cond,Grads]=\
            bdg.Grad_Cond_PF_bridge_back_samp_an(lw_cond,int_Gs_cond,\
            cond_path,seeds_cond,t0,x0,\
            T,bdg.b_ou_1d,theta,theta_fd,bdg.Sig_ou_1d,sigma,sigma_fd,\
            bdg.b_ou_aux,theta_aux,bdg.Sig_ou_aux,sigma_aux,sigma_aux_fd,\
            bdg.r_quasi_normal_1d,[bdg.ou_sd,[theta_aux,sigma_aux]],\
            [bdg.ou_sd,[theta_aux,sigma_aux_fd]],bdg.H_quasi_normal,\
            [bdg.ou_sd,[theta_aux,sigma_aux],theta_aux],\
            [bdg.ou_sd,[theta_aux,sigma_aux_fd],theta_aux],\
            bdg.sampling_ou, [theta_aux,sigma_aux],obs,bdg.log_g_normal_den,sd,\
            bdg.ou_trans_den,[theta_aux,sigma_aux],bdg.Grad_log_aux_trans_ou_new,\
            bdg.ou_trans_den,bdg.Grad_log_G_new,resamp_coef,l,d, N,seed,fd)
            """
            (lw_cond,int_Gs_cond,x_cond,seeds_cond,t0,x0,T,b,A,A_fd,Sig,fi,fi_fd,b_til,A_til,Sig_til,fi_til,\
            fi_til_fd,r,r_pars,r_pars_fd,H,H_pars,H_pars_fd,sample_funct,sample_pars,obs,log_g_den,g_den_par, aux_trans_den,atdp,\
            Grad_log_aux_trans,prop_trans_den, Grad_log_G,resamp_coef, l, d,N,seed,fd_rate,crossed=False):
            """
            seed+=int((int(T/d))*int(int(2**l*d-1)))
            ch_paths[b]=cond_path
            #comp_pf_diffs[b]=np.sum((x_pr_0*pff.norm_logweights(log_weights_0,ax=1)\
            #-x_pr_1*pff.norm_logweights(log_weights_1,ax=1)),axis=1)
            #comp_pf_l[b,1]=np.sum((pff.norm_logweights(log_weights_1,ax=1)*x_pr_1),axis=1)
            #comp_pf_l[b,0]=np.sum((pff.norm_logweights(log_weights_0,ax=1)*x_pr_0),axis=1)           
            grads[b]=Grads
        mcmc[k,i]=ch_paths
        mcmc_mean[k,i]=np.mean(ch_paths,axis=0)
        #pf_diffs[k,i]=np.mean(comp_pf_diffs**2,axis=0)
        #pf_l[k,i]=np.mean(comp_pf_l,axis=0)
        Grads_file[k,i]=np.mean(grads,axis=0)

end=time.time()
print(end-start)
# %%

"""labels=np.array(range(1,16))
i=0
Grads_file=np.reshape(np.loadtxt("Observationsdata/data_grad_bias/Prl_SGD_ou_bridge_Grads_va"+str(labels[i])+".txt",dtype=float),(len(eLes),samples,B,3)) 
ch_paths_file=np.reshape(np.loadtxt("Observationsdata/data_grad_bias/Prl_SGD_ou_bridge_ch_paths_va"+str(labels[i])+".txt",dtype=float),(len(eLes),samples,B,int(T/d)))
"""
v="singleb3v1"
Grads_file=np.reshape(np.loadtxt("Observationsdata/data2/Prl_SGD_ou_bridge_Grads_v"+v+".txt",dtype=float),(len(eLes),samples,B,3)) 
ch_paths_file=np.reshape(np.loadtxt("Observationsdata/data2/Prl_SGD_ou_bridge_ch_paths_v"+v+".txt",dtype=float),(len(eLes),samples,B,int(T/d)))
#%%
#"""
dim=1
dim_o=1
theta,sigma,sd=theta_in,sigma_in,sd_in
#---------------------------------------------
K=np.array([[np.exp(d*theta)]])
G=np.array([[sigma*np.sqrt((np.exp(2*d*theta)-1)/(2*theta))]])
H=np.array([[1]])
D=np.array([[sd]])
Grad_S=np.array([[[0]],[[0]],[[1]]],dtype=float)
Grad_R_sigma_s=(np.exp(2*theta*d)-1)/(2*theta)
Grad_R_theta=(sigma**2/(2*theta**2))*(1-np.exp(2*theta*d)+2*d*theta*np.exp(2*theta*d))
Grad_R=np.zeros((3,1,1),dtype=float)
Grad_R[0,0,0]=Grad_R_theta
Grad_R[1,0,0]=Grad_R_sigma_s
Grad_K=np.array([[[d*np.exp(d*theta)]],[[0]],[[0]]],dtype=float)
x_kf,x_kf_smooth,Grad_log_lik_an=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
x_kf_2=bdg.KF(x0[0],dim,dim_o,K,G,H,D,obs)[0]
Grad_log_lik_an[1,0,0]=2*Grad_log_lik_an[1,0,0]*sigma

#"""
d_times=np.array(range(t0,int(T/d)+1))*d
#weights=pff.norm_logweights(lws,ax=2)
#PF_mean=np.mean(np.sum(x_prs*weights,axis=-1),axis=0)
l=L_max
l_times=np.arange(t0,T,2**(-l))
l_max_times=np.arange(t0,T,2**(-l_max))
#print(times, l_times)
x_mean=np.mean(ch_paths_file,axis=(1,2))[-1]
#print(x_mean.shape)
#plt.plot(d_times[1:],PF_mean,label="PF")
plt.plot(d_times[1:],x_reg,label="True signal")
#plt.plot(l_max_times,x_true[:-1],label="True complete signal")
plt.plot(d_times,x_kf_smooth[:,0],label="KF smooth")
#plt.plot(d_times,x_kf[:,0],label="KF")
plt.plot(d_times[1:], obs,label="Observations")
plt.plot(d_times[1:], x_mean.T,label="PGibbs")
print(obs)
plt.legend()    
# come back
#%%
Grid_p=1
thetas=np.array([theta_in])
sigmas=np.array([sigma_in])
sds=np.array([sd_in])
Grid=np.stack((thetas,sigmas,sds))
theta_aux=thetas+0.2
sigma_aux=sigmas
[theta_0,sigma_0,sd_0]=[theta,sigma,sd]
Grads_eles=np.zeros((len(eLes),3))
dim=1
dim_o=1
for i in range(len(eLes)):
        l_dis=eLes[i]
        #K=np.array([[np.exp(d*theta)]])
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
        x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
        Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
        Grads_eles[i]=Grad_log_lik[:,0,0]
#%%
par_n=2
#(len(eLes),samples,B,3)
bh=int(0*B/100)
print("The gradient is: ",Grad_log_lik_an[par_n])
print(np.mean(Grads_file[:,:,bh:,par_n],axis=(1,2)))
grad_bias=np.abs(np.mean(Grads_file[:,:,bh:,par_n]-Grad_log_lik_an[par_n],axis=(1,2)))
var_grad_bias=np.var(Grads_file[:,:,bh:,par_n]-Grad_log_lik_an[par_n],axis=(1,2))
r_grad_bias=np.abs(np.mean(Grads_file[1:,:,bh:,par_n]-Grads_file[:-1,:,bh:,par_n],axis=(1,2)))
grad_bias_up=grad_bias+np.sqrt(var_grad_bias)*1.96/np.sqrt(samples*(mcmc_links-bh))
grad_bias_lb=grad_bias-np.sqrt(var_grad_bias)*1.96/np.sqrt(samples*(mcmc_links-bh))
#theo_bias=np.abs(Grads_eles[:,par_n]-Grad_log_lik_an[par_n,0,0])
#plt.plot(eLes,theo_bias,label="Theoretical bias")
plt.plot(eLes,grad_bias,label="Gradient bias")
plt.plot(eLes,grad_bias_up,label="UB")
plt.plot(eLes,grad_bias_lb,label="LB")
plt.plot(eLes,2**(eLes[0])/2**(eLes)*grad_bias[0],label="$\Delta_l$")
plt.plot(eLes[1:],r_grad_bias,label="Richardson gradient bias")
plt.yscale("log")
plt.legend()
#%%
i=9
#len(eLes),samples,B,int(T/d))
print(v)
bh=int(0*B/100)
print(np.mean(ch_paths_file[:,:,bh:],axis=(1,2))[:,i])
print(x_kf_smooth[i+1,0])
s_R_bias=np.abs(np.mean(ch_paths_file[1:,:,bh:]-ch_paths_file[:-1,:,bh:],axis=(1,2)))[:,i]  #smoothing richardson bias
sum_s_R_bias=np.sum(np.abs(np.mean(ch_paths_file[1:,:,bh:]-ch_paths_file[:-1,:,bh:],axis=(1,2))),axis=1)
s_bias=np.abs(np.mean(ch_paths_file[:,:,bh:]-x_kf_smooth[1:,0],axis=(1,2)))[:,i]
var_bias=np.var(ch_paths_file[:,:,bh:]-x_kf_smooth[1:,0],axis=(1,2))[:,i]
s_bias_up=s_bias+np.sqrt(var_bias)*1.96/np.sqrt(samples*mcmc_links)
s_bias_lb=s_bias-np.sqrt(var_bias)*1.96/np.sqrt(samples*mcmc_links)
plt.plot(eLes[1:],s_R_bias,label="Smoothing Richardson bias")
#plt.plot(eLes[1:],sum_s_R_bias,label="Sum smoothing Richardson bias")
plt.plot(eLes,s_bias,label="Smoothing bias")
plt.plot(eLes,s_bias_up,label="Smoothing bias up")
plt.plot(eLes,s_bias_lb,label="Smoothing bias lb")
plt.plot(eLes,2**(eLes[0])/2**(eLes)*sum_s_R_bias[0],label="$\Delta_l$")
plt.plot(eLes,2**(eLes[0]/2)/2**(eLes/2)*sum_s_R_bias[0],label="$\Delta_l^{1/2}$")
plt.legend()
print(s_bias_lb)
plt.yscale("log")
    #%%

#%%
# TEST FOR THE FUNCTION C_Grad_Cond_PF_bridge_back_samp
if True:
    x0_sca=1.2
    x0=x0_sca
    l=10
    T=3
    t0=0
    l_d=0
    d=2**(l_d)
    theta_true=-0.25
    sigma_true=0.8
    np.random.seed(40)
    collection_input=[bdg.b_ou_1d,theta_true,bdg.Sig_ou_1d,sigma_true]
    resamp_coef=1
    l_max=10
    x_true= bdg.gen_gen_data_1d(T,x0_sca,l_max,collection_input)
    x_reg=bdg.cut(T,l_max,-l_d,x_true)[1:]
    times=np.array(range(t0,int(T/d)+1))*d
    l_times=np.arange(t0,T,2**(-l))
    l_max_times=np.arange(t0,T,2**(-l_max))
    #plt.plot(times[1:],x_reg,label="True signal")
    #plt.plot(l_max_times,x_true[:-1],label="True complete signal")
    sd_true=1.2
    np.random.seed(67)
    d_times=np.array(range(t0+d,int(T/d)+1))*d
    obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)
    theta=theta_true
    sigma=sigma_true
    theta_aux=theta+0.2
    sigma_aux=sigma
    sd=sd_true
    fd=1e-6
    theta_fd_0=theta_true+fd
    theta_fd_1=theta_true+fd*0.5
    sigma_fd_0=sigma_true+fd
    sigma_fd_1=sigma_true+fd*0.5
    sigma_aux_fd_0=sigma_aux+fd
    sigma_aux_fd_1=sigma_aux+fd*0.5
    start=time.time()
    B=500*15*2
    samples=30
    # interactive 1 samples=100
    N=50
    x0=x0_sca+np.zeros(N)
    seed=2985
    l0=2
    L_max=10
    eLes=np.array(range(l0,L_max+1))

mcmc_mean=np.zeros((len(eLes),samples,2,int(T/d)))
mcmc=np.zeros((len(eLes),samples,B,2,int(T/d)))
grads_mean=np.zeros((len(eLes),samples,2,3))
pf_diffs=np.zeros((len(eLes),samples,int(T/d)))
pf_l=np.zeros((len(eLes),samples,2,int(T/d)))
ori_pf_l=np.zeros((len(eLes),samples,int(T/d)))
resamp_coef=1
#%%
ch_paths_file=np.reshape(np.loadtxt("Observations&data/Prl_C_Grad_chain_ch_paths_v5.txt",dtype=float),(len(eLes),samples,2,int(T/d)))   
Grads_file=np.reshape(np.loadtxt("Observations&data/Prl_C_Grad_chain_Grads_v5.txt",dtype=float),(len(eLes),samples,2,3))
#%%

for k in range(len(eLes)):
    l=eLes[k] 
    for i in range(samples):
        np.random.seed(i+10000)
        #print("Seed feeded to PF_bridge is: ",seed)
        [log_weights,int_Gs,x_pr]=bdg.PF_bridge(t0,x0,T,bdg.b_ou_1d,theta,bdg.Sig_ou_1d,sigma,bdg.b_ou_aux,theta_aux,\
        bdg.Sig_ou_aux,sigma_aux,bdg.r_quasi_normal_1d,[bdg.ou_sd,[theta_aux,sigma_aux]],bdg.H_quasi_normal,\
        [bdg.ou_sd,[theta_aux,sigma_aux],theta_aux],bdg.sampling_ou, [theta_aux,sigma_aux],obs,bdg.log_g_normal_den,sd,\
        bdg.ou_trans_den,[theta_aux,sigma_aux],bdg.ou_trans_den,resamp_coef,l,d, N,seed)
        #x_mean=np.sum(pff.norm_logweights(log_weights,ax=1)*x_pr,axis=-1)
        weights=pff.norm_logweights(log_weights[-1,:])
        #ori_pf_l[k,i]=np.sum(((pff.norm_logweights(log_weights,ax=1))*x_pr),axis=1)
        ori_pf_l[k,i]=np.mean((x_pr),axis=1)
        #print(weights.shape)
        index=np.random.choice(np.array(range(N)))
        cond_path=x_pr[:,index]
        cond_path_0=cond_path
        cond_path_1=cond_path
        cond_log_weights=log_weights[:,index]
        cond_int_G=int_Gs[:,index]
        seeds_cond=np.zeros((int(T/d),2),dtype=int)
        seeds_cond[:,0]=seed+np.array(range(int(T/d)))*int(int(2**l*d-1))
        seeds_cond[:,1]=index*np.ones(int(T/d))
        seeds_cond_0=seeds_cond
        seeds_cond_1=seeds_cond

        ch_paths=np.zeros((B,2,int(T/d)))
        comp_pf_diffs=np.zeros((B,int(T/d)))
        comp_pf_l=np.zeros((B,2,int(T/d)))
        ch_weights=np.zeros((B,2,int(T/d)))
        grads=np.zeros((B,2,3))
        ch_whole_paths=np.zeros((B,2,int(T/d)))
        ch_whole_weights=np.zeros((B,2,int(T/d)))

        seed+=(int(T/d))*int(int(2**l*d-1))
        cond_whole_path=cond_path
        cond_whole_log_weights=cond_log_weights
        for b in range(B):
            print("The sample is: ",i," The batch is: ",b)
            print("The level is: ",l)
            """
            [log_weights_0,log_weights_1,x_pr_0,x_pr_1,cond_log_weights_0,cond_log_weights_1,\
            cond_int_G_0,cond_int_G_1,cond_path_0,cond_path_1,seeds_cond_0,seeds_cond_1,Grads_0,Grads_1]
            """
            [log_weights_0,log_weights_1,x_pr_0,x_pr_1,cond_log_weights_0,cond_log_weights_1,\
            cond_int_G_0,cond_int_G_1,cond_path_0,cond_path_1,seeds_cond_0,seeds_cond_1,Grads_0,Grads_1]=\
            bdg.C_Grad_Cond_PF_bridge_back_samp(\
            cond_path_0,cond_path_1,seeds_cond_0,seeds_cond_1,t0,x0,\
            T,bdg.b_ou_1d,theta,theta,theta_fd_0,theta_fd_1,bdg.Sig_ou_1d,sigma,sigma,sigma_fd_0,sigma_fd_1,\
            bdg.b_ou_aux,theta_aux,theta_aux,bdg.Sig_ou_aux,sigma_aux,sigma_aux,sigma_aux_fd_0,sigma_aux_fd_1,\
            bdg.r_quasi_normal_1d,[bdg.ou_sd,[theta_aux,sigma_aux]],[bdg.ou_sd,[theta_aux,sigma_aux]],\
            [bdg.ou_sd,[theta_aux,sigma_aux_fd_0]],[bdg.ou_sd,[theta_aux,sigma_aux_fd_1]],bdg.H_quasi_normal,\
            [bdg.ou_sd,[theta_aux,sigma_aux],theta_aux],[bdg.ou_sd,[theta_aux,sigma_aux],theta_aux],\
            [bdg.ou_sd,[theta_aux,sigma_aux_fd_0],theta_aux],[bdg.ou_sd,[theta_aux,sigma_aux_fd_1],theta_aux],\
            bdg.rej_max_coup_ou, [theta_aux,sigma_aux,theta_aux,sigma_aux],obs,bdg.log_g_normal_den,sd,sd,\
            bdg.ou_trans_den,[theta_aux,sigma_aux],[theta_aux,sigma_aux],bdg.Grad_log_aux_trans_ou_new,\
            bdg.ou_trans_den,[theta_aux,sigma_aux],[theta_aux,sigma_aux],bdg.Grad_log_G_new,l,d, N,seed,fd,crossed=False)
            print("The shape of Grads_0 is: ",Grads_0.shape)    
            seed+=int((int(T/d))*int(int(2**l*d-1)))
            ch_paths[b]=[cond_path_0,cond_path_1]
            comp_pf_diffs[b]=np.sum((x_pr_0*pff.norm_logweights(log_weights_0,ax=1)\
            -x_pr_1*pff.norm_logweights(log_weights_1,ax=1)),axis=1)
            comp_pf_l[b,1]=np.sum((pff.norm_logweights(log_weights_1,ax=1)*x_pr_1),axis=1)
            comp_pf_l[b,0]=np.sum((pff.norm_logweights(log_weights_0,ax=1)*x_pr_0),axis=1)           
            grads[b]=np.array([Grads_0,Grads_1])
        mcmc[k,i]=ch_paths
        mcmc_mean[k,i]=np.mean(ch_paths,axis=0)
        pf_diffs[k,i]=np.mean(comp_pf_diffs**2,axis=0)
        pf_l[k,i]=np.mean(comp_pf_l,axis=0)
        grads_mean[k,i]=np.mean(grads,axis=0)

end=time.time()
print(end-start)
# %%
dim=1
dim_o=1
print(theta, sigma,sd)
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
x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
#KF(xin,dim,dim_o,K,G,H,D,obs)
x_kf_2=bdg.KF(x0[0],dim,dim_o,K,G,H,D,obs)[0]
Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
d_times=np.array(range(t0,int(T/d)+1))*d
#weights=pff.norm_logweights(lws,ax=2)
#PF_mean=np.mean(np.sum(x_prs*weights,axis=-1),axis=0)
l=L_max
l_times=np.arange(t0,T,2**(-l))
l_max_times=np.arange(t0,T,2**(-l_max))
#print(times, l_times)
x_mean=np.mean(mcmc_mean,axis=1)[-1,1]
#print(x_mean.shape)
#plt.plot(d_times[1:],PF_mean,label="PF")
plt.plot(d_times[1:],x_reg,label="True signal")
#plt.plot(l_max_times,x_true[:-1],label="True complete signal")
plt.plot(d_times,x_kf_smooth[:,0],label="KF smooth")
plt.plot(d_times,x_kf[:,0],label="KF")
plt.plot(d_times[1:], obs,label="Observations")
plt.plot(d_times[1:], x_mean.T,label="PGibbs")
print(obs)
plt.legend()
#%%

Grid_p=1
thetas=np.array([theta])
sigmas=np.array([sigma])
sds=np.array([sd])
Grid=np.stack((thetas,sigmas,sds))
theta_aux=thetas+0.2
sigma_aux=sigmas
[theta_0,sigma_0,sd_0]=[theta,sigma,sd]
Grads_eles=np.zeros((len(eLes),3))
dim=1
dim_o=1
print(eLes)
for i in range(len(eLes)):
        l_dis=eLes[i]
        #K=np.array([[np.exp(d*theta)]])
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
        x_kf,x_kf_smooth,Grad_log_lik_l=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
        Grad_log_lik_l[1,0,0]=2*Grad_log_lik_l[1,0,0]*sigma
        Grads_eles[i]=Grad_log_lik_l[:,0,0]

#%%
par_n=0
lev=1 # lev in {0,1}
#print(grads_mean[:,:,0,par_n])
#print(grads_mean[:,:,1,par_n])
print(Grad_log_lik[par_n])
print((np.mean(Grads_file[:,:,lev,par_n],axis=1)))
print(Grads_eles[:,par_n])
#plt.plot(eLes,np.abs(np.mean(Grads_file[:,:,lev,par_n],axis=1)-Grad_log_lik[par_n])[0],label="Gradient bias")
grad_bias=np.abs(np.mean(Grads_file[:,:,lev,par_n]-Grad_log_lik[par_n],axis=1))
r_grad_bias=np.abs(np.mean(Grads_file[:,:,0,par_n]-Grads_file[:,:,1,par_n],axis=1))

var_grad_bias=np.var(Grads_file[:,:,lev,par_n]-Grad_log_lik[par_n],axis=1)

grad_bias_up=grad_bias+np.sqrt(var_grad_bias)*1.96/np.sqrt(samples)
grad_bias_lb=grad_bias-np.sqrt(var_grad_bias)*1.96/np.sqrt(samples)
theo_bias=np.abs(Grads_eles[:,par_n]-Grad_log_lik[par_n,0,0])
print(theo_bias)
plt.plot(eLes,theo_bias,label="Theoretical bias")
plt.plot(eLes,grad_bias,label="Gradient bias")
plt.plot(eLes,grad_bias_up,label="UB")
plt.plot(eLes,grad_bias_lb,label="LB")

plt.plot(eLes,r_grad_bias,label="Richardson gradient bias")
plt.plot(eLes,2**(eLes[0])/2**(eLes)*grad_bias[0],label="$\Delta_l$")
#plt.plot(eLes,r_grad_bias,label="Richardson gradient bias")
plt.yscale("log")
plt.legend()
#%%
i=-1
s_R_bias=np.abs(np.mean(mcmc_mean[:,:,0]-mcmc_mean[:,:,1],axis=1))[:,i]  #smoothing richardson bias
sum_s_R_bias=np.sum(np.abs(np.mean(mcmc_mean[:,:,0]-mcmc_mean[:,:,1],axis=1)),axis=-1)
s_bias=np.abs(np.mean(mcmc_mean[:,:,0]-x_kf_smooth[1:,0],axis=1))[:,i]
plt.plot(eLes,s_R_bias,label="Smoothing Richardson bias")
plt.plot(eLes,sum_s_R_bias,label="Sum smoothing Richardson bias")
plt.plot(eLes,s_bias,label="Smoothing bias")
plt.plot(eLes,2**(eLes[0])/2**(eLes)*sum_s_R_bias[0],label="$\Delta_l$")
plt.plot(eLes,2**(eLes[0]/2)/2**(eLes/2)*sum_s_R_bias[0],label="$\Delta_l^{1/2}$")
plt.legend()
plt.yscale("log")

#%%
par_n=0
#print(Grads_file[:,:,0,par_n])
#print(Grads_file[:,:,1,par_n])
s_error=np.mean((Grads_file[:,:,0,par_n]-Grads_file[:,:,1,par_n])**2,axis=1)
#s_error_sum=np.sum(np.mean((grads_mean[:,:,0]-grads_mean[:,:,1])**2,axis=1),axis=-1)
plt.plot(eLes,s_error,label="Error")
#plt.plot(eLes,s_error_sum,label="Error sum")
#plt.plot(eLes, 2**(eLes[-1]/2)/2**(eLes/2)*s_error[-1],label="$\Delta_l^{1/2}$")
plt.plot(eLes, 2**(eLes[-1])/2**(eLes)*s_error[-1],label="$\Delta_l$")
plt.legend()
plt.yscale("log")
# %%
i=1
plt.plot(mcmc[0,0,:,-1,:])
#%%

##########################################################################################
##########################################################################################
##########################################################################################

# TEST FOR THE COUPLE PF (NORMAL SAMPLING)

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
collection_input=[bdg.b_ou_1d,theta_true,bdg.Sig_ou_1d,sigma_true]
resamp_coef=1
l_max=10
x_true= bdg.gen_gen_data_1d(T,x0_sca,l_max,collection_input)
x_reg=bdg.cut(T,l_max,-l_d,x_true)[1:]
times=np.array(range(t0,int(T/d)+1))*d
l_times=np.arange(t0,T,2**(-l))
l_max_times=np.arange(t0,T,2**(-l_max))
plt.plot(times[1:],x_reg,label="True signal")
plt.plot(l_max_times,x_true[:-1],label="True complete signal")
sd_true=1
np.random.seed(3)
d_times=np.array(range(t0+d,int(T/d)+1))*d
obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)
plt.plot(d_times, obs,label="Observations")
print(obs)

#%%
theta=theta_true
sigma=sigma_true
theta_aux=theta+0.2
sigma_aux=sigma
sd=sd_true
fd=1e-4
theta_fd=theta_true+fd
sigma_fd=sigma_true+fd
sigma_aux_fd=sigma_aux+fd
start=time.time()
B=10
samples=1000
# interactive 1 samples=100
N=100
x0=x0_sca+np.zeros(N)
seed=2985
l0=2
L_max=8
eLes=np.array(range(l0,L_max+1))
mcmc_mean=np.zeros((len(eLes),samples,2,int(T/d)))
mcmc=np.zeros((len(eLes),samples,B,2,int(T/d)))
grads_mean=np.zeros((len(eLes),samples,2,3))
pf_diffs=np.zeros((len(eLes),samples,int(T/d)))
pf_l=np.zeros((len(eLes),samples,2,int(T/d)))
ori_path_l=np.zeros((len(eLes),samples,2,int(T/d)),dtype=float)
seeds_pf_0=np.zeros((len(eLes),samples,2,int(T/d)),dtype=int)
seeds_pf_1=np.zeros((len(eLes),samples,2,int(T/d)),dtype=int)
pf_mean=np.zeros((len(eLes),samples,2),dtype=float)
resamp_coef=1
for k in range(len(eLes)):
    l=eLes[k] 
    for i in range(samples):
        np.random.seed(i+10000)
        print("l is: ",l)
        print("The sample is: ",i)
        #print("Seed feeded to PF_bridge is: ",seed)

        """
        [log_weights_0,log_weights_1,int_Gs_0,int_Gs_1,x_pr_0,x_pr_1,seeds_0_wp,seeds_1_wp]
        """
        """
        C_PF_bridge(t0,x0,T,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,\
        max_sample_funct,sample_pars,obs,log_g_den,g_den_par, aux_trans_den,atdp,\
        prop_trans_den,ind_prop_trans_par, resamp_coef, l, d,N,seed,crossed=False)
        """

        [log_weights_0,log_weights_1,int_Gs_0,int_Gs_1,x_pr_0,x_pr_1,seeds_0_wp,seeds_1_wp]=\
        bdg.C_PF_bridge(t0,x0,T,bdg.b_ou_1d,theta,bdg.Sig_ou_1d,sigma,bdg.b_ou_aux,theta_aux,\
        bdg.Sig_ou_aux,sigma_aux,bdg.r_quasi_normal_1d,[bdg.ou_sd,[theta_aux,sigma_aux]],\
        bdg.H_quasi_normal,[bdg.ou_sd,[theta_aux,sigma_aux],theta_aux],bdg.rej_max_coup_ou,\
        [theta_aux,sigma_aux,theta_aux,sigma_aux],obs,bdg.log_g_normal_den,sd,\
        bdg.ou_trans_den,[theta_aux,sigma_aux],bdg.ou_trans_den,[theta_aux,sigma_aux],resamp_coef,l,d, N,seed)
        #x_mean=np.sum(pff.norm_logweights(log_weights,ax=1)*x_pr,axis=-1)
        pf_mean[k,i]=[np.mean(x_pr_0[-1]),np.mean(x_pr_1[-1])]
        #ori_pf_l[k,i]=np.sum(((pff.norm_logweights(log_weights,ax=1))*x_pr),axis=1)
        ori_path_l[k,i]=np.copy(np.array([x_pr_0[:,0],x_pr_1[:,0]]))
        seeds_pf_0[k,i]=seeds_0_wp[:,:,0].T
        seeds_pf_1[k,i]=seeds_1_wp[:,:,0].T
        #print(weights.shape)
        seed+=(int(T/d))*int(int(2**l*d-1))
        
# %%
dim=1
dim_o=1
print(theta, sigma,sd)
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
x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
#KF(xin,dim,dim_o,K,G,H,D,obs)
x_kf_2=bdg.KF(x0[0],dim,dim_o,K,G,H,D,obs)[0]
Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
d_times=np.array(range(t0,int(T/d)+1))*d
#weights=pff.norm_logweights(lws,ax=2)
#PF_mean=np.mean(np.sum(x_prs*weights,axis=-1),axis=0)
l=L_max
l_times=np.arange(t0,T,2**(-l))
l_max_times=np.arange(t0,T,2**(-l_max))
#print(times, l_times)
x_mean=np.mean(mcmc_mean,axis=1)[-1,1]
#print(x_mean.shape)
#plt.plot(d_times[1:],PF_mean,label="PF")
plt.plot(d_times[1:],x_reg,label="True signal")
#plt.plot(l_max_times,x_true[:-1],label="True complete signal")
plt.plot(d_times,x_kf_smooth[:,0],label="KF smooth")
plt.plot(d_times,x_kf[:,0],label="KF")
plt.plot(d_times[1:], obs,label="Observations")
plt.plot(d_times[1:], x_mean.T,label="PGibbs")
print(obs)
plt.legend()  
#%%
# for the second moment of the differnce of the coupled pfs
        
sm_pf=np.mean((pf_mean[:,:,0]-pf_mean[:,:,1])**2,axis=1)
plt.plot(eLes,2**(eLes[0])/2**(eLes)*sm_pf[0],label="$\Delta_l$")
plt.plot(eLes,sm_pf,label="Second moment of the difference of the coupled PFs")
plt.legend()
plt.yscale("log")
#%%


# for the bias of the particle filter
bias=np.abs(np.mean(pf_mean[:,:,1]-x_kf[-1,0],axis=1))
plt.plot(eLes,bias,label="PF bias")
plt.plot(eLes,2**(eLes[0])/2**(eLes)*bias[0],label="$\Delta_l$")
plt.legend()
plt.yscale("log")


# %%
##########################################################################################
##########################################################################################
##########################################################################################
# TEST FOR THE COUPLED SGD



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
collection_input=[bdg.b_ou_1d,theta_true,bdg.Sig_ou_1d,sigma_true]
resamp_coef=1
l_max=10
x_true= bdg.gen_gen_data_1d(T,x0_sca,l_max,collection_input)
x_reg=bdg.cut(T,l_max,-l_d,x_true)[1:]
times=np.array(range(t0,int(T/d)+1))*d
l_times=np.arange(t0,T,2**(-l))
l_max_times=np.arange(t0,T,2**(-l_max))
plt.plot(times[1:],x_reg,label="True signal")
plt.plot(l_max_times,x_true[:-1],label="True complete signal")
sd_true=0.6
np.random.seed(3)
d_times=np.array(range(t0+d,int(T/d)+1))*d
obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)
plt.plot(d_times, obs,label="Observations")
print(obs)
#%%
resamp_coef=1
fd=1e-4
theta_in=theta_true-0.4
theta_in_fd=theta_in+fd
sigma_in=sigma_true-0.5
sigma_in_fd=sigma_in+fd
sigma_in_aux=sigma_in
theta_in_aux=theta_in+0.2
sigma_in_aux_fd=sigma_in_aux+fd
sd_in=sd_true
N=50
start=time.time()
mcmc_links=500
SGD_steps=2
B=mcmc_links*SGD_steps
gamma=0.05
alpha=0.01
seed=2393
l=6
x0=x0_sca+np.zeros(N)
ch_paths_0 ,ch_paths_1,pars_0,pars_1, Grads_test_0,Grads_test_1=\
bdg.C_SGD_bridge(t0,x0,T,bdg.b_ou_1d,theta_in,theta_in_fd,bdg.Sig_ou_1d,sigma_in,sigma_in_fd,\
bdg.b_ou_aux,theta_in_aux,bdg.Sig_ou_aux,sigma_in_aux,sigma_in_aux_fd,\
bdg.r_quasi_normal_1d,[bdg.ou_sd,[theta_in_aux,sigma_in_aux]],[bdg.ou_sd,[theta_in_aux,sigma_in_aux_fd]],\
bdg.H_quasi_normal,\
[bdg.ou_sd,[theta_in_aux,sigma_in_aux],theta_in_aux],[bdg.ou_sd,[theta_in_aux,sigma_in_aux_fd],theta_in_aux],\
bdg.rej_max_coup_ou, [theta_in_aux,sigma_in_aux,theta_in_aux,sigma_in_aux],obs,bdg.log_g_normal_den,sd_in,\
bdg.ou_trans_den,[theta_in_aux,sigma_in_aux],bdg.Grad_log_aux_trans_ou_new,\
bdg.ou_trans_den,[theta_in_aux,sigma_in_aux],bdg.Grad_log_G_new,resamp_coef,l,d, N,seed,fd,mcmc_links,SGD_steps,gamma,\
alpha,crossed=False)
"""
C_SGD_bridge(t0,x0,T,b,A_in,A_fd_in,Sig,fi_in,fi_fd_in,b_til,A_til_in,Sig_til,fi_til_in,\
    fi_til_fd_in,r,r_pars,r_pars_fd,H,H_pars,H_pars_fd,max_sample_funct,sample_pars,\
    obs,log_g_den,g_den_par_in, aux_trans_den,atdp,\
    Grad_log_aux_trans,prop_trans_den,ind_prop_trans_par, Grad_log_G,resamp_coef, l, d,N,seed,fd_rate,\
    mcmc_links,SGD_steps,gamma, alpha, \
    crossed=False):

"""
end=time.time()
print(end-start)
#%%
Grads_0=np.zeros((SGD_steps,3))
Grads_1=np.zeros((SGD_steps,3))
print(l)
for i in range(SGD_steps):
            Grad_mcmc_0=np.mean(Grads_test_0[mcmc_links*i:mcmc_links*(i+1)],axis=0)
            Grad_mcmc_1=np.mean(Grads_test_1[mcmc_links*i:mcmc_links*(i+1)],axis=0)

#%%
Grid_p=20
thetas=np.linspace(-1,1,Grid_p)*0.75+theta_true
sigmas=np.linspace(-1,1,Grid_p)*0.75+sigma_true
theta_aux=thetas+0.2
sigma_aux=sigmas
sds=np.linspace(-1,1,Grid_p)*0.5+ sd_true
fd_rate=1e-4
[theta_fd,sigma_fd,sd_fd]=np.array([thetas,sigmas,sds])+fd_rate*(np.zeros((3,Grid_p))+1)
print(thetas,sigmas)
#%%
# IN 2d
Grads=np.zeros((Grid_p,Grid_p,3))
dim=1
sd=sd_true
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
        x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
        Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
        Grads[j,i]=Grad_log_lik[:,0,0]
#%%
#[theta_0,sigma_0,sd_0]=np.array([theta_true,sigma_true,sd_true])+np.array([0.4,-0.3,0])
[theta_0,sigma_0,sd_0]=[theta_in,sigma_in,sd_in]
SGD_steps=10
pars=np.zeros((SGD_steps+1,3))
Grads_test=np.zeros((SGD_steps+1,3))
alpha=0.0001
gamma=0.05
theta=theta_0
sigma=sigma_0
sd=sd_0
pars[0,:]=np.array([theta,sigma,sd])

for b_ind in range(SGD_steps):
    
    #sigma=sigmas[j]
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
    x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
    Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
    Grads_test[b_ind]=Grad_log_lik[:,0,0]
    print(Grads_test[b_ind,:2])
    theta+=gamma*Grads_test[b_ind,0]/(b_ind+1)**(0.5+alpha)
    sigma+=gamma*Grads_test[b_ind,1]/(b_ind+1)**(0.5+alpha)
    #sd+=gamma*Grads_test[b_ind,2]/(b_ind+1)**(0.5+alpha)
    pars[b_ind+1]=np.array([theta,sigma,sd])

#%%
print("pars_0 is: ")
print(pars_0)
print("sd is: ",sd_true)
plt.plot(pars[:,0].T,pars[:,1].T)
plt.plot(pars_0[:,0].T,pars_1[:,1].T)
plt.plot(pars_1[:,0].T,pars_1[:,1].T)
thetas_Grid,sigmas_Grid=np.meshgrid(thetas,sigmas)
plt.quiver(thetas_Grid,sigmas_Grid,Grads[:,:,0],Grads[:,:,1])
print("The starting guesses are: ",theta_0,sigma_0)
print("The actual parameters are: ",theta_true,sigma_true)
max=np.max(Grads[:,:,0]**2+Grads[:,:,1]**2)
min=np.min(Grads[:,:,0]**2+Grads[:,:,1]**2)
print("The maximum gradient is: ",np.sqrt(max), "The minimum gradient is: ",np.sqrt(min))
plt.xlabel("Theta")
plt.ylabel("Sigma")
plt.title("SGD")
#plt.savefig("Gradiend_flow_&_SGD.pdf")
plt.show()
    
#%%
##########################################################################################
##########################################################################################
##########################################################################################

# IN THE FOLLOWING WE TEST THE COUPLED SGD, THE IDEA IS TO CHECK THE SECOND MOMENT OF THE DIFFERENCE OF THE 
#PARAMETERS


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
collection_input=[bdg.b_ou_1d,theta_true,bdg.Sig_ou_1d,sigma_true]
resamp_coef=1
l_max=10
x_true= bdg.gen_gen_data_1d(T,x0_sca,l_max,collection_input)
x_reg=bdg.cut(T,l_max,-l_d,x_true)[1:]
times=np.array(range(t0,int(T/d)+1))*d
l_times=np.arange(t0,T,2**(-l))
l_max_times=np.arange(t0,T,2**(-l_max))
plt.plot(times[1:],x_reg,label="True signal")
plt.plot(l_max_times,x_true[:-1],label="True complete signal")
sd_true=0.6
np.random.seed(3)
d_times=np.array(range(t0+d,int(T/d)+1))*d
obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)
plt.plot(d_times, obs,label="Observations")
print(obs)
#%%

resamp_coef=1
fd=1e-4
theta_in=theta_true-0.4
theta_in_fd=theta_in+fd
sigma_in=sigma_true-0.5
sigma_in_fd=sigma_in+fd
sigma_in_aux=sigma_in
theta_in_aux=theta_in+0.2
sigma_in_aux_fd=sigma_in_aux+fd
sd_in=sd_true
N=50
start=time.time()
mcmc_links=500
SGD_steps=2
B=mcmc_links*SGD_steps
gamma=0.05
alpha=0.01
seed=2393
l=6
x0=x0_sca+np.zeros(N)
ch_paths_0 ,ch_paths_1,pars_0,pars_1, Grads_test_0,Grads_test_1=\
bdg.C_SGD_bridge(t0,x0,T,bdg.b_ou_1d,theta_in,theta_in_fd,bdg.Sig_ou_1d,sigma_in,sigma_in_fd,\
bdg.b_ou_aux,theta_in_aux,bdg.Sig_ou_aux,sigma_in_aux,sigma_in_aux_fd,\
bdg.r_quasi_normal_1d,[bdg.ou_sd,[theta_in_aux,sigma_in_aux]],[bdg.ou_sd,[theta_in_aux,sigma_in_aux_fd]],\
bdg.H_quasi_normal,\
[bdg.ou_sd,[theta_in_aux,sigma_in_aux],theta_in_aux],[bdg.ou_sd,[theta_in_aux,sigma_in_aux_fd],theta_in_aux],\
bdg.rej_max_coup_ou, [theta_in_aux,sigma_in_aux,theta_in_aux,sigma_in_aux],obs,bdg.log_g_normal_den,sd_in,\
bdg.ou_trans_den,[theta_in_aux,sigma_in_aux],bdg.Grad_log_aux_trans_ou_new,\
bdg.ou_trans_den,[theta_in_aux,sigma_in_aux],bdg.Grad_log_G_new,resamp_coef,l,d, N,seed,fd,mcmc_links,SGD_steps,gamma,\
alpha,crossed=False)
"""
C_SGD_bridge(t0,x0,T,b,A_in,A_fd_in,Sig,fi_in,fi_fd_in,b_til,A_til_in,Sig_til,fi_til_in,\
    fi_til_fd_in,r,r_pars,r_pars_fd,H,H_pars,H_pars_fd,max_sample_funct,sample_pars,\
    obs,log_g_den,g_den_par_in, aux_trans_den,atdp,\
    Grad_log_aux_trans,prop_trans_den,ind_prop_trans_par, Grad_log_G,resamp_coef, l, d,N,seed,fd_rate,\
    mcmc_links,SGD_steps,gamma, alpha, \
    crossed=False):

"""
end=time.time()
print(end-start)
###########################################################################
###########################################################################
###########################################################################

# In the following I test the SGD bias and other statistics

#%%
if True:

    x0_sca=1.2
    x0=x0_sca
    l=10
    T=4
    t0=0
    l_d=0
    d=2**(l_d)
    theta_true=-0.3
    sigma_true=1.2
    np.random.seed(7)
    collection_input=[bdg.b_ou_1d,theta_true,bdg.Sig_ou_1d,sigma_true]
    resamp_coef=1
    l_max=10
    x_true= bdg.gen_gen_data_1d(T,x0_sca,l_max,collection_input)
    x_reg=bdg.cut(T,l_max,-l_d,x_true)[1:]
    times=np.array(range(t0,int(T/d)+1))*d
    l_times=np.arange(t0,T,2**(-l))
    l_max_times=np.arange(t0,T,2**(-l_max))
    #plt.plot(times[1:],x_reg,label="True signal")
    #plt.plot(l_max_times,x_true[:-1],label="True complete signal")
    sd_true=1
    np.random.seed(3)
    d_times=np.array(range(t0+d,int(T/d)+1))*d
    obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)
    #plt.plot(d_times, obs,label="Observations")
    theta=theta_true
    sigma=sigma_true
    theta_aux=theta+0.2
    sigma_aux=sigma
    sd=sd_true
    fd=1e-6
    theta_fd=theta_true+fd
    sigma_fd=sigma_true+fd
    sigma_aux_fd=sigma_aux+fd
    gamma=0.5
    alpha=0.5
    start=time.time()
    mcmc_links=500*10
    SGD_steps=1
    B=mcmc_links*SGD_steps
    samples=30
    # interactive 1 samples=100
    N=50
    x0=x0_sca+np.zeros(N)
    l0=3
    L_max=6
    seed=0
    eLes=np.array(range(l0,L_max+1))
#%%
pars_file=np.reshape(np.loadtxt("Observations&data/Prl_SGD_ou_bridge_pars_v5.txt",dtype=float),(len(eLes),samples,SGD_steps+1,3))   
Grads_file=np.reshape(np.loadtxt("Observations&data/Prl_SGD_ou_bridge_Grads_v5.txt",dtype=float),(len(eLes),samples,SGD_steps,3))
#%%
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
x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
#KF(xin,dim,dim_o,K,G,H,D,obs)
x_kf_2=bdg.KF(x0[0],dim,dim_o,K,G,H,D,obs)[0]
Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
#%%
par_n=1
step=0
print("The analytical grad is: ", Grad_log_lik)
#(len(eLes),samples,SGD_steps,3)
Grads_mean=np.mean(Grads_file,axis=1)[:,step,par_n]
print(Grads_mean,Grad_log_lik[par_n])
print(Grads_file.shape,Grad_log_lik.shape)
plt.plot(eLes,np.abs(Grads_mean-Grad_log_lik[par_n,0,0]),label="Gradient bias")
grad_bias=np.abs(Grads_mean-Grad_log_lik[par_n,0,0])
#r_grad_bias=np.abs(np.mean(grads_mean[:,:,0,par_n]-grads_mean[:,:,1,par_n],axis=1))
#plt.plot(eLes,grad_bias,label="Gradient bias")
#plt.plot(eLes,r_grad_bias,label="Richardson gradient bias")
plt.plot(eLes,2**(eLes[-1])/2**(eLes)*grad_bias[-1],label="$\Delta_l$")
#plt.plot(eLes,r_grad_bias,label="Richardson gradient bias")
plt.yscale("log")
plt.legend()
#%%
Grid_p=9
thetas=np.linspace(-1,1,Grid_p)*0.2+theta_in
lsigmas=np.linspace(-1,1,Grid_p)*0.2+ np.log(sigma_in)
lsds=np.linspace(-1,1,Grid_p)*0.2+np.log(sd_in)
Grid=np.stack((thetas,lsigmas,lsds))
theta_aux=thetas+0.2
sigma_aux=sigmas
#fd_rate=1e-4
#[theta_fd,sigma_fd,sd_fd]=np.array([thetas,sigmas,sds])+fd_rate*(np.zeros((3,Grid_p))+1)
#print(thetas,sigmas)
print(Grid.shape)

#%%
[theta_0,sigma_0,sd_0]=[theta_in,sigma_in,sd_in]
x=0
y=1
Grads=np.zeros((Grid_p,Grid_p,3))
dim=1
dim_o=1
for i in range(len(Grid[x])):
    par_x=Grid[x][i]
    for j in range(len(Grid[y])):
        #sigma=sigmas[j]
        par_y=np.exp(Grid[y][j])
        theta=(y==0)*par_y+(x==0)*par_x+ (x!=0)*(y!=0)*theta_0
        sigma=(y==1)*par_y+(x==1)*par_x+ (x!=1)*(y!=1)*sigma_0
        sd=(y==2)*par_y+(x==2)*par_x+ (x!=2)*(y!=2)*sd_0
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
        x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
        Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma*sigma # the last sigma is put in 
        # order to account for the gradient of log sigma instead of sigma
        Grad_log_lik[2,0,0]=Grad_log_lik[2,0,0]*sd
        Grads[j,i]=Grad_log_lik[:,0,0]

#%%
#[theta_0,sigma_0,sd_0]=np.array([theta_true,sigma_true,sd_true])+np.array([0.4,-0.3,0])
[theta_0,sigma_0,sd_0]=[theta_in,sigma_in,sd_in]
SGD_steps=20
pars=np.zeros((SGD_steps+1,3))
Grads_test=np.zeros((SGD_steps+1,3))
#alpha=0.5
#gamma=0.05
theta=theta_0
sigma=sigma_0
sd=sd_0
pars[0,:]=np.array([theta,sigma,sd])
for b_ind in range(SGD_steps):
    #sigma=sigmas[j]
    #print(theta,sigma)
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
    x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
    Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
    Grads_test[b_ind]=Grad_log_lik[:,0,0]
    #print(Grads_test[b_ind,:2])
    theta+=gamma*Grads_test[b_ind,0]/(b_ind+1)**(0.5+alpha)
    sigma+=gamma*Grads_test[b_ind,1]/(b_ind+1)**(0.5+alpha)
    #sd+=gamma*Grads_test[b_ind,2]/(b_ind+1)**(0.5+alpha)
    pars[b_ind+1]=np.array([theta,sigma,sd])
#%%
# len(eLes),samples,2,SGD_steps+1,3)
a=0
b=4
print("pars_0 is: ")
print("sd is: ",sd_true)
ele=-5
new_pars_file=np.zeros((len(eLes),samples,SGD_steps+1,3))
new_pars_file[:,:,:,0]=pars_file[:,:,:,0]
new_pars_file[:,:,:,1]=np.log(pars_file[:,:,:,1])
new_pars_file[:,:,:,2]=np.log(pars_file[:,:,:,2])
# (len(eLes),samples,SGD_steps+1,3)
#plt.plot(pars[a:,0].T,pars[a:,1].T)
plt.plot(new_pars_file[ele,:,a:b,0].T,new_pars_file[ele,:,a:b,1].T)
#plt.plot(pars_file[-1,:,0,a:,0].T,pars_file[-1,:,0,a:,1].T)
thetas_Grid,sigmas_Grid=np.meshgrid(thetas,lsigmas)
plt.quiver(thetas_Grid,sigmas_Grid,Grads[:,:,0],Grads[:,:,1])
print("The starting guesses are: ",theta_0,sigma_0)
print("The actual parameters are: ",theta_true,sigma_true)
max=np.max(Grads[:,:,0]**2+Grads[:,:,1]**2)
min=np.min(Grads[:,:,0]**2+Grads[:,:,1]**2)
print("The maximum gradient is: ",np.sqrt(max), "The minimum gradient is: ",np.sqrt(min))
plt.xlabel("Theta")
plt.ylabel("Sigma")
plt.title("SGD")
#plt.savefig("Gradiend_flow_&_SGD.pdf")
plt.show()
#%%
# FINITE DIFFERENCE ACCURACY. 
# In the following I test the accuracy of finite differences with the goal
# of assessing the rounding error as fd_rate goes to zero.
L=10
eLes=np.arange(L)
fd_0=5e-7
fd_rates=fd_0/2**eLes
# What functions will we consider? polynomial, exponential, logarithmic, square root, cosine

def functs(x):

    return np.array([x**2,np.exp(x),np.log(x),x**(1/2),np.cos(x)])

def der_fu(x):
    return np.array([2*x,np.exp(x),1/x,0.5/x**(1/2),-np.sin(x)])

n_samples=100
samples=1+2*np.random.uniform(size=n_samples)
#samples varies in the first dimension and fds varies in the second
fds=(functs(samples[:,np.newaxis]+fd_rates)-functs(samples[:,np.newaxis]))/fd_rates
ders=der_fu(samples)
errors=np.abs(fds-ders[:,:,np.newaxis])
print(errors,fds)
mean_errors=np.mean(errors,axis=(0,1))

plt.plot(eLes,mean_errors)
plt.yscale("log")
#%%
################################################################################################


# This section is made to compute the the particle filter.

if True:

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
    collection_input=[bdg.b_ou_1d,theta_true,bdg.Sig_ou_1d,sigma_true]
    resamp_coef=1
    l_max=10
    x_true= bdg.gen_gen_data_1d(T,x0_sca,l_max,collection_input)
    x_reg=bdg.cut(T,l_max,-l_d,x_true)[1:]
    times=np.array(range(t0,int(T/d)+1))*d
    l_times=np.arange(t0,T,2**(-l))
    l_max_times=np.arange(t0,T,2**(-l_max))
    #plt.plot(times[1:],x_reg,label="True signal")
    #plt.plot(l_max_times,x_true[:-1],label="True complete signal")
    sd_true=2.1
    np.random.seed(3)
    d_times=np.array(range(t0+d,int(T/d)+1))*d
    obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)
    #plt.plot(d_times, obs,label="Observations")
    theta=theta_true
    sigma=sigma_true
    theta_aux=theta+0.2
    sigma_aux=sigma
    sd=sd_true
    fd=1e-8
    theta_fd=theta_true+fd
    sigma_fd=sigma_true+fd
    sigma_aux_fd=sigma_aux+fd
    start=time.time()
    samples=40
    # interactive 1 samples=100
    N=5000000
    x0=x0_sca+np.zeros(N)
    l0=3
    L_max=11
    eLes=np.array(range(l0,L_max+1))

#%%
v="11"
#log_weights=np.reshape(np.loadtxt("Observationsdata/data1/Prl_PF_bridge_log_weight_v"+v+".txt",dtype=float),(len(eLes),samples,int(T/d),N))
#x_pf=np.reshape(np.loadtxt("Observationsdata/data2/Prl_PF_bridge_x_pf_v"+v+".txt",dtype=float),(len(eLes),samples,int(T/d)))
x_pf=np.reshape(np.loadtxt("Observationsdata/data2/Prl_PF_bridge_x_pf_v"+v+".txt",dtype=float),(len(eLes),samples,int(T/d)))
#int_Gs=np.reshape(np.loadtxt("Observationsdata/data1/Prl_PF_bridge_int_Gs_v"+v+".txt",dtype=float),(len(eLes),samples,int(T/d),N))
#%%
dim=1
dim_o=1
#theta,sigma,sd=theta,sigma,sd
K=np.array([[np.exp(d*theta)]])
G=np.array([[sigma*np.sqrt((np.exp(2*d*theta)-1)/(2*theta))]])
H=np.array([[1]])
D=np.array([[sd]])
Grad_S=np.array([[[0]],[[0]],[[1]]],dtype=float)
Grad_R_sigma_s=(np.exp(2*theta*d)-1)/(2*theta)
Grad_R_theta=(sigma**2/(2*theta**2))*(1-np.exp(2*theta*d)+2*d*theta*np.exp(2*theta*d))
Grad_R=np.zeros((3,1,1),dtype=float)
Grad_R[0,0,0]=Grad_R_theta
Grad_R[1,0,0]=Grad_R_sigma_s
Grad_K=np.array([[[d*np.exp(d*theta)]],[[0]],[[0]]],dtype=float)
x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
x_kf_2=bdg.KF(x0[0],dim,dim_o,K,G,H,D,obs)[0]
Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
#%%
#weights=pff.norm_logweights(log_weights,ax=3)
#x_pf=np.sum(x_pr*weights,axis=3)
#x_pf2=np.mean(x_pr,axis=3)
#%%
i=1
#(len(eLes),samples,int(T/d))
print(np.mean(x_pf,axis=(1))[:,i])
print(x_kf_smooth[i+1])
bias=np.abs(np.mean(x_pf-x_kf_smooth[1:,0],axis=(1))[:,i])
r_bias=np.abs(np.mean(x_pf[1:]-x_pf[:-1],axis=(1))[:,i])
var=np.var(x_pf-x_kf_smooth[1:,0],axis=(1))[:,i]
print(var)
bias_up=bias+np.sqrt(var)*1.96/np.sqrt(samples)
bias_lb=bias-np.sqrt(var)*1.96/np.sqrt(samples)
plt.plot(eLes[1:],r_bias,label="r_bias")
plt.plot(eLes,bias,label="bias")
plt.plot(eLes,bias_up,label="bias_up")
plt.plot(eLes,bias_lb,label="bias_lb")
plt.plot(eLes,2**(eLes[0])/2**(eLes)*bias[0],label="$\Delta_l$")
plt.yscale("log")
plt.legend()
#%%

if True:

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
    collection_input=[bdg.b_ou_1d,theta_true,bdg.Sig_ou_1d,sigma_true]
    resamp_coef=1
    l_max=10
    x_true= bdg.gen_gen_data_1d(T,x0_sca,l_max,collection_input)
    x_reg=bdg.cut(T,l_max,-l_d,x_true)[1:]
    times=np.array(range(t0,int(T/d)+1))*d
    l_times=np.arange(t0,T,2**(-l))
    l_max_times=np.arange(t0,T,2**(-l_max))
    plt.plot(times[1:],x_reg,label="True signal")
    plt.plot(l_max_times,x_true[:-1],label="True complete signal")
    sd_true=2.1
    np.random.seed(3)
    d_times=np.array(range(t0+d,int(T/d)+1))*d
    obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)
    plt.plot(d_times, obs,label="Observations")
    theta=theta_true
    sigma=sigma_true
    theta_aux=theta+0.2
    sigma_aux=sigma
    sd=sd_true
    fd=1e-8
    theta_fd=theta_true+fd
    sigma_fd=sigma_true+fd
    sigma_aux_fd=sigma_aux+fd
    start=time.time()
    B=1
   
    samples=40
    # interactive 1 samples=100
    N=5000000
    x0=x0_sca+np.zeros(N)
    l0=3
    L_max=12
    eLes=np.array(range(l0,L_max+1))
#%%
v="11"
#ch_paths_file_van=np.reshape(np.loadtxt("Observations&data/Prl_SGD_ou_bridge_van_comparison_ch_paths_v"+v+".txt",dtype=float),(samples,B,int(T/d)))
#pars_file_van=np.reshape(np.loadtxt("Observations&data/Prl_SGD_ou_bridge_van_comparison_pars_v"+v+".txt",dtype=float),(samples,SGD_steps+1,3))
#ch_paths_file=np.reshape(np.loadtxt("Observations&data/Prl_SGD_ou_bridge_comparison_ch_paths_v"+v+".txt",dtype=float),(samples,B,int(T/d)))
#pars_file=np.reshape(np.loadtxt("Observations&data/Prl_SGD_ou_bridge_comparison_pars_v"+v+".txt",dtype=float),(samples,SGD_steps+1,3))
ch_paths_file=np.reshape(np.loadtxt("Observationsdata/data1/Prl_PG_chain_ch_paths_v"+v+".txt",dtype=float),((len(eLes),samples,B,int(T/d))))
comp_pf_l=np.reshape(np.loadtxt("Observationsdata/data1/Prl_PG_chain_comp_pf_v"+v+".txt",dtype=float),(len(eLes),samples,B,int(T/d)))
#%%
dim=1
dim_o=1
#theta,sigma,sd=theta,sigma,sd
K=np.array([[np.exp(d*theta)]])
G=np.array([[sigma*np.sqrt((np.exp(2*d*theta)-1)/(2*theta))]])
H=np.array([[1]])
D=np.array([[sd]])
Grad_S=np.array([[[0]],[[0]],[[1]]],dtype=float)
Grad_R_sigma_s=(np.exp(2*theta*d)-1)/(2*theta)
Grad_R_theta=(sigma**2/(2*theta**2))*(1-np.exp(2*theta*d)+2*d*theta*np.exp(2*theta*d))
Grad_R=np.zeros((3,1,1),dtype=float)
Grad_R[0,0,0]=Grad_R_theta
Grad_R[1,0,0]=Grad_R_sigma_s
Grad_K=np.array([[[d*np.exp(d*theta)]],[[0]],[[0]]],dtype=float)
x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
x_kf_2=bdg.KF(x0[0],dim,dim_o,K,G,H,D,obs)[0]
Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma

#%%
i=0
#(len(eLes),samples,B,int(T/d))
pf_l=np.mean(comp_pf_l,axis=(1,2))
print(pf_l.shape)
print(pf_l[:,i])
print(x_kf[i+1])
r_bias=np.abs(pf_l[1:,i]-pf_l[:-1,i])
bias_eles=np.abs(pf_l[:,i]-x_kf[i+1,0])
var_eles=np.var(comp_pf_l-x_kf[1:,0],axis=(1,2))[:,i]
bias_eles_up=bias_eles+np.sqrt(var_eles)*1.96/np.sqrt(samples*B)
bias_eles_lb=bias_eles-np.sqrt(var_eles)*1.96/np.sqrt(samples*B)
plt.plot(eLes[1:],r_bias,label="r_bias")
plt.plot(eLes,bias_eles,label="bias")
plt.plot(eLes,bias_eles_up)
plt.plot(eLes,bias_eles_lb)
plt.plot(eLes,2**(eLes[0])/2**(eLes)*bias_eles[0],label="$\Delta_l$")
plt.yscale("log")
plt.legend()
print(x_kf)
#%%


#%%%

# This iteration is made to analyze the computations of the function Prl_Grad_chain, particularly, 
# the function outputs averaged arrays (no additional dimension for B).

if True:

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
    collection_input=[bdg.b_ou_1d,theta_true,bdg.Sig_ou_1d,sigma_true]
    resamp_coef=1
    l_max=10
    x_true= bdg.gen_gen_data_1d(T,x0_sca,l_max,collection_input)
    x_reg=bdg.cut(T,l_max,-l_d,x_true)[1:]
    times=np.array(range(t0,int(T/d)+1))*d
    l_times=np.arange(t0,T,2**(-l))
    l_max_times=np.arange(t0,T,2**(-l_max))
    plt.plot(times[1:],x_reg,label="True signal")
    plt.plot(l_max_times,x_true[:-1],label="True complete signal")
    sd_true=1
    np.random.seed(3)
    d_times=np.array(range(t0+d,int(T/d)+1))*d
    obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)
    plt.plot(d_times, obs,label="Observations")
    theta=theta_true
    sigma=sigma_true
    theta_aux=theta+0.2
    sigma_aux=sigma
    sd=sd_true
    fd=1e-8
    theta_fd=theta_true+fd
    sigma_fd=sigma_true+fd
    sigma_aux_fd=sigma_aux+fd
    start=time.time()
    B=500*3*2*20
    samples=40
    # interactive 1 samples=100
    N=200
    x0=x0_sca+np.zeros(N)
    l0=3
    L_max=8
    eLes=np.array(range(l0,L_max+1))
#%%
v="5v5"
#ch_paths_file_van=np.reshape(np.loadtxt("Observations&data/Prl_SGD_ou_bridge_van_comparison_ch_paths_v"+v+".txt",dtype=float),(samples,B,int(T/d)))
#pars_file_van=np.reshape(np.loadtxt("Observations&data/Prl_SGD_ou_bridge_van_comparison_pars_v"+v+".txt",dtype=float),(samples,SGD_steps+1,3))
#ch_paths_file=np.reshape(np.loadtxt("Observations&data/Prl_SGD_ou_bridge_comparison_ch_paths_v"+v+".txt",dtype=float),(samples,B,int(T/d)))
#pars_file=np.reshape(np.loadtxt("Observations&data/Prl_SGD_ou_bridge_comparison_pars_v"+v+".txt",dtype=float),(samples,SGD_steps+1,3))
ch_paths_file=np.reshape(np.loadtxt("Observationsdata/data1/Prl_Grad_chain_ch_paths_v"+v+".txt",dtype=float),(len(eLes), samples,int(T/d)))
Grads_file=np.reshape(np.loadtxt("Observationsdata/data1/Prl_Grad_chain_Grads_v"+v+".txt",dtype=float),(len(eLes),samples,3))
#%%
dim=1
dim_o=1
theta,sigma,sd=theta,sigma,sd
K=np.array([[np.exp(d*theta)]])
G=np.array([[sigma*np.sqrt((np.exp(2*d*theta)-1)/(2*theta))]])
H=np.array([[1]])
D=np.array([[sd]])
Grad_S=np.array([[[0]],[[0]],[[1]]],dtype=float)
Grad_R_sigma_s=(np.exp(2*theta*d)-1)/(2*theta)
Grad_R_theta=(sigma**2/(2*theta**2))*(1-np.exp(2*theta*d)+2*d*theta*np.exp(2*theta*d))
Grad_R=np.zeros((3,1,1),dtype=float)
Grad_R[0,0,0]=Grad_R_theta
Grad_R[1,0,0]=Grad_R_sigma_s
Grad_K=np.array([[[d*np.exp(d*theta)]],[[0]],[[0]]],dtype=float)
x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
x_kf_2=bdg.KF(x0[0],dim,dim_o,K,G,H,D,obs)[0]
Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
#%%
mcmc_links=B
#(len(eLes),samples,3)
par_n=2
print(par_n,v)
Grads_mean=np.mean(Grads_file,axis=(1))[:,par_n] # rank 1 dim len(eLes)
Grads_bias=np.abs(Grads_mean-Grad_log_lik[par_n,0,0])
Grads_var=np.var(Grads_file[:,:,par_n]-Grad_log_lik[par_n,0,0],axis=(1))
Grads_bias_up=Grads_bias+np.sqrt(Grads_var)*1.96/np.sqrt(samples)  
Grads_bias_lb=Grads_bias-np.sqrt(Grads_var)*1.96/np.sqrt(samples)  
#print(Grads_bias_lb)
#print(Grads_bias_lb)
#print(Grads_bias_up)
plt.plot(eLes,Grads_bias,label="Gradient bias")
plt.plot(eLes,Grads_bias_up,label="Gradient bias up")
plt.plot(eLes,Grads_bias_lb,label="Gradient bias lb")
print(Grads_mean,Grad_log_lik[par_n])
print(Grads_file.shape,Grad_log_lik.shape)
#grad_bias=np.abs(np.mean(grads_mean[:,:,0,par_n]-Grad_log_lik[par_n],axis=1))
#r_grad_bias=np.abs(np.mean(grads_mean[:,:,0,par_n]-grads_mean[:,:,1,par_n],axis=1))
#plt.plot(eLes,grad_bias,label="Gradient bias")
#plt.plot(eLes,r_grad_bias,label="Richardson gradient bias")
plt.plot(eLes,2**(eLes[-1])/2**(eLes)*Grads_bias[-1],label="$\Delta_l$")
#plt.plot(eLes,r_grad_bias,label="Richardson gradient bias")
plt.yscale("log")
plt.legend()
#%%

#%%

i=4
#len(eLes),samples,int(T/d))


print(v)
print(np.mean(ch_paths_file,axis=(1))[:,i])
print(x_kf_smooth[i+1,0])
s_R_bias=np.abs(np.mean(ch_paths_file[1:,:]-ch_paths_file[:-1,:],axis=(1)))[:,i]  #smoothing richardson bias
sum_s_R_bias=np.sum(np.abs(np.mean(ch_paths_file[1:,:]-ch_paths_file[:-1,:],axis=(1))),axis=1)
s_bias=np.abs(np.mean(ch_paths_file-x_kf_smooth[1:,0],axis=(1)))[:,i]
var_bias=np.var(ch_paths_file-x_kf_smooth[1:,0],axis=(1))[:,i]
s_bias_up=s_bias+np.sqrt(var_bias)*1.96/np.sqrt(samples*mcmc_links)
s_bias_lb=s_bias-np.sqrt(var_bias)*1.96/np.sqrt(samples*mcmc_links)
plt.plot(eLes[1:],s_R_bias,label="Smoothing Richardson bias")
#plt.plot(eLes[1:],sum_s_R_bias,label="Sum smoothing Richardson bias")
plt.plot(eLes,s_bias,label="Smoothing bias")
plt.plot(eLes,s_bias_up,label="Smoothing bias up")
plt.plot(eLes,s_bias_lb,label="Smoothing bias lb")
plt.plot(eLes,2**(eLes[0])/2**(eLes)*sum_s_R_bias[0],label="$\Delta_l$")
plt.plot(eLes,2**(eLes[0]/2)/2**(eLes/2)*sum_s_R_bias[0],label="$\Delta_l^{1/2}$")
plt.legend()
print(s_bias_lb)
plt.yscale("log")

#%%
###########################################################################
###########################################################################
###########################################################################
#%%
if True:

    x0_sca=1.2
    x0=x0_sca
    l=10
    T=10
    t0=0
    l_d=0
    d=2**(l_d)
    theta_true=-0.3
    sigma_true=0.8
    sd_true=0.8
    np.random.seed(7)
    collection_input=[bdg.b_ou_1d,theta_true,bdg.Sig_ou_1d,sigma_true]
    resamp_coef=1
    l_max=10
    x_true= bdg.gen_gen_data_1d(T,x0_sca,l_max,collection_input)
    x_reg=bdg.cut(T,l_max,-l_d,x_true)[1:]
    dim=1
    np.random.seed(1007)
    d_times=np.array(range(t0+d,int(T/d)+1))*d
    obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)
    resamp_coef=1
    N=50
    start=time.time()
    mcmc_links=5
    #mcmc_links=10
    SGD_steps=4
    B=mcmc_links*SGD_steps
    fd=1e-8
    theta_in=-0.8
    sigma_in=1
    sd_in=1
    theta_in_fd=theta_in+fd
    sigma_in_fd=sigma_in+fd
    sigma_in_aux=sigma_in
    theta_in_aux=theta_in+0.2
    sigma_in_aux_fd=sigma_in_aux+fd
    #arg_cm=int(sys.argv[1])
    #arg_cm=32
    samples=200*3*20
    seed=4253#+samples*(arg_cm-1)
    #samples=2
    gamma=0.15
    alpha=0.5
    seed=2393
    x0=x0_sca+np.zeros(N)
    l0=2
    L_max=8
    eLes=np.array(range(l0,L_max+1))
    print(eLes)
#%%
v="rcv2_10"
#ch_paths_file_van=np.reshape(np.loadtxt("Observations&data/Prl_SGD_ou_bridge_van_comparison_ch_paths_v"+v+".txt",dtype=float),(samples,B,int(T/d)))
#pars_file_van=np.reshape(np.loadtxt("Observations&data/Prl_SGD_ou_bridge_van_comparison_pars_v"+v+".txt",dtype=float),(samples,SGD_steps+1,3))
#ch_paths_file=np.reshape(np.loadtxt("Observations&data/Prl_SGD_ou_bridge_comparison_ch_paths_v"+v+".txt",dtype=float),(samples,B,int(T/d)))
#pars_file=np.reshape(np.loadtxt("Observations&data/Prl_SGD_ou_bridge_comparison_pars_v"+v+".txt",dtype=float),(samples,SGD_steps+1,3))
pars_file=np.reshape(np.loadtxt("Observationsdata/data2/Prl_C_SGD_ou_bridge_pars_v"+v+".txt",dtype=float),(len(eLes), samples,SGD_steps+1,3))
Grads_file=np.reshape(np.loadtxt("Observationsdata/data2/Prl_C_SGD_ou_bridge_Grads_v"+v+".txt",dtype=float),(len(eLes),samples,B,3))
#%%

dim=1
dim_o=1
theta,sigma,sd=theta_in,sigma_in,sd_in
K=np.array([[np.exp(d*theta)]])
G=np.array([[sigma*np.sqrt((np.exp(2*d*theta)-1)/(2*theta))]])
H=np.array([[1]])
D=np.array([[sd]])
Grad_S=np.array([[[0]],[[0]],[[1]]],dtype=float)
Grad_R_sigma_s=(np.exp(2*theta*d)-1)/(2*theta)
Grad_R_theta=(sigma**2/(2*theta**2))*(1-np.exp(2*theta*d)+2*d*theta*np.exp(2*theta*d))
Grad_R=np.zeros((3,1,1),dtype=float)
Grad_R[0,0,0]=Grad_R_theta
Grad_R[1,0,0]=Grad_R_sigma_s
Grad_K=np.array([[[d*np.exp(d*theta)]],[[0]],[[0]]],dtype=float)
x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
x_kf_2=bdg.KF(x0[0],dim,dim_o,K,G,H,D,obs)[0]
Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma

#%%

#(len(eLes),samples,SGD_steps+1,3)
#(len(eLes),samples,B,3)
par_n=0
Grads_mean=np.mean(Grads_file[:,:,:mcmc_links],axis=(1,2))[:,par_n] # rank 1 dim len(eLes)
Grads_bias=np.abs(Grads_mean-Grad_log_lik[par_n,0,0])
Grads_var=np.var(Grads_file[:,:,:mcmc_links,par_n]-Grad_log_lik[par_n,0,0],axis=(1,2))
Grads_bias_up=Grads_bias+np.sqrt(Grads_var)*1.96/np.sqrt(samples*mcmc_links)  
Grads_bias_lb=Grads_bias-np.sqrt(Grads_var)*1.96/np.sqrt(samples*mcmc_links)  
#print(Grads_bias_lb)
#print(Grads_bias_lb)
#print(Grads_bias_up)
plt.plot(eLes,Grads_bias,label="Gradient bias")
plt.plot(eLes,Grads_bias_up,label="Gradient bias up")
plt.plot(eLes,Grads_bias_lb,label="Gradient bias lb")
print(Grads_mean,Grad_log_lik[par_n])
print(Grads_file.shape,Grad_log_lik.shape)
#grad_bias=np.abs(np.mean(grads_mean[:,:,0,par_n]-Grad_log_lik[par_n],axis=1))
#r_grad_bias=np.abs(np.mean(grads_mean[:,:,0,par_n]-grads_mean[:,:,1,par_n],axis=1))
#plt.plot(eLes,grad_bias,label="Gradient bias")
#plt.plot(eLes,r_grad_bias,label="Richardson gradient bias")
plt.plot(eLes,2**(eLes[-1])/2**(eLes)*Grads_bias[-1],label="$\Delta_l$")
#plt.plot(eLes,r_grad_bias,label="Richardson gradient bias")
plt.yscale("log")
plt.legend()

#%%
# This space is dedicated to the test of the coupled SGD, and it's bias and coupling in terms of the 
# time discretization.
# Coupled bias and sm 
if True:
    
    x0_sca=1.2
    x0=x0_sca
    l=10
    T=10
    t0=0
    l_d=0
    d=2**(l_d)
    theta_true=-0.3
    sigma_true=0.8
    sd_true=0.8
    np.random.seed(7)
    collection_input=[bdg.b_ou_1d,theta_true,bdg.Sig_ou_1d,sigma_true]
    resamp_coef=1
    l_max=10
    x_true= bdg.gen_gen_data_1d(T,x0_sca,l_max,collection_input)
    x_reg=bdg.cut(T,l_max,-l_d,x_true)[1:]
    dim=1
    np.random.seed(1007)
    d_times=np.array(range(t0+d,int(T/d)+1))*d
    obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)
    resamp_coef=1
    N=50
    start=time.time()
    mcmc_links=5
    #mcmc_links=10
    SGD_steps=4
    B=mcmc_links*SGD_steps
    fd=1e-8
    theta_in=-0.8
    sigma_in=1
    sd_in=1
    theta_in_fd=theta_in+fd
    sigma_in_fd=sigma_in+fd
    sigma_in_aux=sigma_in
    theta_in_aux=theta_in+0.2
    sigma_in_aux_fd=sigma_in_aux+fd
    #arg_cm=int(sys.argv[1])
    #arg_cm=32
    samples=200*3*20
    seed=4253#+samples*(arg_cm-1)
    #samples=2
    gamma=0.15
    alpha=0.5
    seed=2393
    x0=x0_sca+np.zeros(N)
    l0=2
    L_max=8
    eLes=np.array(range(l0,L_max+1))
#%%

labels= [str(i) for i in range(1,13)]
i=0
ch_paths_file=np.reshape(np.loadtxt("Observationsdata/data_grad_bias/Prl_C_SGD_ou_bridge_ch_paths_va"+labels[i]+".txt",dtype=float),(len(eLes),samples,2,B,int(T/d)))
pars_file=np.reshape(\
    np.loadtxt("Observationsdata/data_grad_bias/Prl_C_SGD_ou_bridge_pars_va"+labels[i]+".txt",dtype=float),(len(eLes),samples,2,SGD_steps+1,3)) 
Grads_file=np.reshape(np.loadtxt("Observationsdata/data_grad_bias/Prl_C_SGD_ou_bridge_Grads_va"+labels[i]+".txt",dtype=float),(len(eLes),samples,2,B,3))

for i in range(len(labels[1:])):
    ch_paths_file=np.concatenate((ch_paths_file,np.reshape(\
    np.loadtxt("Observationsdata/data_grad_bias/Prl_C_SGD_ou_bridge_ch_paths_va"+labels[i+1]+".txt",dtype=float),(len(eLes),samples,2,B,int(T/d
    )))),axis=1)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data_grad_bias/Prl_C_SGD_ou_bridge_pars_va"+labels[i+1]+".txt",dtype=float),(len(eLes),samples,2,SGD_steps+1,3)) ),axis=1)  
    Grads_file=np.concatenate((Grads_file,np.reshape(\
    np.loadtxt("Observationsdata/data_grad_bias/Prl_C_SGD_ou_bridge_Grads_va"+labels[i+1]+".txt",dtype=float),(len(eLes),samples,2,B,3))),axis=1)  
print(Grads_file.shape)
#%%
v="rcv2_10"
pars_file=np.reshape(\
    np.loadtxt("Observationsdata/data2/Prl_C_SGD_ou_bridge_pars_v"+v+".txt",dtype=float),(len(eLes),samples,2,SGD_steps+1,3)) 
Grads_file=np.reshape(np.loadtxt("Observationsdata/data2/Prl_C_SGD_ou_bridge_Grads_v"+v+".txt",dtype=float),(len(eLes),samples,2,B,3))
ch_paths_file=np.reshape(np.loadtxt("Observationsdata/data2/Prl_C_SGD_ou_bridge_ch_paths_v"+v+".txt",dtype=float),(len(eLes),samples,2,B,int(T/d)))
#print(pars_file[0,:,1,a:,x])
#%%


dim=1
dim_o=1
theta,sigma,sd=theta_in,sigma_in,sd_in
K=np.array([[np.exp(d*theta)]])
G=np.array([[sigma*np.sqrt((np.exp(2*d*theta)-1)/(2*theta))]])
H=np.array([[1]])
D=np.array([[sd]])
Grad_S=np.array([[[0]],[[0]],[[1]]],dtype=float)
Grad_R_sigma_s=(np.exp(2*theta*d)-1)/(2*theta)
Grad_R_theta=(sigma**2/(2*theta**2))*(1-np.exp(2*theta*d)+2*d*theta*np.exp(2*theta*d))
Grad_R=np.zeros((3,1,1),dtype=float)
Grad_R[0,0,0]=Grad_R_theta
Grad_R[1,0,0]=Grad_R_sigma_s
Grad_K=np.array([[[d*np.exp(d*theta)]],[[0]],[[0]]],dtype=float)
x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
x_kf_2=bdg.KF(x0[0],dim,dim_o,K,G,H,D,obs)[0]
Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma

#%%
#(len(eLes),samples,2,B,3)
par_n=2
lev=1
print(par_n,v, " level is: " , lev)
print(Grads_file.shape)
Grads_mean=np.mean(Grads_file[:,:,:,:mcmc_links],axis=(1,3))[:,lev,par_n] # rank 1 dim len(eLes)
Grads_bias=np.abs(Grads_mean-Grad_log_lik[par_n,0,0])
R_bias_Grads=np.abs(np.mean(Grads_file[:,:,0,:mcmc_links]-Grads_file[:,:,1,:mcmc_links],axis=(1,2))[:,par_n] )
second_R_bias_Grads=np.abs(np.mean(Grads_file[1:,:,lev,:mcmc_links]-Grads_file[:-1,:,lev,:mcmc_links],axis=(1,2))[:,par_n] )
Grads_var=np.var(Grads_file[:,:,lev,:mcmc_links,par_n]-Grad_log_lik[par_n,0,0],axis=(1,2))
Grads_bias_up=Grads_bias+np.sqrt(Grads_var)*1.96/np.sqrt(samples*mcmc_links)  
Grads_bias_lb=Grads_bias-np.sqrt(Grads_var)*1.96/np.sqrt(samples*mcmc_links)  
#print(Grads_bias_lb)
#print(Grads_bias_lb)
#print(Grads_bias_up)
plt.plot(eLes,Grads_bias,label="Gradient bias")
plt.plot(eLes,Grads_bias_up,label="Gradient bias up")
plt.plot(eLes,Grads_bias_lb,label="Gradient bias lb")
plt.plot(eLes,R_bias_Grads,label="R grad")
plt.plot(eLes[1:],second_R_bias_Grads,label="second R grad")
print(Grads_mean,Grad_log_lik[par_n])
print(Grads_file.shape,Grad_log_lik.shape)
print(Grads_bias)
#grad_bias=np.abs(np.mean(grads_mean[:,:,0,par_n]-Grad_log_lik[par_n],axis=1))
#r_grad_bias=np.abs(np.mean(grads_mean[:,:,0,par_n]-grads_mean[:,:,1,par_n],axis=1))
#plt.plot(eLes,grad_bias,label="Gradient bias")
#plt.plot(eLes,r_grad_bias,label="Richardson gradient bias")
plt.plot(eLes,2**(eLes[-1])/2**(eLes)*Grads_bias[-1],label="$\Delta_l$")
#plt.plot(eLes,r_grad_bias,label="Richardson gradient bias")
plt.yscale("log")
plt.legend()

 #%%
# (len(eLes),samples,2,B,int(T/d))
i=9
print(i,v)
lev=1
#print(ch_paths.shape)
print(np.mean(ch_paths_file[:,:,:,:mcmc_links],axis=(1,3))[:,lev,i])
print(x_kf_smooth[i+1,0])
s_R_bias=np.abs(np.mean(ch_paths_file[:,:,1,:mcmc_links]-ch_paths_file[:,:,0,:mcmc_links],axis=(1,2)))[:,i]  #smoothing richardson bias
second_s_R_bias=np.abs(np.mean(ch_paths_file[1:,:,1,:mcmc_links]-ch_paths_file[:-1,:,1,:mcmc_links],axis=(1,2)))[:,i]
#sum_s_R_bias=np.sum(np.abs(np.mean(ch_paths_file[:,:,0]-ch_paths_file[:,:,1],axis=(1,2))),axis=1)
s_bias=np.abs(np.mean(ch_paths_file[:,:,:,:mcmc_links]-x_kf_smooth[1:,0],axis=(1,3)))[:,lev,i]
var_bias=np.var(ch_paths_file[:,:,:,:mcmc_links]-x_kf_smooth[1:,0],axis=(1,3))[:,lev,i]
s_bias_up=s_bias+np.sqrt(var_bias)*1.96/np.sqrt(samples*mcmc_links)
s_bias_lb=s_bias-np.sqrt(var_bias)*1.96/np.sqrt(samples*mcmc_links)
plt.plot(eLes,s_R_bias,label="Smoothing Richardson bias")
#plt.plot(eLes,sum_s_R_bias,label="Sum smoothing Richardson bias")
plt.plot(eLes,s_bias,label="Smoothing bias")
plt.plot(eLes[1:],second_s_R_bias,label="Second R bias")
plt.plot(eLes,s_bias_up,label="Smoothing bias up")
plt.plot(eLes,s_bias_lb,label="Smoothing bias lb")
plt.plot(eLes,2**(eLes[0])/2**(eLes)*s_bias[0],label="$\Delta_l$")
plt.plot(eLes,2**(eLes[0]/2)/2**(eLes/2)*s_bias[0],label="$\Delta_l^{1/2}$")
plt.legend()
plt.yscale("log")
#%%
Grid_p=19
thetas=np.linspace(-1,1,Grid_p)*0.1+pars[-1,0]
sigmas=np.linspace(-1,1,Grid_p)*0.1 +pars[-1,1]
sds=np.linspace(-1,1,Grid_p)*0.1 +pars[-1,1]
Grid=np.stack((thetas,sigmas,sds))
thetass_aux=thetas+0.2
sigmas_aux=sigmas
fd_rate=1e-4
[theta_fd,sigma_fd,sd_fd]=np.array([thetas,sigmas,sds])+fd_rate*(np.zeros((3,Grid_p))+1)
x=0
y=2
print(sd_in,sigma_in)
#print(Grid.shape)
#%%
# IN 2d
Grads=np.zeros((Grid_p,Grid_p,3))
dim=1
dim_o=1
[theta_0,sigma_0,sd_0]=np.array([pars[-1,0],pars[-1,1],pars[-1,2]])  # np.array([theta_in,sigma_in,sd_in])   
for i in range(len(Grid[x])):
    par_x=Grid[x][i]
    for j in range(len(sigmas)):
        #sigma=sigmas[j]
        par_y=Grid[y][j]
        theta=(y==0)*par_y+(x==0)*par_x+ (x!=0)*(y!=0)*theta_0
        sigma=(y==1)*par_y+(x==1)*par_x+ (x!=1)*(y!=1)*sigma_0
        sd=(y==2)*par_y+(x==2)*par_x+ (x!=2)*(y!=2)*sd_0
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
        x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
        Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
        Grads[j,i]=Grad_log_lik[:,0,0]
#%%
#[theta_0,sigma_0,sd_0]=np.array([theta_true,sigma_true,sd_true])+np.array([0.4,-0.3,0])
[theta_0,sigma_0,sd_0]=[theta_in,sigma_in,sd_in]
SGD_steps_an=1000
pars=np.zeros((SGD_steps_an+1,3))
Grads_test=np.zeros((SGD_steps_an+1,3))
gamma_new=0.4
#alpha=0.01
theta=theta_0
sigma= sigma_0
sd=sd_0
pars[0,:]=np.array([theta_in,sigma_in,sd_in])
for b_ind in range(SGD_steps_an):
    #sigma=sigmas[j]
    #print(theta,sigma)
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
    x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
    Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
    Grads_test[b_ind]=Grad_log_lik[:,0,0]
    #print(Grads_test[b_ind,:2])
    #print(Grads_test[b_ind,0])
    theta+=gamma_new*Grads_test[b_ind,0]/(b_ind+1)**(0.5+alpha)#*((x==0)+(y==0))
    sigma*=np.exp((gamma_new*sigma*Grads_test[b_ind,1]/(b_ind+1)**(0.5+alpha)))#*((x==1)+(y==1))
    sd*=np.exp((gamma_new*sd*Grads_test[b_ind,2]/(b_ind+1)**(0.5+alpha)))#*((x==2)+(y==2))
    pars[b_ind+1]=np.array([theta,sigma,sd])
print(pars[-1])
#%%
#len(eLes),samples,2,SGD_steps+1,3)
a=0
lev=0
c=0
e=10
#a=SGD_steps-3
print("pars_0 is: ")
print("sd is: ",sd_true)
#plt.plot(pars[a:,0].T,pars[a:,1].T)
print(pars_file[lev,:,1,a:,y])
plt.plot(pars_file[lev,c:e,1,a:,x].T,pars_file[lev,c:e,1,a:,y].T)
plt.plot(pars_file[lev,c:e,0,a:,x].T,pars_file[lev,c:e,0,a:,y].T)
x_Grid,y_Grid=np.meshgrid(Grid[x],Grid[y])
plt.quiver(x_Grid,y_Grid,Grads[:,:,x],Grads[:,:,y])
plt.xlabel("Theta")
plt.ylabel("Sigma")
plt.title("SGD")
#plt.savefig("Gradiend_flow_&_SGD.pdf")
plt.show()
#%%
# change i \in {0,1,2} to chekc the behaviour of the different parameters.
i=2
#(len(eLes),samples,2,SGD_steps+1,3)
s0=2**0
sm_pars=np.mean((pars_file[:,:,0,s0]-pars_file[:,:,1,s0])**2,axis=1)
#sm_pars=np.mean((pars_file[:,:,0,-1]-pars_file[:,:,1,-1])**2,axis=1)
plt.plot(eLes,sm_pars[:,i],label="Second moment of the difference of the parameters")
plt.plot(eLes,2**(eLes[-1])/2**(eLes)*sm_pars[-1,i],label="$\Delta_l$")
plt.legend()
plt.yscale("log")
print(sm_pars[0,:],2**(eLes[-1])/2**(eLes[0])*sm_pars[-1]*2**eLes[0])
 
#%%
if True:

    x0_sca=1.2
    x0=x0_sca
    l=10
    T=10
    t0=0
    l_d=0
    d=2**(l_d)
    theta_true=-0.1
    sigma_true=0.2
    sd_true=0.8
    np.random.seed(3)
    collection_input=[bdg.b_ou_1d,theta_true,bdg.Sig_ou_1d,sigma_true]
    resamp_coef=1
    l_max=10
    x_true= bdg.gen_gen_data_1d(T,x0_sca,l_max,collection_input)
    x_reg=bdg.cut(T,l_max,-l_d,x_true)[1:]
    times=np.array(range(t0,int(T/d)+1))*d
    l_times=np.arange(t0,T,2**(-l))
    l_max_times=np.arange(t0,T,2**(-l_max))
    np.random.seed(4)
    d_times=np.array(range(t0+d,int(T/d)+1))*d
    obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)
    resamp_coef=1
    N=500
    start=time.time()
    mcmc_links=int(600*6*8*2*0.3)
    #mcmc_links=10
    SGD_steps=1
    B=mcmc_links*SGD_steps
    fd=1e-8
    theta_in=-1
    sigma_in=1
    sd_in=1
    theta_in_fd=theta_in+fd
    sigma_in_fd=sigma_in+fd
    sigma_in_aux=sigma_in
    theta_in_aux=theta_in+0.2
    sigma_in_aux_fd=sigma_in_aux+fd
    
    #arg_cm=int(sys.argv[1])
    #arg_cm=32
    samples=40
    seed=4253#+samples*(arg_cm-1)
    #samples=2
    gamma=0.3
    alpha=0.5
    seed=2393
    x0=x0_sca+np.zeros(N)
    l0=3
    L_max=8
    eLes=np.array(range(l0,L_max+1))
#%%

labels= [str(i) for i in range(1,11)]
i=0
pars_file=np.reshape(\
    np.loadtxt("Observationsdata/data_grad_bias/Prl_SGD_ou_bridge_pars_vsingle10v"+labels[i]+".txt",dtype=float),(len(eLes),samples,SGD_steps+1,3)) 
levels_file=np.reshape(np.loadtxt("Observationsdata/data_grad_bias/Prl_SGD_ou_bridge_Grads_vsingle10v"+labels[i]+".txt",dtype=int),(len(eLes),samples,B,3))

for i in range(len(labels[1:])):
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data_grad_bias/Prl_SGD_ou_bridge_pars_vsingle10v"+labels[i+1]+".txt",dtype=float),(len(eLes),samples,SGD_steps+1,3)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data_grad_bias/Prl_SGD_ou_bridge_Grads_vsingle10v"+labels[i+1]+".txt",dtype=int),(len(eLes),samples,B,3))),axis=0)  
#%%
v="singlec9"
ch_paths_file=np.reshape(np.loadtxt("Observationsdata/data2/Prl_SGD_ou_bridge_ch_paths_v"+v+".txt",dtype=float),(len(eLes),samples,B,int(T/d)))
pars_file=np.reshape(np.loadtxt("Observationsdata/data2/Prl_SGD_ou_bridge_pars_v"+v+".txt",dtype=float),(len(eLes),samples,SGD_steps+1,3))   
Grads_file=np.reshape(np.loadtxt("Observationsdata/data2/Prl_SGD_ou_bridge_Grads_v"+v+".txt",dtype=float),(len(eLes),samples,B,3))
#%%


dim=1
dim_o=1
theta,sigma,sd=theta_in,sigma_in,sd_in
K=np.array([[np.exp(d*theta)]])
G=np.array([[sigma*np.sqrt((np.exp(2*d*theta)-1)/(2*theta))]])
H=np.array([[1]])
D=np.array([[sd]])
Grad_S=np.array([[[0]],[[0]],[[1]]],dtype=float)
Grad_R_sigma_s=(np.exp(2*theta*d)-1)/(2*theta)
Grad_R_theta=(sigma**2/(2*theta**2))*(1-np.exp(2*theta*d)+2*d*theta*np.exp(2*theta*d))
Grad_R=np.zeros((3,1,1),dtype=float)
Grad_R[0,0,0]=Grad_R_theta
Grad_R[1,0,0]=Grad_R_sigma_s
Grad_K=np.array([[[d*np.exp(d*theta)]],[[0]],[[0]]],dtype=float)
x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
x_kf_2=bdg.KF(x0[0],dim,dim_o,K,G,H,D,obs)[0]
Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma

#%%
#len(eLes),samples,B,3)
#samples=40*len(labels)
par_n=0
print(par_n, v)

step=1
Grads_mean=np.mean(Grads_file[:,:,:,:mcmc_links],axis=(1,2))[:,par_n]
Grads_bias=np.abs(Grads_mean-Grad_log_lik[par_n,0,0])
Grads_var=np.var(Grads_file[:,:,:mcmc_links,par_n]-Grad_log_lik[par_n,0,0],axis=(1,2))
Grads_bias_up=Grads_bias+np.sqrt(Grads_var)*1.96/np.sqrt(samples*mcmc_links)  
Grads_bias_lb=Grads_bias-np.sqrt(Grads_var)*1.96/np.sqrt(samples*mcmc_links)  
#print(Grads_bias_lb)

#print(Grads_bias_lb)
#print(Grads_bias_up)
plt.plot(eLes,Grads_bias,label="Gradient bias")
plt.plot(eLes,Grads_bias_up,label="Gradient bias up")
plt.plot(eLes,Grads_bias_lb,label="Gradient bias lb")
print(Grads_mean,Grad_log_lik[par_n])
print(Grads_file.shape,Grad_log_lik.shape)
#grad_bias=np.abs(np.mean(grads_mean[:,:,0,par_n]-Grad_log_lik[par_n],axis=1))
#r_grad_bias=np.abs(np.mean(grads_mean[:,:,0,par_n]-grads_mean[:,:,1,par_n],axis=1))
#plt.plot(eLes,grad_bias,label="Gradient bias")
#plt.plot(eLes,r_grad_bias,label="Richardson gradient bias")
plt.plot(eLes,2**(eLes[-1])/2**(eLes)*Grads_bias[-1],label="$\Delta_l$")
#plt.plot(eLes,r_grad_bias,label="Richardson gradient bias")
plt.yscale("log")
plt.legend()
# In the following I test the coupled SGD bias and other statistics
#%%
#%%
#(len(eLes),samples,B,int(T/d))
i=9
print(i,v)
lev=1
print(ch_paths_file.shape)
#print(np.mean(ch_paths_file,axis=(1,2))[:,i])
print(x_kf_smooth[i+1,0])
s_R_bias=np.abs(np.mean(ch_paths_file[1:]-ch_paths_file[:-1],axis=(1,2)))[:,i]  #smoothing richardson bias
s_bias=np.abs(np.mean(ch_paths_file-x_kf_smooth[1:,0],axis=(1,2)))[:,i]
var_bias=np.var(ch_paths_file-x_kf_smooth[1:,0],axis=(1,2))[:,i]

s_bias_up=s_bias+np.sqrt(var_bias)*1.96/np.sqrt(samples*mcmc_links)
s_bias_lb=s_bias-np.sqrt(var_bias)*1.96/np.sqrt(samples*mcmc_links)
plt.plot(eLes[1:],s_R_bias,label="Smoothing Richardson bias")
#plt.plot(eLes,sum_s_R_bias,label="Sum smoothing Richardson bias")
plt.plot(eLes,s_bias,label="Smoothing bias")
plt.plot(eLes,s_bias_up,label="Smoothing bias up")
plt.plot(eLes,s_bias_lb,label="Smoothing bias lb")
plt.plot(eLes,2**(eLes[0])/2**(eLes)*s_bias[0],label="$\Delta_l$")
#plt.plot(eLes,2**(eLes[0]/2)/2**(eLes/2)*sum_s_R_bias[0],label="$\Delta_l^{1/2}$")
plt.legend()
plt.yscale("log")
##%%
#%%
"""
    x_true= bdg.gen_gen_data_1d(T,x0_sca,l_max,collection_input)
    x_reg=bdg.cut(T,l_max,-l_d,x_true)[1:]
    times=np.array(range(t0,int(T/d)+1))*d
    l_times=np.arange(t0,T,2**(-l))
    l_max_times=np.arange(t0,T,2**(-l_max))
    np.random.seed(5)
    d_times=np.array(range(t0+d,int(T/d)+1))*d
    obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)
    resamp_coef=1
    N=50 
    start=time.time()
    mcmc_links=300
    #mcmc_links=10
    SGD_steps=1
    B=mcmc_links*SGD_steps
    fd=1e-12
    theta_in=-0.67
    sigma_in=1.6
    sd_in=1.2
    theta_in_fd=theta_in+fd
    sigma_in_fd=sigma_in+fd
    sigma_in_aux=sigma_in
    theta_in_aux=theta_in+0.2
    sigma_in_aux_fd=sigma_in_aux+fd
    
    #arg_cm=int(sys.argv[1])
    #arg_cm=32
    samples=40
    seed=4253#+samples*(arg_cm-1)
    #samples=2
    gamma=0.3
    alpha=0.5
    seed=2393
    x0=x0_sca+np.zeros(N)
    l0=3
    L_max=10
    eLes=np.array(range(l0,L_max+1))

"""




if True:

    x0_sca=1.2
    x0=x0_sca
    l=10
    T=5
    t0=0
    l_d=0
    d=2**(l_d)
    theta_true=-0.3
    sigma_true=1.2
    sd_true=0.55
    np.random.seed(7)
    collection_input=[bdg.b_ou_1d,theta_true,bdg.Sig_ou_1d,sigma_true]
    resamp_coef=1
    l_max=10
    x_true= bdg.gen_gen_data_1d(T,x0_sca,l_max,collection_input)
    x_reg=bdg.cut(T,l_max,-l_d,x_true)[1:]
    times=np.array(range(t0,int(T/d)+1))*d
    l_times=np.arange(t0,T,2**(-l))
    l_max_times=np.arange(t0,T,2**(-l_max))
    
    np.random.seed(3)
    d_times=np.array(range(t0+d,int(T/d)+1))*d
    obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)
    resamp_coef=1
    N=100
    start=time.time()
    mcmc_links=50
    SGD_steps=8
    B=mcmc_links*SGD_steps
    fd=1e-8
    theta_in=-1
    sigma_in=1
    sd_in=1
    theta_in_fd=theta_in+fd
    sigma_in_fd=sigma_in+fd
    sigma_in_aux=sigma_in
    theta_in_aux=theta_in+0.2
    sigma_in_aux_fd=sigma_in_aux+fd
    
    samples=40
    #samples=2
    gamma=0.7
    alpha=0.5
    seed=2393
    x0=x0_sca+np.zeros(N)
    l0=2
    L_max=7
    eLes=np.array(range(l0,L_max+1))
    v="rcv1_5"
    
#%%
pars_file=np.reshape(np.loadtxt("Observationsdata/data2/Prl_C_SGD_ou_bridge_pars_v"+v+".txt",dtype=float),(len(eLes),samples,2,SGD_steps+1,3))   
Grads_file=np.reshape(np.loadtxt("Observationsdata/data2/Prl_C_SGD_ou_bridge_Grads_v"+v+".txt",dtype=float),(len(eLes),samples,2,B,3))
#%%
dim=1
dim_o=1
theta,sigma,sd=theta_in,sigma_in,sd_in
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
x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
x_kf_2=bdg.KF(x0[0],dim,dim_o,K,G,H,D,obs)[0]
Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
#%%
#(len(eLes),samples,2,B,3))
par_n=2
lev=0
step=0
Grads_mean=np.mean(Grads_file[:,:,:,:mcmc_links],axis=(1,3))[:,lev,par_n]
Grads_bias=np.abs(Grads_mean-Grad_log_lik[par_n,0,0])
Grads_var=np.var(Grads_file[:,:,lev,:mcmc_links,par_n]-Grad_log_lik[par_n,0,0],axis=(1,2))
Grads_bias_up=Grads_bias+np.sqrt(Grads_var)*1.96/np.sqrt(samples*mcmc_links)  
Grads_bias_lb=Grads_bias-np.sqrt(Grads_var)*1.96/np.sqrt(samples*mcmc_links)  
#print(Grads_bias_lb)
#print(Grads_bias_up)
plt.plot(eLes,Grads_bias,label="Gradient bias")
plt.plot(eLes,Grads_bias_up,label="Gradient bias up")
plt.plot(eLes,Grads_bias_lb,label="Gradient bias lb")


print(Grads_mean,Grad_log_lik[par_n])
print(Grads_file.shape,Grad_log_lik.shape)
#grad_bias=np.abs(np.mean(grads_mean[:,:,0,par_n]-Grad_log_lik[par_n],axis=1))
#r_grad_bias=np.abs(np.mean(grads_mean[:,:,0,par_n]-grads_mean[:,:,1,par_n],axis=1))
#plt.plot(eLes,grad_bias,label="Gradient bias")
#plt.plot(eLes,r_grad_bias,label="Richardson gradient bias")
plt.plot(eLes,2**(eLes[-1])/2**(eLes)*Grads_bias[-1],label="$\Delta_l$")
#plt.plot(eLes,r_grad_bias,label="Richardson gradient bias")
plt.yscale("log")
plt.legend()
#%%
Grid_p=19
   
thetas=np.linspace(-1,1,Grid_p)*0.5+theta_in
sigmas=np.linspace(-1,1,Grid_p)*0.5 +sigma_in
sds=np.linspace(-1,1,Grid_p)*0.1+sd_in
Grid=np.stack((thetas,sigmas,sds))
thetas_aux=thetas+0.2
sigmas_aux=sigmas
fd_rate=1e-4
[theta_fd,sigma_fd,sd_fd]=np.array([thetas,sigmas,sds])+fd_rate*(np.zeros((3,Grid_p))+1)
#%%
# IN 2d
Grads=np.zeros((Grid_p,Grid_p,3))
dim=1
#sd=sd_true
sd=pars[-1,-1]
dim_o=1
for i in range(len(thetas)):
    theta=thetas[i]
    for j in range(len(sigmas)):
        sigma=sigmas[j]
        #print(theta,sigma)
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
        x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
        Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
        Grads[j,i]=Grad_log_lik[:,0,0]
#%%
#[theta_0,sigma_0,sd_0]=np.array([theta_true,sigma_true,sd_true])+np.array([0.4,-0.3,0])
[theta_0,sigma_0,sd_0]=[theta_in,sigma_in,sd_in]
SGD_steps=200
pars=np.zeros((SGD_steps+1,3))
Grads_test=np.zeros((SGD_steps+1,3))
#alpha=0.5
#gamma=0.05
theta=theta_0
sigma=sigma_0
sd=sd_0
pars[0,:]=np.array([theta,sigma,sd])
for b_ind in range(SGD_steps):
    #sigma=sigmas[j]
    #print(theta,sigma)
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
    x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
    Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
    Grads_test[b_ind]=Grad_log_lik[:,0,0]
    #print(Grads_test[b_ind,:2])
    theta+=gamma*Grads_test[b_ind,0]/(b_ind+1)**(0.5+alpha)
    sigma+=gamma*Grads_test[b_ind,1]/(b_ind+1)**(0.5+alpha)
    #sd+=gamma*Grads_test[b_ind,2]/(b_ind+1)**(0.5+alpha)
    pars[b_ind+1]=np.array([theta,sigma,sd])

#%%
# len(eLes),samples,2,SGD_steps+1,3)
a=2
b=-1
ele=-6
print("pars_0 is: ")
print("sd is: ",sd_true)
print(pars_file.shape)
sample=0
plt.plot(pars[a:,0].T,pars[a:,1].T)
#plt.plot(pars_file[ele,:,1,a:b,0].T,pars_file[ele,:,1,a:b,1].T)
plt.plot(pars_file[-1,:,0,a:,0].T,pars_file[-1,:,0,a:,1].T)
thetas_Grid,sigmas_Grid=np.meshgrid(thetas,sigmas)
plt.quiver(thetas_Grid,sigmas_Grid,Grads[:,:,0],Grads[:,:,1])

print("The actual parameters are: ",theta_true,sigma_true)
max=np.max(Grads[:,:,0]**2+Grads[:,:,1]**2)
min=np.min(Grads[:,:,0]**2+Grads[:,:,1]**2)
print("The maximum gradient is: ",np.sqrt(max), "The minimum gradient is: ",np.sqrt(min))
plt.xlabel("Theta")
plt.ylabel("Sigma")
plt.title("SGD")
#plt.savefig("Gradiend_flow_&_SGD.pdf")
plt.show()
#%%
# In the following we check the second moment of the difference of the parameters
# (len(eLes),samples,2,SGD_steps+1,3)
i=1
sgd_step=1
#print(sm_pars[-3])
sm_pars=np.mean((pars_file[:,:,0,sgd_step]-pars_file[:,:,1,sgd_step])**2,axis=1)
plt.plot(eLes,sm_pars[:,i],label="Second moment of the difference of the parameters")
plt.plot(eLes,2**(eLes[-1])/2**(eLes)*sm_pars[-1,i],label="$\Delta_l$")
plt.legend()
plt.yscale("log")
#%%

# IN THIS ITERATION WE CHECK THE COUPLING IN TERMS OF THE LEVELS OF THE 
# SGD STEPS
if True==True:
    x0_sca=1.2
    x0=x0_sca
    l=10
    T=10
    t0=0
    l_d=0
    d=2**(l_d)
    theta_true=-0.3
    sigma_true=0.8
    sd_true=0.8
    np.random.seed(7)
    collection_input=[bdg.b_ou_1d,theta_true,bdg.Sig_ou_1d,sigma_true]
    resamp_coef=1
    l_max=10
    x_true= bdg.gen_gen_data_1d(T,x0_sca,l_max,collection_input)
    x_reg=bdg.cut(T,l_max,-l_d,x_true)[1:]
    times=np.array(range(t0,int(T/d)+1))*d
    l_times=np.arange(t0,T,2**(-l))
    l_max_times=np.arange(t0,T,2**(-l_max))
    np.random.seed(1007)
    d_times=np.array(range(t0+d,int(T/d)+1))*d
    obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)
    resamp_coef=1
    N=50
    start=time.time()
    mcmc_links=5
    SGD_steps=32*16*2*14
    #SGD_steps=2
    B=mcmc_links*SGD_steps
    fd=1e-8
    theta_in=-1
    sigma_in=1
    sd_in=1
    theta_in_fd=theta_in+fd
    sigma_in_fd=sigma_in+fd
    sigma_in_aux=sigma_in
    theta_in_aux=theta_in+0.2
    sigma_in_aux_fd=sigma_in_aux+fd
    samples=40
    gamma=0.2
    alpha=0.5
    seed=2393
    x0=x0_sca+np.zeros(N)
    pars=np.zeros((samples,2,SGD_steps+1,3))
    Grads=np.zeros((samples,2,B,3))
    ch_paths=np.zeros((samples,2,B,int(T/d)))
    inputs=[]
    l=6
    v="rcv2_12"
    plt.plot(times[1:],x_reg,label="True signal")
    plt.plot(l_max_times,x_true[:-1],label="True complete signal")
    plt.plot(times[1:], obs,label="Observations")
# %%
pars_file=np.reshape(np.loadtxt("Observationsdata/data3/Prl_C_SGD_ou_bridge_pars_v"+v+".txt",dtype=float),(samples,2,SGD_steps+1,3))   
Grads_file=np.reshape(np.loadtxt("Observationsdata/data3/Prl_C_SGD_ou_bridge_Grads_v"+v+".txt",dtype=float),(samples,2,B,3))
#%%
# here we plot realizations of the SGD
# realizations of the SGD 
l=8
resamp_coef=1
start=time.time()
SGD_steps_an=2**4
mcmc_links_an=5
gamma=0.2
seed=454654
ch_paths_0 ,ch_paths_1,pars_0,pars_1, Grads_test_0,Grads_test_1=\
bdg.C_SGD_bridge(t0,x0,T,bdg.b_ou_1d,theta_in,theta_in_fd,bdg.Sig_ou_1d,sigma_in,sigma_in_fd,\
bdg.b_ou_aux,theta_in_aux,bdg.Sig_ou_aux,sigma_in_aux,sigma_in_aux_fd,\
bdg.r_quasi_normal_1d,[bdg.ou_sd,[theta_in_aux,sigma_in_aux]],[bdg.ou_sd,[theta_in_aux,sigma_in_aux_fd]],\
bdg.H_quasi_normal,\
[bdg.ou_sd,[theta_in_aux,sigma_in_aux],theta_in_aux],[bdg.ou_sd,[theta_in_aux,sigma_in_aux_fd],theta_in_aux],\
bdg.rej_max_coup_ou, [theta_in_aux,sigma_in_aux,theta_in_aux,sigma_in_aux],obs,bdg.log_g_normal_den,sd_in,\
bdg.ou_trans_den,[theta_in_aux,sigma_in_aux],bdg.Grad_log_aux_trans_ou_new,\
bdg.ou_trans_den,[theta_in_aux,sigma_in_aux],bdg.Grad_log_G_new,bdg.update_pars_ou, resamp_coef,l,d, N,seed,fd,mcmc_links_an,SGD_steps_an,gamma,\
alpha)
"""
C_SGD_bridge(t0,x0,T,b,A_in,A_fd_in,Sig,fi_in,fi_fd_in,b_til,A_til_in,Sig_til,fi_til_in,\
    fi_til_fd_in,r,r_pars,r_pars_fd,H,H_pars,H_pars_fd,max_sample_funct,sample_pars,\
    obs,log_g_den,g_den_par_in, aux_trans_den,atdp,\
    Grad_log_aux_trans,prop_trans_den,ind_prop_trans_par, Grad_log_G,resamp_coef, l, d,N,seed,fd_rate,\
    mcmc_links,SGD_steps,gamma, alpha, \
    crossed=False):

"""
end=time.time()
print(end-start)
#%%
#[theta_0,sigma_0,sd_0]=np.array([theta_true,sigma_true,sd_true])+np.array([0.4,-0.3,0])
[theta_0,sigma_0,sd_0]=[theta_in,sigma_in,sd_in]
SGD_steps_an=30
pars=np.zeros((SGD_steps_an+1,3))
Grads_test=np.zeros((SGD_steps_an+1,3))
gamma_new=0.3
#alpha=0.01
theta=theta_0
sigma= sigma_0
dim=1
dim_o=0
sd=sd_0
pars[0,:]=np.array([theta,sigma,sd])
for b_ind in range(SGD_steps_an):
    #sigma=sigmas[j]
    #print(theta,sigma)
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
    x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
    Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
    Grads_test[b_ind]=Grad_log_lik[:,0,0]
    #print(Grads_test[b_ind,:2])
    #print(Grads_test[b_ind,0])
    theta+=gamma_new*Grads_test[b_ind,0]/(b_ind+1)**(0.5+alpha)#*((x==0)+(y==0))
    sigma*=np.exp((gamma_new*sigma*Grads_test[b_ind,1]/(b_ind+1)**(0.5+alpha)))#*((x==1)+(y==1))
    sd*=np.exp((gamma_new*sd*Grads_test[b_ind,2]/(b_ind+1)**(0.5+alpha)))#*((x==2)+(y==2))
    pars[b_ind+1]=np.array([theta,sigma,sd])
print(pars[-1])
#%%
#[-1.0626674   0.23017365  0.50460076]
Grid_p=19
thetas=np.linspace(-1,1,Grid_p)*0.1+pars[-1,0]
sigmas=np.linspace(-1,1,Grid_p)*(pars[-1,1]+0.00021)*0.1 +pars[-1,1]
sds=np.linspace(-1,1,Grid_p)*(pars[-1,2]+0.00021)*0.1 +pars[-1,2]
Grid=np.stack((thetas,sigmas,sds))
thetass_aux=thetas+0.2
sigmas_aux=sigmas
fd_rate=1e-4
[theta_fd,sigma_fd,sd_fd]=np.array([thetas,sigmas,sds])+fd_rate*(np.zeros((3,Grid_p))+1)
x=0
y=2
#print(Grid.shape)
#%%
# IN 2d
Grads=np.zeros((Grid_p,Grid_p,3))
dim=1
dim_o=1
[theta_0,sigma_0,sd_0]=np.array([theta_true,sigma_true,sd_true])   
[theta_0,sigma_0,sd_0]= pars[-1]
for i in range(len(Grid[x])):
    par_x=Grid[x][i]
    for j in range(len(sigmas)):
        #sigma=sigmas[j]
        par_y=Grid[y][j]
        theta=(y==0)*par_y+(x==0)*par_x+ (x!=0)*(y!=0)*theta_0
        sigma=(y==1)*par_y+(x==1)*par_x+ (x!=1)*(y!=1)*sigma_0
        sd=(y==2)*par_y+(x==2)*par_x+ (x!=2)*(y!=2)*sd_0
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
        x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
        Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
        Grads[j,i]=Grad_log_lik[:,0,0]

#%%
# samples,2,SGD_steps+1,3
lev=1
t=9
a=2**(t-1)
a=0
b=2**t
b=SGD_steps_an
#b=-1
c=0
e=40
print(pars[-1])
print("sd is: ",sd_true)
print(SGD_steps)
plt.plot(pars[a:b,x].T,pars[a:b,y].T)
#plt.plot(pars_file[c:e,lev,a:b,x].T,pars_file[c:e,lev,a:b,y].T)
plt.plot(pars_1[a:b,x].T,pars_1[a:b,y].T,label="path 1")
plt.plot(pars_0[a:b,x].T,pars_0[a:b,y].T,label="path 0")
#plt.plot(pars_file[:,1,a:b,x].T,pars_file[:,1,a:b,y].T)
#plt.plot(pars_file[-1,:,0,a:,0].T,pars_file[-1,:,0,a:,1].T)
x_Grid,y_Grid=np.meshgrid(Grid[x],Grid[y])
plt.quiver(x_Grid,y_Grid,Grads[:,:,x],Grads[:,:,y])
#print("The starting guesses are: ",theta_0,sigma_0)
#print("The actual parameters are: ",theta_true,sigma_true)
#max=np.max(Grads[:,:,0]**2+Grads[:,:,1]**2)
#min=np.min(Grads[:,:,0]**2+Grads[:,:,1]**2)
#print("The maximum gradient is: ",np.sqrt(max), "The minimum gradient is: ",np.sqrt(min))
labels=["Theta","Sigma","SD"]
plt.xlabel(labels[x])
plt.ylabel(labels[y])
plt.title("SGD")
#plt.savefig("Gradiend_flow_&_SGD.pdf")
plt.show()
#%%
#%%
#(samples,2,SGD_steps+1,3)
p_levels=np.array(range(int(np.log2(SGD_steps)+1)))
p_levels_0=np.array(range(int(np.log2(SGD_steps))))
S_plev_0=2**(p_levels_0)
S_plev=2**(p_levels)
#print(S_plev)
pars_plev=pars_file[:,:,S_plev,:]
pars_plev_0=pars_file[:,:,int(SGD_steps/2):,:][:,:,S_plev_0,:]
#print(np.mean(pars_file[:,1,-1],axis=0))
#pars_or_plev=pars[S_plev]
#pars_or_plev_0=pars[int(SGD_steps/2):][S_plev_0]
#print(pars_file[0,-1,:,-1])
#%%
# In the following we check the second moment of the difference of the parameters
print(pars_file[0,1,:,2])
i=0
lev=1
#diff_or_square=(pars_or_plev[1:,i]-pars_or_plev[:-1,i])**2
#diff_or_square_0=(pars_or_plev_0[1:,i]-pars_or_plev_0[:-1,i])**2
#print(diff_or_square)
sm_pars=np.mean((pars_plev[:,lev,1:,i]-pars_plev[:,lev,:-1,i])**2,axis=0)
sm_pars_0=np.mean((pars_plev_0[:,1,1:,i]-pars_plev_0[:,1,:-1,i])**2,axis=0) 
plt.plot(p_levels[1:],sm_pars,label="Second moment of the difference of the parameters")
#plt.plot(p_levels_0[1:],sm_pars_0,label="Sm 0")
#plt.plot(p_levels[1:],diff_or_square,label="GD")
#plt.plot(p_levels_0[1:],diff_or_square_0,label="GD 0")
plt.plot(p_levels[1:],2**(p_levels[1])/2**(p_levels[1:])*sm_pars[1],label="$\Delta_p$")
plt.plot(p_levels[1:],2**(p_levels[-1]/2)/2**(p_levels[1:]/2)*sm_pars[-1],label="$\Delta_p^{1/2}$")
#plt.plot(p_levels[1:],2**(p_levels[-1]*3)/2**(p_levels[1:]*3)*diff_or_square[-1],label="$\Delta_p^{3}$")
plt.legend()
plt.title("$alpha=0.01,SGD_steps=gamma/(N_p)^{1/2}$")
plt.xlabel("$p$")
plt.yscale("log")
#%%
#%%%%%
#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
x0_sca=1.2
x0=x0_sca
l=10
T=5
t0=0
l_d=0
d=2**(l_d)
theta_true=-0.3
sigma_true=0.5
sd_true=0.3
np.random.seed(7)
collection_input=[bdg.b_ou_1d,theta_true,bdg.Sig_ou_1d,sigma_true]
resamp_coef=1
l_max=10
x_true= bdg.gen_gen_data_1d(T,x0_sca,l_max,collection_input)
x_reg=bdg.cut(T,l_max,-l_d,x_true)[1:]
times=np.array(range(t0,int(T/d)+1))*d
l_times=np.arange(t0,T,2**(-l))
l_max_times=np.arange(t0,T,2**(-l_max))
np.random.seed(3)
d_times=np.array(range(t0+d,int(T/d)+1))*d
obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)
resamp_coef=1
N=500
start=time.time()
mcmc_links=500
SGD_steps=10
B=mcmc_links*SGD_steps
fd=1e-8
theta_in=-0.325
theta_in_fd=theta_in+fd
sigma_in=1.275
sigma_in_fd=sigma_in+fd
sigma_in_aux=sigma_in
theta_in_aux=theta_in+0.2
sigma_in_aux_fd=sigma_in_aux+fd
sd_in=sd_true
samples=20
gamma=0.05
alpha=0.01
seed=2393
x0=x0_sca+np.zeros(N)
l=6

#%%
resamp_coef=1
start=time.time()
ch_paths_0 ,ch_paths_1,pars_0,pars_1, Grads_test_0,Grads_test_1=\
bdg.C_SGD_bridge(t0,x0,T,bdg.b_ou_1d,theta_in,theta_in_fd,bdg.Sig_ou_1d,sigma_in,sigma_in_fd,\
bdg.b_ou_aux,theta_in_aux,bdg.Sig_ou_aux,sigma_in_aux,sigma_in_aux_fd,\
bdg.r_quasi_normal_1d,[bdg.ou_sd,[theta_in_aux,sigma_in_aux]],[bdg.ou_sd,[theta_in_aux,sigma_in_aux_fd]],\
bdg.H_quasi_normal,\
[bdg.ou_sd,[theta_in_aux,sigma_in_aux],theta_in_aux],[bdg.ou_sd,[theta_in_aux,sigma_in_aux_fd],theta_in_aux],\
bdg.rej_max_coup_ou, [theta_in_aux,sigma_in_aux,theta_in_aux,sigma_in_aux],obs,bdg.log_g_normal_den,sd_in,\
bdg.ou_trans_den,[theta_in_aux,sigma_in_aux],bdg.Grad_log_aux_trans_ou_new,\
bdg.ou_trans_den,[theta_in_aux,sigma_in_aux],bdg.Grad_log_G_new,resamp_coef,l,d, N,seed,fd,mcmc_links,SGD_steps,gamma,\
alpha,crossed=False)
"""
C_SGD_bridge(t0,x0,T,b,A_in,A_fd_in,Sig,fi_in,fi_fd_in,b_til,A_til_in,Sig_til,fi_til_in,\
    fi_til_fd_in,r,r_pars,r_pars_fd,H,H_pars,H_pars_fd,max_sample_funct,sample_pars,\
    obs,log_g_den,g_den_par_in, aux_trans_den,atdp,\
    Grad_log_aux_trans,prop_trans_den,ind_prop_trans_par, Grad_log_G,resamp_coef, l, d,N,seed,fd_rate,\
    mcmc_links,SGD_steps,gamma, alpha, \
    crossed=False):

"""
end=time.time()
print(end-start)
#%%
Grads_0=np.zeros((SGD_steps,3))
Grads_1=np.zeros((SGD_steps,3))
print(l)
for i in range(SGD_steps):
            Grad_mcmc_0=np.mean(Grads_test_0[mcmc_links*i:mcmc_links*(i+1)],axis=0)
            Grad_mcmc_1=np.mean(Grads_test_1[mcmc_links*i:mcmc_links*(i+1)],axis=0)

#%%
Grid_p=21
thetas=np.linspace(-1,1,Grid_p)*0.3-.8
sigmas=np.linspace(-1,1,Grid_p)*0.3+sigma_true
theta_aux=thetas+0.2
sigma_aux=sigmas
sds=np.linspace(-1,1,Grid_p)*0.1+ sd_true
fd_rate=1e-4
[theta_fd,sigma_fd,sd_fd]=np.array([thetas,sigmas,sds])+fd_rate*(np.zeros((3,Grid_p))+1)
print(thetas,sigmas)
#%%
# IN 2d
Grads=np.zeros((Grid_p,Grid_p,3))
dim=1
sd=sd_true
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
        x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
        Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
        Grads[j,i]=Grad_log_lik[:,0,0]
#%%
#[theta_0,sigma_0,sd_0]=np.array([theta_true,sigma_true,sd_true])+np.array([0.4,-0.3,0])
[theta_0,sigma_0,sd_0]=[theta_in,sigma_in,sd_in]
SGD_steps=SGD_steps
pars=np.zeros((SGD_steps+1,3))
Grads_test=np.zeros((SGD_steps+1,3))
#alpha=0.0001
#gamma=0.05
theta=theta_0
sigma=sigma_0
sd=sd_0
pars[0,:]=np.array([theta,sigma,sd])

for b_ind in range(SGD_steps):
    
    #sigma=sigmas[j]
    #print(theta,sigma)
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
    x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
    Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
    Grads_test[b_ind]=Grad_log_lik[:,0,0]
    #print(Grads_test[b_ind,:2])
    theta+=gamma*Grads_test[b_ind,0]/(b_ind+1)**(0.5+alpha)
    sigma+=gamma*Grads_test[b_ind,1]/(b_ind+1)**(0.5+alpha)
    #sd+=gamma*Grads_test[b_ind,2]/(b_ind+1)**(0.5+alpha)
    pars[b_ind+1]=np.array([theta,sigma,sd])
    
#%%
print("pars_0 is: ")
print(pars_0)
print("sd is: ",sd_true)
plt.plot(pars[:,0].T,pars[:,1].T)
plt.plot(pars_0[:,0].T,pars_0[:,1].T)
plt.plot(pars_1[:,0].T,pars_1[:,1].T)
thetas_Grid,sigmas_Grid=np.meshgrid(thetas,sigmas)
plt.quiver(thetas_Grid,sigmas_Grid,Grads[:,:,0],Grads[:,:,1])
print("The starting guesses are: ",theta_0,sigma_0)
print("The actual parameters are: ",theta_true,sigma_true)
max=np.max(Grads[:,:,0]**2+Grads[:,:,1]**2)
min=np.min(Grads[:,:,0]**2+Grads[:,:,1]**2)
print("The maximum gradient is: ",np.sqrt(max), "The minimum gradient is: ",np.sqrt(min))
plt.xlabel("Theta")
plt.ylabel("Sigma")
plt.title("SGD")
#plt.savefig("Gradiend_flow_&_SGD.pdf")
plt.show()
# %%
### This part of the code is built to test the code with an GD and and SGD which is simply
### a GD with gaussian noise.


if True==True:
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
    collection_input=[bdg.b_ou_1d,theta_true,bdg.Sig_ou_1d,sigma_true]
    resamp_coef=1
    l_max=10
    x_true= bdg.gen_gen_data_1d(T,x0_sca,l_max,collection_input)
    x_reg=bdg.cut(T,l_max,-l_d,x_true)[1:]
    times=np.array(range(t0,int(T/d)+1))*d
    l_times=np.arange(t0,T,2**(-l))
    l_max_times=np.arange(t0,T,2**(-l_max))
    sd_true=0.55
    np.random.seed(3)
    d_times=np.array(range(t0+d,int(T/d)+1))*d
    obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)
    resamp_coef=1
    N=50
    start=time.time()
    mcmc_links=40
    SGD_steps=32*8
    B=mcmc_links*SGD_steps
    fd=1e-4
    theta_in=-0.37-0.3
    theta_in_fd=theta_in+fd
    sigma_in=1.33-0.5
    sigma_in_fd=sigma_in+fd
    sigma_in_aux=sigma_in
    theta_in_aux=theta_in+0.2
    sigma_in_aux_fd=sigma_in_aux+fd
    sd_in=0.55
    samples=40
    gamma=0.1
    alpha=0.5
    seed=2393
    x0=x0_sca+np.zeros(N)
   
    l=5
#%%
Grid_p=20
thetas=np.linspace(-1,1,Grid_p)*0.15+theta_true
sigmas=np.linspace(-1,1,Grid_p)*0.15+sigma_true
sds=np.linspace(-1,1,Grid_p)*0.15+sd_true
Grid=np.stack((thetas,sigmas,sds))
theta_aux=thetas+0.2
sigma_aux=sigmas
fd_rate=1e-4
[theta_fd,sigma_fd,sd_fd]=np.array([thetas,sigmas,sds])+fd_rate*(np.zeros((3,Grid_p))+1)
print(thetas,sigmas)
x=0
y=2
print(Grid.shape)
#%%
# IN 2d
Grads=np.zeros((Grid_p,Grid_p,3))
dim=1
par_left=sd_in
dim_o=1
for i in range(len(Grid[x])):
    par_x=Grid[x][i]
    for j in range(len(sigmas)):
        #sigma=sigmas[j]
        par_y=Grid[y][j]
        theta=(y==0)*par_y+(x==0)*par_x+ (x!=0)*(y!=0)*theta_0
        sigma=(y==1)*par_y+(x==1)*par_x+ (x!=1)*(y!=1)*sigma_0
        sd=(y==2)*par_y+(x==2)*par_x+ (x!=2)*(y!=2)*sd_0
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
        x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
        Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
        Grads[j,i]=Grad_log_lik[:,0,0]
#%%
SGD_steps=32*8*16*4
#[theta_0,sigma_0,sd_0]=np.array([theta_true,sigma_true,sd_true])+np.array([0.4,-0.3,0])
[theta_0,sigma_0,sd_0]=np.array([theta_true,sigma_true,sd_true])+0.2
#SGD_steps=32*16*16
pars=np.zeros((SGD_steps+1,3))
Grads_test=np.zeros((SGD_steps+1,3))
#gamma=0.2
#alpha=0.01
theta=theta_0
sigma=sigma_0
sd=sd_0
theta_random=theta_0
sigma_random=sigma_0
sd_random=sd_0
pars[0,:]=np.array([theta,sigma,sd])
ns=10
pars_random=np.zeros((ns,SGD_steps+1,3))
pars_random[:,0,:]=np.array([theta_random,sigma_random,sd_random])    
sd_noise=5e-1
for b_ind in range(SGD_steps):
    #sigma=sigmas[j]
    #print(theta,sigma)
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
    x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
    Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
    Grads_test[b_ind]=Grad_log_lik[:,0,0]
    theta+=gamma*Grads_test[b_ind,0]/(b_ind+1)**(0.5+alpha)*((x==0)+(y==0))
    sigma*=np.exp((gamma*sigma*Grads_test[b_ind,1]/(b_ind+1)**(0.5+alpha))*((x==1)+(y==1)))
    sd*=np.exp((gamma*sd*Grads_test[b_ind,2]/(b_ind+1)**(0.5+alpha))*((x==2)+(y==2)))
    #sd=(1/sd-sd**2*10*gamma*Grads_test[b_ind,2]/(b_ind+1)**(0.5+alpha))**(-1)    
    pars[b_ind+1]=np.array([theta,sigma,sd])

for i in range(ns):
    for b_ind in range(SGD_steps):

        K=np.array([[np.exp(d*theta_random)]])
        G=np.array([[sigma_random*np.sqrt((np.exp(2*d*theta_random)-1)/(2*theta_random))]])
        H=np.array([[1]])
        D=np.array([[sd_random]])
        #print(K,G**2,H,D)
        Grad_S=np.array([[[0]],[[0]],[[1]]],dtype=float)
        Grad_R_sigma_s=(np.exp(2*theta_random*d)-1)/(2*theta_random)
        Grad_R_theta=(sigma_random**2/(2*theta_random**2))*(1-np.exp(2*theta_random*d)\
        +2*d*theta_random*np.exp(2*theta_random*d))
        Grad_R=np.zeros((3,1,1),dtype=float)
        Grad_R[0,0,0]=Grad_R_theta
        Grad_R[1,0,0]=Grad_R_sigma_s
        Grad_K=np.array([[[d*np.exp(d*theta_random)]],[[0]],[[0]]],dtype=float)
        #print(Grad_K,Grad_R,Grad_S)
        x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
        Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma_random
        Grads_test[b_ind]=Grad_log_lik[:,0,0]
        theta_random+=gamma*(Grads_test[b_ind,0]+np.random.normal(0,sd_noise))/(b_ind+1)**(0.5+alpha)*((x==0)+(y==0))
        sigma_random*=np.exp((gamma*sigma_random*(Grads_test[b_ind,1]+np.random.normal(0,sd_noise))/(b_ind+1)**(0.5+alpha))*((x==1)+(y==1)))
        #sd_random*=np.exp((2*gamma*sd_random*(Grads_test[b_ind,2]+np.random.normal(0,sd_noise))/(b_ind+1)**(0.5+alpha))*((x==2)+(y==2)))
        sd_random=(1/sd_random-sd_random**2*10*gamma*Grads_test[b_ind,2]/(b_ind+1)**(0.5+alpha)*((x==2)+(y==2)))**(-1)    
        pars_random[i,b_ind+1]=np.array([theta_random,sigma_random,sd_random])

#%%
print(alpha)
a=SGD_steps-50
a=0
b=SGD_steps
print("pars_0 is: ")
print("sd is: ",sd_true)
plt.plot(pars[a:b,x].T,pars[a:b,y].T)
#print(pars_random[:,-10:,2])
plt.plot(pars_random[:,a:b,x].T,pars_random[:,a:b,y].T)
x_Grid,y_Grid=np.meshgrid(Grid[x],Grid[y])
plt.quiver(x_Grid,y_Grid,Grads[:,:,x],Grads[:,:,y])
#print("The starting guesses are: ",theta_0,sigma_0)
#print("The actual parameters are: ",theta_true,sigma_true)
#max=np.max(Grads[:,:,0]**2+Grads[:,:,1]**2)
#min=np.min(Grads[:,:,0]**2+Grads[:,:,1]**2)
#print("The maximum gradient is: ",np.sqrt(max), "The minimum gradient is: ",np.sqrt(min))
labels=["Theta","Sigma","SD"]
plt.xlabel(labels[x])
plt.ylabel(labels[y])
plt.title("SGD")
#plt.savefig("Gradiend_flow_&_SGD.pdf")
plt.show()
#%%
#(samples,2,SGD_steps+1,3)
p_levels=np.array(range(int(np.log2(SGD_steps)+1)))
p_levels_0=np.array(range(int(np.log2(SGD_steps))))
S_plev_0=2**(p_levels_0)
S_plev=2**(p_levels)
print(S_plev)
pars_or_plev=pars[S_plev]
pars_or_plev_random=pars_random[:,S_plev]
pars_or_plev_0=pars[int(SGD_steps/2):][S_plev_0]
pars_or_plev_0_random=(pars_random[:,int(SGD_steps/2):])[:,S_plev_0]
#print(pars_file[0,-1,:,-1])
#%%
# In the following we check the second moment of the difference of the parameters
#print(pars_file[0,1,:,2])
i=2
diff_or_square=(pars_or_plev[1:,i]-pars_or_plev[:-1,i])**2
diff_or_square_0=(pars_or_plev_0[1:,i]-pars_or_plev_0[:-1,i])**2
diff_or_square_random=np.mean((pars_or_plev_random[:,1:,i]-pars_or_plev_random[:,:-1,i])**2,axis=0)
diff_or_square_0_random=np.mean((pars_or_plev_0_random[:,1:,i]-pars_or_plev_0_random[:,:-1,i])**2,axis=0)
#print(diff_or_square)
#sm_pars=np.mean((pars_plev[:,1,1:,i]-pars_plev[:,1,:-1,i])**2,axis=0)
#sm_pars_0=np.mean((pars_plev_0[:,1,1:,i]-pars_plev_0[:,1,:-1,i])**2,axis=0) 
#plt.plot(p_levels[1:],sm_pars,label="Second moment of the difference of the parameters")
#plt.plot(p_levels_0[1:],sm_pars_0,label="Sm 0")
print(np.mean((pars_random[:,0,i])**2))
#plt.plot(p_levels[1:],np.zeros(len(p_levels[1:]))+np.mean((pars_random[:,0,i])**2),label="GD")
plt.plot(p_levels[1:],diff_or_square_random,label="random GD")
#plt.plot(p_levels[1:],diff_or_square,label="GD")
#plt.plot(p_levels_0[1:],diff_or_square_0,label="GD 0")
plt.plot(p_levels[1:],2**(p_levels[1])/2**(p_levels[1:])*diff_or_square_random[0],label="$\Delta_p$")
plt.plot(p_levels[1:],2**(p_levels[1]/2)/2**(p_levels[1:]/2)*diff_or_square_random[0],label="$\Delta_p^{1/2}$")
#plt.plot(p_levels[1:],2**(p_levels[-1]*3)/2**(p_levels[1:]*3)*diff_or_square[-1],label="$\Delta_p^{3}$")
plt.legend()
plt.title("SGD_steps=gamma/(N_p)^{1/2}")
plt.xlabel("$p$")
plt.yscale("log")
#%%
# what can of behaviour can we expect from the second moment of the difference of the parameters?

# There are two hyperparaemters of the SGD, the learning rate and the noise. 
# I will study four combinations of these hyperparameters.




### Large learning rate and large noise: The system quickly moves to a zone where the gradient's noise is comparable
# to the analytical gradient, particularly, the error follows a Delta_p^{1/2} behaviour.

### Large leaning rate and small noise: The rates are partitioned in two zones, the first one with 
# a quick rate followed by the second one with a Delta_p^{1/2} behaviour.

### Small learning rate and small noise: I observe several zones with different rates, the first one
# with already a Delta_p^{1/2} behaviour, the second one with a more GD behaviour. 

### Small learning rate and large noise: The system has a small error but the rate is almost constant. 



# %%
############################################################################################
############################################################################################

# UNBIASED TESTS RESULTS FOR THE OU
if True:
    x0_sca=1.2
    x0=x0_sca
    l=10
    T=10
    t0=0
    l_d=0
    d=2**(l_d)
    theta_true=-0.3
    sigma_true=0.8
    sd_true=0.8
    np.random.seed(7)
    collection_input=[bdg.b_ou_1d,theta_true,bdg.Sig_ou_1d,sigma_true]
    resamp_coef=1
    l_max=10
    x_true= bdg.gen_gen_data_1d(T,x0_sca,l_max,collection_input)
    x_reg=bdg.cut(T,l_max,-l_d,x_true)[1:]
    times=np.array(range(t0,int(T/d)+1))*d
    l_times=np.arange(t0,T,2**(-l))
    l_max_times=np.arange(t0,T,2**(-l_max))
    np.random.seed(1007)
    d_times=np.array(range(t0+d,int(T/d)+1))*d
    obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)
    resamp_coef=1
    N=50
    start=time.time()
    mcmc_links=5 
    fd=1e-8
    theta_in=-1
    sigma_in=1
    sd_in=1
    theta_in_fd=theta_in+fd
    sigma_in_fd=sigma_in+fd
    sigma_in_aux=sigma_in
    theta_in_aux=theta_in+0.2
    sigma_in_aux_fd=sigma_in_aux+fd
    gamma=0.2
    alpha=0.5
    x0=x0_sca+np.zeros(N)
    inputs=[]
    pmax=14
    l0=4
    lmax=11  
    beta_l=1
    beta_p=1
    samples=3000
    
    #arg_cm=int(sys.argv[1])
    #arg_cm=32
    #seed=1+40*(3000*0)+samples*(arg_cm-1)
    pars=np.zeros((samples,2,2,3))
    levs=np.zeros((samples,2),dtype=int)
    CL=0.08
    CL0=0.08
    CP=0.2
    CP0=0.07
    s0=2**0
#%%
l_times=np.arange(t0,T,2**(-l))
l_max_times=np.arange(t0,T,2**(-l_max))
plt.plot(times[1:],x_reg,label="True signal")
plt.plot(l_max_times,x_true[:-1],label="True complete signal")
sd_true=5e-1
np.random.seed(3)
obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)
plt.plot(times[1:], obs,label="Observations")
plt.legend()
#%%
    
"""
(t0,x0,T,b,A_0,A_fd_0,Sig,fi_0,fi_fd_0,b_til,A_til_0,Sig_til,fi_til_0,\
    fi_til_fd_0,r,r_pars,r_pars_fd,H,H_pars,H_pars_fd,sample_funct,sample_pars,\
    obs,log_g_den,g_den_par_0, aux_trans_den,atdp,\
    Grad_log_aux_trans,prop_trans_den, Grad_log_G,resamp_coef, l, d,N,seed,fd_rate,\
    mcmc_links,SGD_steps,gamma, alpha, update_pars):
"""
l=5
N=50
fd_rate=1e-8
mcmc_links=10
gamma=0.7
alpha=0.5
SGD_steps=2**7
seed=4233
"""
[ch_paths_v,pars_one_samp_v]=bdg.SGD_bridge_vanilla(t0,x0,T,bdg.b_ou_1d,theta_in,theta_in_fd,bdg.Sig_ou_1d,\
sigma_in,sigma_in_fd,bdg.b_ou_aux,theta_in_aux,bdg.Sig_ou_aux,sigma_in_aux,\
sigma_in_aux_fd,bdg.r_quasi_normal_1d,[bdg.ou_sd,[theta_in_aux,sigma_in_aux]],\
[bdg.ou_sd,[theta_in_aux,sigma_in_aux_fd]],\
bdg.H_quasi_normal,[bdg.ou_sd,[theta_in_aux,sigma_in_aux],theta_in_aux],\
[bdg.ou_sd,[theta_in_aux,sigma_in_aux_fd],theta_in_aux],\
bdg.sampling_ou,[theta_in_aux,sigma_in_aux],\
obs,bdg.log_g_normal_den,sd_in, bdg.ou_trans_den,[theta_in_aux,sigma_in_aux],\
bdg.Grad_log_aux_trans_ou_new,bdg.ou_trans_den, bdg.Grad_log_G_new,resamp_coef, l, d,N,seed,fd_rate,\
mcmc_links,SGD_steps,gamma, alpha,bdg.update_pars_ou)  
"""

#"""
[ch_paths,pars_one_samp]=bdg.SGD_bridge(t0,x0,T,bdg.b_ou_1d,theta_in,theta_in_fd,bdg.Sig_ou_1d,\
sigma_in,sigma_in_fd,bdg.b_ou_aux,theta_in_aux,bdg.Sig_ou_aux,sigma_in_aux,\
sigma_in_aux_fd,bdg.r_quasi_normal_1d,[bdg.ou_sd,[theta_in_aux,sigma_in_aux]],\
[bdg.ou_sd,[theta_in_aux,sigma_in_aux_fd]],\
bdg.H_quasi_normal,[bdg.ou_sd,[theta_in_aux,sigma_in_aux],theta_in_aux],\
[bdg.ou_sd,[theta_in_aux,sigma_in_aux_fd],theta_in_aux],\
bdg.sampling_ou,[theta_in_aux,sigma_in_aux],\
obs,bdg.log_g_normal_den,sd_in, bdg.ou_trans_den,[theta_in_aux,sigma_in_aux],\
bdg.Grad_log_aux_trans_ou_new,bdg.ou_trans_den, bdg.Grad_log_G_new,resamp_coef, l, d,N,seed,fd_rate,\
mcmc_links,SGD_steps,gamma, alpha,bdg.update_pars_ou)  
#"""

"""
[ch_paths_0,ch_paths_1,pars_one_samp_0,pars_one_samp_1,Grad_0,Grad_1]=\
bdg.C_SGD_bridge(t0,x0,T,bdg.b_ou_1d,theta_in,theta_in_fd,bdg.Sig_ou_1d,\
sigma_in,sigma_in_fd,bdg.b_ou_aux,theta_in_aux,bdg.Sig_ou_aux,sigma_in_aux,\
sigma_in_aux_fd,bdg.r_quasi_normal_1d,[bdg.ou_sd,[theta_in_aux,sigma_in_aux]],\
[bdg.ou_sd,[theta_in_aux,sigma_in_aux_fd]],\
bdg.H_quasi_normal,[bdg.ou_sd,[theta_in_aux,sigma_in_aux],theta_in_aux],\
[bdg.ou_sd,[theta_in_aux,sigma_in_aux_fd],theta_in_aux],\
bdg.rej_max_coup_ou,[theta_in_aux,sigma_in_aux,theta_in_aux,sigma_in_aux],\
obs,bdg.log_g_normal_den,sd_in, bdg.ou_trans_den,[theta_in_aux,sigma_in_aux],\
bdg.Grad_log_aux_trans_ou_new,bdg.ou_trans_den,[theta_in_aux,sigma_in_aux],\
bdg.Grad_log_G_new,bdg.update_pars_ou,resamp_coef, l, d,N,seed,fd_rate,\
mcmc_links,SGD_steps,gamma, alpha)  
"""

#%%
ids=["37831923","37831949","37870285","37870318","37948382","37968427","37999267"]

value = np.loadtxt("Observationsdata/displays/test.37831923.1.out", usecols=3)
i = 42  # The integer you want in the file name
filename = f"Observationsdata/displays/test.{ids[6]}.{2}.out" # Produces "data_42.txt"
with open(filename, 'r') as f:
    first_line = next(f).strip()  # "Parallelized processes time: 4227.7228899002075"
float1 = float(first_line.split(":")[1])

#%%
# rcv4_unb_ip_i
samples=3000
v="rcv4_unb_ip_i"
labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])
pars_file=np.reshape(\
    np.loadtxt("Observationsdata/data6/Prl_Unbiased_v"+v+labels[0]+".txt",dtype=float),(samples,2,2,3)) 
levels_file=np.reshape(np.loadtxt("Observationsdata/data6/Prl_Unbiased_levels_v"+v+labels[0]+".txt",dtype=int),(samples,2))
filename=f"Observationsdata/displays/test.{ids[0]}.{1}.out"
#times=np.array([np.loadtxt(filename, usecols=3)])
for i in range(len(labels[1:])):
    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data6/Prl_Unbiased_v"+v+labels[i+1]+".txt",dtype=float),(samples,2,2,3)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data6/Prl_Unbiased_levels_v"+v+labels[i+1]+".txt",dtype=int),(samples,2))),axis=0)  
    
# This version lacks sample 16
v="rcv4_unb_ip_ii"
labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])
for i in range(len(labels)):
    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[1]+"."+labels[i]+".out", usecols=3)])),axis=0)

    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data6/Prl_Unbiased_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,3)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data6/Prl_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  

# This version lacks sample 27
v="rcv4_unb_ip_iii"
labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15","16","17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "28", "29","30"])
for i in range(len(labels)):
    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[2]+"."+labels[i]+".out", usecols=3)])),axis=0)

    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data6/Prl_Unbiased_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,3)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data6/Prl_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  
    # This version lacks sample 18
v="rcv4_unb_ip_iv"
labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15","16","17", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27","28", "29","30"])
for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[3]+"."+labels[i]+".out", usecols=3)])),axis=0)

    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data6/Prl_Unbiased_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,3)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data6/Prl_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  

# This version lacks sample 12 and 28
v="rcv4_unb_ip_vi"
labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "13", "14", "15","16","17", "18","19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "29","30"])
for i in range(len(labels)):

    #filename="Observationsdata/displays/test."+ids[5]+"."+labels[i]+".out"
    #with open(filename, 'r') as f:
    #    first_line = next(f).strip()  # "Parallelized processes time: 4227.7228899002075"
    #    float1 = float(first_line.split(":")[1])
        
    #times=np.concatenate((times,np.array([float1])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data6/Prl_Unbiased_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,3)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data6/Prl_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  
    
# This version lacks sample 14
v="rcv4_unb_ip_vii"
labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11","12", "13", "15","16","17", "18","19", "20"\
    ,"21", "22", "23","25", "26","27","28","29","30"])
for i in range(len(labels)):

    #filename="Observationsdata/displays/test."+ids[6]+"."+labels[i]+".out"
    #with open(filename, 'r') as f:
    #    first_line = next(f).strip()  # "Parallelized processes time: 4227.7228899002075"
    #    float1 = float(first_line.split(":")[1])

    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data6/Prl_Unbiased_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,3)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data6/Prl_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  

#%%
#%%
samples=3000
v="rcv3_unb_ip_i"
labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])
pars_file=np.reshape(\
    np.loadtxt("Observationsdata/data5/Prl_Unbiased_v"+v+labels[0]+".txt",dtype=float),(samples,2,2,3)) 
levels_file=np.reshape(np.loadtxt("Observationsdata/data5/Prl_Unbiased_levels_v"+v+labels[0]+".txt",dtype=int),(samples,2))

for i in range(len(labels[1:])):
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data5/Prl_Unbiased_v"+v+labels[i+1]+".txt",dtype=float),(samples,2,2,3)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data5/Prl_Unbiased_levels_v"+v+labels[i+1]+".txt",dtype=int),(samples,2))),axis=0)  

v="rcv3_unb_ip_ii"

for i in range(len(labels)):
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data5/Prl_Unbiased_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,3)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data5/Prl_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  


v="rcv3_unb_ip_iii" # This version lacks the sample 27
labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "28", "29","30"])


for i in range(len(labels)):
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data5/Prl_Unbiased_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,3)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data5/Prl_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  


v="rcv3_unb_ip_iv"

labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])

for i in range(len(labels)):
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data5/Prl_Unbiased_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,3)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data5/Prl_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  


v="rcv3_unb_ip_v"
# This version lacks the sample 12
labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])

for i in range(len(labels)):
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data5/Prl_Unbiased_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,3)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data5/Prl_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  

v="rcv3_unb_ip_vi" # This version lacks the sample 28
samples=3000

labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11","12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "29","30"])

for i in range(len(labels)):
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data5/Prl_Unbiased_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,3)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data5/Prl_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  

#%%
"""
v="rcv3_unb_ip_vii" 
samples=3000

labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11","12", "13", "15", "16", "17", "18", "19", "20"\
    ,"21", "23", "24", "25", "26", "27","28", "29","30"])

for i in range(len(labels)):
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data5/Prl_Unbiased_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,3)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data5/Prl_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  

"""


#%%

# This space is made to get the ip files
samples=40
v="rcv2_unb_ip_"
labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29"])
pars_file=np.reshape(\
    np.loadtxt("Observationsdata/data4/Prl_Unbiased_v"+v+labels[0]+".txt",dtype=float),(samples,2,2,3)) 
levels_file=np.reshape(np.loadtxt("Observationsdata/data4/Prl_Unbiased_levels_v"+v+labels[0]+".txt",dtype=int),(samples,2))
for i in range(len(labels[1:])):
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data4/Prl_Unbiased_v"+v+labels[i+1]+".txt",dtype=float),(samples,2,2,3)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data4/Prl_Unbiased_levels_v"+v+labels[i+1]+".txt",dtype=int),(samples,2))),axis=0)  

labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])
samples=3000
names=["viii","x","xii"]
for j in range(len(names)):
    for i in range(len(labels)):
        pars_file=np.concatenate((pars_file,np.reshape(\
        np.loadtxt("Observationsdata/data4/Prl_Unbiased_vrcv2_unb_ip_"+names[j]+"_"+labels[i]+".txt",dtype=float),(samples,2,2,3)) ),axis=0)  
        levels_file=np.concatenate((levels_file,np.reshape(\
        np.loadtxt("Observationsdata/data4/Prl_Unbiased_levels_vrcv2_unb_ip_"+names[j]+"_"+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  
samples=3000
names=["ix","xi"]
labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21"])
j=0
for i in range(len(labels)):
        pars_file=np.concatenate((pars_file,np.reshape(\
        np.loadtxt("Observationsdata/data4/Prl_Unbiased_vrcv2_unb_ip_"+names[j]+"_"+labels[i]+".txt",dtype=float),(samples,2,2,3)) ),axis=0)  
        levels_file=np.concatenate((levels_file,np.reshape(\
        np.loadtxt("Observationsdata/data4/Prl_Unbiased_levels_vrcv2_unb_ip_"+names[j]+"_"+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  

labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
])
j=1
for i in range(len(labels)):
        pars_file=np.concatenate((pars_file,np.reshape(\
        np.loadtxt("Observationsdata/data4/Prl_Unbiased_vrcv2_unb_ip_"+names[j]+"_"+labels[i]+".txt",dtype=float),(samples,2,2,3)) ),axis=0)  
        levels_file=np.concatenate((levels_file,np.reshape(\
        np.loadtxt("Observationsdata/data4/Prl_Unbiased_levels_vrcv2_unb_ip_"+names[j]+"_"+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  


labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])
v="rcv2_unb_ip_ll_"
samples=200
for i in range(len(labels)):
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data4/Prl_Unbiased_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,3)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data4/Prl_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  
v="rcv2_unb_ip_lll_"
samples=1000
for i in range(len(labels)):
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data4/Prl_Unbiased_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,3)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data4/Prl_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  

v="rcv2_unb_ip_iv_"
samples=1000
for i in range(len(labels)):
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data4/Prl_Unbiased_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,3)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data4/Prl_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  

v="rcv2_unb_ip_v_"
samples=1000
for i in range(len(labels)):
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data4/Prl_Unbiased_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,3)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data4/Prl_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  
#%%
v="rcv2_unb_ip_vi_"
"""
labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22"])
"""
samples=1000
for i in range(len(labels)):
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data4/Prl_Unbiased_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,3)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data4/Prl_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  
#%%

"""labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])
    """
samples=3000
v="rcv2_unb_ip_vii_"
for i in range(len(labels)):
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data4/Prl_Unbiased_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,3)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data4/Prl_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  
#%%
print(pars_file.shape)

#%%
#labels= [str(i) for i in range(101,110)]
#samples=40
#i=0
v="rcv2_unb"
labels=np.array(["1:40","41:80","81:120","121:140","161:200"])
pars_file=np.reshape(\
    np.loadtxt("Observationsdata/data4/Prl_Unbiased_v"+v+labels[0]+".txt",dtype=float),(samples,2,2,3)) 
levels_file=np.reshape(np.loadtxt("Observationsdata/data4/Prl_Unbiased_levels_v"+v+labels[0]+".txt",dtype=int),(samples,2))
#%%
for i in range(len(labels[1:])):
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data4/Prl_Unbiased_v"+v+labels[i+1]+".txt",dtype=float),(samples,2,2,3)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data4/Prl_Unbiased_levels_v"+v+labels[i+1]+".txt",dtype=int),(samples,2))),axis=0)  
"""        
labels2= [str(i) for i in range(111,191)]

for i in range(len(labels)):
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observations&data/data_unb_ou/Prl_Unbiased_v"+labels[i]+".txt",dtype=float),(samples,2,2,3)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observations&data/data_unb_ou/Prl_Unbiased_levels_v"+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  
"""
#check the sgd equaition, specifically the logvaraibles  
#%%
labels= ["33","34","35","36","37","38"]
samples=200
i=0
pars_file=np.reshape(\
    np.loadtxt("Observations&data/Prl_Unbiased_v"+labels[i]+".txt",dtype=float),(samples,2,2,3)) 
levels_file=np.reshape(np.loadtxt("Observations&data/Prl_Unbiased_levels_v"+labels[i]+".txt",dtype=int),(samples,2))
for i in range(len(labels))[1:]:
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observations&data/Prl_Unbiased_v"+labels[i]+".txt",dtype=float),(samples,2,2,3)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observations&data/Prl_Unbiased_levels_v"+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  

labels= [str(i) for i in range(39,39+32)]
samples=80
i=0
for i in range(len(labels)):
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observations&data/data_unb_ou/Prl_Unbiased_v"+labels[i]+".txt",dtype=float),(samples,2,2,3)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observations&data/data_unb_ou/Prl_Unbiased_levels_v"+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  
        
"""
labels= [str(i) for i in range(39+32,96)]
samples=200
i=0
for i in range(len(labels)):
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observations&data/data_unb_ou/Prl_Unbiased_v"+labels[i]+".txt",dtype=float),(samples,2,2,3)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observations&data/data_unb_ou/Prl_Unbiased_levels_v"+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  
"""        

"""
labels2= ["26","27","28","29","30","31"]
samples=40*25
for i in range(len(labels2)):
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observations&data/Prl_Unbiased_v"+labels2[i]+".txt",dtype=float),(samples,2,2,3)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observations&data/Prl_Unbiased_levels_v"+labels2[i]+".txt",dtype=int),(samples,2))),axis=0) 
print(200*8+40*25*6)
"""
#%%

Grid_p=21
#[-0.36102774  1.32128857  0.43934842]
# -0.48122309 , 1.40604535 , 0.49615104
thetas=np.linspace(-1,1,Grid_p)*0.0005+pars[-1,0]
sigmas=np.linspace(-1,1,Grid_p)*0.0005+  pars[-1,1]
sds=np.linspace(-1,1,Grid_p)*0.0005+ pars[-1,2]
Grid=np.stack((thetas,sigmas,sds))
theta_aux=thetas+0.2
sigma_aux=sigmas
fd_rate=1e-4
[theta_fd,sigma_fd,sd_fd]=np.array([thetas,sigmas,sds])+fd_rate*(np.zeros((3,Grid_p))+1)
print(thetas,sigmas)
print(Grid.shape)
x=1
y=2
#%%
# IN 2d
[theta_0,sigma_0,sd_0]=[theta_in,sigma_in,sd_in]
Grads=np.zeros((Grid_p,Grid_p,3))
dim=1
par_left=sd_in
dim_o=1
for i in range(len(Grid[x])):
    par_x=Grid[x][i]
    for j in range(len(sigmas)):
        #sigma=sigmas[j]
        par_y=Grid[y][j]
        theta=(y==0)*par_y+(x==0)*par_x+ (x!=0)*(y!=0)*pars[-1,0]
        sigma=(y==1)*par_y+(x==1)*par_x+ (x!=1)*(y!=1)*pars[-1,1]
        sd=(y==2)*par_y+(x==2)*par_x+ (x!=2)*(y!=2)*pars[-1,2]
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
        x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
        Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
        Grads[j,i]=Grad_log_lik[:,0,0]
#%%
[theta_0,sigma_0,sd_0]=[theta_in,sigma_in,sd_in]
x=1
y=2
Grads_dis=np.zeros((Grid_p,Grid_p,3))
dim=1
par_left=sd_in
dim_o=1
l_dis=l
for i in range(len(Grid[x])):
    par_x=Grid[x][i]
    for j in range(len(sigmas)):    
        #sigma=sigmas[j]
        par_y=Grid[y][j]
        theta=(y==0)*par_y+(x==0)*par_x+ (x!=0)*(y!=0)*theta_0
        sigma=(y==1)*par_y+(x==1)*par_x+ (x!=1)*(y!=1)*sigma_0
        sd=(y==2)*par_y+(x==2)*par_x+ (x!=2)*(y!=2)*sd_0
        #K=np.array([[np.exp(d*theta)]])
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
        x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
        Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
        Grads_dis[j,i]=Grad_log_lik[:,0,0]

#%%
#[theta_0,sigma_0,sd_0]=np.array([theta_true,sigma_true,sd_true])+np.array([0.4,-0.3,0])
SGD_steps=32*8*16*4*4
#pars=np.zeros((SGD_steps+1,3))
Grads_test=np.zeros((SGD_steps+1,3))
#alpha=0.0001
#gamma=0.05
theta=-1
sigma=1
alpha_new=alpha
gamma_new=0.2
sd=1
dim=1
dim_o=1
theta, sigma, sd = pars[-1,0],pars[-1,1],pars[-1,2]
pars[0,:]=np.copy(pars[-1,:] )#np.array([theta,sigma,sd])
for b_ind in range(SGD_steps):
    #sigma=sigmas[j]
    #print(theta,sigma)
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
    x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
    Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
    Grads_test[b_ind]=Grad_log_lik[:,0,0]
    #print(Grads_test[b_ind,:2])
    theta+=gamma_new*Grads_test[b_ind,0]/(b_ind+1)**(0.5+alpha_new)
    sigma+=gamma_new*Grads_test[b_ind,1]/(b_ind+1)**(0.5+alpha_new)
    sd+=gamma_new*Grads_test[b_ind,2]/(b_ind+1)**(0.5+alpha_new)
    pars[b_ind+1]=np.array([theta,sigma,sd])
print(pars[-1])
#%%
print(pars[-1])
print(pars[-2])

print("sd is: ",sd_true)
a=SGD_steps-5
#a=10
b=1000000
#print(pars.shape)
#plt.plot(pars[a:,x].T,pars[a:,y].T)
#plt.plot(pars_one_samp[a:,x].T,pars_one_samp[a:,y].T,label="Backward")
#plt.plot(pars_one_samp_v[a:,x].T,pars_on   e_samp_v[a:,y].T,label="Vanilla")
#thetas_Grid,sigmas_Grid=np.meshgrid(thetas,sigmas)
plt.plot(pars[a:b,x].T,pars[a:b,y].T,label="True")
x_Grid,y_Grid=np.meshgrid(Grid[x],Grid[y])
#plt.plot(pars_file[a:b,1,1,x],pars_file[a:b,1,1,y],label="Single samples")
#plt.quiver(x_Grid,y_Grid,Grads_dis[:,:,x],Grads_dis[:,:,y])
plt.quiver(x_Grid,y_Grid,Grads[:,:,x],Grads[:,:,y])
#print(pars_one_samp)
#print(pars_one_samp_v[-3:,:])
#print(pars[-3:,:])
print("The actual parameters are: ",theta_true,sigma_true)
labels=["Theta","Sigma","SD"]
plt.xlabel(labels[x])
plt.ylabel(labels[y])
plt.title("SGD")
plt.legend()
#plt.savefig("Gradiend_flow_&_SGD.pdf")
plt.show()
print(pars[-1])
print(Grads[-1,-1,:],Grads[-1,-1,:])

#%%
#%%
# The true parameters are [-0.36102774  1.32128857  0.43934842]
#%%
# The probabilities of the levels are:
eLes=np.arange(l0,lmax+1)
beta=beta_l
q=4
P0=(1+CL*np.sum((q+eLes[1:]-l0)*np.log(q+eLes[1:]-l0)**2/2**(beta*eLes[1:]))/\
(CL0*(q+1)*np.log(q+1)**2))**(-1)

l_cumu=bdg.P_l_cumu_gen(P0,lmax-l0+1,beta,l0)
l=eLes[bdg.sampling(l_cumu)]
l_den=np.zeros(len(eLes))
l_den[0]=P0
l_den[1:]=l_cumu[1:]-l_cumu[:-1]

# cumulative for the number of SGD steps
beta=beta_p
ePes=np.arange(0,pmax+1)
eSes=s0*2**ePes
P0=(1+CP*np.sum((ePes[1:]+q)*np.log(ePes[1:]+q)**2/eSes[1:]**(beta))\
/(CP0*(q+1)*np.log(1+q)**2))**(-1)

p_cumu=bdg.P_p_cumu_gen(P0,pmax,beta,s0)
p_den=np.zeros(len(ePes))
p_den[0]=P0
p_den[1:]=p_cumu[1:]-p_cumu[:-1]
print("The density of l is:",l_den)
print("The density of p is:",p_den)
#%%
# -0.35906652  1.31791581  0.4422513
print(pars_file[40*29+1],pars_file[40*30+1])
print(np.concatenate((levels_file,np.array(range((1240*30-40)))[:,np.newaxis]),axis=1))
#%%
#%%
print(pars_file.shape[0])
print(l_den)
print(p_den)
dist=np.zeros((lmax-l0+1,pmax+1))
for i in range(levels_file.shape[0]):
    dist[levels_file[i,0]-l0,levels_file[i,1]]+=1
print(dist)
#%%
# samples,2,2,3
print("The parameters are:", pars[-1])
print(pars_file.shape)
unb_terms=pars_file/(l_den[levels_file[:,0]-l0,np.newaxis,np.newaxis,np.newaxis]\
*p_den[levels_file[:,1],np.newaxis,np.newaxis,np.newaxis])
#unb_terms=unb_terms[-200:]
a=0
b=-1
est=np.mean(unb_terms[:,1,1]-unb_terms[:,1,0]-(unb_terms[:,0,1]-unb_terms[:,0,0]),axis=0)
print(est)
var_est=np.var(unb_terms[:,1,1]-unb_terms[:,1,0]-(unb_terms[:,0,1]-unb_terms[:,0,0]),axis=0)
print(var_est/((3000*40)))
#%%
np.random.seed(2)
an_mean= np.array([-1.0787,0.8027,0.7301])
print(unb_terms.shape)
samples=519000
print(np.log2(samples/10))
batches=500
ests=np.zeros((batches,3))
costs=np.zeros(batches)
m_costss=np.zeros((16 -1))
mses=np.zeros((16 -1,3))
for i in range(1,16):
    for j in range(batches):

        
        batch_samples=np.random.choice(samples,2**i)
        costs[j]=np.sum(2**(levels_file[batch_samples,0]+levels_file[batch_samples,1]))        
        ests[j]=np.mean(unb_terms[batch_samples,1,1]-unb_terms[batch_samples,1,0]-(unb_terms[batch_samples,0,1]-unb_terms[batch_samples,0,0]),axis=0)
    m_costss[i-1]=np.mean(costs)
    mses[i-1]=np.mean((ests-an_mean)**2,axis=0)

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

#%%
par_n=1
[b_0,b_1]= coef(np.log(mses[:,par_n]),np.log(m_costss))
print(b_0,b_1)  
plt.plot(mses[:,par_n],np.exp(b_0)*mses[:,par_n]**b_1,lw=2,label=rf'$\varepsilon^{{{2*b_1:.2f}}}$',c="coral")
plt.plot(mses[:,par_n],m_costss[0]*mses[0,par_n]/mses[:,par_n],\
lw=2,label="Canonical Monte Carlo rate: "+rf'$\varepsilon^{{-2}}$',color="dodgerblue")    
plt.plot(mses[:,par_n],m_costss[0]*mses[0,par_n]**(3/2)/mses[:,par_n]**(3/2),lw=2,\
label=rf'$\varepsilon^{{-3}}$',c="deepskyblue")
plt.scatter(mses[:,par_n],m_costss,label="Unbiased Bride approach",lw=2,c="coral")

plt.xlabel(r"$\varepsilon^2$",size=14)
plt.ylabel("Cost",size=14)
plt.tick_params(axis="both",          # "x", "y", or "both"
                labelsize=14) 

plt.yscale("log")
plt.xscale("log")
plt.legend(fontsize=12) 
#plt.savefig("OU_Cost_vs_MSE.pdf")
plt.show()
#%%


#%%
plt.plot(np.arange(1,16),mses[:,par_n])
plt.plot(np.arange(1,16),mses[0,par_n]*2**1/2**np.arange(1,16))
plt.yscale("log")
#%%
max_time=np.max(times)
print(max_time/3600) 
my_data = times/3600

# Convert the Python list to a NumPy array (optional but often convenient)
arr = np.array(my_data)

# Create a histogram
plt.hist(arr, bins=100)  # you can adjust the number of bins

# Label axes and show plot
plt.xlabel('Value')
plt.ylabel('Count')
plt.title('Histogram of Positive Numbers')
plt.show()

#%%

test_ar=np.array([[0,1],[1,2],[2,3],[0,2]])
print(np.where(test_ar[:,0]==0))
#%%
p_zeros=np.where(levels_file[:,1]==5)
#print(p_zeros)
print(pars_file[p_zeros[0],0,0])
print(p_zeros[0][2])
p_selected_01=pars_file[p_zeros[0],0,1]
p_selected_11=pars_file[p_zeros[0],1,1]
p_selected_10=pars_file[p_zeros[0],1,0]
#%%
## THIS SPACE IS RESERVED TO FOR THE TEST OF THE TOY MODEL

pmax=5
l0=0
lmax=10
beta_l=1
beta_p=1
samples=16*3
CL=0.03
CL0=0.1
CP0=0.016
CP=0.01
s0=2**3
samples=100
seed=1003
K=100
labels= ["18,19,20,21"]
pars_file=np.zeros((samples*len(labels),2,K))
levels_file=np.zeros((samples*len(labels),2),dtype=int)
i=0
pars_file=np.reshape(np.loadtxt("Observations&data/Prl_Unbiased_v"+labels[i]+".txt",dtype=float),(samples,2,K))   
levels_file=np.reshape(np.loadtxt("Observations&data/Prl_Unbiased_levels_v"+labels[i]+".txt",dtype=int),(samples,2))
#%%
eLes=np.arange(l0,lmax+1)
beta=beta_l
P0=(1+CL*2**(beta*(l0+1))*np.sum((eLes[1:]+1)*np.log(eLes[1:]+1)**2/2**(beta*eLes[1:]))/\
(CL0*(l0+2)*np.log(l0+2)**2))**(-1)
l_cumu=bdg.P_l_cumu_gen(P0,lmax-l0+1,beta,l0)
l_den=np.zeros(len(eLes))
l_den[0]=P0
l_den[1:]=l_cumu[1:]-l_cumu[:-1]

# cumulative for the number of SGD steps
beta=beta_p
ePes=np.arange(0,pmax+1)
eSes=s0*2**ePes
#eSes=16*2*ePes
P0=(1+CP*eSes[1]**(beta)*np.sum((ePes[1:]+1)*np.log(ePes[1:]+1)**2/eSes[1:]**(beta))\
/(CP0*(2)*np.log(2)**2))**(-1)

p_cumu=bdg.P_p_cumu_gen(P0,pmax,beta,s0)
p_den=np.zeros(len(ePes))
p_den[0]=P0
p_den[1:]=p_cumu[1:]-p_cumu[:-1]
print(l_den,p_den)
#%%
print(levels_file)
#%%
unb_terms=pars_file/(l_den[levels_file[:,0]-l0,np.newaxis,np.newaxis])
est=np.mean(unb_terms[:,1]-unb_terms[:,0])
print(est)
print(1/2**5)
var_est=np.var(unb_terms[:,1,1]-unb_terms[:,1,0]-(unb_terms[:,0,1]-unb_terms[:,0,0]),axis=0)
print(var_est/samples)
#%%

#%%
######################################################
######################################################
######################################################
# In the following we compare the vanilla and the backward SGD for the OU

if True:

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
    collection_input=[bdg.b_ou_1d,theta_true,bdg.Sig_ou_1d,sigma_true]
    resamp_coef=1
    l_max=10
    x_true= bdg.gen_gen_data_1d(T,x0_sca,l_max,collection_input)
    x_reg=bdg.cut(T,l_max,-l_d,x_true)[1:]
    times=np.array(range(t0,int(T/d)+1))*d
    l_times=np.arange(t0,T,2**(-l))
    l_max_times=np.arange(t0,T,2**(-l_max))
    sd_true=0.55
    np.random.seed(3)
    d_times=np.array(range(t0+d,int(T/d)+1))*d
    obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)
    resamp_coef=1
    fd=1e-4
    gamma=0.5
    alpha=0.5
    theta_in=-0.67
    sigma_in=1.6
    sd_in=0.7
    theta_in_aux=theta_in+0.2
    sigma_in_aux=sigma_in
    theta_in_fd=theta_in+fd
    sigma_in_fd=sigma_in+fd
    sigma_in_aux_fd=sigma_in_aux+fd
    start=time.time()
    mcmc_links=20
    SGD_steps=2**8
    #SGD_steps=2**1
    B=mcmc_links*SGD_steps

    samples=1
    #samples=2

    # interactive 1 samples=100
    N=50
    x0=x0_sca+np.zeros(N)
    l=10
    seed=0
#%%
"""
(t0,x0,T,b,A_0,A_fd_0,Sig,fi_0,fi_fd_0,b_til,A_til_0,Sig_til,fi_til_0,\
    fi_til_fd_0,r,r_pars,r_pars_fd,H,H_pars,H_pars_fd,sample_funct,sample_pars,\
    obs,log_g_den,g_den_par_0, aux_trans_den,atdp,\
    Grad_log_aux_trans,prop_trans_den, Grad_log_G,resamp_coef, l, d,N,seed,fd_rate,\
    mcmc_links,SGD_steps,gamma, alpha, update_pars):
"""
"""l=6
N=50
x0=x0_sca+np.zeros(N)
fd_rate=1e-10
mcmc_links=100
gamma=0.1
alpha=0.5
SGD_steps=2**1
samples=2
pars_one_samps=np.zeros((samples,SGD_steps+1,3))
pars_one_samps_v=np.zeros((samples,SGD_steps+1,3))
seed=234
"""
v="single1"

#ch_paths_file_van=np.reshape(np.loadtxt("Observations&data/Prl_SGD_ou_bridge_van_comparison_ch_paths_v"+v+".txt",dtype=float),(samples,B,int(T/d)))
#pars_file_van=np.reshape(np.loadtxt("Observations&data/Prl_SGD_ou_bridge_van_comparison_pars_v"+v+".txt",dtype=float),(samples,SGD_steps+1,3))
#ch_paths_file=np.reshape(np.loadtxt("Observations&data/Prl_SGD_ou_bridge_comparison_ch_paths_v"+v+".txt",dtype=float),(samples,B,int(T/d)))
#pars_file=np.reshape(np.loadtxt("Observations&data/Prl_SGD_ou_bridge_comparison_pars_v"+v+".txt",dtype=float),(samples,SGD_steps+1,3))
ch_paths_file=np.reshape(np.loadtxt("Observations&data/Prl_SGD_ou_bridge_ch_paths_v"+v+".txt",dtype=float),(samples,B,int(T/d)))
pars_file=np.reshape(np.loadtxt("Observations&data/Prl_SGD_ou_bridge_pars_v"+v+".txt",dtype=float),(samples,SGD_steps+1,3))
print(pars_file[0,-1])
#%%
for i in range(samples):
    #"""
    seed+=i
    np.random.seed(seed)
    print(theta_in,sigma_in)
    #"""
    [ch_paths_v,pars_one_samp_v]=bdg.SGD_bridge_vanilla(t0,x0,T,bdg.b_ou_1d,theta_in,theta_in_fd,bdg.Sig_ou_1d,\
    sigma_in,sigma_in_fd,bdg.b_ou_aux,theta_in_aux,bdg.Sig_ou_aux,sigma_in_aux,\
    sigma_in_aux_fd,bdg.r_quasi_normal_1d,[bdg.ou_sd,[theta_in_aux,sigma_in_aux]],\
    [bdg.ou_sd,[theta_in_aux,sigma_in_aux_fd]],\
    bdg.H_quasi_normal,[bdg.ou_sd,[theta_in_aux,sigma_in_aux],theta_in_aux],\
    [bdg.ou_sd,[theta_in_aux,sigma_in_aux_fd],theta_in_aux],\
    bdg.sampling_ou,[theta_in_aux,sigma_in_aux],\
    obs,bdg.log_g_normal_den,sd_in, bdg.ou_trans_den,[theta_in_aux,sigma_in_aux],\
    bdg.Grad_log_aux_trans_ou_new,bdg.ou_trans_den, bdg.Grad_log_G_new,resamp_coef, l, d,N,seed,fd_rate,\
    mcmc_links,SGD_steps,gamma, alpha,bdg.update_pars_ou)  
    pars_one_samps_v[i]=pars_one_samp_v

    
    np.random.seed(seed)
    print(theta_in,sigma_in)
    #"""
    [ch_paths,pars_one_samp]=bdg.SGD_bridge(t0,x0,T,bdg.b_ou_1d,theta_in,theta_in_fd,bdg.Sig_ou_1d,\
    sigma_in,sigma_in_fd,bdg.b_ou_aux,theta_in_aux,bdg.Sig_ou_aux,sigma_in_aux,\
    sigma_in_aux_fd,bdg.r_quasi_normal_1d,[bdg.ou_sd,[theta_in_aux,sigma_in_aux]],\
    [bdg.ou_sd,[theta_in_aux,sigma_in_aux_fd]],\
    bdg.H_quasi_normal,[bdg.ou_sd,[theta_in_aux,sigma_in_aux],theta_in_aux],\
    [bdg.ou_sd,[theta_in_aux,sigma_in_aux_fd],theta_in_aux],\
    bdg.sampling_ou,[theta_in_aux,sigma_in_aux],\
    obs,bdg.log_g_normal_den,sd_in, bdg.ou_trans_den,[theta_in_aux,sigma_in_aux],\
    bdg.Grad_log_aux_trans_ou_new,bdg.ou_trans_den, bdg.Grad_log_G_new,resamp_coef, l, d,N,seed,fd_rate,\
    mcmc_links,SGD_steps,gamma, alpha,bdg.update_pars_ou)  
    #"""
    pars_one_samps[i]=pars_one_samp


    #"""
"""
[ch_paths_0,ch_paths_1,pars_one_samp_0,pars_one_samp_1,Grad_0,Grad_1]=\
bdg.C_SGD_bridge(t0,x0,T,bdg.b_ou_1d,theta_in,theta_in_fd,bdg.Sig_ou_1d,\
sigma_in,sigma_in_fd,bdg.b_ou_aux,theta_in_aux,bdg.Sig_ou_aux,sigma_in_aux,\
sigma_in_aux_fd,bdg.r_quasi_normal_1d,[bdg.ou_sd,[theta_in_aux,sigma_in_aux]],\
[bdg.ou_sd,[theta_in_aux,sigma_in_aux_fd]],\
bdg.H_quasi_normal,[bdg.ou_sd,[theta_in_aux,sigma_in_aux],theta_in_aux],\
[bdg.ou_sd,[theta_in_aux,sigma_in_aux_fd],theta_in_aux],\
bdg.rej_max_coup_ou,[theta_in_aux,sigma_in_aux,theta_in_aux,sigma_in_aux],\
obs,bdg.log_g_normal_den,sd_in, bdg.ou_trans_den,[theta_in_aux,sigma_in_aux],\
bdg.Grad_log_aux_trans_ou_new,bdg.ou_trans_den,[theta_in_aux,sigma_in_aux],\
bdg.Grad_log_G_new,bdg.update_pars_ou,resamp_coef, l, d,N,seed,fd_rate,\
mcmc_links,SGD_steps,gamma, alpha)  
"""

# %%
Grid_p=11
thetas=np.linspace(-1,1,Grid_p)*0.2-0.3
lsigmas=np.linspace(-1,1,Grid_p)*0.25-0.3
lsds=np.linspace(-1,1,Grid_p)*0.5+0

Grid=np.stack((thetas,lsigmas,lsds))
print(np.log(sigma_in),np.log(sd_in))
theta_aux=thetas+0.2
sigma_aux=np.exp(lsigmas)
print(np.log(sd_true),np.log(sigma_true))
#fd_rate=1e-4
#[theta_fd,sigma_fd,sd_fd]=np.array([thetas,sigmas,sds])+fd_rate*(np.zeros((3,Grid_p))+1)
#print(thetas,sigmas)

#%%
[theta_0,sigma_0,sd_0]=np.copy(np.array([theta_true,sigma_true,sd_true]))
x=1
y=2
Grads=np.zeros((Grid_p,Grid_p,3))
dim=1
dim_o=1
for i in range(len(Grid[x])):
    par_x=np.exp(Grid[x][i])*(x!=0)+Grid[x][i]*(x==0)
    for j in range(len(Grid[y])):
        #sigma=sigmas[j]
        par_y=np.exp(Grid[y][j])*(y!=0)+Grid[y][j]*(y==0)
        theta=(y==0)*par_y+(x==0)*par_x+ (x!=0)*(y!=0)*theta_0
        sigma=(y==1)*par_y+(x==1)*par_x+ (x!=1)*(y!=1)*sigma_0
        sd=(y==2)*par_y+(x==2)*par_x+ (x!=2)*(y!=2)*sd_0
        K=np.array([[np.exp(d*theta)]])
        G=np.array([[sigma*np.sqrt((np.exp(2*d*theta)-1)/(2*theta))]])
        H=np.array([[1]])
        D=np.array([[sd]])
        Grad_S=np.array([[[0]],[[0]],[[1]]],dtype=float)
        Grad_R_sigma_s=(np.exp(2*theta*d)-1)/(2*theta)
        Grad_R_theta=(sigma**2/(2*theta**2))*(1-np.exp(2*theta*d)+2*d*theta*np.exp(2*theta*d))
        Grad_R=np.zeros((3,1,1),dtype=float)
        Grad_R[0,0,0]=Grad_R_theta
        Grad_R[1,0,0]=Grad_R_sigma_s
        Grad_K=np.array([[[d*np.exp(d*theta)]],[[0]],[[0]]],dtype=float)
        #print(Grad_K,Grad_R,Grad_S)
        x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
        Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma*sigma
        Grad_log_lik[2,0,0]=Grad_log_lik[2,0,0]*sd
        Grads[j,i]=Grad_log_lik[:,0,0]
#%%

[theta_0,sigma_0,sd_0]=[theta_in , sigma_true , sd_true]
x=1
y=2
Grads_dis=np.zeros((Grid_p,Grid_p,3))
dim=1
par_left=sd_in
dim_o=1
l_dis=20
for i in range(len(Grid[x])):
    par_x=Grid[x][i]
    for j in range(len(sigmas)):    
        #sigma=sigmas[j]
        par_y=Grid[y][j]
        theta=(y==0)*par_y+(x==0)*par_x+ (x!=0)*(y!=0)*theta_true
        sigma=(y==1)*par_y+(x==1)*par_x+ (x!=1)*(y!=1)*sigma_true
        sd=(y==2)*par_y+(x==2)*par_x+ (x!=2)*(y!=2)*sd_true
        #K=np.array([[np.exp(d*theta)]])
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
        x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
        Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
        Grads_dis[j,i]=Grad_log_lik[:,0,0]
# %%
print(gamma)
SGD_steps=32*2**9
pars=np.zeros((SGD_steps+1,3))
Grads_test=np.zeros((SGD_steps+1,3))
#alpha=0.0001
#gamma=0.05
theta=theta_in
sigma=sigma_in
alpha_new=0.5
gamma_new=0.04
sd=sd_in
l_dis=l
pars[0]=np.array([theta,sigma,sd])
for b_ind in range(SGD_steps):
    
    #sigma=sigmas[j]
    #print(theta,sigma)
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
    x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
    Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
    
    Grads_test[b_ind]=Grad_log_lik[:,0,0]
    #print(Grads_test[b_ind,:2])
    theta+=gamma_new*Grads_test[b_ind,0]/(b_ind+1)**(0.5+alpha_new)
    sigma+=gamma_new*Grads_test[b_ind,1]/(b_ind+1)**(0.5+alpha_new)
    sd+=gamma_new*Grads_test[b_ind,2]/(b_ind+1)**(0.5+alpha_new)
    pars[b_ind+1]=np.array([theta,sigma,sd])
#%%
print("The final parameters are: ",pars[-1])
#print("sd is: ",sd_true)
#print("log sigma true is:", np.log(sigma_true))
#a=SGD_steps-3
print(pars[0])
a=0
c=0
e=samples
b=SGD_steps
n_pars_file=np.zeros(pars_file.shape)
n_pars=np.zeros(pars.shape)
n_pars_file_van=np.zeros( pars_file_van.shape)
n_pars_file[:,:,1]=np.log(pars_file[:,:,1])
n_pars_file[:,:,2]=np.log(pars_file[:,:,2])
n_pars_file[:,:,0]=pars_file[:,:,0]
n_pars_file_van[:,:,1]=np.log(pars_file_van[:,:,1])
n_pars_file_van[:,:,2]=np.log(pars_file_van[:,:,2])
n_pars_file_van[:,:,0]=pars_file_van[:,:,0]
n_pars[:,1]=np.log(pars[:,1])
n_pars[:,2]=np.log(pars[:,2])
n_pars[:,0]=pars[:,0]

plt.plot(n_pars[a:,x].T,n_pars[a:,y].T)

#plt.plot(n_pars_file_van[c:e,a:b,x].T,n_pars_file_van[c:e,a:b,y].T,label="Vanilla")
plt.plot(n_pars_file[c:e,a:b,x].T,n_pars_file[c:e,a:b,y].T,label="Backward")
#thetas_Grid,sigmas_Grid=np.meshgrid(thetas,sigmas)
x_Grid,y_Grid=np.meshgrid(Grid[x],Grid[y])
#plt.quiver(x_Grid,y_Grid,Grads_dis[:,:,x],Grads_dis[:,:,y])
plt.quiver(x_Grid,y_Grid,Grads[:,:,x],Grads[:,:,y])
labels=["Theta","logSigma","logSD"]
plt.xlabel(labels[x])
plt.ylabel(labels[y])
plt.title("SGD")
plt.legend()
#plt.savefig("Gradiend_flow_&_SGD.pdf")
plt.show()
#%%
i=2
fontsize=14
mse=np.mean((pars_file-pars[-1])**2,axis=0)[2:,i]
mse_van=np.mean((pars_file_van-pars[-1])**2,axis=0)[2:,i]
print(mse[0],mse_van[0])
plt.plot(np.array(range(mse.shape[0]))+2**2,mse,label="Backward")
plt.plot(np.array(range(mse.shape[0]))+2**2,mse_van,label="Vanilla")
plt.yscale("log")
plt.xscale("log")

print(np.mean((pars_file),axis=0)[-1,i])
labels=[r"$\theta$","$\sigma$","$s$"]
plt.title(labels[i],fontsize=fontsize)
plt.ylabel("MSE",fontsize=fontsize)
plt.xlabel("SGD steps",fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.savefig("MSE_SGD_sd.pdf")
#%%
#%%
theta=theta_in
sigma=sigma_in
sd=sd_in
K=np.array([[np.exp(d*theta)]])
G=np.array([[sigma*np.sqrt((np.exp(2*d*theta)-1)/(2*theta))]])
H=np.array([[1]])
D=np.array([[sd]])
Grad_S=np.array([[[0]],[[0]],[[1]]],dtype=float)
Grad_R_sigma_s=(np.exp(2*theta*d)-1)/(2*theta)
Grad_R_theta=(sigma**2/(2*theta**2))*(1-np.exp(2*theta*d)+2*d*theta*np.exp(2*theta*d))
Grad_R=np.zeros((3,1,1),dtype=float)
Grad_R[0,0,0]=Grad_R_theta
Grad_R[1,0,0]=Grad_R_sigma_s
Grad_K=np.array([[[d*np.exp(d*theta)]],[[0]],[[0]]],dtype=float)
#print(Grad_K,Grad_R,Grad_S)
x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma*sigma
Grad_log_lik[2,0,0]=Grad_log_lik[2,0,0]*sd
Grads[j,i]=Grad_log_lik[:,0,0]
j=1
mse=np.mean((np.mean(ch_paths_file[:,mcmc_links*(j-1):mcmc_links*j],axis=1)-x_kf_smooth[1:,0])**2,axis=0)
mse_van=np.mean((np.mean(ch_paths_file_van[:,mcmc_links*(j-1):mcmc_links*j],axis=1)-x_kf_smooth[1:,0])**2,axis=0)
print(np.mean(ch_paths_file[:,mcmc_links*(j-1):mcmc_links*j],axis=1))
print(mse_van)
print(x_kf_smooth)
plt.ylabel("MSE")
plt.xlabel("$t$")

plt.yscale("log")
plt.plot(mse,label="Backward")
plt.plot(mse_van,label="Vanilla")
plt.legend()
print(mse_van[1])
#plt.savefig("MSE_SGD.pdf")
 #%%

# In this iteration I change the parametrization of the gradient field (and the field itself)
# to reflect the SGD in terms of theta, log(sigma) and log(sd).
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################

if True:
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
    collection_input=[bdg.b_ou_1d,theta_true,bdg.Sig_ou_1d,sigma_true]
    resamp_coef=1
    l_max=10
    x_true= bdg.gen_gen_data_1d(T,x0_sca,l_max,collection_input)
    x_reg=bdg.cut(T,l_max,-l_d,x_true)[1:]
    times=np.array(range(t0,int(T/d)+1))*d
    l_times=np.arange(t0,T,2**(-l))
    l_max_times=np.arange(t0,T,2**(-l_max))
    sd_true=0.55
    np.random.seed(3)
    d_times=np.array(range(t0+d,int(T/d)+1))*d
    obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)
    #obs=np.array([])
    resamp_coef=1
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
#%%
l=7
N=50
x0=x0_sca+np.zeros(N)
fd_rate=fd
mcmc_links=100
gamma=0.1
alpha=0.5
SGD_steps=2**4
samples=10
pars_one_samps=np.zeros((samples,SGD_steps+1,3))
pars_one_samps_v=np.zeros((samples,SGD_steps+1,3))
seed=234
#%%
start=time.time()
for i in range(samples):
    #"""
    seed=i
    np.random.seed(seed)
    print(theta_in,sigma_in)
    #"""
    [ch_paths_v,pars_one_samp_v]=bdg.SGD_bridge_vanilla(t0,x0,T,bdg.b_ou_1d,theta_in,theta_in_fd,bdg.Sig_ou_1d,\
    sigma_in,sigma_in_fd,bdg.b_ou_aux,theta_in_aux,bdg.Sig_ou_aux,sigma_in_aux,\
    sigma_in_aux_fd,bdg.r_quasi_normal_1d,[bdg.ou_sd,[theta_in_aux,sigma_in_aux]],\
    [bdg.ou_sd,[theta_in_aux,sigma_in_aux_fd]],\
    bdg.H_quasi_normal,[bdg.ou_sd,[theta_in_aux,sigma_in_aux],theta_in_aux],\
    [bdg.ou_sd,[theta_in_aux,sigma_in_aux_fd],theta_in_aux],\
    bdg.sampling_ou,[theta_in_aux,sigma_in_aux],\
    obs,bdg.log_g_normal_den,sd_in, bdg.ou_trans_den,[theta_in_aux,sigma_in_aux],\
    bdg.Grad_log_aux_trans_ou_new,bdg.ou_trans_den, bdg.Grad_log_G_new,resamp_coef, l, d,N,seed,fd_rate,\
    mcmc_links,SGD_steps,gamma, alpha,bdg.update_pars_ou)  
    pars_one_samps_v[i]=pars_one_samp_v

    seed=i
    np.random.seed(seed)
    print(theta_in,sigma_in)
    #"""
    [ch_paths,pars_one_samp]=bdg.SGD_bridge(t0,x0,T,bdg.b_ou_1d,theta_in,theta_in_fd,bdg.Sig_ou_1d,\
    sigma_in,sigma_in_fd,bdg.b_ou_aux,theta_in_aux,bdg.Sig_ou_aux,sigma_in_aux,\
    sigma_in_aux_fd,bdg.r_quasi_normal_1d,[bdg.ou_sd,[theta_in_aux,sigma_in_aux]],\
    [bdg.ou_sd,[theta_in_aux,sigma_in_aux_fd]],\
    bdg.H_quasi_normal,[bdg.ou_sd,[theta_in_aux,sigma_in_aux],theta_in_aux],\
    [bdg.ou_sd,[theta_in_aux,sigma_in_aux_fd],theta_in_aux],\
    bdg.sampling_ou,[theta_in_aux,sigma_in_aux],\
    obs,bdg.log_g_normal_den,sd_in, bdg.ou_trans_den,[theta_in_aux,sigma_in_aux],\
    bdg.Grad_log_aux_trans_ou_new,bdg.ou_trans_den, bdg.Grad_log_G_new,resamp_coef, l, d,N,seed,fd_rate,\
    mcmc_links,SGD_steps,gamma, alpha,bdg.update_pars_ou)  
    #"""
    pars_one_samps[i]=pars_one_samp
end=time.time()
print(end-start)

#%%
Grid_p=9
thetas=np.linspace(-1,1,Grid_p)*0.1+theta_in
lsigmas=np.linspace(-1,1,Grid_p)*0.1+ np.log(sigma_in)
lsds=np.linspace(-1,1,Grid_p)*0.1+np.log(sd_in)
Grid=np.stack((thetas,lsigmas,lsds))
theta_aux=thetas+0.2
sigma_aux=np.log(lsigmas)
#fd_rate=1e-4
#[theta_fd,sigma_fd,sd_fd]=np.array([thetas,sigmas,sds])+fd_rate*(np.zeros((3,Grid_p))+1)
#print(thetas,sigmas)
print(Grid.shape)

#%%
[theta_0,sigma_0,sd_0]=[theta_in,sigma_in,sd_in]
x=0
y=1
Grads=np.zeros((Grid_p,Grid_p,3))
dim=1
par_left=sd_in
dim_o=1
for i in range(len(Grid[x])):
    
    par_x=np.exp(Grid[x][i])*(x!=0)+Grid[x][i]*(x==0)
    for j in range(len(Grid[y])):
        #sigma=sigmas[j]
        par_y=np.exp(Grid[y][j])*(y!=0)+Grid[y][j]*(y==0)
        theta=(y==0)*par_y+(x==0)*par_x+ (x!=0)*(y!=0)*theta_in
        sigma=(y==1)*par_y+(x==1)*par_x+ (x!=1)*(y!=1)*sigma_in
        sd=(y==2)*par_y+(x==2)*par_x+ (x!=2)*(y!=2)*sd_in
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
        x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
        Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma*sigma # the last sigma is put in 
        # order to account for the gradient of log sigma instead of sigma
        Grad_log_lik[2,0,0]=Grad_log_lik[2,0,0]*sd
        Grads[j,i]=Grad_log_lik[:,0,0]

#%%


[theta_0,sigma_0,sd_0]=[theta_in , sigma_true , sd_true]
x=0
y=2
Grads_dis=np.zeros((Grid_p,Grid_p,3))
dim=1
par_left=sd_in
dim_o=1
l_dis=20
for i in range(len(Grid[x])):
    par_x=np.exp(Grid[x][i])
    for j in range(len(Grid[y])):    
        #sigma=sigmas[j]
        par_y=np.exp(Grid[y][j])
        theta=(y==0)*par_y+(x==0)*par_x+ (x!=0)*(y!=0)*theta_true
        sigma=(y==1)*par_y+(x==1)*par_x+ (x!=1)*(y!=1)*sigma_true
        sd=(y==2)*par_y+(x==2)*par_x+ (x!=2)*(y!=2)*sd_true
        #K=np.array([[np.exp(d*theta)]])
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
        x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
        Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma*sigma # the last sigma is put in 
        # order to account for the gradient of log sigma instead of sigma
        Grad_log_lik[2,0,0]=Grad_log_lik[2,0,0]*sd
        Grads_dis[j,i]=Grad_log_lik[:,0,0]

#%%

a=0
c=0
e=-1
new_pars_one_samps=np.zeros(pars_one_samps.shape)
new_pars_one_samps[:,:,1]=np.log(pars_one_samps[:,:,1])
new_pars_one_samps[:,:,2]=np.log(pars_one_samps[:,:,2])
new_pars_one_samps[:,:,0]=pars_one_samps[:,:,0]
new_pars_one_samps_v=np.zeros(pars_one_samps_v.shape)
new_pars_one_samps_v[:,:,1]=np.log(pars_one_samps_v[:,:,1])
new_pars_one_samps_v[:,:,2]=np.log(pars_one_samps_v[:,:,2])
new_pars_one_samps_v[:,:,0]=pars_one_samps_v[:,:,0]
#plt.plot(pars[a:,x].T,pars[a:,y].T)
plt.plot(new_pars_one_samps[c:e,a:,x].T,new_pars_one_samps[c:e,a:,y].T,label="Backward")
plt.plot(new_pars_one_samps_v[c:e,a:,x].T,new_pars_one_samps_v[c:e,a:,y].T,label="Vanilla")
#thetas_Grid,sigmas_Grid=np.meshgrid(thetas,sigmas)
x_Grid,y_Grid=np.meshgrid(Grid[x],Grid[y])
plt.quiver(x_Grid,y_Grid,Grads[:,:,x],Grads[:,:,y])
#plt.quiver(x_Grid,y_Grid,Grads[:,:,x],Grads[:,:,y])
#print(pars_one_samp[:,:])
#print(pars_one_samp_v[:,:])
#print(pars[-3:,:])
print("The actual parameters are: ",theta_true,sigma_true)
labels=["Theta","log(Sigma)","log(SD)"]
plt.xlabel(labels[x])
plt.ylabel(labels[y])
plt.title("SGD")
plt.legend()
#plt.savefig("Gradiend_flow_&_SGD.pdf")
plt.show()
# %%

## HERE WE TEST THE KALMAN FILTER

if True:

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
    collection_input=[bdg.b_ou_1d,theta_true,bdg.Sig_ou_1d,sigma_true]
    resamp_coef=1
    l_max=10
    x_true= bdg.gen_gen_data_1d(T,x0_sca,l_max,collection_input)
    x_reg=bdg.cut(T,l_max,-l_d,x_true)[1:]
    times=np.array(range(t0,int(T/d)+1))*d
    l_times=np.arange(t0,T,2**(-l))
    l_max_times=np.arange(t0,T,2**(-l_max))
    plt.plot(times[1:],x_reg,label="True signal")
    plt.plot(l_max_times,x_true[:-1],label="True complete signal")
    sd_true=2.1
    np.random.seed(3)
    d_times=np.array(range(t0+d,int(T/d)+1))*d
    obs=bdg.gen_obs(x_reg,bdg.g_normal_1d,sd_true)
    plt.plot(d_times, obs,label="Observations")
    theta=theta_true
    sigma=sigma_true
    theta_aux=theta+0.2
    sigma_aux=sigma
    sd=sd_true
    fd=1e-8
    theta_fd=theta_true+fd
    sigma_fd=sigma_true+fd
    sigma_aux_fd=sigma_aux+fd
    start=time.time()
    B=1
   
    samples=40
    # interactive 1 samples=100
    N=500000
    x0=x0_sca+np.zeros(N)
    l0=3
    L_max=10
    eLes=np.array(range(l0,L_max+1))
#%%
dim=1
dim_o=1
#theta,sigma,sd=theta,sigma,sd
K=np.array([[np.exp(d*theta)]])
G=np.array([[sigma*np.sqrt((np.exp(2*d*theta)-1)/(2*theta))]])
H=np.array([[1]])
D=np.array([[sd]])
Grad_S=np.array([[[0]],[[0]],[[1]]],dtype=float)
Grad_R_sigma_s=(np.exp(2*theta*d)-1)/(2*theta)
Grad_R_theta=(sigma**2/(2*theta**2))*(1-np.exp(2*theta*d)+2*d*theta*np.exp(2*theta*d))
Grad_R=np.zeros((3,1,1),dtype=float)
Grad_R[0,0,0]=Grad_R_theta
Grad_R[1,0,0]=Grad_R_sigma_s
Grad_K=np.array([[[d*np.exp(d*theta)]],[[0]],[[0]]],dtype=float)
x_kf,x_kf_smooth,Grad_log_lik=bdg.KF_Grad_lik(x0[0],dim,dim_o,K,G,H,D,obs,Grad_K,Grad_R,Grad_S)
x_kf_2=bdg.KF(x0[0],dim,dim_o,K,G,H,D,obs)[0]
Grad_log_lik[1,0,0]=2*Grad_log_lik[1,0,0]*sigma
print(x_kf)
print(obs.shape)


#%%
import numpy as np

class KalmanFilter:
    def __init__(self, F, B, H, Q, R, x0, P0):
        self.F = F  # State transition model
        self.B = B  
        self.H = H  
        self.Q = Q  
        self.R = R  
        self.x = x0  
        self.P = P0  
    def predict(self, u):
        # Predict the state and state covariance
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x
    def update(self, z):
        # Compute the Kalman gain
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))      
        # Update the state estimate and covariance matrix
        y = z - np.dot(self.H, self.x)  
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
        return self.x, self.P
# Example usage
F = K
B = np.array([[0]])     
H = np.array([[1]])         
Q = G**2
R = D**2
# Initial state and covariance
x0 = np.array([[1.2]]) 
P0 = np.array([[0]]) 
# Create Kalman Filter instance
kf = KalmanFilter(F, B, H, Q, R, x0, P0)
# Predict and update with the control input and measurement
u = np.array([[0]])  
z = np.array([[obs[0]]]) 
# Predict step
predicted_state = kf.predict(u)
#print("Predicted state:\n", predicted_state)
# Update step
updated_state, updated_cov = kf.update(z)
print("Updated state:\n", updated_state)   
#_____________________________________________________________________________________________________________

kf = KalmanFilter(F, B, H, Q, R, updated_state, updated_cov)
# Predict and update with the control input and measurement
u = np.array([[0]])  
z = np.array([[obs[1]]]) 
# Predict step
predicted_state = kf.predict(u)
#print("Predicted state:\n", predicted_state)
# Update step
updated_state, updated_cov = kf.update(z)
print("Updated state:\n", updated_state)    
#%%



#%%
###################################################################################################
################################################################################################################################################################################################################################
################################################################################################################################################################################################################################

################################################################################################################################################################################################################################
# IN THE FOLLOWING WE TEST THE ALGORITHM FOR THE RED KANGAROO DATA

# The file can be obtained as
rkdata= np.loadtxt("Kangaroo_data.txt")
print(rkdata.shape)
print(rkdata[0,2]-rkdata[-1,2])
# %%
plt.scatter(range(len(rkdata[:,2])-1),rkdata[1:,2]-  rkdata[:-1,2])
min_d=np.min(rkdata[1:,2]-  rkdata[:-1,2])
max_d=np.max(rkdata[1:,2]-  rkdata[:-1,2])
print(min_d,max_d)
# %%
# The policy we are going to use regarding the time step is the following, we choose the smallest interval
# then we divide that into 8, that gives us the first delta_l.
Delta_l=min_d/8
print(Delta_l)
intervals=rkdata[1:,2]-  rkdata[:-1,2]
partits=np.rint(intervals/Delta_l)
deltas=intervals/partits
print(deltas)
# %%
#Let's make this into a function

def get_deltas(data):
    Delta_l=np.min(data[1:]-  data[:-1])/8
    intervals=data[1:]-  data[:-1]
    partits=np.rint(intervals/Delta_l)
    deltas=intervals/partits
    return deltas
# %%
# IN THE FOLLOWING WE TEST THE BRIDGE ALGORITHM FOR THE RED KANGAROO MODEL.
# We choose the intial distriubtion as the equilibrium distribution (gamma) such that 
# the hidden model will have the same distribution. 

# 1) Sett the parameters:
N=1000
the1=1
the2=1
the3=1
dist_params=[the1,the2,the3]
# 2) Get the sampling from the initial distribution

in_dis=bdg.gamma_sampling(N,dist_params)


# 3) Get the jump, x_pr

x_pr=bdg.gamma_sampling(N,dist_params)+ind_dis

# 4) Get the bridge
t0=0
x0=in_dis
T=2.3
x_p=x_pr
l=10
# b=bdg.b_log
A=[the1,the2,the3]
# Sig=bdg.Sig_gbm_1d
fi=the3
# b_til=bdg.b_log_aux




# 
# Bridge_1d(t0,x0,T,x_p,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,l,d,N,seed\
#    ,crossed=False,backward=False,j=False,fd=False,N_pf=False,cond_seed=False,an=False,Dt=False)


# int_G,int_psi_the,int_psi_sig,x_the_mid,x_sig_mid
# %%
# THIS SPACE IS RESERVED TO THE TEST OF BASIC FUNCTIONS.

#%%

# Test of b_log
N=3
the1=2
the2=1
the3=0.5
dist_params=[the1,the2,the3]
x=bdg.gamma_sampling(N,dist_params)
print(x)
drift=bdg.b_log(x,dist_params)
print(drift)
print((the3**2/2+the1-x*the2)*x)
# The test seems to be working fine
# What will happen when one of these are negative? 

#%%
# Test of b_log_aux and update_log_functs
# def update_log_functs(pars_0,pars_1,t0,x0_0,x0_1,T,x_pr_0,x_pr_1,levels=2)
pars_0=dist_params
pars_1=dist_params
t0=1
T=3.2
x0_0=x
x0_1=x
x_pr_0=x+bdg.gamma_sampling(N,dist_params)
x_pr_1=x_pr_0
up=bdg.update_log_functs(pars_0,pars_1,t0,x0_0,x0_1,T,x_pr_0,x_pr_1,levels=1)
a,b,fi=up
print(bdg.b_log_aux(t0+(T-t0)/2,x_pr_0,[a,b])/x_pr_0,bdg.b_log(x_pr_0,dist_params))
print(bdg.b_log_aux(T,x_pr_0,[a,b])/x_pr_0,bdg.b_log(x_pr_0,dist_params))
print(bdg.b_log_aux(t0,x,[a,b])/x,bdg.b_log(x_pr_0,dist_params))
# The functions update_log_functs and b_log_aux are¿ working properly.
#%%
# Test of r_log and H_log

# def r_log_normal(t,x,T,x_pr,pars)
N=3
the1=2
the2=1
the3=0.5
dist_params=[the1,the2,the3]
t0=1.2
T=3.2
x=bdg.gamma_sampling(N,dist_params)
x_pr=x+bdg.gamma_sampling(N,dist_params)
pars=bdg.update_log_functs(dist_params,dist_params,t0,x,x,T,x_pr,x_pr,levels=1)
t=2.54
print(bdg.r_log_normal(t,x,T,x_pr,pars))                                   

# mechanically it seems like the function is working properly
#%%
print(bdg.H_log_normal(t,x,T,x_pr,pars))

# %%
# Test of the CIR proposal transition
N=10000000
the1=2
the2=1
the3=0.5
dist_params=[the1,the2,the3]
t0=1.2
T=3.2
x0=1.4+np.zeros(N)
x=bdg.gamma_sampling(N,dist_params)
chi=bdg.sampling_CIR(x,N,T-t0,dist_params)
#print(x,chi)

# %%
# comparison between the CIR proposal and the exact transition
# trans_noncentral_chisquare(t0,x0,T,x_pr,pars):
# sampling_CIR(x0,N,d,sample_pars):
samples = bdg.sampling_CIR(x0, N,T-t0,dist_params)

# 2. Create a histogram of the samples (normalized to form a density)
plt.figure(figsize=(8,5))
counts, bin_edges, _ = plt.hist(samples, bins=1000, density=True,alpha=0.5, label="Sampled histogram")
# 3. Evaluate the PDF on a grid of points covering the sample range
x_grid = np.linspace(bin_edges[0], bin_edges[-1], 300)  # e.g. 300 points
#print(t0,x0,)
pdf_vals = bdg.trans_noncentral_chisquare(t0,x0[0],T,x_grid, dist_params)
#print(pdf_vals)
# 4. Overlay the theoretical PDF curve
plt.plot(x_grid, pdf_vals, 'r-', linewidth=2, label="Theoretical PDF")
# 5. Label and show
plt.xlabel("x")
plt.ylabel("Density")
plt.title("Check: Histogram of Samples vs. Theoretical PDF")
plt.legend()
plt.show()
# %%
# Test for the noninformative particle filter

def log_g_den_nonin(obs,x_pr,g_den_par,crossed=False):

    return 1

#%%


N=50
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
seed=1
obs=np.array([1.4,2,3])
obs_times=np.array([1.2,2,3.5])
start=time.time()
bdg.Gen_PF_bridge(bdg.gamma_sampling,in_dist_pars, bdg.b_log,dist_params,\
    bdg.Sig_gbm_1d,the3,bdg.b_log_aux,A_til,bdg.Sig_gbm_1d,fi_til,bdg.r_log_normal,\
    r_pars,bdg.H_log_normal,H_pars,bdg.update_log_functs,\
    bdg.sampling_CIR,dist_params,obs,obs_times,log_g_den_nonin,\
    1, bdg.trans_log_normal,1,bdg.trans_noncentral_chisquare, resamp_coef, l, d,N,seed)
end=time.time()
print("Time taken for the noninformative particle filter: ",end-start)
# %%
if True:
    N=1000000
    the1=2
    the2=1
    the3=0.5
    dist_params=[the1,the2,the3]
    in_dist_pars=dist_params
    #t0=1.2
    #T=3.2
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
    start=time.time()
    l0=1
    Lmax=8
    eLes=np.array(range(l0,Lmax+1))
    samples=400
    meanss=np.zeros((len(eLes),samples,len(obs)))
    smss=np.zeros((len(eLes),samples,len(obs)))    
    inputs=[]

#meanss=np.zeros((len(eLes),samples,len(obs)))
#smss=np.zeros((len(eLes),samples,len(obs)))    
# %%
v="mean_check_lognormal_prop_1"

smss=np.reshape(np.loadtxt("Observationsdata/data6/Prl_Gen_PF_bridge_levels_smss"+v+".txt",\
dtype=float),(len(eLes),samples,len(obs)))
meanss=np.reshape(np.loadtxt("Observationsdata/data6/Prl_Gen_PF_bridge_meanss"+v+".txt",\
dtype=float),(len(eLes),samples,len(obs)))
#smss=np.reshape(np.loadtxt("Observationsdata/Prl_Gen_PF_bridge_levels_v"+v+".txt",\
#dtype=float),(len(eLes),samples,len(obs)))
#meanss=np.reshape(np.loadtxt("Observationsdata/Prl_Gen_PF_bridge_v"+v+".txt",\
#dtype=float),(len(eLes),samples,len(obs)))
# %%
x_means_means=np.mean(meanss,axis=(1))
var_x_means=np.var(meanss,axis=(1))
x_sm_means=np.mean(smss,axis=1)
var_x_sm=np.var(smss,axis=1)
the1,the2,the3= dist_params
w=the3**2/2+the1
xi=the2
sigma=the3
alpha=2*w/sigma**2-1
theta=sigma**2/(2*xi)
print(alpha,theta)
#%%
the1,the2,the3= dist_params
w=the3**2/2+the1
xi=the2
sigma=the3
alpha=2*w/sigma**2-1
theta=sigma**2/(2*xi)
print(alpha,theta)
counts, bin_edges, _ = plt.hist(x_pr[0], bins=1000, density=True,alpha=0.5, label="Sampled histogram")
#3. Evaluate the PDF on a grid of points covering the sample range
x_grid = np.linspace(bin_edges[0], bin_edges[-1], 300)  # e.g. 300 points
#print(t0,x0,)
pdf_vals = gamma.pdf(x_grid,alpha,scale=theta)
plt.plot(x_grid, pdf_vals, 'r-', linewidth=2, label="Theoretical PDF")

#%%
t=-1
bias=np.abs(x_means_means-theta*alpha)
#print(bias)
sm_bias=np.abs(x_sm_means-(theta**2*alpha+(theta*alpha)**2))
sm_bias_ub=sm_bias+np.sqrt(var_x_sm)*1.96/np.sqrt(samples)
bias_ub=bias+np.sqrt(var_x_means)*1.96/np.sqrt(samples)
bias_lb=bias-np.sqrt(var_x_means)*1.96/np.sqrt(samples)
sm_bias_lb=sm_bias-np.sqrt(var_x_sm)*1.96/np.sqrt(samples)
plt.plot(eLes,bias[:,t],label="Bias")
plt.plot(eLes,bias_ub[:,t],label="Bias UB")
plt.plot(eLes,bias_lb[:,t],label="Bias LB")
plt.plot(eLes,sm_bias[:,t],label="sm bias")
plt.plot(eLes,sm_bias_ub[:,t],label="sm bias UB")
plt.plot(eLes,sm_bias_lb[:,t],label="sm bias LB")
plt.plot(eLes,bias[0,t]*2**eLes[0]/2**eLes)
plt.yscale("log")
plt.legend()
print(meanss[:,:2,:])

#%%
print(meanss[:,:2,:])
# %%
## Test for the rej_max_coup_CIR funciton
#rej_max_coup_CIR(x0,x1,N,d,sample_pars)
N=3
the1=2
the2=1
the3=0.5
dist_params=[the1,the2,the3]
t0=1.2
T=3.2
np.random.seed(1)
x0=bdg.gamma_sampling(dist_params,N)
x1=bdg.gamma_sampling(dist_params,N)
print(x0,x1)
#%%
B=10000
data0=np.zeros((B,N))
data1=np.zeros((B,N))
sample_pars=[the1,the2,the3,the1,the2,the3]
for b in range(B):
    data0[b],data1[b]=bdg.rej_max_coup_CIR(x0,x1,N,T-t0,sample_pars)
    
print(data0,data1)
# %%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_scatter_and_marginals(X, Y, pdf_x, pdf_y, bins=30):
    """
    Create a figure showing:
      1) Scatter of (X, Y)
      2) Histogram of X (above the scatter), overlaid with pdf_x
      3) Histogram of Y (to the right of the scatter), overlaid with pdf_y
    
    Parameters
    ----------
    X, Y : array_like, shape (n_samples,)
        Samples of the two random variables.
    pdf_x : function
        A function pdf_x(x_array) -> array of the same shape as x_array,
        giving the PDF for X.
    pdf_y : function
        A function pdf_y(y_array) -> array of the same shape as y_array,
        giving the PDF for Y.
    bins : int
        Number of bins for the histograms.
    """

    # 1. Basic checks
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape.")

    # 2. Set up the figure with GridSpec for 3 axes:
    #    - ax_main: the main scatter plot
    #    - ax_xhist: histogram for X across the top
    #    - ax_yhist: histogram for Y on the right
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(4, 4, figure=fig)
    
    # Main scatter plot: from row 1 to 4, col 0 to 3
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    # Top histogram: row 0, col 0 to 3
    ax_xhist = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    # Right histogram: row 1 to 4, col 3
    ax_yhist = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

    # 3. Scatter plot of (X, Y)
    ax_main.scatter(X, Y, alpha=0.5, s=20)
    ax_main.set_xlabel("X")
    ax_main.set_ylabel("Y")

    # 4. Histogram of X (top)
    ax_xhist.hist(X, bins=bins, density=True, alpha=0.5, color="C0")
    # Evaluate pdf_x on a grid covering the range of X
    x_grid = np.linspace(X.min(), X.max(), 200)
    px = pdf_x(x_grid)  # user-supplied pdf function
    ax_xhist.plot(x_grid, px, "r-", lw=2, label="pdf_x")
    ax_xhist.set_ylabel("Density (X)")
    ax_xhist.legend()
    # Hide x tick labels on ax_xhist to reduce clutter
    plt.setp(ax_xhist.get_xticklabels(), visible=False)

    # 5. Histogram of Y (right), oriented horizontally
    ax_yhist.hist(Y, bins=bins, density=True, alpha=0.5, color="C0", orientation="horizontal")
    # Evaluate pdf_y on a grid covering the range of Y
    y_grid = np.linspace(Y.min(), Y.max(), 200)
    py = pdf_y(y_grid)  # user-supplied pdf function
    # Plot the pdf_y vertically: we invert axes usage by swapping coords
    ax_yhist.plot(py, y_grid, "r-", lw=2, label="pdf_y")
    ax_yhist.set_xlabel("Density (Y)")
    ax_yhist.legend()
    # Hide y tick labels on ax_yhist
    plt.setp(ax_yhist.get_yticklabels(), visible=False)

    # 6. Adjust layout
    #    Typically we turn off some extra space between subplots
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------
# Example usage (mocking user-defined PDFs):
if __name__ == "__main__":
    particle=1
    # Generate sample data, say from normal distributions as an example
    np.random.seed(123)
    X = data0[:,particle]
    Y = data1[:,particle]
    
    # Suppose we define their "theoretical" PDFs (for demonstration):
    import math
    def pdf_x(x):
        # Normal with mean=2, std=1
        return bdg.trans_noncentral_chisquare(t0,x0[particle],T,x, dist_params)

    def pdf_y(y):
        # Normal with mean=5, std=2
        return bdg.trans_noncentral_chisquare(t0,x1[particle],T,y, dist_params)

    # Plot
    plot_scatter_and_marginals(X, Y, pdf_x, pdf_y, bins=30)


# %%
# test for the couple particle filter
if True:
    N=1000000
    the1=2
    the2=1
    the3=0.5
    dist_params=[the1,the2,the3]
    rej_dist_params=[the1,the2,the3,the1,the2,the3]
    in_dist_pars=dist_params
    w=the3**2/2+the1
    xi=the2
    sigma=the3
    alpha=2*w/sigma**2-1
    theta=sigma**2/(2*xi)
    #t0=1.2
    #T=3.2
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
    start=time.time()
    l0=2
    Lmax=9
    eLes=np.array(range(l0,Lmax+1))
    samples=100
    print("mean is:",theta*alpha)
    print("second moment is:",theta**2*alpha+(theta*alpha)**2)
#%%
v="v1_2"
x_means=np.reshape(np.loadtxt("Observationsdata/Prl_Gen_C_PF_bridge_v"+v+".txt",\
dtype=float),(2,len(eLes[:-2]),samples,len(obs)))
x_sm=np.reshape(np.loadtxt("Observationsdata/Prl_Gen_C_PF_bridge_levels_v"+v+".txt",\
dtype=float),(2,len(eLes[:-2]),samples,len(obs)))
v="v1_3"
x_means=np.concatenate((x_means,np.reshape(np.loadtxt("Observationsdata/Prl_Gen_C_PF_bridge_v"+v+".txt",\
dtype=float),(2,len(eLes[-2:]),samples,len(obs)))),axis=1)
x_sm=np.concatenate((x_sm,np.reshape(np.loadtxt("Observationsdata/Prl_Gen_C_PF_bridge_levels_v"+v+".txt",\
dtype=float),(2,len(eLes[-2:]),samples,len(obs)))),axis=1)

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
plt.plot(eLes,x_means_means[0,-1]*2**eLes[0]/2**eLes,label="$\Delta_l$")
plt.plot(eLes,x_means_means[-1,-1]*2**(eLes[-1]/2)/2**(eLes/2),label="$\Delta_l^{1/2}$")
plt.xlabel("$l$")
plt.yscale("log")
plt.legend()
#%%
level=1
#(2,len(eLes),samples,len(obs))
x_means_means=np.mean(x_means,axis=(2))[level]
print(x_means_means[:,-1])
var_x_means=np.var(x_means,axis=(2))[level]
x_sm_means=np.mean(x_sm,axis=(2))[level]
var_x_sm=np.var(x_sm,axis=(2))[level]
print(theta*alpha)
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
#%%
a=1e-3+(0.1)**2/2
print(2*(np.exp(a*51)-np.exp(a))/(np.exp(a)-1))

 # %%

# TEST FOR THE GENERAL CONDITIONAL PARTICLE FILETER WITH BACKWARD SAMPLING
if True:

    N=30
    the1=2
    the2=1
    the3=0.5
    the4=1
    the1,the2,the3,the4=2.397, 4.429e-3, 0.84, 17.36
    w=the3**2/2+the1
    xi=the2
    sigma=the3
    alpha=2*w/sigma**2-1
    theta=sigma**2/(2*xi)
    dist_params=[the1,the2,the3]
    rej_dist_params=[the1,the2,the3,the1,the2,the3]
    in_dist_pars=dist_params
    #t0=1.2
    #T=3.2
    A_til=dist_params
    fi_til=the3
    r_pars=1
    resamp_coef=1
    l=2
    d=1
    H_pars=1
    seed=6
    rkdata= np.loadtxt("Kangaroo_data.txt")
    obs=rkdata[:,:2]
    #obs=np.array([1.4,2,3,4.7,5.3,6.5])
    obs_times=rkdata[:,2]
    print(bdg.get_deltas(obs_times))
    print(np.log2( bdg.get_deltas(obs_times)))
    #obs_times=np.array([1.2,2,3.5,4.7,5.3,6.5])/3
    start=time.time()
    samples=40
    #the4=1    
#%%
#"""
#%%
seed=1
N=50
l=6
print(l)

start=time.time()
[log_weights,int_Gs,x_pr,indices]=bdg.Gen_PF_bridge(bdg.gamma_sampling,in_dist_pars, bdg.b_log,dist_params,\
        bdg.Sig_gbm_1d,the3,bdg.b_log_aux,A_til,bdg.Sig_aux_gbm_1d,fi_til,bdg.r_log_normal,\
        r_pars,bdg.H_log_normal,H_pars,bdg.update_log_functs,\
        bdg.sampling_prop_log_normal,dist_params,obs,obs_times,bdg.log_g_nbino_den,\
        the4, bdg.trans_log_normal,1, bdg.trans_prop_log_normal, resamp_coef, l, d,N,seed)
end=time.time()
print("Time taken for the particle filter: ",end-start)
print(log_weights.shape)
weights=pff.norm_logweights(log_weights,ax=1)
fmean=np.mean(x_pr,axis=1)  
#print(fmean.shape)    
#plt.plot(fmean)
#plt.scatter(obs)
#%%
#"""
start=time.time()
B=10
samples=10
d=1
T=len(obs_times)*d
Lmax=5
l0=5
eLes=np.array(range(l0,Lmax+1))
seed=436
mcmc_mean=np.zeros((len(eLes),samples,int(T/d)))
resamp_coef=1
for j in range(len(eLes)):
    l=eLes[j]
    print("l is ",l)
    for i in range(samples):
        seed+=1
        np.random.seed(i)
        #print("Seed feeded to PF_bridge is: ",seed)
        [log_weights,int_Gs,x_pr,indices]=bdg.Gen_PF_bridge(bdg.gamma_sampling,in_dist_pars, bdg.b_log,dist_params,\
        bdg.Sig_gbm_1d,the3,bdg.b_log_aux,A_til,bdg.Sig_aux_gbm_1d,fi_til,bdg.r_log_normal,\
        r_pars,bdg.H_log_normal,H_pars,bdg.update_log_functs,\
        bdg.sampling_CIR,dist_params,obs,obs_times,bdg.log_g_den_nonin,\
        the4, bdg.trans_log_normal,1, bdg.trans_noncentral_chisquare, resamp_coef, l, d,N,seed)
        #print(log_weights)
        #x_mean=np.sum(pff.norm_logweights(log_weights,ax=1)*x_pr,axis=-1)
        weights=pff.norm_logweights(log_weights[-1])
        #print(weights.shape)
        index=np.random.choice(np.array(range(N)))
        cond_path=x_pr[:,index]
        cond_log_weights=log_weights[:,index]
        clwn=np.copy(cond_log_weights)
        cond_int_G=int_Gs[:,index]
        cign=np.copy(cond_int_G)
        seeds_cond=np.zeros((int(T/d)-1,2),dtype=int)
        seeds_cond[:,0]=seed+np.array(range(int(T/d)-1))*int(int(2**l-1))
        seeds_cond[:,1]=index*np.ones(int(T/d)-1)
        scn=np.copy(seeds_cond)
        ch_paths=np.zeros((B,int(T/d)))
        ch_weights=np.zeros((B,int(T/d)))
        ch_whole_paths=np.zeros((B,int(T/d)))
        ch_whole_weights=np.zeros((B,int(T/d)))


        cond_whole_path=cond_path
        cwpn=cond_path
        cond_whole_log_weights=cond_log_weights
        
        for b in range(B):
            print("sample iteration: ",i," chain iteration: ",b)
            seed+=int((int(T/d))*int(int(2**l-1)))
            np.random.seed(seed)
            [log_weights,x_pr,cond_log_weights,int_Gs_cond,cond_path,seeds_cond]=\
            bdg.Gen_Cond_PF_bridge_back_samp(bdg.gamma_sampling,in_dist_pars, cond_path,seeds_cond,\
            bdg.b_log,dist_params,\
            bdg.Sig_gbm_1d,the3,bdg.b_log_aux,bdg.Sig_aux_gbm_1d,bdg.r_log_normal,\
            bdg.H_log_normal, bdg.update_log_functs,\
            bdg.sampling_prop_log_normal,[the1,the2,the3],obs,obs_times,bdg.Grad_log_g_nonin,\
            the4, bdg.trans_log_normal, bdg.trans_prop_log_normal, resamp_coef, l, d,N,seed)
            #(in_dist,in_dist_pars,x_cond,seeds_cond,t0,x0,T,b,A,Sig,\
            #fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,update_func,sample_funct,sample_pars,\
            #obs,obs_times,log_g_den,g_den_par, aux_trans_den,atdp, prop_trans_den, resamp_coef, l, d,N,seed):
            #(bdg.gamma_sampling,in_dist_pars, bdg.b_log,dist_params,\
            #bdg.Sig_gbm_1d,the3,bdg.b_log_aux,A_til,bdg.Sig_aux_gbm_1d,fi_til,bdg.r_log_normal,\
            #r_pars,bdg.H_log_normal,H_pars,bdg.update_log_functs,\
            #bdg.sampling_CIR,dist_params,obs,obs_times,log_g_den_nonin,\
            #1, bdg.trans_log_normal,1, bdg.trans_noncentral_chisquare, resamp_coef, l, d,N,seed)
            #seed=int((int(T/d))*int(int(2**l*d-1)))
            ch_paths[b]=cond_path
            ch_weights[b]=cond_log_weights
            #print("seed conditionals are:",seeds_cond)
        mcmc_mean[j,i]=np.mean(ch_paths,axis=0)


end=time.time()
print(end-start)
#%%
# (len(eLes),samples,int(T/d))

#mcmc_mean_2=mcmc_mean
#%%
print(cond_log_weights)
tro_seeds_cond=seeds_cond
tro_cond_path=cond_path
tro_seeds=seed
print(tro_cond_path[1:]-tro_cond_path[:-1])
# %%
# np.zeros((len(eLes),samples,2,int(T/d)))
t=-1
Lmax=4
l0=1
eLes=np.array(range(l0,Lmax+1))
mean=np.mean(mcmc_mean,axis=(1))
mean_2=np.mean(mcmc_mean_2,axis=(1))
print(mean[:,t],alpha*theta)
print(mean_2[:,t],alpha*theta)

var=np.var(mcmc_mean,axis=(1))
var_2=np.var(mcmc_mean_2,axis=(1))
bias=np.abs(mean-theta*alpha)   
bias_2=np.abs(mean_2-theta*alpha)   
bias_ub=bias+np.sqrt(var)*1.96/np.sqrt(samples)
bias_lb=bias-np.sqrt(var)*1.96/np.sqrt(samples)
bias_ub_2=bias_2+np.sqrt(var_2)*1.96/np.sqrt(samples)
bias_lb_2=bias_2-np.sqrt(var_2)*1.96/np.sqrt(samples)
print(bias_lb_2[:,t])
plt.plot(eLes,bias[:,t],label="Bias")
plt.plot(eLes,bias_ub[:,t],label="Bias UB")
plt.plot(eLes,bias_lb[:,t],label="Bias LB")

plt.plot(eLes,bias_2[:,t],label="Bias_2")
plt.plot(eLes,bias_ub_2[:,t],label="Bias UB 2")
plt.plot(eLes,bias_lb_2[:,t],label="Bias LB 2")


plt.yscale("log")
plt.legend()
# %%
if True:
    """samples=40
    B=1
    rkdata= np.loadtxt("Kangaroo_data.txt")
    obs=rkdata[:3,:2]
    obs_times=rkdata[:3,2]
    
    #obs=np.array([1.4,2,3])
    #obs_times=np.array([1.2,2,3.5])/5

    #arg_cm=int(sys.argv[1])
    inputs=[]
    start=time.time()
    d=1
    T=len(obs_times)*d
    p=15
    l=10
    N0=30
    eNes=N0*2**np.array(range(p))
    d=1
    T=len(obs_times)*d"""

    N=100
    samples=40
    B=5000
    rkdata= np.loadtxt("Kangaroo_data.txt")
    obs=rkdata[:3,:2]
    obs_times=rkdata[:3,2]
    
    #obs=np.array([1.4,2,3])
    #obs_times=np.array([1.2,2,3.5])/5

    #arg_cm=int(sys.argv[1])
    inputs=[]
    start=time.time()
    l0=4
    Lmax=10
    eLes=np.array(range(l0,Lmax+1))
    #eLes=eNes
    mcmcs=np.zeros((samples,len(eLes),int(T/d)))
    the1=2
    the2=1
    the3=0.5
    the1,the2,the3,the4=2.397, 4.429e-3, 0.84, 17.36
    w=the3**2/2+the1
    xi=the2
    sigma=the3
    alpha=2*w/sigma**2-1
    theta=sigma**2/(2*xi)
    print("mean is: ",alpha*theta)
    dist_params=[the1,the2,the3]
    
    in_dist_pars=dist_params
    A_til=1
    fi_til=the3
    r_pars=1
    resamp_coef=1
    H_pars=1
    d=1
    start=time.time()    
    T=len(obs_times)*d
# %%
v="GSL_ip_ii"
mcmcs=np.reshape(np.loadtxt("Observationsdata/data6/Prl_Gen_smoother_logarithmic_v"+v+".txt",\
dtype=float),(samples,len(eLes),int(T/d)))
#%%

v="GSL16_ip_ii"
labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])

mcmcs=np.reshape(\
    np.loadtxt("Observationsdata/data7/Prl_Gen_smoother_logarithmic_v"+v+labels[0]+".txt",dtype=float),(samples,len(eLes),int(T/d))) 
pf_means=np.reshape(\
    np.loadtxt("Observationsdata/data7/Prl_Gen_smoother_logarithmic_pf_means_v"+v+labels[0]+".txt",dtype=float),(samples,len(eLes),B,int(T/d))) 


for i in range(len(labels[1:])):
    mcmcs=np.concatenate((mcmcs,np.reshape(\
    np.loadtxt("Observationsdata/data7/Prl_Gen_smoother_logarithmic_v"+v+labels[i+1]+".txt",dtype=float),(samples,len(eLes),int(T/d))) ),axis=0)  

    pf_means=np.concatenate((pf_means,np.reshape(\
    np.loadtxt("Observationsdata/data7/Prl_Gen_smoother_logarithmic_pf_means_v"+v+labels[i+1]+".txt",dtype=float),(samples,len(eLes),B,int(T/d))) ),axis=0)  
    

#%%
v="GSL9_ip_i"
labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])

mcmcs=np.reshape(\
    np.loadtxt("Observationsdata/data7/Prl_Gen_smoother_logarithmic_v"+v+labels[0]+".txt",dtype=float),(samples,len(eLes),int(T/d))) 
for i in range(len(labels[1:])):
    mcmcs=np.concatenate((mcmcs,np.reshape(\
    np.loadtxt("Observationsdata/data7/Prl_Gen_smoother_logarithmic_v"+v+labels[i+1]+".txt",dtype=float),(samples,len(eLes),int(T/d))) ),axis=0)  

v="GSL9_ip_ii"

for i in range(len(labels)):
    mcmcs=np.concatenate((mcmcs,np.reshape(\
    np.loadtxt("Observationsdata/data7/Prl_Gen_smoother_logarithmic_v"+v+labels[i]+".txt",dtype=float),(samples,len(eLes),int(T/d))) ),axis=0)  



#%%

v="GSL6_ip_i"
labels=np.array(["1", "2", "3"  ,"5", "6", "7", "8", "9", "10", \
"11", "12",  "17", "18", "19"\
    ,"21",  "24", "25", "26", "27", "28", "29","30"])

mcmcs=np.reshape(\
    np.loadtxt("Observationsdata/data7/Prl_Gen_smoother_logarithmic_v"+v+labels[0]+".txt",dtype=float),(samples,len(eLes),int(T/d))) 
for i in range(len(labels[1:])):
    mcmcs=np.concatenate((mcmcs,np.reshape(\
    np.loadtxt("Observationsdata/data7/Prl_Gen_smoother_logarithmic_v"+v+labels[i+1]+".txt",dtype=float),(samples,len(eLes),int(T/d))) ),axis=0)  


v="GSL6_ip_ii"
labels=np.array(["1", "2", "3"  ,"5", "6", "7", "8", "9", "10", \
"11", "12",  "17", "18", "19"\
    ,"21",  "24", "25", "26", "27", "28", "29","30"])

for i in range(len(labels)):
    mcmcs=np.concatenate((mcmcs,np.reshape(\
    np.loadtxt("Observationsdata/data7/Prl_Gen_smoother_logarithmic_v"+v+labels[i]+".txt",dtype=float),(samples,len(eLes),int(T/d))) ),axis=0)  



#%%

v="GSL5_ip_i"
labels=np.array(["1", "2", "3"  ,"5", "6", "7", "8", "9", "10", \
"11", "12",  "17", "18", "19"\
    ,"21",  "24", "25", "26", "27", "28", "29","30"])

mcmcs=np.reshape(\
    np.loadtxt("Observationsdata/data7/Prl_Gen_smoother_logarithmic_v"+v+labels[0]+".txt",dtype=float),(samples,len(eLes),int(T/d))) 
for i in range(len(labels[1:])):
    mcmcs=np.concatenate((mcmcs,np.reshape(\
    np.loadtxt("Observationsdata/data7/Prl_Gen_smoother_logarithmic_v"+v+labels[i+1]+".txt",dtype=float),(samples,len(eLes),int(T/d))) ),axis=0)  

#%%


v="GSL4_ip_i"
labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])

mcmcs=np.reshape(\
    np.loadtxt("Observationsdata/data7/Prl_Gen_smoother_logarithmic_v"+v+labels[0]+".txt",dtype=float),(samples,len(eLes),int(T/d))) 

for i in range(len(labels[1:])):
    mcmcs=np.concatenate((mcmcs,np.reshape(\
    np.loadtxt("Observationsdata/data7/Prl_Gen_smoother_logarithmic_v"+v+labels[i+1]+".txt",dtype=float),(samples,len(eLes),int(T/d))) ),axis=0)  

v="GSL4_ip_ii"
for i in range(len(labels)):
    mcmcs=np.concatenate((mcmcs,np.reshape(\
    np.loadtxt("Observationsdata/data7/Prl_Gen_smoother_logarithmic_v"+v+labels[i]+".txt",dtype=float),(samples,len(eLes),int(T/d))) ),axis=0)  

v="GSL4_ip_iii"
for i in range(len(labels)):
    mcmcs=np.concatenate((mcmcs,np.reshape(\
    np.loadtxt("Observationsdata/data7/Prl_Gen_smoother_logarithmic_v"+v+labels[i]+".txt",dtype=float),(samples,len(eLes),int(T/d))) ),axis=0)  


v="GSL4_ip_iv"
for i in range(len(labels)):
    mcmcs=np.concatenate((mcmcs,np.reshape(\
    np.loadtxt("Observationsdata/data7/Prl_Gen_smoother_logarithmic_v"+v+labels[i]+".txt",dtype=float),(samples,len(eLes),int(T/d))) ),axis=0)  



#%%
v="GSL2_ip_ii"

for i in range(len(labels)):
    mcmcs=np.concatenate((mcmcs,np.reshape(\
    np.loadtxt("Observationsdata/data7/Prl_Gen_smoother_logarithmic_v"+v+labels[i]+".txt",dtype=float),(samples,len(eLes),int(T/d))) ),axis=0)  

v="GSL2_ip_iii"

for i in range(len(labels)):
    mcmcs=np.concatenate((mcmcs,np.reshape(\
    np.loadtxt("Observationsdata/data7/Prl_Gen_smoother_logarithmic_v"+v+labels[i]+".txt",dtype=float),(samples,len(eLes),int(T/d))) ),axis=0)  
v="GSL2_ip_iv"

for i in range(len(labels)):
    mcmcs=np.concatenate((mcmcs,np.reshape(\
    np.loadtxt("Observationsdata/data7/Prl_Gen_smoother_logarithmic_v"+v+labels[i]+".txt",dtype=float),(samples,len(eLes),int(T/d))) ),axis=0)  

v="GSL2_ip_v"

for i in range(len(labels)):
    mcmcs=np.concatenate((mcmcs,np.reshape(\
    np.loadtxt("Observationsdata/data7/Prl_Gen_smoother_logarithmic_v"+v+labels[i]+".txt",dtype=float),(samples,len(eLes),int(T/d))) ),axis=0)  


v="GSL2_ip_vi"
samples=80*5
for i in range(len(labels)):
    mcmcs=np.concatenate((mcmcs,np.reshape(\
    np.loadtxt("Observationsdata/data7/Prl_Gen_smoother_logarithmic_v"+v+labels[i]+".txt",dtype=float),(samples,len(eLes),int(T/d))) ),axis=0)  


#%%

v="GSL_ip_iii"
labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])

mcmcs=np.reshape(\
    np.loadtxt("Observationsdata/data6/Prl_Gen_smoother_logarithmic_v"+v+labels[0]+".txt",dtype=float),(samples,len(eLes),int(T/d))) 

for i in range(len(labels[1:])):
    mcmcs=np.concatenate((mcmcs,np.reshape(\
    np.loadtxt("Observationsdata/data6/Prl_Gen_smoother_logarithmic_v"+v+labels[i+1]+".txt",dtype=float),(samples,len(eLes),int(T/d))) ),axis=0)  


#%%
v="GSL_ip_iv"
labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])

for i in range(len(labels)):
    mcmcs=np.concatenate((mcmcs,np.reshape(\
    np.loadtxt("Observationsdata/data6/Prl_Gen_smoother_logarithmic_v"+v+labels[i]+".txt",dtype=float),(samples,len(eLes),int(T/d))) ),axis=0)  


v="GSL_ip_v"
labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])

for i in range(len(labels)):
    mcmcs=np.concatenate((mcmcs,np.reshape(\
    np.loadtxt("Observationsdata/data6/Prl_Gen_smoother_logarithmic_v"+v+labels[i]+".txt",dtype=float),(samples,len(eLes),int(T/d))) ),axis=0)  

v="GSL_ip_vi"
labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])

for i in range(len(labels)):
    mcmcs=np.concatenate((mcmcs,np.reshape(\
    np.loadtxt("Observationsdata/data6/Prl_Gen_smoother_logarithmic_v"+v+labels[i]+".txt",dtype=float),(samples,len(eLes),int(T/d))) ),axis=0)  

v="GSL_ip_vii"
labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])

for i in range(len(labels)):
    mcmcs=np.concatenate((mcmcs,np.reshape(\
    np.loadtxt("Observationsdata/data6/Prl_Gen_smoother_logarithmic_v"+v+labels[i]+".txt",dtype=float),(samples,len(eLes),int(T/d))) ),axis=0)  

v="GSL_ip_viii"
labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])

for i in range(len(labels)):
    mcmcs=np.concatenate((mcmcs,np.reshape(\
    np.loadtxt("Observationsdata/data6/Prl_Gen_smoother_logarithmic_v"+v+labels[i]+".txt",dtype=float),(samples,len(eLes),int(T/d))) ),axis=0)  


v="GSL_ip_ix"
labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])

for i in range(len(labels)):
    mcmcs=np.concatenate((mcmcs,np.reshape(\
    np.loadtxt("Observationsdata/data6/Prl_Gen_smoother_logarithmic_v"+v+labels[i]+".txt",dtype=float),(samples,len(eLes),int(T/d))) ),axis=0)  


v="GSL_ip_x"
labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])

for i in range(len(labels)):
    mcmcs=np.concatenate((mcmcs,np.reshape(\
    np.loadtxt("Observationsdata/data6/Prl_Gen_smoother_logarithmic_v"+v+labels[i]+".txt",dtype=float),(samples,len(eLes),int(T/d))) ),axis=0)  

# %%
t=-3
print(pf_means.shape)
pf_mean=np.mean(pf_means,axis=(0,2))
pf_var=np.var(pf_means,axis=(0,2))
print(pf_mean)
pf_bias=np.abs(pf_mean-theta*alpha)
pf_bias_ub=pf_bias+np.sqrt(pf_var)*1.96/np.sqrt(samples*B)
pf_bias_lb=pf_bias-np.sqrt(pf_var)*1.96/np.sqrt(samples*B)
plt.plot(eLes,pf_bias[:,t],label="PF Bias") 
plt.plot(eLes,pf_bias_ub[:,t],label="PF Bias UB")
plt.plot(eLes,pf_bias_lb[:,t],label="PF Bias LB")
#plt.plot(eLes,pf_bias[-1,t]*np.sqrt(eLes[-1])/np.sqrt(eLes),label="$1/N$")

plt.legend()
plt.yscale("log")
#plt.xscale("log")
#%%

t=-1
print("the mean is: ",theta*alpha)
mean=np.mean(mcmcs,axis=(0))
print(mean.shape)
print(mean)
var=np.var(mcmcs,axis=(0)) 
bias=np.abs(mean-theta*alpha)   
bias_ub=bias+np.sqrt(var)*1.96/np.sqrt(samples)
bias_lb=bias-np.sqrt(var)*1.96/np.sqrt(samples)
plt.plot(eLes,bias[:,t],label="Bias")
plt.plot(eLes,bias_ub[:,t],label="Bias UB")
plt.plot(eLes,bias_lb[:,t],label="Bias LB")
#print(bias_lb[:,t ])
plt.plot(eLes,bias[0,t]*2**eLes[0]/2**eLes,label="$\Delta_l$")
plt.yscale("log")
#plt.xscale("log")
plt.xlabel("$l$")
plt.legend()
#%%
#%%
# TEST FOR THE GENERAL COUPLED CONDITIONAL PARTICLE FILETER WITH BACKWARD SAMPLING
if True:
    samples=40
    B=1
    rkdata= np.loadtxt("Kangaroo_data.txt")
    obs=rkdata[:3,:2]
    obs_times=rkdata[:3,2]
    
    #obs=np.array([1.4,2,3,4.7])
    #obs_times=np.array([1.2,2,3.5,4.7])/3
    inputs=[]
    start=time.time()
    #l0=4
    #Lmax=9
    N0=30
    p=10
    eNes=N0*2**np.array(range(p))
    #arg_cm=int(sys.argv[1])
    l=6
    d=1
    H_pars=1
    seed=6
    #rkdata= np.loadtxt("Kangaroo_data.txt")
    #obs=rkdata[:,:2]
    #obs_times=rkdata[:,2]
    obs=np.array([1.4,2,3,4.7,5.3,6.5])
    obs_times=np.array([1.2,2,3.5,4.7,5.3,6.5])/3
    start=time.time()
    samples=40
    print("mean is: ",alpha*theta)
    #the4=1    

#"""
#%%
seed=1
[log_weights,int_Gs,x_pr]=bdg.Gen_PF_bridge(bdg.gamma_sampling,in_dist_pars, bdg.b_log,dist_params,\
        bdg.Sig_gbm_1d,the3,bdg.b_log_aux,A_til,bdg.Sig_aux_gbm_1d,fi_til,bdg.r_log_normal,\
        r_pars,bdg.H_log_normal,H_pars,bdg.update_log_functs,\
        bdg.sampling_prop_log_normal,dist_params,obs,obs_times,bdg.log_g_nbino_den,\
        the4, bdg.trans_log_normal,1, bdg.trans_prop_log_normal, resamp_coef, l, d,N,seed)

#%%
#"""

start=time.time()
B=2
samples=3
d=1
T=len(obs_times)*d
l0=4
Lmax=4
eLes=np.array(range(l0,Lmax+1))
seed=436
print(seed)
mcmc_mean=np.zeros((2,len(eLes),samples,2,int(T/d)))
resamp_coef=1
for j in range(len(eLes)):
    l=eLes[j]
    print("l is ",l)
    for i in range(samples):
        seed+=1
        np.random.seed(i)
        #print("Seed feeded to PF_bridge is: ",seed)
        [log_weights,int_Gs,x_pr]=bdg.Gen_PF_bridge(bdg.gamma_sampling,in_dist_pars, bdg.b_log,dist_params,\
        bdg.Sig_gbm_1d,the3,bdg.b_log_aux,A_til,bdg.Sig_aux_gbm_1d,fi_til,bdg.r_log_normal,\
        r_pars,bdg.H_log_normal,H_pars,bdg.update_log_functs,\
        bdg.sampling_CIR,dist_params,obs,obs_times,bdg.log_g_den_nonin,\
        the4, bdg.trans_log_normal,1, bdg.trans_noncentral_chisquare, resamp_coef, l, d,N,seed)
        #print(log_weights)
        #x_mean=np.sum(pff.norm_logweights(log_weights,ax=1)*x_pr,axis=-1)
        weights=pff.norm_logweights(log_weights[-1])
        #print(weights.shape)
        index=np.random.choice(np.array(range(N)))
        cond_path_0=x_pr[:,index]
        cond_path_1=np.copy(cond_path_0)
        cond_log_weights=log_weights[:,index]
        clwn=np.copy(cond_log_weights)
        cond_int_G=int_Gs[:,index]
        cign=np.copy(cond_int_G)
        seeds_cond=np.zeros((int(T/d)-1,2),dtype=int)
        seeds_cond[:,0]=seed+np.array(range(int(T/d)-1))*int(int(2**l-1))
        seeds_cond[:,1]=index*np.ones(int(T/d)-1)
        seeds_cond_0=np.copy(seeds_cond)
        seeds_cond_1=np.copy(seeds_cond)
        scn=np.copy(seeds_cond)
        ch_paths=np.zeros((2,B,int(T/d)))
        ch_weights=np.zeros((2,B,int(T/d)))
        ch_whole_paths=np.zeros((B,int(T/d)))
        ch_whole_weights=np.zeros((B,int(T/d)))
       
        for b in range(B):
            print("sample iteration: ",i," chain iteration: ",b)
            seed+=int((int(T/d))*int(int(2**l-1)))
            np.random.seed(seed)
            print(seed)
            [log_weights_0,log_weights_1,x_pr_0,x_pr_1,cond_log_weights_0,cond_log_weights_1,\
            int_Gs_cond_0,int_Gs_cond_1,cond_path_0,cond_path_1,seeds_cond_0,seeds_cond_1]=\
            bdg.Gen_C_Cond_PF_bridge_back_samp(bdg.rej_max_coup_gamma_in_dist,coup_in_dist_pars,\
            cond_path_0,cond_path_1 ,seeds_cond_0,\
            seeds_cond_1,bdg.b_log,dist_params_0, dist_params_1,bdg.Sig_gbm_1d,the3,the3,\
            bdg.b_log_aux,bdg.Sig_aux_gbm_1d,\
            bdg.r_log_normal,bdg.H_log_normal,bdg.update_log_functs,\
            bdg.rej_max_coup_log_normal,[the1,the2,the3,the1,the2,the3],\
            obs,obs_times,bdg.log_g_den_nonin,the4,the4,bdg.trans_log_normal,\
            bdg.trans_prop_log_normal,[the1,the2,the3],[the1,the2,the3], resamp_coef, l, d,N,seed)


            #[log_weights,x_pr,cond_log_weights,int_Gs_cond,cond_path,seeds_cond]=\
            #bdg.Gen_Cond_PF_bridge_back_samp(bdg.gamma_sampling,in_dist_pars, cond_path,seeds_cond,\
            #bdg.b_log,dist_params,\
            #bdg.Sig_gbm_1d,the3,bdg.b_log_aux,A_til,bdg.Sig_aux_gbm_1d,fi_til,bdg.r_log_normal,\
            #r_pars,bdg.H_log_normal,H_pars,\
            #bdg.update_log_functs,\
            #bdg.sampling_prop_log_normal,[the1,the2,the3],obs,obs_times,bdg.log_g_nbino_den,\
            #the4, bdg.trans_log_normal,1, bdg.trans_prop_log_normal, resamp_coef, l, d,N,seed)
            
            ch_paths[:,b]=np.array([cond_path_0,cond_path_1])
            ch_weights[:,b]=np.array([cond_log_weights_0,cond_log_weights_1])
                   
            #print("seed conditionals are:",seeds_cond)
        mcmc_mean[:,j,i,0]=np.mean(ch_paths,axis=1)
        mcmc_mean[:,j,i,1]=np.mean(ch_weights,axis=1)
end=time.time()

# %%
if True:
    N=5000
    samples=40
    B=1
    rkdata= np.loadtxt("Kangaroo_data.txt")
    obs=rkdata[:3,:2]
    obs_times=rkdata[:3,2]
    
    #obs=np.array([1.4,2,3,4.7])
    #obs_times=np.array([1.2,2,3.5,4.7])/3
    inputs=[]
    start=time.time()
    l0=4
    Lmax=10
    N0=30
    p=13
    eNes=N0*2**np.array(range(p))
    #arg_cm=int(sys.argv[1])
    #l=6
    
    d=1
    T=len(obs_times)*d
    eLes=np.array(range(l0,Lmax+1))
    eLes=eNes
    
#%%

d=1
T=len(obs_times)*d
v="GCSL4_ip_iv"
labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10",\
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])
mcmcs=np.reshape(\
    np.loadtxt("Observationsdata/data9/Prl_Gen_C_smoother_logarithmic_v"+v+labels[0]+".txt",dtype=float),(samples,2,B,len(eLes),int(T/d))) 
pf_means=np.reshape(\
    np.loadtxt("Observationsdata/data9/Prl_Gen_C_smoother_logarithmic_pf_means_v"+v+labels[0]+".txt",dtype=float),(samples,2,B,len(eLes),int(T/d))) 

for i in range(len(labels[1:])):
    mcmcs=np.concatenate((mcmcs,np.reshape(\
    np.loadtxt("Observationsdata/data9/Prl_Gen_C_smoother_logarithmic_v"+v+labels[i+1]+".txt",dtype=float),(samples,2,B,len(eLes),int(T/d))) ),axis=0)  
    pf_means=np.concatenate((pf_means,np.reshape(\
    np.loadtxt("Observationsdata/data9/Prl_Gen_C_smoother_logarithmic_pf_means_v"+v+labels[i+1]+".txt",dtype=float),(samples,2,B,len(eLes),int(T/d))) ),axis=0)  
#%%
l0=10
Lmax=11
eLes=np.array(range(l0,Lmax+1))

samples=40
T=len(obs_times)*d
v="2ip_iv"
labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])
mcmcs2=np.reshape(\
    np.loadtxt("Observationsdata/data6/Prl_Gen_C_smoother_logarithmic_v"+v+labels[0]+".txt",dtype=float),(samples,2,len(eLes),int(T/d))) 

for i in range(len(labels[1:])):
    mcmcs2=np.concatenate((mcmcs2,np.reshape(\
    np.loadtxt("Observationsdata/data6/Prl_Gen_C_smoother_logarithmic_v"+v+labels[i+1]+".txt",dtype=float),(samples,2,len(eLes),int(T/d))) ),axis=0)  
#%%
mcmcs=np.concatenate((mcmcs,mcmcs2),axis=2)
print(mcmcs.shape)  
l0=4
Lmax=11
eLes=np.array(range(l0,Lmax+1))
#%%
v="2ip_ii"
labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])

for i in range(len(labels)):
    mcmcs=np.concatenate((mcmcs,np.reshape(\
    np.loadtxt("Observationsdata/data6/Prl_Gen_C_smoother_logarithmic_v"+v+labels[i]+".txt",dtype=float),(samples,2,len(eLes),int(T/d))) ),axis=0)  


v="2ip_iii"
for i in range(len(labels)):
    mcmcs=np.concatenate((mcmcs,np.reshape(\
    np.loadtxt("Observationsdata/data6/Prl_Gen_C_smoother_logarithmic_v"+v+labels[i]+".txt",dtype=float),(samples,2,len(eLes),int(T/d))) ),axis=0)  

print(mcmcs.shape)    

#mcmcs=np.reshape(np.loadtxt("Observations&data/Prl_Gen_C_smoother_logarithmic_v"+v+".txt",dtype=float),(samples,2,len(eLes),int(T/d)))
#Grads=np.reshape(np.loadtxt("Observations&data/Prl_SGD_bridge_Grads_vtest6.txt",dtype=float),(samples,B,3))

#%%
# For the samples obtained in ibex
# (samples,2,len(eLes),int(T/d))
#%%
# (samples,2,B,len(eLes),int(T/d))
t=-2
lev=1
print(pf_means.shape)
#print(pf_means[:2,0,0,:2])
#print(pf_means[:2,1,0,:2])
pf_mean=np.mean(pf_means,axis=(0,2))
pf_var=np.var(pf_means,axis=(0,2))
print(pf_mean[lev,:,t])
print(pf_var[lev,:,t])
pf_bias=np.abs(pf_mean-theta*alpha)
pf_bias_ub=pf_bias+np.sqrt(pf_var)*1.96/np.sqrt(samples*B)
pf_bias_lb=pf_bias-np.sqrt(pf_var)*1.96/np.sqrt(samples*B)
plt.plot(eLes,pf_bias[lev,:,t],label="PF Bias") 
plt.plot(eLes,pf_bias_ub[lev,:,t],label="PF Bias UB")
plt.plot(eLes,pf_bias_lb[lev,:,t],label="PF Bias LB")
plt.plot(eLes,pf_bias[lev,-1,t]*np.sqrt(eLes[-1])/np.sqrt(eLes),label="$1/N^{1/2}$")
plt.xscale("log")
plt.legend()
plt.yscale("log")

#%%
#%%
# (samples,2,B,len(eLes),int(T/d))
t=-1
lev=0
mean=np.mean(mcmcs,axis=(0,2))[lev]
#print(mcmcs[:2,0,0,:3])
#print(mcmcs[:2,1,0,:3])

print(alpha*theta)
print(mean[:,t])
var=np.var(mcmcs,axis=(0,2))[lev] #len(eLes),int(T/d)
print(var[:,t])
bias=np.abs(mean-theta*alpha)   
bias_ub=bias+np.sqrt(var)*1.96/np.sqrt(mcmcs.shape[0])
bias_lb=bias-np.sqrt(var)*1.96/np.sqrt(mcmcs.shape[0])
print(eLes.shape,bias.shape)
#plt.plot(eLes,bias[0,t]*2**eLes[0]/2**eLes,label="$\Delta_l$")
plt.plot(eLes,bias[:,t],label="Bias")
plt.plot(eLes,bias_ub[:,t],label="Bias UB")
plt.plot(eLes,bias_lb[:,t],label="Bias LB")
plt.yscale("log")
plt.xscale("log")   
plt.legend()
#%%
# For the coupling
# (samples,2,B,len(eLes),int(T/d))
t=-3
print(mcmcs[1,0,:30,-1])
print(mcmcs[1,1,:30,-1])
#%%
a=110
b=115
sm=np.mean(np.mean((mcmcs[:,0,a:b]-mcmcs[:,1,a:b])**2,axis=1),axis=0)
var_sm=np.var(np.mean((mcmcs[:,0,a:b]-mcmcs[:,1,a:b])**2,axis=1),axis=0)
sm_ub=sm+np.sqrt(var_sm)*1.96/np.sqrt(mcmcs.shape[0])
sm_lb=sm-np.sqrt(var_sm)*1.96/np.sqrt(mcmcs.shape[0])
plt.plot(eLes,sm[:,t],label="sm")
plt.plot(eLes,sm_ub[:,t],label="sm UB")
plt.plot(eLes,sm_lb[:,t],label="sm LB")
plt.plot(eLes,sm[-1,t]*2**(eLes[-1]/2)/2**(eLes/2),label="$\Delta_l^{1/2}$")
plt.plot(eLes,sm[0,t]*2**(eLes[0])/2**(eLes),label="$\Delta_l$")
plt.yscale("log")
#plt.xscale("log")
plt.legend()
#%%

#%%
# For the samples obtained in bridge_test.py
# (2,len(eLes),samples,2,int(T/d))
# (samples,2,len(eLes),int(T/d))
t=-1
lev=1
mean=np.mean(mcmcs,axis=(0))[lev,:]
print(mean[:,t])
var=np.var(mcmcs,axis=(0))[lev,:]
bias=np.abs(mean-theta*alpha)   
bias_ub=bias+np.sqrt(var)*1.96/np.sqrt(samples)
bias_lb=bias-np.sqrt(var)*1.96/np.sqrt(samples)
plt.plot(eLes,bias[:,t],label="Bias")
plt.plot(eLes,bias_ub[:,t],label="Bias UB")
plt.plot(eLes,bias_lb[:,t],label="Bias LB")
plt.yscale("log")
plt.legend()
# %%
# In the following we test the function Gen_Grad_Cond_PF_bridge, we do it considering noninformative 
# observaitons, i.e. the observation likelihood does not dependn on the state.
if True:
    #N=50
    the1=2
    the2=1
    the3=0.5
    the4=1
    the1,the2,the3,the4=2.397, 4.429e-3, 0.84, 17.36
    dist_params=[the1,the2,the3]
    fd=1e-10
    the1_fd=the1+fd
    the2_fd=the2+fd
    the3_fd=the3+fd

    dist_params_fd=np.array([[the1_fd,the2,the3],[the1,the2_fd,the3],[the1,the2,the3_fd]])
    w=the3**2/2+the1
    xi=the2
    sigma=the3
    alpha=2*w/sigma**2-1
    theta=sigma**2/(2*xi)
    
    rej_dist_params=[the1,the2,the3,the1,the2,the3]
    in_dist_pars=dist_params
    dist_params_0=dist_params
    dist_params_1=dist_params
    coup_in_dist_pars=[the1,the2,the3,the1,the2,the3]
    #t0=1.2
    #T=3.2
    A_til=dist_params
    fi_til=the3
    r_pars=1
    resamp_coef=1
    l=2
    d=1
    H_pars=1
    seed=6
    rkdata= np.loadtxt("Kangaroo_data.txt")
    obs=rkdata[:3,:2]
    obs_times=rkdata[:3,2]
    #obs=np.array([1.4,2,3,4.7,5.3,6.5])
    #obs_times=np.array([1.2,2,3.5,4.7,5.3,6.5])/3
    start=time.time()
    samples=40
    print("mean is: ",alpha*theta)
    #the4=1

print(dist_params_fd)
print(dist_params_fd[0,0],dist_params_fd[1,0])
# %%
def Grad_log_g_nonin(x, y, the4):
    return 0
start=time.time()
N=50
B=1000
samples=2
d=1
T=len(obs_times)*d
Lmax=6
l0=4
eLes=np.array(range(l0,Lmax+1))
seed=4232868
mcmc_mean=np.zeros((len(eLes),samples,2,int(T/d)))
resamp_coef=1
Gradss=np.zeros((len(eLes),samples,4))
for j in range(len(eLes)):
    l=eLes[j]
    print("l is ",l)
    for i in range(samples):
        seed+=1
        np.random.seed(i)

        
        #print("Seed feeded to PF_bridge is: ",seed)
        [log_weights,int_Gs,x_pr]=bdg.Gen_PF_bridge(bdg.gamma_sampling,in_dist_pars, bdg.b_log,dist_params,\
        bdg.Sig_gbm_1d,the3,bdg.b_log_aux,A_til,bdg.Sig_aux_gbm_1d,fi_til,bdg.r_log_normal,\
        r_pars,bdg.H_log_normal,H_pars,bdg.update_log_functs,\
        bdg.sampling_CIR,dist_params,obs,obs_times,bdg.log_g_den_nonin,\
        the4, bdg.trans_log_normal,1, bdg.trans_noncentral_chisquare, resamp_coef, l, d,N,seed)
        #print(log_weights)
        #x_mean=np.sum(pff.norm_logweights(log_weights,ax=1)*x_pr,axis=-1)
        weights=pff.norm_logweights(log_weights[-1])
        #print(weights.shape)
        index=np.random.choice(np.array(range(N)))
        cond_path=x_pr[:,index]
        cond_log_weights=log_weights[:,index]
        clwn=np.copy(cond_log_weights)
        cond_int_G=int_Gs[:,index]
        cign=np.copy(cond_int_G)
        seeds_cond=np.zeros((int(T/d)-1,2),dtype=int)
        seeds_cond[:,0]=seed+np.array(range(int(T/d)-1))*int(int(2**l-1))
        seeds_cond[:,1]=index*np.ones(int(T/d)-1)
        scn=np.copy(seeds_cond)
        ch_paths=np.zeros((B,int(T/d)))
        ch_weights=np.zeros((B,int(T/d)))
        ch_whole_paths=np.zeros((B,int(T/d)))
        ch_whole_weights=np.zeros((B,int(T/d)))
        cond_whole_path=cond_path
        cwpn=cond_path
        cond_whole_log_weights=cond_log_weights
        Grad_sum=0



        for b in range(B):
            print("sample iteration: ",i," chain iteration: ",b)
            seed+=int((int(T/d))*int(int(2**l-1)))
            np.random.seed(seed)
            
            #[log_weights,x_pr, cond_log_weights,cond_int_G,cond_path,seeds_cond,Grads]

            #(in_dist,in_dist_pars,Grad_log_in_dist,\
            #x_cond,seeds_cond,b,A,A_fd,Sig,fi,fi_fd,b_til,Sig_til,r,\
            #H,update_func,sample_funct,sample_pars,obs,obs_times,log_g_den,g_den_par,\

            #aux_trans_den,Grad_log_aux_trans,prop_trans_den, Grad_log_G,resamp_coef, l, d,N,seed,\
            #fd_rate)
      
            [log_weights,x_pr,cond_log_weights,int_Gs_cond,cond_path,seeds_cond,Grads]=\
            bdg.Gen_Grad_Cond_PF_bridge_back_samp(bdg.gamma_sampling,in_dist_pars, bdg.Grad_log_gamma_in_dist, \
            cond_path,seeds_cond,bdg.b_log,dist_params,dist_params_fd,\
            bdg.Sig_gbm_1d,the3,the3_fd,bdg.b_log_aux,bdg.Sig_aux_gbm_1d,bdg.r_log_normal,\
            bdg.H_log_normal, bdg.update_log_functs,\
            bdg.sampling_prop_log_normal,[the1,the2,the3],obs,obs_times,bdg.log_g_den_nonin,\
            1, bdg.trans_log_normal, bdg.Grad_trans_log_normal,bdg.trans_prop_log_normal,\
            bdg.Grad_log_g_nonin,resamp_coef, l, d,N,seed,fd)
            Grad_sum+=Grads
            ch_paths[b]=cond_path
            ch_weights[b]=cond_log_weights
            #print("seed conditionals are:",seeds_cond)
        Gradss[j,i]=Grad_sum/B
        mcmc_mean[j,i,0]=np.mean(ch_paths,axis=0)
        mcmc_mean[j,i,1]=np.mean(ch_whole_paths,axis=0)
end=time.time()
print(end-start)
    #%%
print(Gradss)
mean=np.mean(Gradss,axis=(1))
var=np.var(Gradss,axis=(1))
mean_ub=mean+np.sqrt(var)*1.96/np.sqrt(samples)
mean_lb=mean-np.sqrt(var)*1.96/np.sqrt(samples)
print(mean_lb)
print(mean)
print(mean_ub)
#%%
#  (len(eLes),samples,2,int(T/d))
t=-3
mean=np.mean(mcmc_mean[:,:,0],axis=(1))
print(theta*alpha)  
print(mean[:,t])    
var=np.var(mcmc_mean[:,:,0],axis=(1))
bias=np.abs(mean-theta*alpha)   
bias_ub=bias+np.sqrt(var)*1.96/np.sqrt(samples)
bias_lb=bias-np.sqrt(var)*1.96/np.sqrt(samples)
plt.plot(eLes,bias[:,t],label="Bias")
plt.plot(eLes,bias_ub[:,t],label="Bias UB")
plt.plot(eLes,bias_lb[:,t],label="Bias LB")
plt.yscale("log")
plt.legend()

#%%
#%%
# save point
if True:


    samples=40
    B=2000*7
    rkdata= np.loadtxt("Kangaroo_data.txt")
    obs=rkdata[:3,:2]
    obs_times=rkdata[:3,2]
    #obs=np.array([1.4,2,3,4.7,5.3,6.5])
    #obs_times=np.array([1.2,2,3.5,4.7,5.3,6.5])/3
    inputs=[]
    start=time.time()
    l0=3
    Lmax=9
    #arg_cm=int(sys.argv[1])
    #arg_cm=32
    d=1
    T=len(obs_times)*d
    eLes=np.array(range(l0,Lmax+1))
    
    the1,the2,the3,the4=2.397, 4.429e-3, 0.84, 17.36
    dist_params=[the1,the2,the3]
    in_dist_pars=dist_params
    fd=1e-10
    the1_fd=the1+fd
    the2_fd=the2+fd
    the3_fd=the3+fd
    dist_params_fd=np.array([[the1_fd,the2,the3],[the1,the2_fd,the3],[the1,the2,the3_fd]])
    #seed=1+17*samples*len(eLes)*(arg_cm-1)+  17*samples*len(eLes)*30*(0)
#seed=1+17*samples*len(eLes)*(arg_cm-1)+  17*samples*len(eLes)*30*(0)
#%%
v="GS_ip_xi"#+str(arg_cm)
mcmcs=np.reshape(np.loadtxt("Observationsdata/Prl_Grad_smooth_mcmc_v"+v+".txt",dtype=float),(samples,len(eLes),int(T/d)))
Grads=np.reshape(np.loadtxt("Observationsdata/Prl_Grad_smooth_Grads_v"+v+".txt",dtype=float),(samples,len(eLes),4))
#%%

v="GGS2_4_ip_i"
labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])
print(labels.shape)
mcmcs=np.reshape(\
    np.loadtxt("Observationsdata/data11/Plr_Gen_Grad_smooth_2_mcmc_v"+v+labels[0]+".txt",dtype=float),(samples,len(eLes),int(T/d))) 
Grads=np.reshape(np.loadtxt("Observationsdata/data11/Plr_Gen_Grad_smooth_2_Grads_v"+v+labels[0]+".txt",dtype=float),(samples,len(eLes),4))
for i in range(len(labels[1:])):
    mcmcs=np.concatenate((mcmcs,np.reshape(\
    np.loadtxt("Observationsdata/data11/Plr_Gen_Grad_smooth_2_mcmc_v"+v+labels[i+1]+".txt",dtype=float),(samples,len(eLes),int(T/d))) ),axis=0)  
    Grads=np.concatenate((Grads,np.reshape(\
    np.loadtxt("Observationsdata/data11/Plr_Gen_Grad_smooth_2_Grads_v"+v+labels[i+1]+".txt",dtype=float),(samples,len(eLes),4))) ,axis=0)
#%%

v="GS_ip_xi"
labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])
print(labels.shape)
mcmcs=np.reshape(\
    np.loadtxt("Observationsdata/data10/Prl_Grad_smooth_mcmc_v"+v+labels[0]+".txt",dtype=float),(samples,len(eLes),int(T/d))) 
Grads=np.reshape(np.loadtxt("Observationsdata/data10/Prl_Grad_smooth_Grads_v"+v+labels[0]+".txt",dtype=float),(samples,len(eLes),4))
for i in range(len(labels[1:])):
    mcmcs=np.concatenate((mcmcs,np.reshape(\
    np.loadtxt("Observationsdata/data10/Prl_Grad_smooth_mcmc_v"+v+labels[i+1]+".txt",dtype=float),(samples,len(eLes),int(T/d))) ),axis=0)  
    Grads=np.concatenate((Grads,np.reshape(\
    np.loadtxt("Observationsdata/data10/Prl_Grad_smooth_Grads_v"+v+labels[i+1]+".txt",dtype=float),(samples,len(eLes),4))) ,axis=0)

#%%
par=2
print(Grads[:,2,par])
mask=np.isnan(Grads[:,2,par])
indices=np.where(mask==True)
print(indices)
#print(Grads[781,2,:])
#print(780/30)
Grads_mean=np.abs(np.mean(Grads,axis=0))
Grads_var=np.var(Grads,axis=0)
print(Grads_mean)
Grads_mean_ub=Grads_mean+np.sqrt(Grads_var)*1.96/np.sqrt(samples)
Grads_mean_lb=Grads_mean-np.sqrt(Grads_var)*1.96/np.sqrt(samples)
plt.plot(eLes,Grads_mean[:,par],label="Grad")
plt.plot(eLes,Grads_mean_ub[:,par],label="Grad UB")
plt.plot(eLes,Grads_mean_lb[:,par],label="Grad LB")
plt.plot(eLes,Grads_mean[-1,par]*2**(eLes[-1])/2**(eLes),label="$\Delta_l$")
plt.yscale("log")
plt.legend()
plt.show()
# could this be related to the change of the log_weight function? 
#%%
t=-3
lev=1
mean=np.mean(mcmcs,axis=(0))
print(mean[:,t])
var=np.var(mcmcs,axis=(0))
bias=np.abs(mean-theta*alpha)   
bias_ub=bias+np.sqrt(var)*1.96/np.sqrt(samples)
bias_lb=bias-np.sqrt(var)*1.96/np.sqrt(samples)
plt.plot(eLes,bias[:,t],label="Bias")
plt.plot(eLes,bias_ub[:,t],label="Bias UB")
plt.plot(eLes,bias_lb[:,t],label="Bias LB")
plt.plot(eLes,bias[0,t]*2**(eLes[0])/2**(eLes),label="$\Delta_l$")
plt.yscale("log")
plt.legend()

#%%
#%%

# TEST FOR THE GENERAL COUPLED GRADEINT OF THE CONDITIONAL PARTICLE FILETER WITH BACKWARD SAMPLING
if True:
    N=50
    the1=2
    the2=1
    the3=0.5
    the4=1
    fd=1e-6
    the1,the2,the3,the4=2.397, 4.429e-1, 0.84, 17.36
    the3_fd=the3+fd
    the3_fd_1=the3+fd/2
    w=the3**2/2+the1
    xi=the2
    sigma=the3
    alpha=2*w/sigma**2-1
    theta=sigma**2/(2*xi)
    dist_params=np.array([the1,the2,the3])
    dist_params_fd=np.array([[the1+fd,the2,the3],[the1,the2+fd,the3],[the1,the2,the3+fd]])
    dist_params_fd_1=np.array([[the1+fd/2,the2,the3],[the1,the2+fd/2,the3],[the1,the2,the3+fd/2]])
    rej_dist_params=[the1,the2,the3,the1,the2,the3]
    in_dist_pars=dist_params
    dist_params_0=dist_params
    dist_params_fd_0=dist_params_fd
    dist_params_1=dist_params
    
    coup_in_dist_pars=[the1,the2,the3,the1,the2,the3]
    #t0=1.2
    #T=3.2
    A_til=dist_params
    fi_til=the3
    r_pars=1
    resamp_coef=1
    l=2
    d=1
    H_pars=1
    seed=6
    #rkdata= np.loadtxt("Kangaroo_data.txt")
    #obs=rkdata[:,:2]
    #obs_times=rkdata[:,2]
    obs=np.array([1.4,2,3,4.7,5.3,6.5])
    obs_times=np.array([1.2,2,3.5,4.7,5.3,6.5])/3
    start=time.time()
    samples=40
    print("mean is: ",alpha*theta)



#"""
#%%
#"""
start=time.time()
B=40
samples=10
d=1
T=len(obs_times)*d
l0=3
Lmax=3
eLes=np.array(range(l0,Lmax+1))
seed=436
print(seed)
mcmc_mean=np.zeros((2,len(eLes),samples,2,int(T/d)))
Grads_mean=np.zeros((2,len(eLes),samples,4))
resamp_coef=1
for j in range(len(eLes)):
    l=eLes[j]
    print("l is ",l)
    for i in range(samples):
        seed+=1
        np.random.seed(i)
        #print("Seed feeded to PF_bridge is: ",seed)
        [log_weights,int_Gs,x_pr]=bdg.Gen_PF_bridge(bdg.gamma_sampling,in_dist_pars, bdg.b_log,dist_params,\
        bdg.Sig_gbm_1d,the3,bdg.b_log_aux,A_til,bdg.Sig_aux_gbm_1d,fi_til,bdg.r_log_normal,\
        r_pars,bdg.H_log_normal,H_pars,bdg.update_log_functs,\
        bdg.sampling_CIR,dist_params,obs,obs_times,bdg.log_g_den_nonin,\
        the4, bdg.trans_log_normal,1, bdg.trans_noncentral_chisquare, resamp_coef, l, d,N,seed)
        #print(log_weights)
        #x_mean=np.sum(pff.norm_logweights(log_weights,ax=1)*x_pr,axis=-1)
        weights=pff.norm_logweights(log_weights[-1])
        #print(weights.shape)
        index=np.random.choice(np.array(range(N)))
        cond_path_0=x_pr[:,index]
        cond_path_1=np.copy(cond_path_0)
        cond_log_weights=log_weights[:,index]
        clwn=np.copy(cond_log_weights)
        cond_int_G=int_Gs[:,index]
        cign=np.copy(cond_int_G)
        seeds_cond=np.zeros((int(T/d)-1,2),dtype=int)
        seeds_cond[:,0]=seed+np.array(range(int(T/d)-1))*int(int(2**l-1))
        seeds_cond[:,1]=index*np.ones(int(T/d)-1)
        seeds_cond_0=np.copy(seeds_cond)
        seeds_cond_1=np.copy(seeds_cond)
        scn=np.copy(seeds_cond)
        ch_paths=np.zeros((2,B,int(T/d)))
        Grads=np.zeros((2,B,4))
        ch_weights=np.zeros((2,B,int(T/d)))
        ch_whole_paths=np.zeros((B,int(T/d)))
        ch_whole_weights=np.zeros((B,int(T/d)))

        for b in range(B):
            print("sample iteration: ",i," chain iteration: ",b)
            seed+=int((int(T/d))*int(int(2**l-1)))
            np.random.seed(seed)
            print(seed)
            [log_weights_0,log_weights_1,x_pr_0,x_pr_1,cond_log_weights_0,cond_log_weights_1,\
            int_Gs_cond_0,int_Gs_cond_1,cond_path_0,cond_path_1,seeds_cond_0,seeds_cond_1,\
            Grads_0,Grads_1]=\
            bdg.Gen_C_Grad_Cond_PF_bridge_back_samp(bdg.rej_max_coup_gamma_in_dist,\
            coup_in_dist_pars,bdg.Grad_log_gamma_in_dist,cond_path_0,cond_path_1 ,seeds_cond_0,\
            seeds_cond_1,bdg.b_log,dist_params_0, dist_params_1,dist_params_fd_0,\
            dist_params_fd_1,bdg.Sig_gbm_1d,the3,the3,the3_fd,the3_fd_1,bdg.b_log_aux,bdg.Sig_aux_gbm_1d,\
            bdg.r_log_normal,bdg.H_log_normal,bdg.update_log_functs,\
            bdg.rej_max_coup_log_normal,[the1,the2,the3,the1,the2,the3],\
            obs,obs_times,bdg.log_g_den_nonin,the4,the4,bdg.trans_log_normal,\
            bdg.Grad_trans_log_normal, bdg.trans_prop_log_normal,[the1,the2,the3],[the1,the2,the3],\
            bdg.Grad_log_G, resamp_coef, l, d,N,seed,fd)
            #[log_weights,x_pr,cond_log_weights,int_Gs_cond,cond_path,seeds_cond]=\
            #bdg.Gen_Cond_PF_bridge_back_samp(bdg.gamma_sampling,in_dist_pars, cond_path,seeds_cond,\
            #bdg.b_log,dist_params,\
            #bdg.Sig_gbm_1d,the3,bdg.b_log_aux,A_til,bdg.Sig_aux_gbm_1d,fi_til,bdg.r_log_normal,\
            #r_pars,bdg.H_log_normal,H_pars,\
            #bdg.update_log_functs,\
            #bdg.sampling_prop_log_normal,[the1,the2,the3],obs,obs_times,bdg.log_g_nbino_den,\
            #the4, bdg.trans_log_normal,1, bdg.trans_prop_log_normal, resamp_coef, l, d,N,seed)
            Grads[:,b]=np.array([Grads_0,Grads_1])
            ch_paths[:,b]=np.array([cond_path_0,cond_path_1])
            ch_weights[:,b]=np.array([cond_log_weights_0,cond_log_weights_1])
                   
            #print("seed conditionals are:",seeds_cond)
        mcmc_mean[:,j,i,0]=np.mean(ch_paths,axis=1)
        mcmc_mean[:,j,i,1]=np.mean(ch_weights,axis=1)
        Grads_mean[:,j,i]=np.mean(Grads,axis=1)
end=time.time()
# %%
#  (2,len(eLes),samples,4)
lev=0
Grads_means=np.abs(np.mean(Grads_mean,axis=(2)))
Grads_var=np.var(Grads_mean,axis=(2))
Grads_means_ub=Grads_means+np.sqrt(Grads_var)*1.96/np.sqrt(samples)
Grads_means_lb=Grads_means-np.sqrt(Grads_var)*1.96/np.sqrt(samples) 
#plt.plot(eLes,Grads_means[lev,:,0],label="Grad")
#plt.plot(eLes,Grads_means_ub[lev,:,0],label="Grad UB")
#plt.plot(eLes,Grads_means_lb[lev,:,0],label="Grad LB")
#plt.yscale("log")   
print(Grads_means)
print(Grads_means_lb,Grads_means_ub)    
#%%
    
if True:
    N=50
    samples=40
    B=500
    rkdata= np.loadtxt("Kangaroo_data.txt")
    obs_or=rkdata[:3,:2]
    obs_times=rkdata[:3,2]
    the1,the2,the3,the4=2.397, 4.429e-3, 0.84, 17.36
    dist_params=[the1,the2,the3]

    #obs=np.array([1.4,2,3,4.7,5.3,6.5])
    #obs_times=np.array([1.2,2,3.5,4.7,5.3,6.5])/3
    inputs=[]
    fd=1e-10
    the3_fd=the3+fd 

    dist_params=np.array([the1,the2,the3])
    dist_params_fd=np.array([[the1+fd,the2,the3],[the1,the2+fd,the3],[the1,the2,the3+fd]])
    dist_params_fd_1=np.array([[the1+fd/2,the2,the3],[the1,the2+fd/2,the3],[the1,the2,the3+fd/2]])
    rej_dist_params=[the1,the2,the3,the1,the2,the3]
    in_dist_pars=dist_params
    dist_params_0=dist_params
    dist_params_fd_0=dist_params_fd
    dist_params_1=dist_params
    
    coup_in_dist_pars=[the1,the2,the3,the1,the2,the3]
    #t0=1.2
    #T=3.2
    A_til=dist_params
    fi_til=the3
    r_pars=1
    resamp_coef=1
    d=1
    H_pars=1
    

    start=time.time()
    l0=4
    Lmax=8
    #arg_cm=int(sys.argv[1])
    #arg_cm=32
    d=1
    T=len(obs_times)*d
    eLes=np.array(range(l0,Lmax+1))

#%%
v="GGCS_ip_ii"
labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])
print(labels.shape)
mcmcs=np.reshape(\
    np.loadtxt("Observationsdata/data12/Prl_Gen_C_Grad_logarithmic_mcmc_v"+v+labels[0]+".txt",dtype=float),(samples,len(eLes),2,B,int(T/d))) 
Grads=np.reshape(np.loadtxt("Observationsdata/data12/Prl_Gen_C_Grad_logarithmic_Grads_v"+v+labels[0]+".txt",dtype=float),(samples,len(eLes),2,B,4))
for i in range(len(labels[1:])):
    mcmcs=np.concatenate((mcmcs,np.reshape(\
    np.loadtxt("Observationsdata/data12/Prl_Gen_C_Grad_logarithmic_mcmc_v"+v+labels[i+1]+".txt",dtype=float),(samples,len(eLes),2,B,int(T/d))) ),axis=0)  
    Grads=np.concatenate((Grads,np.reshape(\
    np.loadtxt("Observationsdata/data12/Prl_Gen_C_Grad_logarithmic_Grads_v"+v+labels[i+1]+".txt",dtype=float),(samples,len(eLes),2,B,4))) ,axis=0)



# %%

# (samples,len(eLes),2,B,4)
par=0
lev=0
Grads_mean=np.abs(np.mean(Grads,axis=(0,3)))[:,lev]
Grads_var=np.var(Grads,axis=(0,3))[:,lev]
print(Grads_mean)
Grads_mean_ub=Grads_mean+np.sqrt(Grads_var)*1.96/np.sqrt(samples*B)
Grads_mean_lb=Grads_mean-np.sqrt(Grads_var)*1.96/np.sqrt(samples*B)
plt.plot(eLes,Grads_mean[:,par],label="Grad")
plt.plot(eLes,Grads_mean_ub[:,par],label="Grad UB")
plt.plot(eLes,Grads_mean_lb[:,par],label="Grad LB")
plt.plot(eLes,Grads_mean[-1,par]*np.sqrt(2**(eLes[-1]))/np.sqrt(2**(eLes)),label="$\Delta_l$")
plt.yscale("log")
plt.legend()
plt.show()
#%%

if True:
    
    fd=1e-10
    rkdata= np.loadtxt("Kangaroo_data.txt")
    obs=rkdata[:,:2]
    #obs_or=rkdata[:,:2]
    #obs=np.zeros((3,2))
    #obs[0,:]=obs_or[0,:]
    #obs[1,:]=obs_or[-2,:]
    #obs[2,:]=obs_or[-1,:]
    obs_times=rkdata[:,2]
    #obs_times_or=rkdata[:,2]
    #obs_times=np.zeros((3))
    #obs_times[0]=obs_times_or[0]
    #obs_times[1]=obs_times_or[-2]
    #obs_times[2]=obs_times_or[-1]+30
    print("obs times are: ",obs_times)
    #obs=np.array([1.4,2,3,4.7,5.3,6.5])
    deltas=bdg.get_deltas(obs_times)
    print("deltas are: ",deltas)
    smean=np.mean(obs)  
    svar=np.var(obs)
    the1=2
    the2=the1/smean
    the3=1
    #the4=np.sqrt(smean+the3**2/(2*the2))/np.sqrt(svar/smean-1-the3**2/(2*the2))    
    the4=10
    the1,the2,the3,the4=2.397, 4.429e-3, 0.84, 17.36
    
    #the1,the2,the3,the4 =1.47761682e+00 ,2.77612500e-03, 6.75328450e-01, 1.86235184e+01
    the3_fd_1=the3+fd/2
    w=the3**2/2+the1
    xi=the2
    sigma=the3
    alpha=2*w/sigma**2-1
    theta=sigma**2/(2*xi)
    dist_params=np.array([the1,the2,the3])
    dist_params_fd=np.array([[the1+fd,the2,the3],[the1,the2+fd,the3],[the1,the2,the3+fd]])
    dist_params_fd_1=np.array([[the1+fd/2,the2,the3],[the1,the2+fd/2,the3],[the1,the2,the3+fd/2]])
    rej_dist_params=[the1,the2,the3,the1,the2,the3]
    in_dist_pars=dist_params
    dist_params_0=dist_params
    dist_params_fd_0=dist_params_fd
    dist_params_1=dist_params
    coup_in_dist_pars=[the1,the2,the3,the1,the2,the3]
    #t0=1.2
    #T=3.2
    A_til=dist_params
    fi_til=the3
    r_pars=1
    resamp_coef=1
    d=1
    H_pars=1 
#%%

#%%
l=3
N=50
mcmc_links=1
SGD_steps=100
gamma=0.000
gammas=[2,3,0.6,10]
alpha=0.5
ast=0
K=2*(2**ast-1)  
start=time.time()
seed=382
#the1,the2,the3,the4=2.397, 4.429e-3, 0.84, 17.36
tests=20
B=SGD_steps*mcmc_links
ch_pathss=np.zeros((tests,SGD_steps+K,int(len(obs_times)/d)))
for i in range(tests):
            print("step is: ",i)
            seed+=1
            ch_paths,pars =bdg.Gen_SGD_bridge(bdg.gamma_sampling,in_dist_pars, bdg.Grad_log_gamma_in_dist, \
            bdg.b_log,dist_params, bdg.Sig_gbm_1d,the3,bdg.b_log_aux,bdg.Sig_aux_gbm_1d,bdg.r_log_normal,\
            bdg.H_log_normal, bdg.update_log_functs,\
            bdg.sampling_prop_log_normal_2,[the1,the2,the3],obs,obs_times,bdg.log_g_nbino_den,\
            the4, bdg.trans_log_normal, bdg.Grad_trans_log_normal,bdg.trans_prop_log_normal_2,\
            bdg.Grad_log_g_nbino,resamp_coef, l, d,N,seed,fd,mcmc_links,SGD_steps,gamma,gammas, alpha,ast, bdg.update_pars_logis)
            ch_pathss[i]=ch_paths

end=time.time()
print("Time for SGD is: ",end-start)
#%%
iterCount=1000
N_0=399
x_matlab=np.reshape(np.loadtxt("Observationsdata/X_sm_2.txt",dtype=float,delimiter=','),(iterCount,N_0 ,len(obs_times)))
#%%
x_m_est=np.mean(x_matlab,axis=1)
x_m_var=np.var(x_m_est,axis=0)
x_m_sing_var=np.var(x_matlab,axis=(0,1))
#%%
samples=40
SGD_steps=400
K=0
v="GSb_comparison_i"
labels=np.array(["1", "2", "3" ,"4","5","6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15","16","17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])
ch_pathss=np.reshape(\
    np.loadtxt("Observationsdata/data20/Prl_Gen_SGD_bridge_ch_paths_v"+v+labels[0]+".txt",dtype=float),(samples,SGD_steps+K,int(len(obs_times)/d))) 

for i in range(len(labels[1:])):
    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    ch_pathss=np.concatenate((ch_pathss,np.reshape(\
    np.loadtxt("Observationsdata/data20/Prl_Gen_SGD_bridge_ch_paths_v"+v+labels[i]+".txt",dtype=float),(samples,SGD_steps+K,int(len(obs_times)/d))) \
    ),axis=0)  
   

#%%
ch_paths_mean=np.mean(ch_pathss,axis=1)
ch_paths_var=np.var(ch_paths_mean,axis=0)
ch_paths_sing_var=np.var(ch_pathss,axis=(0,1))
#plt.plot(obs_times,ch_paths_var,label="var path back")
#plt.plot(obs_times,x_m_var,label="var path matlab")
plt.plot(obs_times,ch_paths_sing_var,label="var path sing back")
plt.plot(obs_times,x_m_sing_var,label="var path sing matlab")
#plt.plot(obs_times,np.mean(ch_paths_mean,axis=0),label="mean path")
#plt.plot(obs_times,obs,label="Observations")
plt.xlabel("Time")
plt.ylabel("Var")
plt.title("Variance of estimation of the mean of smoothing distribution")
plt.legend()
#%%


 #%%
# the1,the2,the3,the4=2.397, 4.429e-3, 0.84, 17.36
par_1=0
par_2=1
k=0
labels=["$\\theta_1$","$\\theta_2$","$\\theta_3$","$\\theta_4$"]
plt.plot(pars[k:,par_1],pars[k:,par_2],label="pars")
plt.xlabel(labels[par_1])
plt.ylabel(labels[par_2])
#%%
eNes=2**(np.array(range(0,int(np.log2(SGD_steps)-1))))
print(eNes)
par_1=3
diffs=np.abs(pars[eNes[1:]]-pars[eNes[:-1]])
plt.plot(eNes[1:],diffs[:,par_1],label="Diff")
plt.plot(eNes[1:],diffs[0,par_1]*np.sqrt((eNes[1]))/np.sqrt((eNes[1:])),label="$1/N^{1/2}$")
plt.xscale("log")
plt.yscale("log")
#%%


#%%
k=1
mean_paths=np.mean(ch_paths[-k:],axis=0)
plt.plot(obs_times,mean_paths,label="Mean path")
plt.plot(obs_times,obs,label="Observations")
#%%

# %%
#pars_s=pars
# %%
if True:
    samples=40
    start=time.time()
    N=200
    l0=13
    Lmax=13
    mcmc_links=1
    SGD_steps=500
    B=SGD_steps*mcmc_links
    gamma=0.005
    gammas=[2,3,0.6,6]
    alpha=0.5
    ast=2
    K=2*(2**ast-1)  
    #arg_cm=int(sys.argv[1])
    arg_cm=1
    d=1
    T=len(obs_times)*d
    eLes=np.array(range(l0,Lmax+1))
#%%
# This data is meant to be the true data (approximation with 
# a large level of time discretization and number of SGD steps)

v="GSb_True_hf4_ip_v"#+str(arg_cm)
pars=np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_SGD_bridge_pars_v"+v+".txt",dtype=float),(samples,len(eLes),SGD_steps+1+ast,4)) 

#ch_paths=np.reshape(\
#    np.loadtxt("Observationsdata/data16/Prl_Gen_SGD_bridge_ch_paths_v"+v+".txt",dtype=float),(samples,len(eLes),B+K,int(T/d))) 

#%%
SGD_steps=150
v="GSb_True_i"#+str(arg_cm)
pars3=np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_SGD_bridge_pars_v"+v+".txt",dtype=float),(samples,len(eLes),SGD_steps+1+ast,4)) 

ch_paths3=np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_SGD_bridge_ch_paths_v"+v+".txt",dtype=float),(samples,len(eLes),B+K,int(T/d))) 


#%%
v="GSb5_ip_ii"
labels=np.array(["32"])
labels=np.array(["1", "2", "3","4" ,"5", "6", "7", "8","9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20",\
    "21", "22","23",  "24", "25", "26", "27", "28", "29","30"])

print(labels.shape)
pars=np.reshape(\
    np.loadtxt("Observationsdata/data15/Prl_Gen_SGD_bridge_pars_v"+v+labels[0]+".txt",dtype=float),(samples,len(eLes),SGD_steps+1+ast,4)) 

#ch_paths=np.reshape(\
#    np.loadtxt("Observationsdata/data15/Prl_Gen_SGD_bridge_ch_paths_v"+v+labels[0]+".txt",dtype=float),(samples,len(eLes),B+K,int(T/d))) 
#%%

for i in range(len(labels[1:])):
    pars=np.concatenate((pars,np.reshape(\
    np.loadtxt("Observationsdata/data15/Prl_Gen_SGD_bridge_pars_v"+v+labels[i+1]+".txt",dtype=float),(samples,len(eLes),SGD_steps+1+ast,4)) ),axis=0)

    #ch_paths=np.concatenate((np.reshape(ch_paths,\
    #np.loadtxt("Observationsdata/Prl_Gen_SGD_bridge_ch_paths_v"+v+labels[i+1]+".txt",dtype=float),(samples,len(eLes),B+K,int(T/d)))),axis=0)



#%%
"""
labels=np.array(["1", "2", "3","4" ,"5", "6", "7", "8","9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20",\
    "21", "22","23",  "24", "25", "26", "27", "28", "29","30"])
"""

v="GSb8_ip_i"
labels=np.array(["1", "2", "3","4" ,"5", "6", "7", "8","9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20",\
    "21", "22","23",  "24", "25", "26", "27", "28", "29","30"])
print(labels.shape)
pars=np.reshape(\
    np.loadtxt("Observationsdata/data15/Prl_Gen_SGD_bridge_pars_v"+v+labels[0]+".txt",dtype=float),(samples,len(eLes),SGD_steps+1+ast,4)) 

#ch_paths=np.reshape(\
#    np.loadtxt("Observationsdata/data14/Prl_Gen_SGD_bridge_ch_paths_v"+v+labels[0]+".txt",dtype=float),(samples,len(eLes),B+K,int(T/d))) 

for i in range(len(labels[1:])):
    pars=np.concatenate((pars,np.reshape(\
    np.loadtxt("Observationsdata/data15/Prl_Gen_SGD_bridge_pars_v"+v+labels[i+1]+".txt",dtype=float),(samples,len(eLes),SGD_steps+1+ast,4)) ),axis=0)

    #ch_paths=np.concatenate((np.reshape(ch_paths,\
    #np.loadtxt("Observationsdata/data14/Prl_Gen_SGD_bridge_ch_paths_v"+v+labels[i+1]+".txt",dtype=float),(samples,len(eLes),B+K,int(T/d)))),axis=0)
    
#pars[:,:,:,2]=pars[:,:,:,2]**2/2
#%%
# 2.2889894e+00 4.2530250e-03 8.0981455e-01 1.9565340e+01
#2.01901483e+00 3.77027500e-03 7.88904675e-01 1.95046647e+01
# previous iteration
print(pars.shape)
labels=["$\\theta_1$","$\\theta_2$","$\\theta_3$","$\\theta_4$"]
# the1,the2,the3,the4=1.97707570e+00, 3.67902500e-03, 7.68023450e-01, 1.94123391e+01
par1=0
par2=1
ele=0
k=100
k0=k+5
k0=0
#k0=int(np.shape(pars)[0]/2)
k1=np.shape(pars)[0]
#k1=k+10
#k1=-1
s1=0
s2=5 #np.shape(pars)[2]    
print(pars.shape)
plt.plot(pars[k0:k1,ele,s1:s2,par1].T,pars[k0:k1,ele,s1:s2,par2].T,label="pars")  
#plt.plot(pars3[k0:k1,ele,s1:s2,par1].T,pars3[k0:k1,ele,s1:s2,par2].T,label="pars",c="red")  
s1=-3
s2=-1
plt.plot(pars[k0:k1,ele,s1:s2,par1].T,pars[k0:k1,ele,s1:s2,par2].T,lw=1,c="black")  
#plt.plot(pars3[k0:k1,ele,s1:s2,par1].T,pars3[k0:k1,ele,s1:s2,par2].T,lw=1,c="green")  
plt.xlabel(labels[par1])
plt.ylabel(labels[par2])
mean_pars=np.mean(pars[:,ele],axis=0)
print(mean_pars[-1]) 
# [1.73275005e+00 3.23095000e-03 7.10832200e-01 1.90700065e+01]
#%%
# Coupling
par1=3
c=0
s0=2**c
eNes=s0*2**(np.array(range(0,int(np.log2(SGD_steps)-c+1))))
sm=np.mean((pars[:,ele,eNes[:-1],par1]-pars[:,ele,eNes[1:],par1])**2,axis=0)
plt.plot(eNes[:-1],sm,label="Coupling")
plt.plot(eNes[:-1],sm[-1]*(eNes[-1]**(1/2))/((eNes[1:]**(1/2))),label="$1/N^{1/2}$")
plt.plot(eNes[:-1],sm[-1]*(eNes[-1])/((eNes[1:])),label="$1/N$")
plt.legend()
plt.yscale("log")
plt.xscale("log")
CP=3e-8
#%%
par1=1
print(np.var(pars[:,ele,1,par1],axis=0))
CP0=3.28e-7
#%%
# aquí
par1=3
c=0
s0=2**c
eNes=s0*2**(np.array(range(0,int(np.log2(SGD_steps)-c+1))))
vars=np.var(pars[:,ele,eNes,par1],axis=0)
plt.plot(eNes,vars,label="Var")
plt.plot(eNes,vars[-1]*(eNes[-1]**(1/2))/((eNes**(1/2))),label="$1/N^{1/2}$")
plt.plot(eNes,vars[-1]*(eNes[-1])/((eNes)),label="$1/N$")
plt.legend()
plt.yscale("log")
plt.xscale("log")
#%%
par1=1
eNes=2**(np.array(range(0,int(np.log2(SGD_steps))+1)))
bias=np.abs(np.mean(pars[:,ele,eNes[1:],par1]-pars[:,ele,eNes[:-1],par1],axis=0))
plt.plot(eNes[1:],bias,label="Bias")
plt.plot(eNes[1:],bias[-1]*(eNes[-1]**(1/2))/((eNes[1:]**(1/2))),label="$1/N^{1/2}$")
plt.plot(eNes[1:],bias[-1]*(eNes[-1])/((eNes[1:])),label="$1/N$")
plt.legend()
plt.yscale("log")
plt.xscale("log")
#%%
plt.plot(eNes, (np.log(eNes)-np.log(50/22)))
plt.xscale("log")

#%%
# In this cell we compute the line where the log-likelihood is maximized

# Take the last points of the pars
pars_last=pars[:,-1,-1,:]
# Get the coefficients of the linear regression
par1=0
par2=1
[b0,b1]=bdg.coef(pars_last[:,par1],pars_last[:,par2])
# Plot the line and compare with the last points of the pars
plt.plot(pars_last[:,par1],pars_last[:,par2],label="pars")
xs=np.arange(-1,8,1)
print(xs)
plt.plot(xs,b0+b1*xs,label="Regression line")
print(b1**(-1))
print("The coefficients of the linear regression are: ",b0,b1)
#%%
# CLT shows that the rate of the bias is 1/N^{alpha+1/2}(alpha in this case is equal to 1) for the SGD steps.
par1=0
c=0
s0=2**c
eNes=s0*2**(np.array(range(0,int(np.log2(SGD_steps)-c))))
diffs=np.mean(np.abs(pars[:,ele,eNes[1:],par1]-pars[:,ele,eNes[:-1],par1]),axis=0)
print(diffs.shape)
plt.plot(eNes[1:],diffs,label="Diff")
plt.plot(eNes[1:],diffs[-1]*(eNes[-1]**(1/2))/((eNes[1:]**(1/2))),label="$1/N^{1/2}$")
plt.plot(eNes[1:],diffs[-1]*(eNes[-1])/((eNes[1:])),label="$1/N$")
plt.legend()
plt.yscale("log")
plt.xscale("log")
print(np.mean((pars[:,ele,eNes[1:],:]-pars[:,ele,eNes[:-1],:])**2,axis=0)[0,:]**2)
#%%
par1=0
c=0
s0=2**c
eNes=s0*2**(np.array(range(0,int(np.log2(SGD_steps)-c))))
vars=np.var(pars[:,ele,eNes,par1],axis=0)
plt.plot(eNes,vars,label="Var")
plt.plot(eNes,vars[-1]*(eNes[-1]**(1/2))/((eNes**(1/2))),label="$1/N^{1/2}$")
#plt.plot(eNes,vars[-1]*(eNes[-1])/((eNes)),label="$1/N$")
plt.legend()
plt.yscale("log")
plt.xscale("log")


# %%
#(samples,len(eLes),B,4 )
par=0
Grads_mean=np.abs(np.mean(Grads,axis=(0,2)))
Grads_var=np.var(Grads,axis=(0,2))
print(Grads_mean)
Grads_mean_ub=Grads_mean+np.sqrt(Grads_var)*1.96/np.sqrt(samples*B)
Grads_mean_lb=Grads_mean-np.sqrt(Grads_var)*1.96/np.sqrt(samples*B)
plt.plot(eLes,Grads_mean[:,par],label="Grad")
plt.plot(eLes,Grads_mean_ub[:,par],label="Grad UB")
plt.plot(eLes,Grads_mean_lb[:,par],label="Grad LB")
plt.plot(eLes,Grads_mean[-1,par]*np.sqrt(2**(eLes[-1]))/np.sqrt(2**(eLes)),label="$\Delta_l$")
plt.yscale("log")
plt.legend()  

plt.show()



# %%
seed=3
np.random.seed(seed)
T=5
theta=-1
sigma=6
sigma_obs=2
l=5
dt=1/2**l
x_0=1
or_path=np.zeros((int(T/dt)+1))
or_path[0]=x_0
observations=np.zeros(T)
for i in range(int(T/dt)):
    or_path[i+1]=or_path[i]+theta*dt+sigma*np.sqrt(dt)*np.random.normal(0,1)
for i in range(T):
    observations[i]=or_path[int((i+1)/dt)]+np.random.normal(0,1)*(sigma_obs)

plt.plot(np.arange(0,T+dt,dt),or_path,label="Original path")
plt.plot(np.arange(0,T,1)+1,observations,label="Observations")

#%%
seed=3
N=3
x_pf=np.zeros((N,int(T/dt)+1))
x_pf[:,0]=x_0
x_pf_resamp=np.zeros((N,int(T/dt)+1))   
x_pf_resamp[:,0]=x_0
for i in range(T):

    #sampling
    for j in range(int(1/dt)):
        x_pf[:,int(i/dt)+j+1]=x_pf[:,int(i/dt)+j]+theta*dt+sigma*np.sqrt(dt)*np.random.normal(0,1,N)
    #resampling
    if i<T-1:   
        weights=ss.norm.pdf(observations[i],loc=x_pf[:,int((i+1)/dt)],scale=sigma_obs) 
        w_normalized=weights/np.sum(weights)    
        index=np.random.choice(np.array(range(N)),N,p=w_normalized)
        x_pf_resamp[:,int(i/dt)+1:int((i+1)/dt)+1]=x_pf[:,int(i/dt)+1:int((i+1)/dt)+1].copy()
        x_pf_resamp[:,:int((i+1)/dt)+1]=x_pf_resamp[index,:int((i+1)/dt)+1].copy()
        x_pf[:,int((i+1)/dt)]=x_pf[index,int((i+1)/dt)].copy()
    else:
        x_pf_resamp[:,int(i/dt)+1:int((i+1)/dt)+1]=x_pf[:,int(i/dt)+1:int((i+1)/dt)+1].copy()

#%%
colors=["blue","red","black"]
for i in range(T):
    for j in range(3):
        plt.plot(np.arange(i,i+1,dt),x_pf[j,int(i/dt):int((i+1)/dt)].T,c=colors[j])
        plt.scatter(np.zeros(1)+i+1-dt, x_pf[j,int((i+1)/dt-1)],c=colors[j])   

#plt.plot(np.arange(0,T+dt,dt),x_pf.T,label="PF path")
plt.legend()
#plt.savefig("all_paths.pdf")
#%%
plt.plot(np.arange(0,T+dt,dt),x_pf_resamp.T)    
print(np.array(range(T))+1,(x_pf_resamp.T)[(np.array(range(T))+1)*int(1/dt)])
for i in range(N):

    plt.scatter(np.array(range(T))+1,(x_pf_resamp.T)[(np.array(range(T))+1)*int(1/dt),i],c="black")
#plt.savefig("resampled_paths.pdf")
# %%
# Here we test the coupled SGD

if True:
    fd=1e-10
    rkdata= np.loadtxt("Kangaroo_data.txt")
    obs=rkdata[:,:2]
    obs_times=rkdata[:,2]
    smean=np.mean(obs)  
    svar=np.var(obs)
    smean=np.mean(obs)  
    svar=np.var(obs)
    the1=2
    the2=the1/smean
    the3=1
    #the4=np.sqrt(smean+the3**2/(2*the2))/np.sqrt(svar/smean-1-the3**2/(2*the2))    
    the4=1
    
    #the1,the2,the3,the4=2.397, 4.429e-3, 0.84, 17.36
    the3_fd=the3+fd
    the3_fd_1=the3+fd/2
    w=the3**2/2+the1
    xi=the2
    sigma=the3
    alpha=2*w/sigma**2-1
    theta=sigma**2/(2*xi)
    dist_params=np.array([the1,the2,the3])
    dist_params_fd=np.array([[the1+fd,the2,the3],[the1,the2+fd,the3],[the1,the2,the3+fd]])
    dist_params_fd_1=np.array([[the1+fd/2,the2,the3],[the1,the2+fd/2,the3],[the1,the2,the3+fd/2]])
    rej_dist_params=[the1,the2,the3,the1,the2,the3]
    in_dist_pars=dist_params
    dist_params_0=dist_params
    dist_params_fd_0=dist_params_fd
    dist_params_1=dist_params
    
    coup_in_dist_pars=[the1,the2,the3,the1,the2,the3]
    #t0=1.2
    #T=3.2
    A_til=dist_params
    fi_til=the3
    r_pars=1
    resamp_coef=1
    
    d=1
    H_pars=1

#%%
l=6
mcmc_links=1
SGD_steps=0
gamma=0.01
#gammas=[1,5e-1,2e-1,1]
#gammas=[0,3e-1,5e-1,3] # This set gave good resutls
gammas=[2,3,0.6,3]
samples=40
arg_cm=1
eLes=np.array([0])
seed=1+17*samples*len(eLes)*(arg_cm-1)+  17*samples*len(eLes)*30*(0)+23
alpha=0.01
N=64
ast=0
start=time.time()
[ch_paths_0,ch_paths_1,pars_0,pars_1 ]=\
bdg.Gen_C_SGD_bridge(bdg.rej_max_coup_gamma_in_dist,\
            coup_in_dist_pars,bdg.Grad_log_gamma_in_dist,\
            bdg.b_log, dist_params,bdg.Sig_gbm_1d,the3,bdg.b_log_aux,bdg.Sig_aux_gbm_1d,\
            bdg.r_log_normal,bdg.H_log_normal,bdg.update_log_functs,\
            bdg.rej_max_coup_log_normal_2,[the1,the2,the3,the1,the2,the3],\
            obs,obs_times,bdg.log_g_nbino_den,the4,bdg.trans_log_normal,\
            bdg.Grad_trans_log_normal, bdg.trans_prop_log_normal_2,[the1,the2,the3],\
            bdg.Grad_log_g_nbino, resamp_coef, l, d,N,seed,fd,mcmc_links,\
            SGD_steps,gamma,gammas, alpha, ast,bdg.update_pars_logis)
end=time.time()
print("Time for SGD is: ",end-start)    
#%%
par_1=0
par_2=1
labels=["$\\theta_1$","$\\theta_2$","$\\theta_3$","$\\theta_4$"]
plt.plot(pars_0[:,par_1].T,pars_0[:,par_2].T,label="pars_0")
plt.plot(pars_1[:,par_1].T,pars_1[:,par_2].T,label="pars_1")
plt.xlabel(labels[par_1])
plt.ylabel(labels[par_2])
plt.legend()

#%%
if True:
    samples=40
    l0=13
    Lmax=13
    N=200
    mcmc_links=1
    SGD_steps=10
    B=mcmc_links*SGD_steps
    gamma=0.005
    #gammas=[1,5e-1,2e-1,1]
    #gammas=[0,3e-1,5e-1,3] # This set gave good resutls
    gammas=[2,3,1,6]
    ast=2
    K=2*(2**ast-1)
    alpha=0.1
    #arg_cm=int(sys.argv[1])
    arg_cm=0
    d=1
    T=len(obs_times)*d
    eLes=np.array(range(l0,Lmax+1))
#%%


v="GCSb_True_ip_i"
pars_0=np.reshape(\
    np.loadtxt("Observationsdata/data17/Prl_Gen_C_SGD_bridge_pars_0_v"+v+".txt",dtype=float),(samples,len(eLes),SGD_steps+1+ast,4)) 

#ch_paths_0=np.reshape(\
#    np.loadtxt("Observationsdata/data14/Prl_Gen_SGD_bridge_ch_paths_0_v"+v+labels[0]+".txt",dtype=float),(samples,len(eLes),B+K,int(T/d))) 
#
pars_1=np.reshape(\
    np.loadtxt("Observationsdata/data17/Prl_Gen_C_SGD_bridge_pars_1_v"+v+".txt",dtype=float),(samples,len(eLes),SGD_steps+1+ast,4)) 

#%%
v="GCSb8_ip_ii"
labels=np.array(["1", "2", "3","4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14","15", "16","17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26","27", "28", "29","30"])
print(labels.shape)
pars_0=np.reshape(\
    np.loadtxt("Observationsdata/data17/Prl_Gen_C_SGD_bridge_pars_0_v"+v+labels[0]+".txt",dtype=float),(samples,len(eLes),SGD_steps+1+ast,4)) 

#ch_paths_0=np.reshape(\
#    np.loadtxt("Observationsdata/data14/Prl_Gen_SGD_bridge_ch_paths_0_v"+v+labels[0]+".txt",dtype=float),(samples,len(eLes),B+K,int(T/d))) 
#
pars_1=np.reshape(\
    np.loadtxt("Observationsdata/data17/Prl_Gen_C_SGD_bridge_pars_1_v"+v+labels[0]+".txt",dtype=float),(samples,len(eLes),SGD_steps+1+ast,4)) 
#%%
#ch_paths_1=np.reshape(\
#    np.loadtxt("Observationsdata/data14/Prl_Gen_SGD_bridge_ch_paths_1_v"+v+labels[0]+".txt",dtype=float),(samples,len(eLes),B+K,int(T/d))) #

for i in range(len(labels[1:])):
    pars_0=np.concatenate((pars_0,np.reshape(\
    np.loadtxt("Observationsdata/data17/Prl_Gen_C_SGD_bridge_pars_0_v"+v+labels[i+1]+".txt",dtype=float),(samples,len(eLes),SGD_steps+1+ast,4)) ),axis=0)

    pars_1=np.concatenate((pars_1,np.reshape(\
    np.loadtxt("Observationsdata/data17/Prl_Gen_C_SGD_bridge_pars_1_v"+v+labels[i+1]+".txt",dtype=float),(samples,len(eLes),SGD_steps+1+ast,4)) ),axis=0)

    #ch_paths_0=np.concatenate((ch_paths_0,np.reshape(\
    #np.loadtxt("Observationsdata/data14/Prl_Gen_SGD_bridge_ch_paths_0_v"+v+labels[i+1]+".txt",dtype=float),(samples,len(eLes),B+K,int(T/d)))),axis=0)
    #
    #
    #ch_paths_1=np.concatenate((ch_paths_1,np.reshape(\
    #np.loadtxt("Observationsdata/data14/Prl_Gen_SGD_bridge_ch_paths_1_v"+v+labels[i+1]+".txt",dtype=float),(samples,len(eLes),B+K,int(T/d)))),axis=0)
# %%
#  samples,len(eLes),SGD_steps+1,4
print(pars_1.shape)
labels=["$\\theta_1$","$\\theta_2$","$\\theta_3$","$\\theta_4$"]
# the1,the2,the3,the4=1.97707570e+00, 3.67902500e-03, 7.68023450e-01, 1.94123391e+01
#the1,the2,the3,the4 =1.45742747e+00 ,2.74535000e-03, 6.71387000e-01, 1.86269606e+01
par1=2
par2=3
ele=0
k=100
k0=k+5
k0=0
#k0=int(np.shape(pars)[0]/2)
k1=np.shape(pars_1)[0]
#k1=k+10
#k1=-1
s1=0
s2=-1 #np.shape(pars)[2]    
print(pars_1.shape)
plt.plot(pars_1[k0:k1,ele,s1:s2,par1].T,pars_1[k0:k1,ele,s1:s2,par2].T,label="pars")  
#plt.plot(pars3[k0:k1,ele,s1:s2,par1].T,pars3[k0:k1,ele,s1:s2,par2].T,label="pars",c="red")  
s1=-3
s2=-1
plt.plot(pars_1[k0:k1,ele,s1:s2,par1].T,pars_1[k0:k1,ele,s1:s2,par2].T,lw=1,c="black")  
#plt.plot(pars3[k0:k1,ele,s1:s2,par1].T,pars3[k0:k1,ele,s1:s2,par2].T,lw=1,c="green")  
plt.xlabel(labels[par1])
plt.ylabel(labels[par2])
mean_pars=np.mean(pars_1[:,ele],axis=0)
print(mean_pars[-1]) 
#%%
# Here we print the difference between the parameters 
# BIAS
step=-1
par=1
par_diff=np.abs(np.mean(pars_0-pars_1  ,axis=0)[:,step])
par_diff_var=np.var(pars_0-pars_1  ,axis=0)[:,step]
par_diff_ub=par_diff+np.sqrt(par_diff_var)*1.96/np.sqrt(samples)
par_diff_lb=par_diff-np.sqrt(par_diff_var)*1.96/np.sqrt(samples)
plt.plot(eLes,par_diff[:,par],label="Diff")
plt.plot(eLes,par_diff_ub[:,par],label="Diff UB")
plt.plot(eLes,par_diff_lb[:,par],label="Diff LB")
plt.plot(eLes,par_diff[0,par]*2**(eLes[0])/2**(eLes),label="$\Delta_l$")
plt.plot(eLes,par_diff[0,par]*2**(eLes[0]/2)/2**(eLes/2),label="$\Delta_l^{1/2}$")
plt.yscale("log")
plt.legend()
plt.show()
print("par_diff is: ",par_diff[0,par]*2**(4/2))
#%%
#%%
# SECOND MOMENT
step=-1
par=1
sms=np.mean((pars_0-pars_1)**2  ,axis=0)[:,step] # len(eLes),4
sms_var=np.var((pars_0-pars_1)**2  ,axis=0)[:,step] 
sms_ub=sms+np.sqrt(sms_var)*1.96/np.sqrt(samples)
sms_lb=sms-np.sqrt(sms_var)*1.96/np.sqrt(samples)
plt.plot(eLes,sms[:,par],label="Diff")
plt.plot(eLes,sms_ub[:,par],label="Diff UB")
plt.plot(eLes,sms_lb[:,par],label="Diff LB")
plt.plot(eLes,sms[-1,par]*np.sqrt(2**(eLes[-1]))/np.sqrt(2**(eLes)),label="$\Delta_l^{1/2}$")
plt.yscale("log")
plt.legend()
plt.show()
CL=sms[0,par]*2**(eLes[0]/2)
CL=1.566092672222229e-07
print("CL is: ",CL)
#%%
step=-1
par=1
var_pars=np.var(pars_0,axis=0)[:,step]
plt.plot(eLes,var_pars[:,par],label="Diff")
plt.yscale("log")
print(var_pars  )
CL0=3e-8
#%%
# UNBIASED TESTING
   
if True:   
    fd=1e-10
    rkdata= np.loadtxt("Kangaroo_data.txt")
    obs=rkdata[:,:2]
    obs_times=rkdata[:,2]
    smean=np.mean(obs)  
    svar=np.var(obs)
    the1=2
    the2=the1/smean
    the3=1
    #the4=np.sqrt(smean+the3**2/(2*the2))/np.sqrt(svar/smean-1-the3**2/(2*the2))    
    the4=1
    #the1,the2,the3,the4=2.397, 4.429e-3, 0.84, 17.36
    the3_fd=the3+fd
    the3_fd_1=the3+fd/2
    w=the3**2/2+the1
    xi=the2
    sigma=the3
    alpha=2*w/sigma**2-1
    theta=sigma**2/(2*xi)
    dist_params=np.array([the1,the2,the3])
    dist_params_fd=np.array([[the1+fd,the2,the3],[the1,the2+fd,the3],[the1,the2,the3+fd]])
    dist_params_fd_1=np.array([[the1+fd/2,the2,the3],[the1,the2+fd/2,the3],[the1,the2,the3+fd/2]])
    rej_dist_params=[the1,the2,the3,the1,the2,the3]
    in_dist_pars=dist_params
    dist_params_0=dist_params
    dist_params_fd_0=dist_params_fd
    dist_params_1=dist_params
    
    coup_in_dist_pars=[the1,the2,the3,the1,the2,the3]
    #t0=1.2
    #T=3.2
    A_til=dist_params
    fi_til=the3
    r_pars=1
    resamp_coef=1
    
    d=1
    H_pars=1
   
#%%

if True:
    samples=100
    #samples=1
    N=200
    mcmc_links=1
    #SGD_steps=10000
    #B=SGD_steps*mcmc_links
    gamma=0.005
    gammas=[2,3,0.6,6]
    alpha=0.5
    ast=2
    K=2*(2**ast-1)  
    CL=1.5e-7
    CL0=3e-8
    CP=3e-8
    CP0=3.28e-8
    s0=2**0
    
    l0=3
    lmax=8
    pmax=lmax+3
    beta_l=1/2
    beta_p=1
    #arg_cm=int(sys.argv[1])
    #arg_cm=32
    d=1
    T=len(obs_times)*d
#%%
ids=["38917951"]
value = np.loadtxt("Observationsdata/displays/test.37831923.1.out", usecols=3)
i = 42  # The integer you want in the file name
filename = f"Observationsdata/displays/test.{ids[0]}.{2}.out" # Produces "data_42.txt"
with open(filename, 'r') as f:
    first_line = next(f).strip()  # "Parallelized processes time: 4227.7228899002075"
float1 = float(first_line.split(":")[1])

labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])

#%%
ids=[39548488, 39548569, 39568214,  39568660,    39583719,    39597363,    39597398,\
         39620862,   39620825,  39624120,  39626149,39626155,39626161,39626168,39626174,\
         39636631,39636652,39636658,39639678,    39639682,    39639691,    39639693,39639696,\
          39639704,    39639709,    39682098,    39683540,    39683545,    39683598,\
          39683609,    39683627,    39683637,39683641,39683643,    39683658,    39683666,\
          39683672,    39683690,    39683701]
print(len(ids))
#%%
from pathlib import Path

# ----------------- customise these -----------------------------------
#ids = [39682098, 40123456, 40987654]      # put all your job IDs here
remote_dir = "bridge/displays"            # remote parent directory
n_per_id = 30                             # files per ID (1..30)
outfile = Path("filelist.txt")            # output list file
# ---------------------------------------------------------------------

with outfile.open("w", encoding="utf-8") as f:
    for job_id in ids:
        for j in range(1, n_per_id + 1):
            rel_path = f"{remote_dir}/test.{job_id}.{j}.out"
            f.write(rel_path + "\n")

print(f"Wrote {outfile} with {len(ids) * n_per_id} paths.")


#%%
from pathlib import Path
import re
import csv

# ---------------------------------------------------------------------
# customise these
n_per_id = 30                               # files per ID (1..30)

root = Path("displays")    # local parent directory
outfile = Path("parallel_times.csv")        # CSV output
# ---------------------------------------------------------------------

pat = re.compile(r"Parallelized processes time:\s*([-+]?\d+(?:\.\d+)?)")

results, missing = [], []

for job_id in ids:
    for j in range(1, n_per_id + 1):
        f = root / f"test.{job_id}.{j}.out"
        if not f.exists():
            missing.append((f, "file not found"))
            continue

        try:
            lines = f.read_text(errors="ignore").splitlines()
            # guard against empty files
            for line in reversed(lines):
                m = pat.search(line)
                if m:
                    results.append((f.name, float(m.group(1))))
                    break
            else:
                missing.append((f, "phrase not found"))
        except Exception as e:
            missing.append((f, f"read error: {e}"))

# -------- write CSV ---------------------------------------------------
with outfile.open("w", newline="") as csvf:
    csv.writer(csvf).writerows([["file", "parallel_time"], *results])

print(f"✓ Extracted {len(results)} values → {outfile}")
print(f"⚠ Skipped {len(missing)} files (see 'missing' list)")

# optional: inspect a few problems
for f, reason in missing[:10]:
    print(f"  {f}  [{reason}]")
if len(missing) > 10:
    print("  …")
#
computation_times=np.array(results)[:,1]
computation_times = np.asarray(computation_times, dtype=float)
print(computation_times)
#%%
ct_hours= computation_times / 3600
plt.hist(ct_hours, bins=50)
plt.yscale("log")
#%%
print(ct_hours.shape,39*30)

#%%

v="GU7_ip_i"
labels=np.array(["1", "2", "3" ,"4","5","6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15","16","17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])
pars_file=np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[0]+".txt",dtype=float),(samples,2,2,4)) 
levels_file=np.reshape(np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[0]+".txt",dtype=int),(samples,2))
#filename=f"Observationsdata/displays/test.{ids[0]}.{1}.out"
#times=np.array([np.loadtxt(filename, usecols=3)])
for i in range(len(labels[1:])):
    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i+1]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i+1]+".txt",dtype=int),(samples,2))),axis=0)  


v="GU7_ip_ii"

for i in range(len(labels)):
    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  


v="GU7_ip_iii"

for i in range(len(labels)):
    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  



v="GU7_ip_iv"

for i in range(len(labels)):
    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  




print(len(labels))
v="GU7_ip_v"

for i in range(len(labels)):
    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  


v="GU7_ip_vi"

for i in range(len(labels)):
    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  



v="GU7_ip_vii"

for i in range(len(labels)):
    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  


v="GU7_ip_viii"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  



v="GU7_ip_ix"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  


v="GU7_ip_x"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  


v="GU7_ip_xi"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  
    


v="GU7_ip_xii"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  
    
v="GU7_ip_xiii"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  

v="GU7_ip_xiv"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  
    
v="GU7_ip_xv"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  
    

v="GU7_ip_xvi"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  

v="GU7_ip_xvii"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  


v="GU7_ip_xviii"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  



v="GU7_ip_xix"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  

v="GU7_ip_xx"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  

v="GU7_ip_xxi"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  

v="GU7_ip_xxii"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  


v="GU7_ip_xxiii"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  



v="GU7_ip_xxiv"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  


v="GU7_ip_xxv"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  



v="GU7_ip_xxvi"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  


v="GU7_ip_xxvii"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  


v="GU7_ip_xxviii"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  



v="GU7_ip_xxix"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  



v="GU7_ip_xxx"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  


v="GU7_ip_xxxi"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  

v="GU7_ip_xxxii"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  

v="GU7_ip_xxxiii"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  

v="GU7_ip_xxxiv"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  



v="GU7_ip_xxxv"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  



v="GU7_ip_xxxvi"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  



v="GU7_ip_xxxvii"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  


v="GU7_ip_xxxviii"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  

v="GU7_ip_xxxix"

for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  
#%%
print(pars_file.shape)
#%%

v="GU3_ip_xvii"

labels=np.array(["1", "2", "3" ,"4","5",  "7", "8", "10", \
"11", "12", "13", "14", "16","17", "18", "19"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])


for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  
    
v="GU3_ip_xviii"
labels=np.array(["1",  "3" ,"4","5",  "7", "8", "9", "10", \
"11", "12", "13", "15","17", "18", "19", "20"\
    ,"21", "22", "23",  "25", "26", "27", "28", "29","30"])


for i in range(len(labels)):

    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  
    

#%%
v="GU2a_ip_iv"


labels=np.array(["1", "2", "3" ,"4" ,"5", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18",  "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])


for i in range(len(labels)):
    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  

#%%

#%%
v="GU1_ip_iii"

labels=np.array(["1", "2", "3" ,"4" ,"5", "7", "8", "9", "10", \
"11", "12", "13", "14", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])


for i in range(len(labels)):
    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  




v="GU1_ip_iv"

labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "24", "25", "26", "27", "28", "29","30"])

for i in range(len(labels)):
    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  




v="GU1_ip_v"

labels=np.array(["1", "2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "25", "26", "27", "28", "29","30"])

for i in range(len(labels)):
    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  


"""
v="GU1_ip_vi"

labels=np.array(["2", "3" ,"4" ,"5", "6", "7", "8", "9", "10", \
"11", "12", "13", "14", "15", "16", "17", "18", "19", "20"\
    ,"21", "22", "23", "24", "25", "26", "27", "28", "29","30"])

for i in range(len(labels)):
    #times=np.concatenate((times,np.array([np.loadtxt(\
    #"Observationsdata/displays/test."+ids[0]+"."+labels[i+1]+".out", usecols=3)])),axis=0)
    pars_file=np.concatenate((pars_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_pars_v"+v+labels[i]+".txt",dtype=float),(samples,2,2,4)) ),axis=0)  
    levels_file=np.concatenate((levels_file,np.reshape(\
    np.loadtxt("Observationsdata/data16/Prl_Gen_Unbiased_levels_v"+v+labels[i]+".txt",dtype=int),(samples,2))),axis=0)  
"""



print(levels_file[:20])


# %%
# The probabilities of the levels are:

eLes=np.arange(l0,lmax+1)
beta=beta_l
q=4
P0=(1+CL*np.sum((q+eLes[1:]-l0)*np.log(q+eLes[1:]-l0)**2/2**(beta*eLes[1:]))/\
(CL0*(q+1)*np.log(q+1)**2))**(-1)

l_cumu=bdg.P_l_cumu_gen(P0,lmax-l0+1,beta,l0)
l=eLes[bdg.sampling(l_cumu)]
l_den=np.zeros(len(eLes))
l_den[0]=P0
l_den[1:]=l_cumu[1:]-l_cumu[:-1]

# cumulative for the number of SGD steps
beta=beta_p
ePes=np.arange(0,pmax+1)
eSes=s0*2**ePes
P0=(1+CP*np.sum((ePes[1:]+q)*np.log(ePes[1:]+q)**2/eSes[1:]**(beta))\
/(CP0*(q+1)*np.log(1+q)**2))**(-1)

p_cumu=bdg.P_p_cumu_gen(P0,pmax,beta,s0)
p_den=np.zeros(len(ePes))
p_den[0]=P0
p_den[1:]=p_cumu[1:]-p_cumu[:-1]
print("The levels are: ",eSes,eLes )
print("The density of l is:",l_den)
print("The density of p is:",p_den)
#%%
sample_levels_f=np.zeros((len(eLes),len(ePes)))
for i in range(levels_file.shape[0]):
    if (levels_file[i,0]-l0== 7 and levels_file[i,1]==11):# or\
        #(levels_file[i,0]-l0== 7 and levels_file[i,1]==8)or\
        #(levels_file[i,0]-l0== 6 and levels_file[i,1]==9):
        print("Found a sample with",levels_file[i,0]-l0,levels_file[i,1])
        print("The sample is: ",pars_file[i])
    sample_levels_f[levels_file[i,0]-l0,levels_file[i,1]]+=1
print("The sample levels frequencies are: ",sample_levels_f)

# %%
print(pars_file.shape[0])
print(l_den)
print(p_den)
dist=np.zeros((lmax-l0+1,pmax+1))
for i in range(levels_file.shape[0]):
    dist[levels_file[i,0]-l0,levels_file[i,1]]+=1
print(dist)
#%%
par1=2
par2=3
plt.scatter(pars_file[:,1,1,par1],pars_file[:,1,1,par2],label="pars_1")

#%%
# samples,2,2,3
#print("The parameters are:", pars[-1])
print(pars_file.shape)
unb_terms=pars_file/(l_den[levels_file[:,0]-l0,np.newaxis,np.newaxis,np.newaxis]\
*p_den[levels_file[:,1],np.newaxis,np.newaxis,np.newaxis])
#unb_terms=unb_terms[-200:]
a=0
b=-1
est=np.mean(unb_terms[:,1,1]-unb_terms[:,1,0]-(unb_terms[:,0,1]-unb_terms[:,0,0]),axis=0)
print(est)
var_est=np.var(unb_terms[:,1,1]-unb_terms[:,1,0]-(unb_terms[:,0,1]-unb_terms[:,0,0]),axis=0)
print(var_est/((pars_file.shape[0])))
#%%
np.log2(pars_file.shape[0])
#%%
# seed=5 z=14
# seed=13 z=16
# seed=23 z=16
# seed=34 z=16
np.random.seed(14)
seeds= np.array(range(13,14))
for seed in seeds:
    print("The seed is: ",seed)
    np.random.seed(seed)
    
    
    1.92039179e+01 
    an_mean=np.array([1.96063975e+00, 3.65980000e-03, 7.62025325e-01,1.89265234e+01])
    #an_mean=np.array([1.96549048e+00 ,3.66775000e-03 ,7.63895425e-01 ,1.92039179e+01])
    # an_mean=np.array([1.73275005e+00,3.23095000e-03 ,7.10832200e-01, 1.8e+01])
    #an_mean=np.array([2.14165185e+00 ,4.00555183e-03, 8.65705696e-01, 1.91018741e+01])
    #an_mean=np.array([1.45742747e+00, 2.74535000e-03 ,6.71387000e-01, 1.86269606e+01])
    #print(unb_terms.shape)
    samples_total=pars_file.shape[0]   
    #print(np.log2(samples_total/10))
    #print(np.log2(samples_total))
    z=16
    batches=8
    ests=np.zeros((batches,4))
    costs=np.zeros(batches)
    times=np.zeros(batches)
    m_times=np.zeros((z-1))
    v_times=np.zeros((z-1))
    m_costss=np.zeros((z -1))
    mses=np.zeros((z -1,4))
    biases=np.zeros((z -1,4))
    vars=np.zeros((z -1,4))
    for i in range(1,z):
        for j in range(batches):
            batch_samples=np.random.choice(samples_total,2**(i+1),replace=False)
            costs[j]=np.sum(2**(levels_file[batch_samples,0]+levels_file[batch_samples,1]))        
            ests[j]=np.mean(unb_terms[batch_samples,1,1]-unb_terms[batch_samples,1,0]-(unb_terms[batch_samples,0,1]-unb_terms[batch_samples,0,0]),axis=0)
            #print("The times are: ",ct_hours[batch_samples//samples])
            times[j]=np.max(ct_hours[batch_samples//samples])
        m_times[i-1]= np.mean(times)
        v_times[i-1]= np.var(times)
        m_costss[i-1]=np.mean(costs)
        mses[i-1]=np.mean((ests-an_mean)**2,axis=0)
        biases[i-1]=np.mean(ests,axis=0)-an_mean
        vars[i-1]=np.var(ests,axis=0)
#%%

    #print(v_times)
    par_n=3
    [b_0,b_1]= bdg.coef(np.log(mses[:,par_n]),np.log(m_costss))
    #print(b_0,b_1)
    #print(mses[:,par_n])
    #plt.plot(biases[:,par_n]**2,m_costss,label="Unbiased Bride approach",ls='--',lw=2,marker='o',markersize=5)
    #plt.plot(vars[:,par_n],m_costss,label="vars",ls='--',lw=2,marker='o',markersize=5)
    plt.plot(mses[:,par_n],np.exp(b_0)*mses[:,par_n]**b_1,lw=2,\
    label=rf'$\varepsilon^{{{2*b_1:.2f}}}$',color="coral")
    plt.plot(mses[:,par_n],m_costss[0]*mses[0,par_n]/mses[:,par_n],lw=2,\
    label="Canonical Monte Carlo rate: "+rf'$\varepsilon^{{-2}}$',color="dodgerblue")    
    plt.plot(mses[:,par_n],m_costss[0]*mses[0,par_n]**(3/2)/mses[:,par_n]**(3/2),\
    lw=2,label=rf'$\varepsilon^{{-3}}$',color="deepskyblue")

    plt.scatter(mses[:,par_n],m_costss,label="Unbiased Bride approach",lw=2,marker='o',color="coral")
    plt.xlabel(r"$\varepsilon^2$",size=14)
    plt.ylabel("Cost",size=14)
    plt.tick_params(axis="both",   # "x", "y", or "both"
                    labelsize=14)
    plt.yscale("log")
    plt.xscale("log")
    plt.legend(fontsize=12)
    #plt.savefig("Logistic_Cost_vs_MSE.pdf")
    plt.show()
#%%
#%%
par_n=1
[b_0,b_1]= bdg.coef(np.log(mses[:,par_n]),np.log(m_times))
print(b_0,b_1)
print(mses[:,par_n])
print(m_times)
#plt.plot(mses[:,par_n],np.exp(b_0)*mses[:,par_n]**b_1,lw=2,label=rf'$\varepsilon^{{{2*b_1:.2f}}}$')
#plt.plot(mses[:,par_n],m_times[0]*mses[0,par_n]/mses[:,par_n],lw=2,label="Canonical Monte Carlo rate: "+rf'$\varepsilon^{{-2}}$')    
#plt.plot(mses[:,par_n],m_times[0]*mses[0,par_n]**(3/2)/mses[:,par_n]**(3/2),lw=2,label=rf'$\varepsilon^{{-3}}$')
#plt.plot(MSEs_van[:,par_n],ct_hours_van,label="Vanilla approach",ls='--',lw=2,marker='o',markersize=5)
#plt.plot(mses[:,par_n],m_times/2,label="Unbiased Bride approach",ls='--',lw=2,marker='o',markersize=5)
plt.plot(np.sum(MSEs_van,axis=1),ct_hours_van,label="Bridge approach",ls='--',lw=2,marker='+',markersize=5)
plt.plot(np.sum(mses,axis=1),m_times/2,label="Unbiased Bride approach",ls='-.',lw=2,marker='o',markersize=5)

plt.xlabel("MSE, "+r"$\varepsilon^2$",size=14)
plt.ylabel("Mean time",size=14)
plt.tick_params(axis="both",   # "x", "y", or "both"
                labelsize=14)
#plt.yscale("log")
plt.xscale("log")
plt.legend()
#plt.savefig("Cost_vs_MSE.pdf")
plt.show()
#%%
# %%

# In this section we will compute the MSE of the vanilla estimator (not unbiase)

if True:   
    fd=1e-10
    rkdata= np.loadtxt("Kangaroo_data.txt")
    obs=rkdata[:,:2]
    obs_times=rkdata[:,2]
    
    smean=np.mean(obs)  
    svar=np.var(obs)
    the1=2
    the2=the1/smean
    the3=1
    #the4=np.sqrt(smean+the3**2/(2*the2))/np.sqrt(svar/smean-1-the3**2/(2*the2))    
    the4=1
    #the1,the2,the3,the4=2.397, 4.429e-3, 0.84, 17.36
    the3_fd=the3+fd
    the3_fd_1=the3+fd/2
    w=the3**2/2+the1
    xi=the2
    sigma=the3
    alpha=2*w/sigma**2-1
    theta=sigma**2/(2*xi)
    dist_params=np.array([the1,the2,the3])
    dist_params_fd=np.array([[the1+fd,the2,the3],[the1,the2+fd,the3],[the1,the2,the3+fd]])
    dist_params_fd_1=np.array([[the1+fd/2,the2,the3],[the1,the2+fd/2,the3],[the1,the2,the3+fd/2]])
    rej_dist_params=[the1,the2,the3,the1,the2,the3]
    in_dist_pars=dist_params
    dist_params_0=dist_params
    dist_params_fd_0=dist_params_fd
    dist_params_1=dist_params
    
    coup_in_dist_pars=[the1,the2,the3,the1,the2,the3]
    #t0=1.2
    #T=3.2
    A_til=dist_params
    fi_til=the3
    r_pars=1
    resamp_coef=1
    
    d=1
    H_pars=1
    
    
#%%
ids_van=[39635547,    39635567,    39635569,    39635574,    39635601,    39635612,    39635621]
#%%
#%%
from pathlib import Path

# ----------------- customise these -----------------------------------
#ids = [39682098, 40123456, 40987654]      # put all your job IDs here
remote_dir = "bridge/displays"            # remote parent directory
                      # files per ID (1..30)
outfile = Path("filelist.txt")            # output list file
# ---------------------------------------------------------------------

with outfile.open("w", encoding="utf-8") as f:
    for job_id in ids_van:
        
            rel_path = f"{remote_dir}/test.{job_id}.out"
            f.write(rel_path + "\n")

print(f"Wrote {outfile} with {len(ids_van) } paths.")


#%%
from pathlib import Path
import re
import csv

# ---------------------------------------------------------------------
# customise these
                        # files per ID (1..30)

root = Path("displays")    # local parent directory
outfile = Path("parallel_times.csv")        # CSV output
# ---------------------------------------------------------------------

pat = re.compile(r"Parallelized processes time:\s*([-+]?\d+(?:\.\d+)?)")

results, missing = [], []

for job_id in ids_van:
    
        f = root / f"test.{job_id}.out"
        if not f.exists():
            missing.append((f, "file not found"))
            continue

        try:
            lines = f.read_text(errors="ignore").splitlines()
            # guard against empty files
            for line in reversed(lines):
                m = pat.search(line)
                if m:
                    results.append((f.name, float(m.group(1))))
                    break
            else:
                missing.append((f, "phrase not found"))
        except Exception as e:
            missing.append((f, f"read error: {e}"))

# -------- write CSV ---------------------------------------------------
with outfile.open("w", newline="") as csvf:
    csv.writer(csvf).writerows([["file", "parallel_time"], *results])

print(f"✓ Extracted {len(results)} values → {outfile}")
print(f"⚠ Skipped {len(missing)} files (see 'missing' list)")

# optional: inspect a few problems
for f, reason in missing[:10]:
    print(f"  {f}  [{reason}]")
if len(missing) > 10:
    print("  …")

computation_times_van=np.array(results)[:,1]
computation_times_van = np.asarray(computation_times, dtype=float)
print(computation_times)
#%%
ct_hours_van= computation_times_van / 3600
plt.hist(ct_hours_van, bins=50)
plt.yscale("log")
#%%
if True:
    samples=40
    N=200
    l0=111
    Lmax=11
    mcmc_links=1
    gamma=0.005
    gammas=[2,3,0.6,10]
    alpha=0.5
    ast=2
    K=2*(2**ast-1)  
    #arg_cm=int(sys.argv[1])
    arg_cm=32
    d=1
    T=len(obs_times)*d
    eLes=np.array(range(l0,Lmax+1))

# %%
eLes=[0]
v="GSbVan9_i"
pars_file_van=np.reshape(\
    np.loadtxt("Observationsdata/data18/Prl_Gen_SGD_bridge_pars_v"+v+".txt",dtype=float),\
    (samples,len(eLes),4)) 


v="GSbVan9_ii"

pars_file_van=np.concatenate((pars_file_van,np.reshape(\
    np.loadtxt("Observationsdata/data18/Prl_Gen_SGD_bridge_pars_v"+v+".txt",dtype=float),\
    (samples,len(eLes),4))),axis=1) 

v="GSbVan9_iii"

pars_file_van=np.concatenate((pars_file_van,np.reshape(\
    np.loadtxt("Observationsdata/data18/Prl_Gen_SGD_bridge_pars_v"+v+".txt",dtype=float),\
    (samples,len(eLes),4))),axis=1) 


v="GSbVan9_iv"

pars_file_van=np.concatenate((pars_file_van,np.reshape(\
    np.loadtxt("Observationsdata/data18/Prl_Gen_SGD_bridge_pars_v"+v+".txt",dtype=float),\
    (samples,len(eLes),4))),axis=1) 

v="GSbVan9_v"

pars_file_van=np.concatenate((pars_file_van,np.reshape(\
    np.loadtxt("Observationsdata/data18/Prl_Gen_SGD_bridge_pars_v"+v+".txt",dtype=float),\
    (samples,len(eLes),4))),axis=1) 



v="GSbVan9_vi"

pars_file_van=np.concatenate((pars_file_van,np.reshape(\
    np.loadtxt("Observationsdata/data18/Prl_Gen_SGD_bridge_pars_v"+v+".txt",dtype=float),\
    (samples,len(eLes),4))),axis=1) 

v="GSbVan9_vii"

pars_file_van=np.concatenate((pars_file_van,np.reshape(\
    np.loadtxt("Observationsdata/data18/Prl_Gen_SGD_bridge_pars_v"+v+".txt",dtype=float),\
    (samples,len(eLes),4))),axis=1) 

#%%

v="GSbVan7_viii"

pars_file_van=np.concatenate((pars_file_van,np.reshape(\
    np.loadtxt("Observationsdata/data18/Prl_Gen_SGD_bridge_pars_v"+v+".txt",dtype=float),\
    (samples,len(eLes),4))),axis=1) 


#%%
ele=-1
par1=2
par2=3
for i in range(pars_file_van.shape[1]):
    plt.scatter(pars_file_van[:,i,par1],pars_file_van[:,i,par2],label="level_"+str(i+1))

plt.legend()
 #%%

l0=3
Lmax=9
eLes=np.array(range(l0,Lmax+1))
labels=np.array(["\theta_1", "\theta_2", "\theta_3", "\theta_4"])
par=3
print("par is: ",par)
#plt.plot(eLes,pars_file[:,0,par],label=labels[par]+" at level "+str(l0))
print(np.mean(pars_file_van,axis=0))


#,1.89265234e+01 
pars_true=[1.96549048e+00, 3.66775000e-03, 7.63895425e-01,1.92039179e+01]
MSEs=np.mean((pars_file_van-pars_true)**2,axis=0)
var=np.var(pars_file_van,axis=0)
bias=np.abs( np.mean(pars_file_van,axis=0)-pars_true)
print("The mean is: ",np.mean(pars_file_van,axis=0))
#plt.plot(eLes[:],bias[:,par]**2,label="bias^2")
#plt.plot(eLes[:],bias[-1,par]**2*2**(eLes[-1]*2)/2**(2*eLes[:]),label="$\Delta_l$")
#plt.plot(eLes[:],var[:,par],label="var")
plt.plot(eLes[:],MSEs[:,par],label="MSE")
#plt.plot
plt.yscale("log")
plt.xlabel("Level $l$")
plt.ylabel("MSE, $\\varepsilon^2$")
plt.legend()

#%%

l0=3
Lmax=9
eLes=np.array(range(l0,Lmax+1))
labels=np.array(["\theta_1", "\theta_2", "\theta_3", "\theta_4"])
par=3
print("par is: ",par)
#plt.plot(eLes,pars_file[:,0,par],label=labels[par]+" at level "+str(l0))
#pars_true=[1.96549048e+00, 3.66775000e-03, 7.63895425e-01, 1.92039179e+01]
pars_true=[1.96063975e+00 ,3.65980000e-03, 7.62025325e-01, 1.92265234e+01]
MSEs_van=np.mean((pars_file_van-pars_true)**2,axis=0)
var=np.var(pars_file_van,axis=0)
bias=np.abs( np.mean(pars_file_van,axis=0)-pars_true)
print("The mean is: ",np.mean(pars_file_van,axis=0))

#plt.plot(eLes[:],bias[:,par]**2,label="bias^2")
#plt.plot(eLes[:],bias[-1,par]**2*2**(eLes[-1]*2)/2**(2*eLes[:]),label="$\Delta_l$")
#plt.plot(eLes[:],var[:,par],label="var")
plt.plot(ct_hours_van,MSEs_van[:,par],label="MSE")
#plt.plot
plt.yscale("log")
plt.xscale("log")
plt.xlabel("Level $l$")
plt.ylabel("MSE, $\\varepsilon^2$")
plt.legend()

#%%
plt.plot(eLes,MSEs[:,par],label="$\\theta_1$")


# %%
SGD_steps=10000
eLes=np.arange(3,10)
print(eLes)
eNes=2**(3+eLes)
print(eNes)
# %%
ePes=np.arange(0,11)
eLes=np.ceil(ePes-2*np.log2(3/1.3))
print(-2*np.log2(3/1.3))
print(eLes,ePes)
print(eLes,2**ePes)
print(ePes-2*np.log2(3/1.3))
# %%
# THE FOLLOWING SECTION IS DEDICATED TO THE PLOT OF A BRIDGE TO ADD IT TO A PRESENTATION

from numba import njit, prange

@njit(parallel=True, fastmath=True)
def em_logistic_batch(t0, x0, T, the1, the2, the3, l, seed=0):
    """
    Euler–Maruyama for the logistic SDE on *many* paths.
    x0 : 1-D array of initial values, length N
    Returns (steps+1, N) array.
    """
    np.random.seed(seed)

    Dt    = 1.0 / (2 ** l)
    steps = int((T - t0) / Dt)
    N     = x0.size

    # pre-allocate output
    x = np.empty((steps + 1, N), dtype=np.float64)
    x[0, :] = x0

    # pre-generate all Gaussian noise:  steps × N
    dW = np.random.standard_normal((steps, N)) * np.sqrt(Dt)

    # constants for drift
    drift_c1 = the3**2 / 2.0

    # main loop over time, *parallel* in paths
    for n in range(steps):
        for j in prange(N):
            xi   = x[n, j]
            drift = the1/the3 - the2/the3 * np.exp(the2*xi)   # your chosen drift
            # alternative classic logistic drift:
            # drift = (drift_c1 + the1 - the2*xi) * xi
            x[n+1, j] = xi + drift * Dt + the3 * xi * dW[n, j]

    return x

#%%
if True:
    fd=1e-10
    rkdata= np.loadtxt("Kangaroo_data.txt")
    obs=rkdata[:,:2]
    #obs_or=rkdata[:,:2]
    #obs=np.zeros((3,2))
    #obs[0,:]=obs_or[0,:]
    #obs[1,:]=obs_or[-2,:]
    #obs[2,:]=obs_or[-1,:]
    obs_times=rkdata[:,2]
    print("the size is:",obs_times.shape)
    #obs_times_or=rkdata[:,2]
    #obs_times=np.zeros((3))
    #obs_times[0]=obs_times_or[0]
    #obs_times[1]=obs_times_or[-2]
    #obs_times[2]=obs_times_or[-1]+30
    print("obs times are: ",obs_times)
    #obs=np.array([1.4,2,3,4.7,5.3,6.5])
    deltas=bdg.get_deltas(obs_times)
    print("deltas are: ",deltas)
    smean=np.mean(obs)  
    svar=np.var(obs)
    the1=2
    the2=the1/smean
    the3=1
    #the4=np.sqrt(smean+the3**2/(2*the2))/np.sqrt(svar/smean-1-the3**2/(2*the2))    
    the4=10
    the1,the2,the3,the4=2.397, 4.429e-3, 0.84, 17.36
    
    #the1,the2,the3,the4 =1.47761682e+00 ,2.77612500e-03, 6.75328450e-01, 1.86235184e+01
    the3_fd_1=the3+fd/2
    w=the3**2/2+the1
    xi=the2
    sigma=the3
    alpha=2*w/sigma**2-1
    theta=sigma**2/(2*xi)
    dist_params=np.array([the1,the2,the3])
    dist_params_fd=np.array([[the1+fd,the2,the3],[the1,the2+fd,the3],[the1,the2,the3+fd]])
    dist_params_fd_1=np.array([[the1+fd/2,the2,the3],[the1,the2+fd/2,the3],[the1,the2,the3+fd/2]])
    rej_dist_params=[the1,the2,the3,the1,the2,the3]
    in_dist_pars=dist_params
    dist_params_0=dist_params
    dist_params_fd_0=dist_params_fd
    dist_params_1=dist_params
    coup_in_dist_pars=[the1,the2,the3,the1,the2,the3]
    #t0=1.2
    #T=3.2
    A_til=dist_params
    fi_til=the3
    r_pars=1
    resamp_coef=1
    d=1
    H_pars=1

#%%
t0=0
T=5
the1,the2,the3,the4=2.397, 4.429e-3, 0.84, 17.36
theta=[the1,the2,the3]
#the4=np.array([[10]])
theta_aux=1
sigma=the3
sigma_aux=sigma
l=5
d=1
N=50
x0=0+np.zeros(N)
x_p=0+np.zeros(N)

dim=1
seed=1 
# t0,x0,T,x_p,b,A,Sig,fi,b_til,A_til,Sig_til,fi_til,r,r_pars,H,H_pars,l,d,N,seed
n_tests=10
"""
start=time.time()

for i in range(n_tests):
    pars_0=[np.array([the1,the2,the3]),the3]
    [A_til,fi_til,r_pars,H_pars,\
    atdp]=bdg.update_log_functs(pars_0,pars_0,t0,x0,x0,T,x_p,x_p,levels=1)
    x=bdg.Bridge_1d_opt(t0,x0,T,x_p,bdg.b_log,[the1,the2,the3],bdg.Sig_gbm_1d,sigma,bdg.b_log_aux,A_til,bdg.Sig_aux_gbm_1d,\
    fi_til,bdg.r_log_normal,r_pars,bdg.H_log_normal,\
    H_pars,l,d,N,seed)
    

end=time.time()
"""
mcmc_links=1
SGD_steps=10
gamma=0.005
gammas=[2,3,0.6,10]
alpha=0.5
ast=0
K=2*(2**ast-1) 

start=time.time()

for i in range(n_tests):
    ch_paths,pars =bdg.Gen_SGD_bridge(bdg.gamma_sampling,in_dist_pars, bdg.Grad_log_gamma_in_dist, \
    bdg.b_log,dist_params, bdg.Sig_gbm_1d,the3,bdg.b_log_aux,bdg.Sig_aux_gbm_1d,bdg.r_log_normal,\
    bdg.H_log_normal, bdg.update_log_functs,\
    bdg.sampling_prop_log_normal_2,[the1,the2,the3],obs,obs_times,bdg.log_g_nbino_den,\
    the4, bdg.trans_log_normal, bdg.Grad_trans_log_normal,bdg.trans_prop_log_normal_2,\
    bdg.Grad_log_g_nbino,resamp_coef, l, d,N,seed,fd,mcmc_links,SGD_steps,gamma,gammas, alpha,ast, bdg.update_pars_logis)

end=time.time()




print("The time is: ",end-start)
#%%
    
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
#"""
#%%
print(192.548015832901/0.770317)
#%%
#print(x)  
x[-1]=x_p
times=np.linspace(t0,T,2**l+1)
fig,ax=plt.subplots()
ax.plot(times,x)
ax.axis("off")
ax.set_xticks([])
ax.set_yticks([])
#plt.savefig("bridge_path.pdf")
plt.show()
# %%
l=6
N=100
mcmc_links=1
SGD_steps=10
gamma=0.005
gammas=[2,3,0.6,10]
alpha=0.5
ast=0
K=2*(2**ast-1)  
start=time.time()
seed=382
#the1,the2,the3,the4=2.397, 4.429e-3, 0.84, 17.36
ch_paths,pars =bdg.Gen_SGD_bridge(bdg.gamma_sampling,in_dist_pars, bdg.Grad_log_gamma_in_dist, \
            bdg.b_log,dist_params, bdg.Sig_gbm_1d,the3,bdg.b_log_aux,bdg.Sig_aux_gbm_1d,bdg.r_log_normal,\
            bdg.H_log_normal, bdg.update_log_functs,\
            bdg.sampling_prop_log_normal_2,[the1,the2,the3],obs,obs_times,bdg.log_g_nbino_den,\
            the4, bdg.trans_log_normal, bdg.Grad_trans_log_normal,bdg.trans_prop_log_normal_2,\
            bdg.Grad_log_g_nbino,resamp_coef, l, d,N,seed,fd,mcmc_links,SGD_steps,gamma,gammas, alpha,ast, bdg.update_pars_logis)
end=time.time()
#%%
v="test"
times=np.loadtxt("Observationsdata/data19/EM_computations"+v+".txt",dtype=float)
print(np.mean(times),np.var(times))
print(times.shape)
# %%
