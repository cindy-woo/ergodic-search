#!/usr/bin/env python
# coding: utf-8

# In[7]:


#!pip install jax==v0.2.20
# get_ipython().system('pip install jax==0.2.12 jaxlib==0.1.67')


# In[6]:


import jax
import jax.numpy as np
import jax.random as jnp_random
import numpy as onp
from jax import grad, jacrev, jacfwd, jit, vmap, partial
from jax.scipy.special import logsumexp
from jax.lax import scan
from jax.experimental import optimizers
from jax.numpy import concatenate as cat
import timeit

import random
import pickle as pkl
import scipy

from IPython.display import clear_output
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Dynamics and target distribution
# ----
# 
# Here we define the dynamics and the target distribution.
# 
# The dynamics are defined as the constrained continuous time dynamical system
# $$
#     \dot{x} = f(x, u) = \tanh(u)
# $$
# where $x(t) \in \mathbb{R}^2$ and $u \in \mathbb{R}^2$ is constrained to the continuous set $[-1,1]^2$
# 
# The target distribution, for which information is distributed within the continuous search space $X \in [0,L]^2 \subset \mathbb{R}^2$, is defined as
# 
# $$
#     p(x) = \sum_{i=1}^3 \eta_i \exp \left( \Vert x - c_i \Vert ^2_{\Sigma_i^{-1}}\right)
# $$
# 
# where $p(x): X \to \mathbb{R}^+$, and $\eta_i, c_i, \Sigma_i$ are the normalizing factor the Gaussian center, and the Gaussian variance respectively.

# In[ ]:


@jit
def p(x):
    return np.exp(-190.5 * np.sum((x[:2] - 0.27)**2)) \
                + np.exp(-180.5 * np.sum((x[:2] - 0.75)**2)) \
                + np.exp(-160.5 * np.sum((x[:2] - np.array([0.23, 0.75]))**2)) #\
                #+ np.exp(-160.5 * np.sum((x[:2] - np.array([0.75, 0.23]))**2))
    #np.exp(-180.5 * np.sum((x[:2] - 0.75)**2)) \
           #     + np.exp(-160.5 * np.sum((x[:2] - np.array([0.23, 0.75]))**2))
    #np.exp(-190.5 * np.sum((x[:2] - 0.27)**2)) \
           #     + np.exp(-180.5 * np.sum((x[:2] - 0.75)**2)) \
           #     + np.exp(-160.5 * np.sum((x[:2] - np.array([0.23, 0.75]))**2)) \
           #     + np.exp(-160.5 * np.sum((x[:2] - np.array([0.75, 0.23]))**2))
@jit
def f(x, u): # dynamics using discrete time eulerintegration
    xnew = x[:2] + u[:2]
    return xnew, xnew


# Helper Functions
# ----
# 
# Below are a few helper functions that we will use throughout. These primarily define a sigmoid function and a orthonormalizing factor $h_k$ for the ergodic metric.

# In[ ]:


def sigmoid(x):
    return (1 + np.exp(-x))**(-1)

def get_hk(k): # normalizing factor for basis function
    _hk = (2. * k + onp.sin(2 * k))/(4 * k)
    _hk[onp.isnan(_hk)] = 1.
    return onp.sqrt(onp.prod(_hk))


# Ergodic metric and sample-weighted ergodic metric
# ----
# 
# Here we define a series of functions to compute the following ergodic metric:
# 
# $$
#     \mathcal{E}(x(t), p) = \sum_{k \in \mathbb{N}^v}\Lambda_k \left( c_k(x(t)) - \phi_k(p) \right)^2
# $$
# where
# $$
# c_k(x(t)) = \frac{1}{T}\int_0^T F_k(x(t))dt, \text{ and } \phi_k = \int_X p(s)F_k(s)ds
# $$
# and $$F_k(x) = \frac{1}{h_k} \prod_{i=1}^v \cos \left( \frac{k_i \pi x_i}{L_i} \right).$$
# 
# We can optimize for when to choose sample in the ergodic metric by weighing the samples $x(t)$ using _sample weights_ $\lambda(t) \in [0,1]$ which are jointly `optimized` with the sample trajectory $x(t)$. The augmentation to the ergodic metric is as follows:
# 
# $$
#     \mathcal{E}(x(t), \lambda(t), p) = \sum_{k \in \mathbb{N}^v}\Lambda_k \left( c_k(x(t), \lambda(t)) - \phi_k(p) \right)^2
# $$
# where
# $$
# c_k(x(t)) = \frac{1}{\int_t \lambda(t) dt}\int_0^T \lambda(t) F_k(x(t))dt
# $$
# and $ 0 \le \lambda(t) \le 1$.

# In[ ]:


def fk(x, k): # basis function
    return np.prod(np.cos(x*k))

fk_vmap = lambda _x, _k: vmap(fk, in_axes=(0,None))(_x, _k)
def get_ck(tr, k):
    ck = np.mean(vmap(partial(fk_vmap, tr))(k), axis=1)
    ck = ck / hk
    return ck

def get_ck_weighted(tr, k, lam):
    ck = np.dot(vmap(partial(fk_vmap, tr[:,:2]))(k), sigmoid(5*lam))#(1.-tr[:,2])*sig)
    ck = ck/ck[0]
    return ck

# ergodic metric + other costs----
# @jit
def fourier_ergodic_loss(u, x0, phik, k):
    xf, tr = scan(f, x0, u[:,:2])

    # this sigma tranformation    ensures that lambda is between [0,1]
#     lam = sigmoid(5*u[:,2])
    lam = u[:,2]
    ck = np.dot(vmap(partial(fk_vmap, tr[:,:2]))(k), lam)#(1.-tr[:,2])*sig)
    ck = ck / ck[0]
#     ck = ck / hk
    # barrier cost to ensure that x(t) stays within X
    barr_cost = 100*np.sum(np.maximum(0, tr-1)**2 + np.maximum(0, -tr)**2)
    lam_barr_cost = 100*np.sum(np.maximum(0, lam-1)**2 + np.maximum(0, -lam)**2)
    return np.sum(lamk*np.square(phik - ck)) + 0.1 * np.mean(u[:,:2]**2) + barr_cost \
            + 0.001*np.sum(np.abs(lam)) + lam_barr_cost# l1 cost to promote sparsity


dl = jit(grad(fourier_ergodic_loss))


# Visualization of target distribution
# ---

# In[ ]:


X,Y = np.meshgrid(*[np.linspace(0,1)]*2)
_s = np.stack([X.ravel(), Y.ravel()]).T
plt.contour(X, Y, vmap(p)(_s).reshape(X.shape))
# plt.axis('equal')
plt.axis('square')


# Compute the Fourier basis modes and the orthonormalization factors
# ----

# In[ ]:


k1, k2 = np.meshgrid(*[np.arange(0, 20, step=1)]*2)

k = np.stack([k1.ravel(), k2.ravel()]).T
# k = np.pi*k
# lamk = (1.+np.linalg.norm(k,axis=1)**2)**(-4./2.)
lamk = np.exp(-0.8 * np.linalg.norm(k, axis=1))
# lamk = np.ones((len(k), 1))
hk = []
for ki in k:
    hk.append(get_hk(ki))
hk = np.array(hk)


# Compute $\phi_k$'s and reconstruct to verify
# ---

# In[ ]:


phik = np.dot(vmap(fk_vmap, in_axes=(None, 0))(_s, k), vmap(p)(_s))
phik = phik/phik[0]
# phik = phik/np.array(hk)

phik_recon = np.dot(phik, vmap(fk_vmap, in_axes=(None, 0))(_s, k)).reshape(X.shape)
plt.contour(X, Y, phik_recon)
plt.axis('square')


# Set up initial conditions and optimization states
# ----

# In[ ]:


x0 = np.array([0.54,0.3])

num_samples = 100

# U AND \LAMBDA ARE ABORBED INTO THE SAME VARIABLE FOR SIMPLICITY
u = np.zeros((num_samples,2))
u = cat([u, np.ones((num_samples,1))], axis=1)
# key = jnp_random.PRNGKey(0)
# u = jnp_random.normal(key, shape=(100, 3)) * 0.01 #+ np.array([0.5,0.5, 0.5])

# veryify that initial conditions are well defined by passing them through the
# ergodic loss
fourier_ergodic_loss(u, x0, phik, k)

# here we use adam as the `optimize`r but that is not required
opt_init, opt_update, get_params = optimizers.adam(1e-3)
opt_state = opt_init(u)


# Optimization loop
# ---
# 
# The optimizaiton problem we are solving is the following:
# 
# $$
# \begin{align}
# && {\arg\min}_{\lambda(t), u(t) \forall t \in [0, T]} \mathcal{E}(x(t)) + \mathcal{J}_\text{barrier}(x(t)) + \int_t \vert \lambda(t) \vert dt \\
# && \text{ subject to } \dot{x} = f(x, u), 0 < \lambda(t) < 1, x(0) = x_0 \\
# \end{align}
# $$
# 
# where $\mathcal{J}_\text{barrier}$ is a barrier cost function and the last term is an $\ell_1$ penalty.

# In[ ]:


log = []
dif = 100
old_u = []
i = 0
for i in range(1500):
#while dif > 0.000005:
    g = dl(get_params(opt_state), x0, phik, k)
    opt_state = opt_update(i, g, opt_state)

    if (i+1) % 20 == 0:
        u = get_params(opt_state)
        log.append(fourier_ergodic_loss(u, x0, phik, k).copy())
        clear_output(wait=True)
        xf, tr = scan(f, x0, u)

        clear_output(wait=True)
        #dif = abs(np.average(old_u-u))
        plt.contourf(X, Y, phik_recon)
        #plt.colorbar()
        plt.scatter(tr[:,0],tr[:,1], c=5*sigmoid(5*u[:,2]), cmap = 'plasma')
        #plt.plot(tr[:,0], tr[:,1], 'r', linewidth='5')
        plt.axis('square')
#         plt.ylim(0,1)
#         plt.xlim(0,1)
        plt.pause(0.0001)
        i+=1
        plt.show()
        #print(dif)
    #old_u = u
    #i+=1
#print(dif)
#print(i)


# Find information map reconstruction based on chosen sensor measurements

# In[ ]:


ck = get_ck(tr, k)
ck_recon = np.dot(ck, vmap(fk_vmap, in_axes=(None, 0))(_s, k)).reshape(X.shape)
#ck_recon = ck_recon/ck_recon.sum()
N = 25
percent = 0.00000005
threshold = percent * np.amax(ck_recon)
print(np.amax(ck_recon))
#ck = get_ck_weighted(tr, k, u[:,2])

#plt.contour(X, Y, ck_recon)
color = len(u[:,2])*[0]
idx = sorted(range(len(u[:,2])), key = lambda sub: u[sub,2])[-N:]
print(idx)
for i in range(100):
  if i in idx:
    color[i] = 1
  else:
    color[i] = 0.3
  #if i % (100/N) == 0:
  #  color[i] = 1
  #print(tr[i,0])
  #print(ck_recon[int(tr[i,0]),int(tr[i,1])])
  #if ck_recon[int(tr[i,0]*50),int(tr[i,1]*50)] >= threshold:
  #  color[i] = 1
  #else:
  #  color[i] = 0
print(len(color))
#plt.scatter(tr[:,0],tr[:,1], c=color)#u[:,2])
#plt.axis('square')
color[-1] = 0
color = np.array(color)
ck = get_ck_weighted(tr, k, color)
ck_recon = np.dot(ck, vmap(fk_vmap, in_axes=(None, 0))(_s, k)).reshape(X.shape)
plt.contourf(X, Y, ck_recon)
plt.scatter(tr[:,0],tr[:,1], c=color, cmap = 'Reds')
plt.colorbar()
plt.axis('square')
plt.show()
#print(tr)

print(color.sum())


# In[ ]:


plt.plot(u[:,2])


# In[ ]:


plt.plot(color)


# In[ ]:


color = np.array(color)
ck = get_ck_weighted(tr, k, color)
ck_recon = np.dot(ck, vmap(fk_vmap, in_axes=(None, 0))(_s, k)).reshape(X.shape)
plt.contourf(X, Y, phik_recon)
plt.axis('square')


# In[ ]:


u = get_params(opt_state)
new_u = cat([np.reshape(u[:,0],(num_samples,1)), np.reshape(u[:,1],(num_samples,1))], axis=1)
color = color.reshape((100,1))
u = cat([new_u, color], axis=1)
print(fourier_ergodic_loss(u, x0, phik, k))


# In[ ]:


import math

MSE = np.square(np.subtract(phik_recon,ck_recon)).mean()

RMSE = math.sqrt(MSE)

print(RMSE)


# In[ ]:


'''
from math import inf

ck_recon = np.where(ck_recon<0, 0, ck_recon)
phik_recon = np.where(phik_recon<0, 0, phik_recon)
ck_recon = ck_recon/ck_recon.sum()
phik_recon = phik_recon/phik_recon.sum()
print(KLdivergence(phik_recon, ck_recon))
print(scipy.special.kl_div(phik_recon, ck_recon).shape)
kl = scipy.special.kl_div(phik_recon, ck_recon)
kl_avg = 0
c = 0
for i in range(50):
  for j in range(50):
    if kl[i,j] != inf:
      kl_avg += kl[i,j]
      c += 1
kl_avg = kl_avg/c
print(kl_avg)
print(np.sum(ck_recon))
print(np.sum(phik_recon))
'''


# In[ ]:


x1 = [0.5009582,  0.4638466,  0.42890435, 0.39627847, 0.36603352, 0.338169,
 0.31264037, 0.28937837, 0.26830527, 0.24934609, 0.23243527, 0.21751955,
 0.20455787, 0.19351958, 0.18438184, 0.17712665, 0.17173818, 0.16820066,
 0.1664968,  0.16660704, 0.16850941, 0.17218018, 0.177595,   0.18473077,
 0.19356775, 0.20409201, 0.21629784, 0.23018959, 0.2457827,  0.26310295,
 0.28218374, 0.30306005, 0.3257594,  0.35028937, 0.3766224,  0.4046801,
 0.4343183,  0.46531746, 0.49738,    0.5301372,  0.5631655,  0.59600997,
 0.62821096, 0.6593292,  0.68896574, 0.71677506, 0.74246985, 0.76582,
 0.7866476,  0.80481917, 0.82023793, 0.83283633, 0.8425699,  0.84941214,
 0.8533511,  0.8543867,  0.85252905, 0.84779805, 0.8402231,  0.82984394,
 0.8167125,  0.80089504, 0.7824756,  0.7615605,  0.7382837,  0.7128131,
 0.6853564,  0.6561665,  0.62554413, 0.59383625, 0.5614287,  0.52873284,
 0.4961658,  0.46412793, 0.43297943, 0.40302098, 0.37448075, 0.3475096,
 0.32218412, 0.29851577, 0.27646387, 0.25595,    0.2368718,  0.21911465,
 0.20256132, 0.18709862, 0.17262188, 0.1590374,  0.14626338, 0.13422982,
 0.12287768, 0.11215756, 0.1020283,  0.09245559, 0.08341053, 0.07486862,
 0.06680891, 0.05921401, 0.05207235, 0.04539538]

y1 = [0.288897,   0.27877545, 0.26968658, 0.26167005, 0.25475484, 0.24896075,
 0.24429998, 0.24077863, 0.238398,   0.23715562, 0.23704585, 0.2380604,
 0.24018843, 0.24341652, 0.24772853, 0.25310537, 0.25952464, 0.2669604,
 0.27538282, 0.28475782, 0.2950471,  0.3062079,  0.31819302, 0.330951,
 0.34442636, 0.35855988, 0.3732891,  0.38854897, 0.4042722,  0.4203901,
 0.43683293, 0.45353055, 0.4704128,  0.48740983, 0.50445235, 0.5214717,
 0.5384,     0.5551704,  0.5717175,  0.5879779,  0.60389113, 0.6194009,
 0.6344562,  0.64901227, 0.66303146, 0.6764833,  0.68934435, 0.70159775,
 0.7132322,  0.7242412,  0.734622,   0.74437505, 0.75350314, 0.7620108,
 0.76990414, 0.7771901,  0.78387654, 0.7899719,  0.7954851,  0.8004254,
 0.8048026,  0.8086266,  0.8119078,  0.8146572,  0.81688637, 0.8186078,
 0.8198354,  0.8205849,  0.8208744,  0.82072496, 0.8201614,  0.8192124,
 0.8179109,  0.8162941,  0.8144026,  0.81228,    0.8099716,  0.8075233,
 0.80498075, 0.80238813, 0.7997876,  0.79721856, 0.7947177,  0.7923187,
 0.79005206, 0.7879454,  0.7860237,  0.7843093,  0.7828225,  0.7815813,
 0.7806019,  0.7798992,  0.77948636, 0.77937573, 0.7795785,  0.7801051,
 0.78096545, 0.78216875, 0.78358686, 0.78514504]

x2 = [0.52273184, 0.5052034,  0.48748836, 0.46966305, 0.4518058,  0.43399552,
 0.4163109,  0.39882925, 0.38162556, 0.3647719,  0.34833667, 0.33238426,
 0.31697473, 0.30216363, 0.28800204, 0.27453673, 0.2618102,  0.24986109,
 0.23872435, 0.22843164, 0.21901163, 0.21049036, 0.20289148, 0.19623666,
 0.19054578, 0.18583724, 0.18212812, 0.17943436, 0.17777096, 0.17715196,
 0.1775906,  0.17909919, 0.18168913, 0.18537071, 0.19015293, 0.19604316,
 0.2030468,  0.2111667,  0.22040263, 0.23075047, 0.24220133, 0.2547405,
 0.26834637, 0.28298903, 0.2986291,  0.3152164,  0.3326887,  0.35097092,
 0.36997464, 0.3895981,  0.40972707, 0.43023643, 0.4509924,  0.47185573,
 0.49268517, 0.5133414,  0.5336909,  0.5536094,  0.5729847,  0.5917192,
 0.6097313,  0.6269558,  0.64334404, 0.6588631,  0.67349476, 0.68723387,
 0.7000868,  0.7120697,  0.72320694, 0.7335295,  0.7430734,  0.7518786,
 0.75998783, 0.7674456,  0.7742977,  0.7805903,  0.7863696,  0.79168165,
 0.79657173, 0.80108446, 0.80526346, 0.80915135, 0.81278956, 0.8162183,
 0.81947666, 0.82260245, 0.8256322,  0.82860136, 0.83154416, 0.83449376,
 0.83748233, 0.8405411,  0.84370047, 0.84698987, 0.850438,   0.8540728,
 0.85792154, 0.86201066, 0.8663654,  0.8710078 ]

y2 = [0.29427227, 0.28828633, 0.28210235, 0.27578047, 0.2693803,  0.2629602,
 0.25657678, 0.25028455, 0.24413553, 0.23817903, 0.2324616,  0.22702698,
 0.22191617, 0.21716757, 0.21281719, 0.2088988,  0.20544422, 0.2024835,
 0.20004521, 0.1981566,  0.19684379, 0.196132,   0.19604568, 0.19660859,
 0.19784397, 0.19977458, 0.20242274, 0.20581038, 0.20995905, 0.21488978,
 0.22062315, 0.22717907, 0.2345766,  0.2428338,  0.2519673,  0.2619921,
 0.27292076, 0.28476298, 0.29752475, 0.3112073,  0.32580596, 0.3413088,
 0.35769504, 0.37493336, 0.39298013, 0.41177765, 0.43125266, 0.45131534,
 0.4718588,  0.49275956, 0.513879,   0.535066,   0.55616057, 0.57699823,
 0.59741545, 0.61725456, 0.63636875, 0.65462667, 0.67191553, 0.6881433,
 0.70324004, 0.7171574,  0.7298682,  0.74136424, 0.7516546,  0.7607631,
 0.76872575, 0.7755886,  0.7814052,  0.78623503, 0.7901414,  0.79319006,
 0.795448,   0.7969825,  0.79786044, 0.7981476,  0.7979084,  0.7972057,
 0.7961004,  0.7946517,  0.7929169,  0.79095155, 0.7888095,  0.78654295,
 0.78420275, 0.78183854, 0.77949876, 0.77723104, 0.77508235, 0.773099,
 0.771327,   0.7698122,  0.7686003,  0.7677369,  0.7672679,  0.7672391,
 0.7676964,  0.76868534, 0.77025026, 0.7724284 ]

color2 = len(u[:,2])*[0]
for i in range(len(color)):
  #if i in idx:
  #  color[i] = 1
  if i % (100/N) == 0:
    color2[i] = 1

u = get_params(opt_state)
clear_output(wait=True)
xf, tr = scan(f, x0, u)
plt.contourf(X, Y, phik_recon)
#plt.colorbar()
plt.scatter(x2,y2, c='orange', alpha=0.5)
plt.scatter(tr[:,0],tr[:,1], c=color, cmap='Reds') #c=5*sigmoid(5*u[:,2]), cmap = 'plasma')
#plt.scatter(tr[:,0],tr[:,1], c=color2, cmap='plasma')
plt.colorbar()
#plt.plot(tr[:,0], tr[:,1], 'r', linewidth='5')
plt.axis('square')
#         plt.ylim(0,1)
#         plt.xlim(0,1)
plt.show()


# In[ ]:





# In[ ]:


u = get_params(opt_state)
xf, tr = scan(f, x0, u)
x = np.array(x)
y = np.array(y)
tr = np.stack((x,y),axis=1)
ck = get_ck_weighted(tr, k, color)
#ck = get_ck(traj, k)
ck_recon = np.dot(ck, vmap(fk_vmap, in_axes=(None, 0))(_s, k)).reshape(X.shape)
plt.contourf(X, Y, ck_recon)
plt.scatter(x,y, c='yellow', alpha=0.5) #cmap = 'plasma')
plt.colorbar()
plt.axis('square')
plt.show()


# In[ ]:


#plt.plot(u[:,2])


# In[ ]:


#temp = []
#for i in u[:,2]:
#  if i > 0.025:
#    temp.append(1)
#  else:
#    temp.append(0)

#plt.plot(temp)


# In[ ]:


#print(fourier_ergodic_loss(u, x0, phik, k))


# In[ ]:


#plt.contourf(X, Y, ck_recon)
#plt.axis('square')


# In[ ]:


#noise = onp.random.normal(0, .5, ck_recon.shape)
#new = ck_recon + noise
#plt.contourf(X, Y, new)
#plt.axis('square')
#print(ck_recon)
#print(noise)

