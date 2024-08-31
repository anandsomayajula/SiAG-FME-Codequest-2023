import random
from amm import amm
import numpy as np

import matplotlib.pyplot as plt
#plt.style.use('paper.mplstyle')

import seaborn as sns
sns.set_theme(style="ticks")

import statsmodels.api as sm
import matplotlib.ticker as mtick
from params import params



def objective(xs_0):
    """ Fix the seed """
    np.random.seed(params['seed'])
    
    """ Initialise the pools """
    Rx0   = params['Rx0']
    Ry0   = params['Ry0']
    phi   = params['phi']
    
    
    
    pools = amm(Rx=Rx0, Ry=Ry0, phi=phi)

    """ Swap and mint """
    l    = pools.swap_and_mint(xs_0)
    """ Simulate 1000 paths of trading in the pools """
    batch_size = params['batch_size']
    T          = params['T']
    kappa      = params['kappa']
    p          = params['p']
    sigma      = params['sigma']

    end_pools, Rx_t, Ry_t, v_t, event_type_t, event_direction_t =\
            pools.simulate( kappa = kappa, p = p, sigma = sigma, T = T, batch_size = batch_size)

    #print('Reserves in asset X for scenario 0:', end_pools[0].Rx)
    #print('Reserves in asset Y for scenario 0:', end_pools[0].Ry)
    x_T = np.zeros(batch_size)
    for k in range(batch_size):
        x_T[k] = np.sum(end_pools[k].burn_and_swap(l))

    ## CVAR:
    x_0      = np.sum(xs_0)
    log_ret  = np.log(x_T) - np.log(x_0)
    r_t = np.mean(log_ret)/T*100
#     print('Average performance     :', r_t)
#     print('Std. Dev. of performance:', np.std(log_ret)/np.sqrt(T)*100)


    # compute cvar
    alpha = params['alpha'] 
    qtl   = -np.quantile(log_ret, 1-alpha)
    cvar  = np.mean(-log_ret[-log_ret>=qtl])
#     print(cvar)
    return cvar, r_t




def derivatives(xs_0):
     #perturbance
     epsilon = 1e-3
     gradients=[]
 
     
     perturbations=np.zeros(params['N_pools'])
       
     for i in range(params['N_pools']):

# Perturb each element in the array of weights and calculate the centered derivative with respect to the objective function. The objective is a combination of cvar and performance. The aim here is to minimize it -> Smaller cvar and larger performance will reduce the objective.
            
         perturbations[i] = epsilon
         cv_1, r_t1= objective(xs_0+perturbations)
         comb1 = cv_1/r_t1
         cv_2, r_t2 =  objective(xs_0-perturbations)
         comb2 = cv_2/r_t2
         gradients_i = (comb1-comb2)/(2*epsilon)
         gradients.append(gradients_i)
         perturbations[i]=0
     return np.array(gradients)
     
def gradient_descent(xs_0):
    learning_rate = 100
    prev_cvar=0
    prev_perf=1
    
    #Stopping Condition
    tol = 0.005
    
    #Perturbation to ensure all the weights are at least non-negative
    noise = 1e-2
    while True:
        gradients = derivatives(xs_0)
       
        #Update Step
        xs_0 = xs_0-learning_rate*gradients
               
        for j in range(params['N_pools']):
            #If any of the weights is less than 0.02, increase it to that threshold.
            if xs_0[j] <noise:
                xs_0[j] = noise
            
        # Scale the weights
        scale = np.sum(xs_0)
        xs_0 = xs_0*(10/scale)
        
        #Stopping condition to check for convergence if gradient descent method.
        cvar, perf = objective(xs_0)
        if abs(cvar/perf - prev_cvar/prev_perf)<tol:
            return xs_0
        else:
            prev_cvar = cvar
            prev_perf = perf
    return xs_0


#Initialize weights to a random number between 0 and 1. Scale weights to sum to 10. 

xs_0= np.zeros(params['N_pools'])
for i in range(params['N_pools']):
    xs_0[i]=random.random()
    
scale = np.sum(xs_0)

xs_0 = xs_0 * (10/scale)
best_x = gradient_descent(xs_0)

print(best_x)
print(objective(best_x))

theta = best_x/10
print(theta)

