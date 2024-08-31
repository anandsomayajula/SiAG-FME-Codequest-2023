# -*- coding: utf-8 -*-
"""
Created on Fri Nov  21 17:26:27 2023

@author: jaimungal
"""

import numpy as np
from collections import deque
import copy

from tqdm import tqdm

class amm():

    def __init__(self, Rx, Ry,  phi):
        """
        instantiate the class

        Parameters
        ----------
        Rx : array (K,)
            initial reservese of token-X in each pool
        Ry : array (K,)
            initial reservese of token-Y in each pool
        phi : array (K,)
            the pool fee

        Returns
        -------
        None.

        """

        assert (len(Rx) == len(Ry)) & (len(Ry)==len(phi)), "length of Rx, Ry, and phi must be the same."

        self.Rx = 1*Rx
        self.Ry = 1*Ry
        self.phi = 1*phi
        self.N = len(self.Rx)

        # number of LP tokens for each pool
        self.L = np.sqrt(self.Rx*self.Ry)

        # the trader begins with no LP tokens
        self.l = np.zeros(len(self.L))

    def swap_x_to_y(self, x, quote=False):
        """
        swap token-X for token-Y across all pools simulataneously

        Parameters
        ----------
        x : array (K,)
            the amount of token-X to swap in each AMM pool.
        quote: bool, optional
            deafult is False.
            If False, then pool states are updated.
            If True, pool states are not updated.

        Returns
        -------
        y : array (K,)
            the amount of token-Y you receive from each pool.

        """

        # ********************
        #     fill in code
        

        y= x * (1-self.phi)*self.Ry/(self.Rx+(1-self.phi)*x)
        if quote==False:
            self.Rx = self.Rx + x
            self.Ry = self.Ry-y
        # ********************

        return y

    def swap_y_to_x(self, y, quote=False):
        """
        swap token-Y for token-X across all pools simulataneously

        Parameters
        ----------
        y : array (K,)
            the amount of token-Y to swap in each AMM pool.
        quote: bool, optional
            deafult is False.
            If False, then pool states are updated.
            If True, pool states are not updated.

        Returns
        -------
        x : array (K,)
            the amount of token-X you receive from each pool.

        """

        # ********************
        #     fill in code
        x= y * (1-self.phi)*self.Rx/(self.Ry+(1-self.phi)*y)
        if quote==False:
            self.Rx = self.Rx - x
            self.Ry = self.Ry+y
        # ********************

        return x

    def mint(self, x, y):
        """
        mint LP tokens across all pools

        Parameters
        ----------
        x : array (K,)
            amount of token-X submitted to each pool.
        y : array (K,)
            amount of token-Y submitted to each pool.

        Returns
        -------
        l : array (K,)
            The amount of LP tokens you receive from each pool.

        """

        for k in range(len(self.Rx)):
            assert np.abs(((x[k]/y[k])-self.Rx[k]/self.Ry[k])) < 1e-9, "pool " + str(k) + " has incorrect submission of tokens"

        # ********************
        #     fill in code
        
        self.l =  self.l+(x*self.L)/self.Rx
        
        self.Rx  = self.Rx+x
        self.Ry = self.Ry + y
        self.L = self.L + self.l
        # ********************

        return self.l

    def swap_and_mint(self, x):
        """
        a method that determines the correct amount of y for each x within the corresponding pool
        to swap and then mint tokens with the reamaing x and the y you received

        Parameters
        ----------
        x : array (K,)
            amount of token-X you have for each pool.

        Returns
        -------
        l : array (K,)
            The amount of LP tokens you receive from each pool.

        """

        # ********************
        #     fill in code

        theta = 1 + ((2 - self.phi) * self.Rx / (2 * (1 - self.phi) * x)) * (1 - np.sqrt(1 + 4 * (x / self.Rx) * ((1 - self.phi) / ((2 - self.phi) ** 2))))
        x_swap = (1-theta)*x
#         y = x * (1-self.phi)*self.Ry/(self.Rx+(1-self.phi)*x)
        y = self.swap_x_to_y(x_swap)
        self.l = self.mint(theta*x, y)
        
        # ********************

        return self.l

    def burn_and_swap(self, l):
        """
        a method that burns your LP tokens, then swaps y to x and returns only x

        Parameters
        ----------
        l : array (K,)
            amount of LP tokens you have for each pool.

        Returns
        -------
        x : array (K,)
            The amount of token-x you receive at the end.

        """

        # ********************
        #     fill in code
        # x = l*self.Rx/self.L 
        # y = l*self.Ry/self.L 
        # self.Rx = self.Rx - x
        # self.Ry = self.Ry - y
        # self.L = self.L - l
        x,y = self.burn(l)
        tot = np.sum(y)
        y = np.full(len(x), tot)
        
        vals = self.swap_y_to_x(y, quote=True)
        
        pools_max = 0
        pool_track = 0
        for i in range(len(vals)):
            if vals[i]>pools_max:
                pools_max = vals[i]
                pool_track = i
    
        y = np.zeros(len(y))
        y[pool_track]= tot
        
        vals_act = self.swap_y_to_x(y, quote=False)
        total_x = np.sum(vals_act)+np.sum(x)    
                
             
               
        
        # ********************

        return total_x

    def burn(self, l):
        """
        burn LP tokens across all pools

        Parameters
        ----------
        l : array (K,)
            amount of LP tokens to burn

        Returns
        -------
        x : array (K,)
            The amount of token-X received across
        y : array (K,)
            The amount of token-Y received across

        """

        for k in range(len(self.L)):
            assert l[k] <= self.l[k], "you have insufficient LP tokens"

        # ********************
        #     fill in code
        x = l*self.Rx/self.L 
        y = l*self.Ry/self.L 
        self.Rx = self.Rx - x
        self.Ry = self.Ry - y
        self.L = self.L - l
        self.l = self.l-l
        # ********************

        return x, y

    def simulate(self, kappa, p, sigma, T=1, batch_size=256):
        """
        Simulate trajectories of all AMM pools simultanesouly.

        Parameters
        ----------
        kappa : array (K+1,)
            rate of arrival of swap events X->Y and Y->X.
            kappa[0,:] is for a common event across all pools
        p : array (K+1,2)
            probability of swap X to Y event conditional on an event arriving.
            p[0,:] is for a common event across all pools
        sigma : array (K+1,2)
            standard deviation of log volume of swap events.
            sigma[0,:] is for a common event across all pools
        T : float, optional: default is 1.
            The amount of (calendar) time to simulate over
        batch_size : int, optional, default is 256.
            the number of paths to generate.

        Returns
        -------
        pools : deque, len=batch_size
            Each element of the list is the pool state at the end of the simulation for that scenario
        Rx_t : deque, len= batch_size
            Each element of the list contains a sequence of arrays.
            Each array shows the reserves in token-X for all AMM pools after each transaction.
        Ry_t : deque, len=batch_size
            Each element of the list contains a sequence of arrays.
            Each array shows the reserves in token-Y for all AMM pools after each transaction.
        v_t : deque, len=batch_size
            Each element of the list contains a sequence of arrays.
            Each array shows the volumes of the transaction sent to the various AMM pools -- the transaction is
            either a swap X for Y or swap Y for X for a single pool, or across all pools at once
        event_type_t : deque, len=batch_size
            Each element of the list contains a sequence of arrays.
            Each array shows the event type. event_type=0 if it is a swap sent to all pools simultaneously,
            otherwise, the swap was sent to pool event_type
        event_direction_t : deque, len=batch_size
            Each element of the list contains a sequence of arrays.
            Each array shows the directi of the swap.
            event_direction=0 if swap X -> Y
            event_direction=1 if swap Y -> X

        """

        # used for generating Poisson random variables for all events
        sum_kappa = np.sum(kappa)

        # used for thinning the Poisson process
        pi = kappa/sum_kappa

        # store the list of reservese generated by the simulation
        def make_list(batch_size):
          x = deque(maxlen=batch_size)
          x = [None] * batch_size
          return x

        Rx_t = make_list(batch_size)
        Ry_t = make_list(batch_size)
        v_t = make_list(batch_size)
        event_type_t = make_list(batch_size)
        event_direction_t = make_list(batch_size)
        pools = make_list(batch_size)

        for k in tqdm(range(batch_size)):

            N = np.random.poisson(lam = sum_kappa*T)

            Rx = np.zeros((N,len(self.Rx)))
            Ry = np.zeros((N,len(self.Rx)))
            v = np.zeros((N,len(self.Rx)))
            event_type = np.zeros(N, int)
            event_direction = np.zeros(N, int)

            pools[k] = copy.deepcopy(self)

            for j in range(N):

                # generate the type of event associated with each event
                event_type[j] = np.random.choice(len(kappa), p=pi)

                # the direction of the swap 0 = x-> y, 1 = y -> x
                event_direction[j] = int(np.random.rand()< p[event_type[j]])

                if event_direction[j] == 0:
                    mu = np.zeros(len(pools[k].Rx)) # deposit X and get Y
                else:
                    mu = np.log(pools[k].Ry/pools[k].Rx) # deposit Y and get X

                if event_type[j] == 0:
                    # there is a swap across all venues
                    v[j,:] = np.exp((mu-0.5*sigma[0]**2) + sigma[0]*np.random.randn() )

                else:

                    # there is a swap only on a specific venue
                    v[j,:] = np.zeros(len(pools[k].Rx))
                    mu = mu[event_type[j]-1]
                    v[j,event_type[j]-1] = np.exp((mu-0.5*sigma[event_type[j]]**2) \
                                                   + sigma[event_type[j]]*np.random.randn() )

                if event_direction[j] == 0:
                    pools[k].swap_x_to_y(v[j,:]) # submit X and get Y
                else:
                    pools[k].swap_y_to_x(v[j,:]) # submit Y and get X

                Rx[j,:] = 1*pools[k].Rx
                Ry[j,:] = 1*pools[k].Ry

            Rx_t[k] = Rx
            Ry_t[k] = Ry
            v_t[k] = v
            event_type_t[k] = event_type
            event_direction_t[k] = event_direction

        return pools, Rx_t, Ry_t, v_t, event_type_t, event_direction_t
