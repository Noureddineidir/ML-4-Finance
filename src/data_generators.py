import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import math 

from tensorflow import keras
from scipy.stats import norm
from numpy.polynomial.hermite import hermgauss

import matplotlib.pyplot as plt

import urllib
import os

class DPDEGenerator(keras.utils.Sequence):
    """ Create batches of random points for the network training. """

    def __init__(self, batch_size):
        """ Initialise the generator by saving the batch size. """
        self.batch_size = batch_size
      
    def __len__(self):
        """ Describes the number of points to create """
        return self.batch_size
    
    def __getitem__(self, idx):
        """ Get one batch of random points in the interior of the domain to 
        train the PDE residual and with initial time to train the initial value.
        """
        data_train_interior = np.random.uniform(
            normalised_min, normalised_max, [self.batch_size, dimension_total]) 

        t_train_initial = normalised_min * np.ones((self.batch_size, 1))
        s_and_p_train_initial = np.random.uniform(
            normalised_min, normalised_max,
            [self.batch_size, dimension_state + dimension_parameter])
        
        data_train_initial = np.concatenate(
            (t_train_initial, s_and_p_train_initial), axis=1)
        return ([data_train_interior, data_train_initial])

class DPDEGeneratorDelta(keras.utils.Sequence):
    """ Create batches of random points for the network training. Also compute the "exact" delta
     with respect to the first underlying after the option was priced with a reference pricing.
     """

    def __init__(self, batch_size):
        """ Initialise the generator by saving the batch size. """
        self.batch_size = batch_size
      
    def __len__(self):
        """ Describes the number of points to create """
        return self.batch_size
    
    def __getitem__(self, idx):
        """ Get one batch of random points in the interior of the domain to 
        train the PDE residual and with initial time to train the initial value.
        """
        data_train_interior = np.random.uniform(
            normalised_min, normalised_max, [self.batch_size, dimension_total]) 

        t_train_initial = normalised_min * np.ones((self.batch_size, 1))
        s_and_p_train_initial = np.random.uniform(
            normalised_min, normalised_max,
            [self.batch_size, dimension_state + dimension_parameter])
        
        data_train_initial = np.concatenate(
            (t_train_initial, s_and_p_train_initial), axis=1)
        
        riskfree_rate_interior = transform_to_riskfree_rate(data_train_interior[:, 3:4])
        volatility1_interior = transform_to_volatility(data_train_interior[:, 4:5])
        volatility2_interior = transform_to_volatility(data_train_interior[:, 5:6])
        correlation_interior = transform_to_correlation(data_train_interior[:, 6:7])
        s1_interior = tf.math.exp(transform_to_logprice(data_train_interior[:, 1:2]))
        s2_interior = tf.math.exp(transform_to_logprice(data_train_interior[:, 2:3]))
        t_interior = transform_to_time(data_train_interior[:, 0:1])
        ds = 0.01 

                
        exact_solution_interior = [exact_solution(t=t_interior[i], s1=s1_interior[i], s2=s2_interior[i], riskfree_rate=riskfree_rate_interior[i], 
               volatility1=volatility1_interior[i], volatility2=volatility2_interior[i], correlation=correlation_interior[i]) for i in range(0,data_train_interior[:,0:1].shape[0])]
        exact_solution_interior= np.reshape(exact_solution_interior, (len(exact_solution_interior),1))

        exact_solution_interior_shifted = [exact_solution(t=t_interior[i], s1= (s1_interior[i] + ds), s2=s2_interior[i], riskfree_rate=riskfree_rate_interior[i], 
               volatility1=volatility1_interior[i], volatility2=volatility2_interior[i], correlation=correlation_interior[i]) for i in range(0,data_train_interior[:,0:1].shape[0])]
        exact_solution_interior_shifted = np.reshape(exact_solution_interior_shifted, (len(exact_solution_interior),1))

        exact_delta_interior = (exact_solution_interior_shifted - exact_solution_interior)/ds

        return [data_train_interior, data_train_initial, exact_delta_interior]

