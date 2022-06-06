import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import math 

from tensorflow import keras
from scipy.stats import norm
import scipy.stats as stats

from numpy.polynomial.hermite import hermgauss

import matplotlib.pyplot as plt

import urllib
import os

def decompose_covariance_matrix(t, volatility1, volatility2, correlation):
    """ Decompose covariance matrix as in Lemma 3.1 of Bayer et. al (2018). """
    sigma_det = (1-correlation**2) * volatility1**2 * volatility2**2
    sigma_sum = (volatility1**2 + volatility2**2 
                  - 2*correlation*volatility1*volatility2)

    ev1 = volatility1**2 - correlation*volatility1*volatility2
    ev2 = -(volatility2**2 - correlation*volatility1*volatility2)
    ev_norm = np.sqrt(ev1**2 + ev2**2)

    eigenvalue = volatility1**2 + volatility2**2 - 2*sigma_det/sigma_sum

    v_mat = np.array([ev1, ev2]) / ev_norm
    d = t * np.array([sigma_det/sigma_sum, eigenvalue])
    return d, v_mat

def one_dimensional_exact_solution(
        t, s, riskfree_rate, volatility, strike_price):
    """ Standard Black-Scholes formula """

    d1 = (1 / (volatility*np.sqrt(t))) * (
            np.log(s/strike_price) 
            + (riskfree_rate + volatility**2/2.) * t
        )
    d2 = d1 - volatility*np.sqrt(t)
    return (norm.cdf(d1) * s 
            - norm.cdf(d2) * strike_price * np.exp(-riskfree_rate*t))

def exact_solution(
    t, s1, s2, riskfree_rate, volatility1, volatility2, correlation):
    """ Compute the option price of a European basket call option. """
    if t == 0:
        return np.maximum(0.5*(s1+s2) - strike_price, 0)

    d, v = decompose_covariance_matrix(
        t, volatility1, volatility2, correlation)
    
    beta = [0.5 * s1 * np.exp(-0.5*t*volatility1**2),
            0.5 * s2 * np.exp(-0.5*t*volatility2**2)]
    integration_points, integration_weights = hermgauss(33)

    # Transform points and weights
    integration_points = np.sqrt(2*d[1]) * integration_points.reshape(-1, 1)
    integration_weights = integration_weights.reshape(1, -1) / np.sqrt(np.pi)

    h_z = (beta[0] * np.exp(v[0]*integration_points)
           + beta[1] * np.exp(v[1]*integration_points))

    evaluation_at_integration_points = one_dimensional_exact_solution(
        t=1, s=h_z * np.exp(0.5*d[0]), 
        strike_price=np.exp(-riskfree_rate * t) * strike_price, 
        volatility=np.sqrt(d[0]), riskfree_rate=0.
        )
    
    solution = np.matmul(integration_weights, evaluation_at_integration_points)
    
    return solution[0, 0]


def get_random_points_of_interest(nr_samples, 
                    t_min_interest=t_min_interest,
                    t_max_interest=t_max_interest,
                    s_min_interest=s_min_interest,
                    s_max_interest=s_max_interest,
                    parameter_min_interest_normalised=normalised_min,
                    parameter_max_interest_normalised=normalised_max):
    """ Get a number of random points within the defined domain of interest. """
    t_sample = np.random.uniform(t_min_interest, t_max_interest, 
                                 [nr_samples, 1])
    t_sample_normalised = normalise_time(t_sample)

    s_sample = np.random.uniform(
        s_min_interest, s_max_interest, [nr_samples, dimension_state])
    s1_sample = s_sample[:, 0:1]
    s2_sample = s_sample[:, 1:2]
    x_sample_normalised = normalise_logprice(np.log(s_sample))

    parameter_sample_normalised = np.random.uniform(
        normalised_min, normalised_max, [nr_samples, dimension_parameter])
    data_normalised = np.concatenate(
        (t_sample_normalised, x_sample_normalised, parameter_sample_normalised),
        axis=1
        )

    riskfree_rate_sample = transform_to_riskfree_rate(
        parameter_sample_normalised[:, 0])
    volatility1_sample = transform_to_volatility(
        parameter_sample_normalised[:, 1])
    volatility2_sample = transform_to_volatility(
        parameter_sample_normalised[:, 2])
    correlation_sample = transform_to_correlation(
        parameter_sample_normalised[:, 3])
    
    return data_normalised, t_sample.reshape(-1), s1_sample.reshape(-1), \
            s2_sample.reshape(-1), riskfree_rate_sample, volatility1_sample, \
            volatility2_sample, correlation_sample


def get_points_for_plot_at_fixed_time(t_fixed=t_max,
                s_min_interest=s_min_interest, s_max_interest=s_max_interest,
                riskfree_rate_fixed=riskfree_rate_eval,
                volatility1_fixed=volatility1_eval,
                volatility2_fixed=volatility2_eval,
                correlation_fixed=correlation_eval,
                n_plot=nr_samples_surface_plot):
    """ Get the spacial and normalised values for surface plots 
    at fixed time and parameter, varying both asset prices. 
    """
    s1_plot = np.linspace(s_min_interest, s_max_interest, n_plot).reshape(-1,1)
    s2_plot = np.linspace(s_min_interest, s_max_interest, n_plot).reshape(-1,1)
    [s1_plot_mesh, s2_plot_mesh] = np.meshgrid(s1_plot, s2_plot, indexing='ij')

    x1_plot_mesh_normalised = normalise_logprice(
        np.log(s1_plot_mesh)).reshape(-1,1)

    x2_plot_mesh_normalised = normalise_logprice(
        np.log(s2_plot_mesh)).reshape(-1,1)

    t_mesh = t_fixed  * np.ones((n_plot**2, 1))
    t_mesh_normalised = normalise_time(t_mesh)

    parameter1_mesh_normalised = (normalise_riskfree_rate(riskfree_rate_fixed) 
                                                      * np.ones((n_plot**2, 1)))
    parameter2_mesh_normalised = (normalise_volatility(volatility1_fixed) 
                                                      * np.ones((n_plot**2, 1)))
    parameter3_mesh_normalised = (normalise_volatility(volatility2_fixed) 
                                                      * np.ones((n_plot**2, 1)))
    parameter4_mesh_normalised = (normalise_correlation(correlation_fixed) 
                                                      * np.ones((n_plot**2, 1)))

    x_plot_normalised = np.concatenate((t_mesh_normalised,
                                        x1_plot_mesh_normalised,
                                        x2_plot_mesh_normalised,
                                        parameter1_mesh_normalised, 
                                        parameter2_mesh_normalised,
                                        parameter3_mesh_normalised, 
                                        parameter4_mesh_normalised), axis=1)

    
    return s1_plot_mesh, s2_plot_mesh, x_plot_normalised

def localisation(t, s1, s2, riskfree_rate=riskfree_rate_eval):
    """ Return the value of the localisation used in the network. """
    return 1/localisation_parameter * np.log(1 +
                    np.exp(localisation_parameter * (
                        0.5*(s1+s2) - np.exp(-riskfree_rate*t)*strike_price))
                    )

# Using the generalization of Implied Vol for 2 or more assets

import scipy.stats as stats

def BS_Call(S,T,r,sigma,K):
  d_1 = (np.log(S/K) + (r+sigma**2/2)*T)/(sigma*np.sqrt(T))
  d_2 = d_1 - sigma * np.sqrt(T)
  return(S*stats.norm.cdf(d_1)-K*np.exp(-r*T)*stats.norm.cdf(d_2))

def BS_Vega_Call(S,K,T,r,sigma):
  d_1 = (np.log(S/K) + (r+sigma**2/2)*T)/(sigma*np.sqrt(T))
  return S * np.sqrt(T)*stats.norm.pdf(d_1)

# To compute the implied_volatility, we use Newton-Raphson method
def Implied_Volatility(Price,S,K,r,T):
  try:
    Implied_vol = 0.5
    Target = 10**(-4)
    Keep_computing = True
    while Keep_computing:
      aux = Implied_vol
      Implied_vol = Implied_vol - (BS_Call(S,T,r,Implied_vol,K) - Price)/BS_Vega_Call(S,K,T,r,Implied_vol)
      if abs(BS_Call(S,T,r,Implied_vol,K) - Price) < Target:
        Keep_computing = False
    return(Implied_vol)
  except:
    return(Implied_Volatility_bis(Price,S,K,r,T))

def Implied_Volatility_bis(Price,S,K,r,T):
  Implied_vol = 0.9
  for i in range(2,8):
    if BS_Call(S,T,r,Implied_vol,K) - Price > 0:
      while BS_Call(S,T,r,Implied_vol,K) - Price > 0:
        Implied_vol = Implied_vol - 10**(-i)
    else:
      while BS_Call(S,T,r,Implied_vol,K) - Price < 0:
        Implied_vol = Implied_vol + 10**(-i)
  if Implied_vol < 0.05:
    print(Implied_vol,Price,BS_Call(S,T,r,Implied_vol,K),S,K,r,T)
  return(Implied_vol)
