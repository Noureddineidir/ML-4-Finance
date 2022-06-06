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

from src.highway_network import HighwayLayer
from src.data_generators import DPDEGenerator, DPDEGeneratorDelta
from src.train_models import transform_ab_to_cd, transform_to_logprice, transform_to_time, normalise_logprice, normalise_time, transform_to_riskfree_rate, transform_to_volatility, transform_to_correlation, normalise_riskfree_rate, normalise_correlation
from src. train_models import DPDEModel, DPDEModelGreeks
from src.evaluation_function import decompose_covariance_matrix, one_dimensional_exact_solution, exact_solution, get_random_points_of_interest, get_points_for_plot_at_fixed_time, localisation, Implied_Volatility, Implied_Volatility_bis

np.random.seed(42)

load_model = True
nr_samples_surface_plot = 21
nr_samples_scatter_plot = 1000
nr_samples_error_calculation = 10000

np.random.seed(42)

# Model parameters. Re-train model after any changes.
s_min_interest = 25
s_max_interest = 150
t_min_interest = 0.5
t_max_interest = 4.

riskfree_rate_min = 0.1
riskfree_rate_max = 0.3
riskfree_rate_eval = 0.2

volatility_min = 0.1
volatility_max = 0.3
volatility1_eval = 0.1
volatility2_eval = 0.3

correlation_min = 0.2
correlation_max = 0.8
correlation_eval = 0.5

strike_price = 100.


#Internal Parameters and normalization 
dimension_state = 2
dimension_parameter = 4
dimension_total = 1 + dimension_state + dimension_parameter

t_min = 0.
t_max = t_max_interest
s_max = strike_price * (1 + 3*volatility_max*t_max)
x_max = np.log(s_max)
x_min = 2*np.log(strike_price) - x_max

normalised_max = 1
normalised_min = -1


nr_nodes_per_layer = 90
initial_learning_rate =  0.001
localisation_parameter = 1/10

n_train = 1000
nr_epochs = 601


t_min_interest_normalised = normalise_time(t_min_interest)
t_max_interest_normalised = normalise_time(t_max_interest)

diff_dx = (normalised_max-normalised_min) / (x_max-x_min) 
diff_dt = (normalised_max-normalised_min) / (t_max-t_min)

riskfree_rate_eval_normalised = normalise_riskfree_rate(riskfree_rate_eval)
volatility1_eval_normalised = normalise_volatility(volatility1_eval)
volatility2_eval_normalised = normalise_volatility(volatility2_eval)
correlation_eval_normalised = normalise_correlation(correlation_eval)


'''
data = DPDEGeneratorDelta(n_train)
_, _, delta_interior = data[0]

print("Biggest delta : {}".format(np.max(delta_interior))) #lower than 0.5, greater than 0 sounds ok
print("Some deltas : {}".format(delta_interior[:5]))
print("Lowest delta : {}".format(np.min(delta_interior)))
'''

'''
s1_plot_mesh, s2_plot_mesh, x_plot_normalised = get_points_for_plot_at_fixed_time()

test_solution = exact_solution(t=4., s1=100., s2=100., riskfree_rate=0.2, 
               volatility1=0.1, volatility2=0.3, correlation=0.5)
assert(np.abs(test_solution - 55.096796282039364) < 1e-10)
'''

if __name__ == "__main__":
    %%time
#TRAINING ORIGINAL MODEL WITH CLASSIC PDE

    if load_model:
        # Load model from local folder. If it is not availabe, download it.
        os.makedirs('model/variables', exist_ok=True)
        url_base = 'https://github.com/LWunderlich/DeepPDE/raw/main/TwoAssetsExample/'
        filename = 'model/saved_model.pb'
        if not os.path.isfile(filename):
            urllib.request.urlretrieve(url_base + filename, filename)

        filename = 'model/variables/variables.data-00000-of-00001'
        if not os.path.isfile(filename):
            urllib.request.urlretrieve(url_base + filename, filename)

        filename = 'model/variables/variables.index'
        if not os.path.isfile(filename):
            urllib.request.urlretrieve(url_base + filename, filename)

        model = keras.models.load_model('model')   
    else:
        # Create and train model from scratch. 
        inputs = keras.Input(shape=(dimension_total,))
        outputs = create_network(inputs)
        model = DPDEModel(inputs=inputs, outputs=outputs)
        batch_generator = DPDEGenerator(n_train)
        model.compile(optimizer=tf.keras.optimizers.Adam(initial_learning_rate))
        callback = tf.keras.callbacks.EarlyStopping(
            'loss', patience=50, restore_best_weights=True)

        model.fit(x=batch_generator, epochs=nr_epochs, steps_per_epoch=10,
                              callbacks=[callback])

    # plot code are available on notebook
    