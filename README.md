# ML-4-Finance
ENSAE Course

#ML-4-Finance-


This repo covers the Machine Learning for finance course assignement provided at ENSAE. A report and a notebook were written for the project. 

---------------------------------------------------------------------------------------
## Topic

The aim this course was to present both traditional and complex ML methods and their applications in the financial world. Part of the final evaluation consists in the reading, presentation and implementation of a financial paper.

 ### Original Paper
 [The Deep Parametric PDE Method: Application to Option Pricing](https://arxiv.org/abs/2012.06211) (available in this repo) deals with solving partial derivatives equations (PDEs) based on a neural network approach. The authors propose the deep parametric PDE method to solve high-dimensional parametric partial differential equations. A single neural network approximates the solution of a whole family of PDEs after being trained without the need of sample solutions. As a practical application, they compute option prices in the multivariate Black-Scholes model. After a single training phase, the prices for different time, state and model parameters are available in milliseconds.
 
 
 ### Notebook 
 In our work, we try to replicate the model presented by the [authors](https://github.com/LWunderlich/DeepPDE). Moreover, we propose to evaluate the model performance from the point of view of implied volatility. Our second contribution consists in modifying the original loss in order to give a better approximation of the greeks, particularly the Delta (price option derivative w.r.t. the underlying price). Most of our code can be found in our [Deep_PDEs_Greeks.ipynb](https://github.com/Noureddineidir/ML-4-Finance/blob/6b0ef72fb3036fcb2c0ef78039a52764665c20af/Deep_PDEs_Greeks.ipynb) and we saved our model's weight in this [folder](https://github.com/Noureddineidir/ML-4-Finance/tree/main/checkpoint) of the repo for people wanting to try and improve it.
 
The notebook is built in the following way : 
 * First part introduces the multivariate Black-Scholes framework and its main parameters (number of underlyings, volatilty and underlyings prices range etc.) as well as useful functions for both normalizing the input to feed the deep Neural Network and functions for evaluating our network (reference pricer) and visualize the errors.

* Second part proposes to build both the authors networks and our network with modified loss based on user's parameters. We also reproduce the authors charts for evaluating the model errors.

* Finally, last part goes further into the model evaluation by comparing both NNs sensitivty with respect to the underlying price and from a point a view of implied volatility.

### Report

The model presentation and our main results are fully available in our [final report](https://github.com/Noureddineidir/ML-4-Finance/blob/6b0ef72fb3036fcb2c0ef78039a52764665c20af/Rapport_OULID_SCHAEFFER_VIALARD.pdf) (in french only). 
 

---------------------------------------------------------------------------
