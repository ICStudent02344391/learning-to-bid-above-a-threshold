# Script to save the results of an important number of simulations of the sequential halving algorithm
"""
This script is separated from the other because the computation time is high and it is better 
to save the results once computed.
"""

# Module importation
import numpy as np
import numpy.random as rndm
import pickle as pkl
import sys
import os
sys.path.append(os.path.abspath("../src"))
import helper_functions as func
import algorithms

sigma = 0.02 # keep same sigma
delta = [0.5, 0.8, 0.99, 0.999] # delta values tested
rndm.seed(123) 
n_iter = 1000 # simulation number
partition = np.linspace(0, 1, 31)
taus = [0.3, 0.5, 0.7] # tau tested
prob_delta = {}
for test, tau in enumerate(taus):
    opti_arm = np.where(partition == func.get_optimal_mu(partition, 
                                                         tau, 
                                                         sigma,
                                                         unif_range = True)[1][0])[0][0]
    prob_delta[f"test_bernstein_{test}"] = {"tau": tau, 
                                      "sigma": sigma,
                                      "deltas": delta,
                                      "success_rates": np.zeros((len(delta), n_iter))}
    prob_delta[f"test_hoeffding_{test}"] = {"tau": tau, 
                                      "sigma": sigma,
                                      "deltas": delta,
                                      "success_rates": np.zeros((len(delta), n_iter))}

    for i, d in enumerate(delta):
        for iter in range(n_iter):
            arm_selected_bern = algorithms.sequential_halving_algorithm(tau, 
                                                                   sigma, 
                                                                   partition, 
                                                                   delta = d)[-2]
            arm_selected_hoeff = algorithms.sequential_halving_algorithm(tau, 
                                                                   sigma, 
                                                                   partition, 
                                                                   delta = d,
                                                                   bernstein = False)[-2]
            prob_delta[f"test_bernstein_{test}"]["success_rates"][i, iter] = (arm_selected_bern == opti_arm)
            prob_delta[f"test_hoeffding_{test}"]["success_rates"][i, iter] = (arm_selected_hoeff == opti_arm)

with open("results/success_rates_tau.pkl", "wb") as file:
	pkl.dump(prob_delta, file) 

# Test of different sigmas
tau = 0.5 # keep same tau
partition = np.linspace(0, 1, 31)
sigmas = [0.01, 0.02, 0.03] # sigma tested
prob_delta = {}
for test, sigma in enumerate(sigmas):
    opti_arm = np.where(partition == func.get_optimal_mu(partition, 
                                                         tau, 
                                                         sigma,
                                                         unif_range = True)[1][0])[0][0]
    prob_delta[f"test_bernstein_{test}"] = {"tau": tau, 
                                      "sigma": sigma,
                                      "deltas": delta,
                                      "success_rates": np.zeros((len(delta), n_iter))}
    prob_delta[f"test_hoeffding_{test}"] = {"tau": tau, 
                                      "sigma": sigma,
                                      "deltas": delta,
                                      "success_rates": np.zeros((len(delta), n_iter))}

    for i, d in enumerate(delta):
        for iter in range(n_iter):
            arm_selected_bern = algorithms.sequential_halving_algorithm(tau, 
                                                                   sigma, 
                                                                   partition, 
                                                                   delta = d)[-2]
            arm_selected_hoeff = algorithms.sequential_halving_algorithm(tau, 
                                                                   sigma, 
                                                                   partition, 
                                                                   delta = d,
                                                                   bernstein = False)[-2]
            prob_delta[f"test_bernstein_{test}"]["success_rates"][i, iter] = (arm_selected_bern == opti_arm)
            prob_delta[f"test_hoeffding_{test}"]["success_rates"][i, iter] = (arm_selected_hoeff == opti_arm)

with open("results/success_rates_sigma.pkl", "wb") as file:
	pkl.dump(prob_delta, file) 
