# Script to plot the regret of UCB algorithm given different delta
"""
This script is separated from the other because the computation time is high and it is better 
to save the results once computed.
Here, regrets of the three tested algorithms are plotted given different partitions.
"""
# Module importation
import numpy as np
import numpy.random as rndm
import pickle as pkl
import sys
import os
sys.path.append(os.path.abspath("../src"))
import helper_functions as func
import algorithms as algo

tau = 0.5
sigma = 0.02
J= 30
delta = [0.2, 0.4, 0.6, 0.8]
nb_simu = 50
max_iter = 1000
regret_UCB_delta = {}
partition = np.linspace(0,1, J+1)
opt_mu = func.get_optimal_mu(partition, tau, sigma)
array_mu = np.zeros(J)
for arm in range(J):
    array_mu[arm] = func.get_mu(partition[arm], partition[arm + 1], tau, sigma)
regret_UCB_delta = {"Hoeffding": [],
                    "Bernstein": [],
                    "delta": delta}
for d in delta:
    list_regret_hoeff = []
    list_regret_bern = []
    for _ in range(nb_simu):
        selected_UCB_bern = algo.UCB_algorithm(tau, 
                                        sigma, 
                                        partition, 
                                        delta = d,
                                        itermax = max_iter)[0]
        selected_UCB_hoeff = algo.UCB_algorithm(tau, 
                                        sigma, 
                                        partition, 
                                        delta = d,
                                        itermax = max_iter,
                                        bernstein = False)[0]
        list_regret_bern.append(np.cumsum(opt_mu - np.array([array_mu[arm] for arm in selected_UCB_bern])))
        list_regret_hoeff.append(np.cumsum(opt_mu - np.array([array_mu[arm] for arm in selected_UCB_hoeff])))
    regret_UCB_delta["Hoeffding"].append(list_regret_hoeff)
    regret_UCB_delta["Bernstein"].append(list_regret_bern)
with open("results/regret_UCB_delta.pkl", "wb") as file:
    pkl.dump(regret_UCB_delta, file) 
