# Script to save the different algorithm regrets
"""
This script is separated from the other because the computation time is high and it is better 
to save the results once computed.
Here, regrets of the three tested algorithms are plotted given different partitions.
"""

# Module importation
import numpy as np
import numpy.random as rndm
import sys
import os
import pickle as pkl
import time
sys.path.append(os.path.abspath("../src"))
import helper_functions as func
import algorithms as algo

rndm.seed(123)

J_list = [10, 50, 100] # arm number tested
max_iter = 10000 # maximal number of rounds
nb_simu = 50 # number of replications of the algorithms
tau = 0.5 # mean of the threshold distribution
sigma = 0.02 # standard deviation of the threshold distribution
regret_sequential_halving = {"Bernstein": {},
                             "Hoeffding": {}}
regret_UCB = {"Bernstein": {},
              "Hoeffding": {}}
# Exploration / exploitation for epsilon-greedy
epsilon_list =[lambda t, J: J / t,
               lambda t, J: (J / t) ** 2,
               lambda t, J: (J / t) ** (1 / 2)]
epsilon_label = [r"$J/t$",
                 r"$(J/t)^2$",
                 r"$(J/t)^{1 / 2}$"]
regret_eps_greedy = {r"$J/t$": {},
                 r"$(J/t)^2$": {},
                 r"$(J/t)^{1 / 2}$": {}}

# Run the algorithms
for J in J_list:
    print(J)
    partition = np.linspace(0,1, J+1)
    # get the optimal arm
    opt_mu = func.get_optimal_mu(partition, tau, sigma)
    # get the expected reward for every arm
    array_mu = np.zeros(J)
    regret_sequential_halving["Bernstein"][f"partition_{J}"] = []
    regret_sequential_halving["Hoeffding"][f"partition_{J}"] = []
    regret_UCB["Bernstein"][f"partition_{J}"] = []
    regret_UCB["Hoeffding"][f"partition_{J}"] = []
    for label in epsilon_label:
        regret_eps_greedy[label][f"partition_{J}"] = []
    for arm in range(J):
        array_mu[arm] = func.get_mu(partition[arm], partition[arm + 1], tau, sigma)
    avg_time_bern = 0
    avg_time_hoeff = 0
    for _ in range(nb_simu):
        # sequential halving algorithm
        ## Bernstein's bound
        time_sha_bern = time.time()
        res_sha_bern = algo.sequential_halving_algorithm(tau, 
                                                sigma, 
                                                partition, 
                                                itermax = max_iter, 
                                                delta = 0.5)
        time_sha_bern = time.time() - time_sha_bern
        avg_time_bern += 1 / nb_simu * time_sha_bern
        selected_sha_bern = res_sha_bern[0]
        opti_sha_bern = res_sha_bern[-2]
        if len(selected_sha_bern) < max_iter:
            selected_sha_bern += [opti_sha_bern] * (max_iter - len(selected_sha_bern))
        regret_sequential_halving["Bernstein"][f"partition_{J}"].append(np.cumsum(opt_mu - np.array([array_mu[arm] for arm in selected_sha_bern])))
        ## Hoeffding's bound
        time_sha_hoeff = time.time()
        res_sha_hoeff = algo.sequential_halving_algorithm(tau, 
                                                sigma, 
                                                partition, 
                                                itermax  = max_iter, 
                                                delta = 0.5,
                                                bernstein = False)
        time_sha_hoeff = time.time() - time_sha_hoeff
        avg_time_hoeff += 1 / nb_simu * time_sha_hoeff
        selected_sha_hoeff = res_sha_hoeff[0]
        opti_sha_hoeff = res_sha_hoeff[-2]
        if len(selected_sha_hoeff) < max_iter:
            selected_sha_hoeff += [opti_sha_hoeff] * (max_iter - len(selected_sha_hoeff))
        regret_sequential_halving["Hoeffding"][f"partition_{J}"].append(np.cumsum(opt_mu - np.array([array_mu[arm] for arm in selected_sha_hoeff])))
        # UCB algorithm
        ## Bernstein's bound
        selected_UCB_bern = algo.UCB_algorithm(tau, 
                                        sigma, 
                                        partition, 
                                        delta = 0.6,
                                        itermax = max_iter)[0]
        regret_UCB["Bernstein"][f"partition_{J}"].append(np.cumsum(opt_mu - np.array([array_mu[arm] for arm in selected_UCB_bern])))
        ## Hoeffding's bound
        selected_UCB_hoeff = algo.UCB_algorithm(tau, 
                                        sigma, 
                                        partition, 
                                        delta = 0.6,
                                        itermax = max_iter,
                                        bernstein = False)[0]
        regret_UCB["Hoeffding"][f"partition_{J}"].append(np.cumsum(opt_mu - np.array([array_mu[arm] for arm in selected_UCB_hoeff])))
        # epsilon greedy algorithm
        for label_index, ratio in enumerate(epsilon_list):
            selected_eps_greedy = algo.eps_greedy(tau, 
                                                sigma, 
                                                partition,
                                                ratio,
                                                itermax = max_iter)[0]
            regret_eps_greedy[epsilon_label[label_index]][f"partition_{J}"].append(np.cumsum(opt_mu - np.array([array_mu[arm] for arm in selected_eps_greedy])))
    print(f"Average time sequential halving algo with Bernstein's bound (J = {J}): {avg_time_bern}")
    print(f"Average time sequential halving algo with Hoeffding's bound (J = {J}): {avg_time_hoeff}")
with open("results/regret_sequential_halving.pkl", "wb") as file:
	pkl.dump(regret_sequential_halving, file) 
with open("results/regret_UCB.pkl", "wb") as file:
	pkl.dump(regret_UCB, file) 
with open("results/regret_eps_greedy.pkl", "wb") as file:
	pkl.dump(regret_eps_greedy, file) 
