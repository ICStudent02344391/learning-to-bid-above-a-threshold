# Zooming adaptative sequential halving algorithm
"""
A zooming adaptative sequential halving algorithm is written in this script.
The algorithm is tested for different threshold distributions.
"""

# Module importation
import numpy as np
import numpy.random as rndm
import sys
import os
import pickle as pkl
import time
sys.path.append(os.path.abspath("../src"))
import algorithms as algo
import helper_functions as func
rndm.seed(123)

def zooming_sequential_halving(tau, sigma, partition_size = 4, budget = 5e4, delta = 0.05, seed = None, bernstein = True, oracle = False):
    """
    Zooming Adaptative Sequential Halving algorithm
    :param tau: mean of the threshold distribution
    :type tau: float positive
    :param sigma: standard deviation of the threshold distribution
    :type sigma: float positive
    :param partition_size: number of arms for each successive partition update 
    :type partition_size: int
    :param budget: maximum number of rounds
    :type budget: int
    :param delta: confidence parameter
    :type delta: float between 0 and 1
    :param seed: random seed
    :type seed: int
    :param bernstein: indicator Bernstein's bounds (True) or Hoeffding's bounds (False)
    :type bernstein: boolean
    :param oracle: indicator compute the theoritical variance (True) 
    or the empirical variance (False) when Bernstein = True
    :type oracle: boolean
    :returns: list of the selected intervals becoming narrower, approximation of the optimal value
    """
    list_zoom = [] # list of intervals
    partition = np.linspace(0, 1, partition_size + 1) # initial partition
    t = 0
    best_range = partition[0], partition[-1]
    while t < budget:
        # Run sequential halving algorithm on the current partition
        results = algo.sequential_halving_algorithm(tau, sigma, partition, delta, seed, budget - t, bernstein, oracle)
        t += results[-1] # update number of rounds
        if t < budget:
            # Update bid space and partition 
            left_index, right_index = results[-3][0], results[-3][2]
            best_range = (partition[results[-2]], partition[results[-2]+1])
            partition = np.linspace(partition[left_index], partition[right_index+1], partition_size + 1)
            list_zoom.append((partition[0], partition[-1]))
    list_zoom.append(best_range)
    return list_zoom
# Results of zooming algo
zooming_results = {}
# Illustration of the algorithm
tau = 0.6
sigma = 0.02
list_mu = []
partition = np.linspace(0,1, 501)
for i in range (len(partition) - 1):
    list_mu.append(func.get_mu(partition[i], 
                               partition[i+1], 
                               tau, 
                               sigma))
zooming_results["illustration"] = {}
zooming_results["illustration"]["zooming"] = zooming_sequential_halving(tau, 
                                                                        sigma, 
                                                                        partition_size = 10, 
                                                                        budget = 25e3, 
                                                                        bernstein=True, 
                                                                        delta = 0.5)
zooming_results["illustration"]["expected_rewards"] = list_mu
# Test of the algorithm for different values of sigma and tau
zooming_results["results"] = []

for tau in [0.3, 0.5, 0.7]:
    for sigma in [0.01, 0.02, 0.03]:
        dico_zoom = {}
        list_mu = []
        partition = np.linspace(0,1, 5001)
        exp_reward_old = 0
        index_arm = 0
        exp_reward_new = func.get_mu(partition[index_arm], 
                                     partition[index_arm + 1], 
                                     tau, 
                                     sigma)
        while exp_reward_new >= exp_reward_old or exp_reward_new < 1e-10:
            index_arm += 1
            exp_reward_old = exp_reward_new
            exp_reward_new = func.get_mu(partition[index_arm], 
                                         partition[index_arm + 1], 
                                         tau, 
                                         sigma)
        index_opti_arm = index_arm - 1
        dico_zoom["tau"] = tau
        dico_zoom["sigma"] = sigma
        list_zooms = []
        dico_zoom["approx"] = []
        dico_zoom["expected_reward_approx"] = []
        for _ in range(30):
            tuple_zoom = zooming_sequential_halving(tau, 
                                                    sigma, 
                                                    partition_size = 8, 
                                                    budget = 25e3, 
                                                    bernstein=True,
                                                    delta = 0.5)[-1]
            dico_zoom["approx"].append((round(tuple_zoom[0], 4), round(tuple_zoom[1], 4)))
            dico_zoom["expected_reward_approx"].append(func.get_mu(tuple_zoom[0], 
                                                  tuple_zoom[1], 
                                                  tau, 
                                                  sigma))
        dico_zoom["true_range"] = (partition[index_opti_arm], partition[index_opti_arm + 1])
        dico_zoom["expected_reward_true"] = exp_reward_old
        zooming_results["results"].append(dico_zoom)
with open("results/zooming_results.pkl", "wb") as file:
	pkl.dump(zooming_results, file) 