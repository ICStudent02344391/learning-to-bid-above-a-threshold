# Script to produce the graphs of the thesis

# Module importation
import numpy as np
import numpy.random as rndm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
import os
import pickle as pkl
# Function importation
sys.path.append(os.path.abspath("../src"))
import helper_functions as func
import algorithms

# Colors
blue = "#003E74"
red = "#A51900"
green = "#66A40A"
graph_colors = ["#00ACD7", "#002147", "#A51900", "#66A40A"]
# Set seed
rndm.seed(123)

# ==========================================================================================
# Illustration of attempts to bid above a threshold
# ==========================================================================================
"""
Random (normal) bids are made, if they are above the threshold distribution, they are colo-
red in green, otherwise in red.
"""
T = 20 # number of rounds
plt.rcParams["figure.figsize"] = (12, 4)
thresholds = rndm.normal(0.5, 0.05, T) # threshold samples
bids = rndm.normal(0.5, 2 * 0.05, T) # bid samples
# Indicators bid above/below threshold
bids_above = [bids[i] if bids[i] > thresholds[i] else None for i in range(len(thresholds))]
bids_below = [bids[i] if bids[i] < thresholds[i] else None for i in range(len(thresholds))]
# Plot
plt.scatter(range(0,T), thresholds, s = 10, c = blue, alpha= 0.7, label = r"Threshold $\tau_t$ (unknown)")
plt.scatter(range(0, T), bids_above, s = 10, c = green, alpha= 0.7, label = r"Bid accepted $b_t$")
plt.scatter(range(0, T), bids_below, s = 10, c = red, alpha= 0.7, label = r"Bid refused $b_t$")
plt.ylim((0.2, 0.9))
plt.xlim((-1, 21))
plt.xlabel(r"$t$")
plt.ylabel("Bids and thresholds")
plt.xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
plt.legend()
plt.grid(axis = "y", alpha = 0.3)
plt.savefig("../figures/bid_threshold.pdf")
plt.close()

# ==========================================================================================
# Expected reward of arm j given different sigma
# ==========================================================================================
"""
Step plots of expected rewards given the ranges of the partition
"""
list_mu = [] # list of expected rewards for each arm of the partition
c = 0 # choose the colour
K = 50 # K arms (at least 3)
partition = np.linspace(0, 1, K+1) # create the partition
plt.rcParams["figure.figsize"] = (10,4)
tau = 0.3 # threshold mean
for sigma in [0.01, 0.05, 0.1]: # threshold standard deviation
    for i in range (len(partition)-1):
        list_mu.append(func.get_mu(partition[i], partition[i+1], tau, sigma)) # get the reward
    # Plot
    plt.step(np.arange(K)+0.5, list_mu, label = r"$\tau =$" + str(tau) + r"; $\sigma=$" + str(sigma), color = graph_colors[c])
    c += 1 # update colour
    list_mu = [] # reset list of expected rewards
# Plot
plt.grid(axis= "y", alpha = 0.3)
plt.legend()
plt.xlabel(r"arm $j$")
plt.ylabel(r"$\mu_j$")
plt.title(r"Expected reward of arm j given different $\tau$ and $\sigma$")
plt.savefig("../figures/expect_rew.pdf")
plt.close()

# ==========================================================================================
# Variances of the reward distribution
# ==========================================================================================
"""
Step plots of the variance of the expected rewards given different threshold distribution
and partitions.
"""
list_v = [] # list of variances
fig, ax = plt.subplots(2,1, figsize = (10,8))
tupple_tau_sig = [(0.2, 0.05), (0.5, 0.05), (0.2, 0.01)] # different tau and sigma
c = 0
for tau_sig in tupple_tau_sig: 
    list_v = [0]
    for j in range(len(partition) - 1):
        tau = tau_sig[0]
        sig = tau_sig[1]
        list_v.append(func.get_variance(partition[j], partition[j+1], tau, sig))
    ax[0].step(np.arange(K+1), list_v, label = r"$\tau =$" + str(tau) + r"; $\sigma=$" + str(sig), color = graph_colors[c])
    c+=1
ax[0].set_xlabel("arm j")
ax[0].legend()
ax[0].set_title("Variances of the reward distribution for a partition of {K} arms")
# New partition of 300 arms
c = 0
K = 300 # K arms (at least 3) 
partition = np.linspace(0, 1, K+1)
for tau_sig in tupple_tau_sig:
    list_v = [0]
    for j in range(len(partition) - 1):
        tau = tau_sig[0]
        sig = tau_sig[1]
        list_v.append(func.get_variance(partition[j], partition[j+1], tau, sig))
    ax[1].step(np.arange(K+1), list_v, label = r"$\tau =$" + str(tau) + r"; $\sigma=$" + str(sig), color = graph_colors[c])
    c+=1
ax[1].legend()
ax[1].set_xlabel("arm j")
ax[1].set_title(f"Variances of the reward distribution for a partition of {K} arms")
plt.tight_layout()
plt.savefig("../figures/variance_rew_partitions.pdf")
plt.close()

# ==========================================================================================
# Activation rules adaptative sequential halving algorithm
# ==========================================================================================
"""
Illustration of arm selections for two configurations of the sequential halving algorithm.
Peak and descending configurations. 
"""
list_mu = []
c = 0
K = 50
plt.rcParams["figure.figsize"] = (10,6)
partition = np.linspace(0, 1, K+1)
sigma = 0.05
tau = 0.3
fig, axes = plt.subplots(2,2)
for i in range (len(partition)-1):
    list_mu.append(func.get_mu(partition[i], partition[i+1], tau, sigma))
axes[0,0].bar([14.5, 20.5, 26.5],[list_mu[14], list_mu[20], list_mu[26]], color = blue, label = "Played at $ep$")
axes[0,0].set_title("Peak configuration at $ep$")
axes[0,0].set_xticks([])
axes[0,1].bar([24.5, 28.5, 32.5],[list_mu[24], list_mu[28], list_mu[32]], color = blue, label = "Played at $ep$")
axes[0,1].set_title("Descending configuration at $ep$")
axes[0,1].set_xticks([])
axes[1,0].bar([17.5, 23.5],[list_mu[17], list_mu[23]], color = green, label = "Activated at $ep+1$")
axes[1,0].set_title("Halving at $ep+1$")
axes[1,1].bar([20.5],[list_mu[20]], color = green, label = "Activated at $ep+1$")
axes[1,1].set_title("Left shift at $ep+1$")
axes[1,0].bar([20.5],[list_mu[20]], color = blue)
axes[1,0].set_xlabel("arm $j$")
axes[1,1].bar([24.5, 28.5],[list_mu[24], list_mu[28]], color = blue)
axes[1,0].bar([14.5, 26.5],[list_mu[14], list_mu[26]], color = red, label = "Deactivated at $ep+1$")
axes[1,1].bar([32.5],[list_mu[32]], color = red, label = "Deactivated at $ep+1$")
axes[1,1].set_xlabel("arm $j$")
for ax in axes.flatten(): 
    ax.bar(np.arange(K) + 0.5,list_mu, color = graph_colors[1], alpha = 0.3)
    ax.grid(axis= "y", alpha = 0.3)
    ax.legend()
fig.suptitle("Rule for activating new arms between $ep$ and $ep+1$")
plt.tight_layout()
plt.subplots_adjust(wspace = 0.2)
plt.savefig("../figures/sequential_halving.pdf")
plt.close()

# ==========================================================================================
# Stopping criterion adaptative sequential halving algorithm
# ==========================================================================================
"""
Illustration of the stopping criterion based on the UCB and LCB of the sequential halving
algorihtm.
"""

list_mu = []
c = 0
K = 20
plt.rcParams["figure.figsize"] = (12,5)
partition = np.linspace(0, 1, K+1)
sigma = 0.02
tau = 0.3
fig, axes = plt.subplots(1,2)
for i in range (len(partition)-1):
    list_mu.append(func.get_mu(partition[i], partition[i+1], tau, sigma))
axes[0].bar([6.5, 7.5, 8.5], 
            [list_mu[6] + 0.03, list_mu[7] - 0.01, list_mu[8] + 0.01], 
            0.95, 
            color = blue, 
            label = "Empirical expected rewards", 
            alpha = 0.6)
axes[0].set_title("Algorithm not stopped")
axes[0].errorbar([6.5, 7.5, 8.5], 
               [list_mu[6] + 0.03, list_mu[7] - 0.01, list_mu[8] + 0.01], 
               yerr = [0.05, 0.05, 0.05],
               lw = 0.01,
               fmt = ".",
               color = red,
               capsize = 5,
               label = r"$LCB_t^{(j^*)} < UCB_t^{(j^* \pm 1)}$")
axes[1].bar([6.5, 7.5, 8.5],
            [list_mu[6], list_mu[7], list_mu[8]], 
            0.95, 
            color = blue, 
            label = "Empirical expected rewards", 
            alpha = 0.6)
axes[1].set_title("Algorithm stopped")
axes[1].errorbar([6.5, 7.5, 8.5], 
               [list_mu[6], list_mu[7], list_mu[8]], 
               yerr = [0.02, 0.02, 0.01],
               lw = 0.01,
               fmt = ".",
               color = green,
               capsize = 5,
               label = r"$LCB_t^{(j^*)} > UCB_t^{(j^* \pm 1)}$")
for ax in axes.flatten(): 
    ax.bar(np.arange(K) + 0.5, list_mu, 1, color = graph_colors[1], alpha = 0.2)
    ax.grid(axis= "y", alpha = 0.3)
    ax.legend()
    ax.set_xticks([6.5, 7.5, 8.5], [r"$j^*-1$    ", r"$j^*$", r"    $j^*+1$"], size = 10)
    ax.set_xlabel("arm $j$")
fig.suptitle("Stopping criterion for the adaptative sequential halving algorithm")
plt.tight_layout()
plt.subplots_adjust(wspace = 0.2)
plt.savefig("../figures/stopping_criterion.pdf")
plt.close()
# ==========================================================================================
# Number of times the top three arms have to be played
# ==========================================================================================
"""
Plot the number of iterations needed so that the lower bound of the optimal arm is greater 
than the upper bound of the second best arm (for Hoeffding's and Bernstein's inequality).
"""
delta = np.arange(0.01, 1, 0.01)
fig, axes = plt.subplots(3, 2, figsize = (10, 8))
tau = 0.5
sigma = 0.02
for c, J in enumerate([10, 50, 100]):
    partition = np.linspace(0, 1, J + 1)
    axes[0,0].plot(delta, func.n_bernstein(partition, tau, sigma, delta), c = graph_colors[c])
    axes[0,0].set_title(r"$n_{\text{Bernstein}}$ for $\tau =$" +  str(tau) + " and $\sigma = $" + str(sigma))
for c, J in enumerate([10, 50, 100]):
    partition = np.linspace(0, 1, J + 1)
    axes[0,1].plot(delta, func.n_hoeffding(partition, tau, sigma, delta), c = graph_colors[c])
    axes[0,1].set_title(r"$n_{\text{Hoeffding}}$ for $\tau =$" +  str(tau) + " and $\sigma = $" + str(sigma))
tau = 0.2
sigma = 0.02
for c, J in enumerate([10, 50, 100]):
    partition = np.linspace(0, 1, J + 1)
    axes[1,0].plot(delta, func.n_bernstein(partition, tau, sigma, delta), c = graph_colors[c])
    axes[1,0].set_title(r"$n_{\text{Bernstein}}$ for $\tau =$" +  str(tau) + " and $\sigma = $" + str(sigma))
for c, J in enumerate([10, 50, 100]):
    partition = np.linspace(0, 1, J + 1)
    axes[1,1].plot(delta, func.n_hoeffding(partition, tau, sigma, delta), c = graph_colors[c])
    axes[1,1].set_title(r"$n_{\text{Hoeffding}}$ for $\tau =$" +  str(tau) + " and $\sigma = $" + str(sigma))
tau = 0.5
sigma = 0.05
for c, J in enumerate([10, 50, 100]):
    partition = np.linspace(0, 1, J + 1)
    axes[2,0].plot(delta, func.n_bernstein(partition, tau, sigma, delta), c = graph_colors[c])
    axes[2,0].set_title(r"$n_{\text{Bernstein}}$ for $\tau =$" +  str(tau) + " and $\sigma = $" + str(sigma))
for c, J in enumerate([10, 50, 100]):
    partition = np.linspace(0, 1, J + 1)
    axes[2,1].plot(delta, func.n_hoeffding(partition, tau, sigma, delta), c = graph_colors[c])
    axes[2,1].set_title(r"$n_{\text{Hoeffding}}$ for $\tau =$" +  str(tau) + " and $\sigma = $" + str(sigma))
for ax in axes.flatten():
    ax.set_xlabel(r"$\delta$")
    ax.grid(axis="y", alpha = 0.5)
for ax in axes[:, 0]:
    ax.set_ylabel(r"$n_{\text{Bernstein}}$")
for ax in axes[:, 1]:
    ax.set_ylabel(r"$n_{\text{Hoeffding}}$")
custom_lines = [Line2D([0], [0], color = graph_colors[0], lw=4),
                Line2D([0], [0], color = graph_colors[1], lw=4),
                Line2D([0], [0], color= graph_colors[2], lw=4)]
for ax in axes.flatten():
    ax.legend(custom_lines, ["$J = 10$", "$J = 50$", "$J = 100$"])
plt.tight_layout()
plt.savefig("../figures/n_bernstein_hoeffding.pdf")
plt.close()

# ==========================================================================================
# Success rates
# ==========================================================================================
"""
Plot the number of iterations needed so that the lower bound of the optimal arm is greater 
than the upper bound of the second best arm (for Hoeffding's and Bernstein's inequality).
"""
with open("results/success_rates_tau.pkl", "rb") as file:
	prob_delta_tau = pkl.load(file)
with open("results/success_rates_sigma.pkl", "rb") as file:
	prob_delta_sigma = pkl.load(file)
bar_width = 0.2
for nb_plot, prob_delta in enumerate([prob_delta_tau, prob_delta_sigma]):
    fig, ax = plt.subplots(2, figsize = (10,7))
    for test in range(3):
        tau = prob_delta[f"test_bernstein_{test}"]["tau"]
        sigma = prob_delta[f"test_bernstein_{test}"]["sigma"]
        delta = prob_delta[f"test_bernstein_{test}"]["deltas"]
        delta_axis = np.arange(len(delta))
        nb_simu = len(prob_delta[f"test_bernstein_{test}"]["success_rates"][0])
        conf_success = 1.96 * np.std(prob_delta[f"test_bernstein_{test}"]["success_rates"], axis = 1) / np.sqrt(nb_simu)
        success_rate = np.mean(prob_delta[f"test_bernstein_{test}"]["success_rates"], axis = 1)
        ax[0].bar(delta_axis - 0.2 + bar_width * test, 
                success_rate,
                bar_width, 
                label = r"$\tau = $" + str(tau) + r"; $\sigma = $" + str(sigma),
                color = graph_colors[test])
        ax[0].errorbar(delta_axis - 0.2 + bar_width * test, 
                    success_rate, 
                    yerr = conf_success,
                    lw = 0.01,
                    fmt = ".",
                    color = "black")
    ax[0].set_title(f"Success rate of Sequential Halving Algorithm given $\delta$ with 95% confidence interval (Bernstein's bound)")
    for test in range(3):
        tau = prob_delta[f"test_hoeffding_{test}"]["tau"]
        sigma = prob_delta[f"test_hoeffding_{test}"]["sigma"]
        delta = prob_delta[f"test_hoeffding_{test}"]["deltas"]
        delta_axis = np.arange(len(delta))
        nb_simu = len(prob_delta[f"test_hoeffding_{test}"]["success_rates"][0])
        conf_success = 1.96 * np.std(prob_delta[f"test_hoeffding_{test}"]["success_rates"], axis = 1) / np.sqrt(nb_simu)
        success_rate = np.mean(prob_delta[f"test_hoeffding_{test}"]["success_rates"], axis = 1)
        ax[1].bar(delta_axis - 0.2 + bar_width * test, 
                success_rate,
                bar_width, 
                label =  r"$\tau = $" + str(tau) + r"; $\sigma = $" + str(sigma), 
                color = graph_colors[test])
        ax[1].errorbar(delta_axis - 0.2 + bar_width * test, 
                    success_rate, 
                    yerr = conf_success,
                    lw = 0.01,
                    fmt = ".",
                    color = "black")
    ax[1].set_title(f"Success rate of Sequential Halving Algorithm given $\delta$ with 95% confidence interval (Hoeffding's bound)")
    for a in ax.flatten():
        a.set_xticks(delta_axis, delta) 
        a.set_xlabel(r"$\delta$") 
        a.set_ylabel("Success rate")
        a.grid(axis = "y", alpha = 0.5) 
        a.legend()
    plt.tight_layout()
    if nb_plot == 0:
        plt.savefig("../figures/delta_test_tau.pdf")
    else:
        plt.savefig("../figures/delta_test_sigma.pdf")
    plt.close()

# ==========================================================================================
# Regret plots
# ==========================================================================================
"""
Regret plots of the epsilon-greedy algorithm
"""
# List of functions for the exploration/exploitation ratio
epsilon_label = [r"$J/t$",
                 r"$(J/t)^2$",
                 r"$(J/t)^{1 / 2}$"]
sigma = 0.02
tau = 0.5
fig, axes = plt.subplots(3, 1, figsize = (10,8))
J_list = [10, 50, 100]
with open("results/regret_eps_greedy.pkl", "rb") as file:
	regret_eps_greedy = pkl.load(file)
custom_lines = []
for color_index in range(len(graph_colors)):
    custom_lines.append(Line2D([0], [0], color = graph_colors[color_index], lw=4))
for nb_plot, J in enumerate(J_list):
    for reg in range(len(regret_eps_greedy[epsilon_label[0]][f"partition_{J}"])):
        for label_index, label in enumerate(epsilon_label):
            axes[nb_plot].plot(regret_eps_greedy[label][f"partition_{J}"][reg],
                                c = graph_colors[label_index], 
                                alpha = 0.5)
    axes[nb_plot].legend(custom_lines, epsilon_label)
    axes[nb_plot].set_title(r"$J =$" + str(J))
for ax in axes.flatten():
    ax.set_xlabel("round $t$")
    ax.set_ylabel("Regret")
    ax.grid(axis="y", alpha = 0.5)
plt.suptitle(r"Comparison of regret for $\tau =$" + str(tau) + r" and $\sigma =$" + str(sigma) + r"($\epsilon$-greedy)")
plt.tight_layout()
plt.savefig("../figures/regret_eps_greedy.pdf")
plt.close()
"""
Regret plots UCB algo given different delta
"""
with open("results/regret_UCB_delta.pkl", "rb") as file:
	regret_UCB_delta = pkl.load(file)
tau = 0.5
sigma = 0.02
fig, axes = plt.subplots(2, 2, figsize = (10,8))
custom_lines = [Line2D([0], [0], color = graph_colors[1], lw=4),
                Line2D([0], [0], color = graph_colors[2], lw=4)]
for n_plot, ax in enumerate(axes.flatten()):
    ax.set_title(r"$\delta =$" + str(regret_UCB_delta["delta"][n_plot]))
    for simu in range(50): 
        ax.plot(regret_UCB_delta["Bernstein"][n_plot][simu],
                c = graph_colors[1], 
                alpha = 0.5)
        ax.plot(regret_UCB_delta["Hoeffding"][n_plot][simu],
                c = graph_colors[2], 
                alpha = 0.5)
    ax.legend(custom_lines, ["Bernstein's bound","Hoeffding's bound"])
    ax.set_xlabel("round $t$")
    ax.set_ylabel("Regret")
    ax.grid(axis="y", alpha = 0.5)
plt.suptitle(r"Comparison of regret of the UCB algorithm for $\tau =$" + str(tau) + r" and $\sigma =$" + str(sigma))
plt.tight_layout()
plt.savefig("../figures/UCB_delta.pdf")
plt.close()

"""
Regret plots of the algorithms given different partitions
"""
# Figures of the algorithms regrets for Bernstein's bound (first column is the three algos,
# second is UCB and sequential halving to better see their regrets) 
fig, axes = plt.subplots(3, 2, figsize = (12,10))
with open("results/regret_sequential_halving.pkl", "rb") as file:
	regret_sha = pkl.load(file)
with open("results/regret_UCB.pkl", "rb") as file:
	regret_UCB = pkl.load(file)
custom_lines = [Line2D([0], [0], color = graph_colors[1], lw=4),
                Line2D([0], [0], color= graph_colors[0], lw=4),
                Line2D([0], [0], color = graph_colors[2], lw=4)]
for nb_plot, J in enumerate(J_list):
    for reg in range(len(regret_sha["Hoeffding"][f"partition_{J}"])):
        axes[nb_plot, 0].plot(regret_sha["Hoeffding"][f"partition_{J}"][reg],
                        c = graph_colors[0], 
                        alpha = 0.5)
    for reg in range(len(regret_UCB["Hoeffding"][f"partition_{J}"])):
        axes[nb_plot, 1].plot(regret_UCB["Hoeffding"][f"partition_{J}"][reg],
                        c = graph_colors[2], 
                        alpha = 0.5)
    for col in range(2):
        for reg in range(len(regret_sha["Bernstein"][f"partition_{J}"])):
            axes[nb_plot, col].plot(regret_sha["Bernstein"][f"partition_{J}"][reg],
                                c = graph_colors[1], 
                                alpha = 0.5)
        axes[nb_plot, col].set_title(r"$J =$" + str(J))    
    axes[0,1].legend(custom_lines, ["sequential halving (Bernstein)", "sequential halving (Hoeffding)", "UCB (Hoeffding)"])
for ax in axes.flatten():
    ax.set_xlabel("round $t$")
    ax.set_ylabel("Regret")
    ax.grid(axis="y", alpha = 0.5)
    ax.set_ylim(0, 100)
plt.suptitle(r"Comparison of regret for $\tau =$" + str(tau) + r" and $\sigma =$" + str(sigma))
plt.tight_layout()
plt.savefig("../figures/regret_all.pdf")
plt.close()

# Figures of the algorithms regrets for Hoeffding's bound (sequential halving and UCB algorithms)
fig, axes = plt.subplots(3, figsize = (10,8))
custom_lines = [Line2D([0], [0], color = graph_colors[1], lw=4),
                Line2D([0], [0], color = graph_colors[2], lw=4)]
for nb_plot, J in enumerate(J_list):
    for reg in range(len(regret_sha["Hoeffding"][f"partition_{J}"])):
        axes[nb_plot].plot(regret_sha["Hoeffding"][f"partition_{J}"][reg],
                        c = graph_colors[1], 
                        alpha = 0.5)
    for reg in range(len(regret_UCB["Hoeffding"][f"partition_{J}"])):
        axes[nb_plot].plot(regret_UCB["Hoeffding"][f"partition_{J}"][reg],
                        c = graph_colors[2], 
                        alpha = 0.5)
    axes[nb_plot].set_title(r"$J =$" + str(J))    
    axes[nb_plot].legend(custom_lines, ["Sequential halving", "UCB"])
for ax in axes.flatten():
    ax.set_xlabel("round $t$")
    ax.set_ylabel("Regret")
    ax.grid(axis="y", alpha = 0.5)
plt.suptitle(r"Comparison of regret for $\tau =$" + str(tau) + r" and $\sigma =$" + str(sigma) + " (Hoeffding's bounds)")
plt.tight_layout()
plt.savefig("../figures/regret_hoeffding.pdf")
plt.close()
fig, axes = plt.subplots(3, figsize = (10,8))
for nb_plot, J in enumerate(J_list):
    for reg in range(len(regret_sha["Bernstein"][f"partition_{J}"])):
        axes[nb_plot].plot(regret_sha["Bernstein"][f"partition_{J}"][reg],
                        c = graph_colors[1], 
                        alpha = 0.5)
    for reg in range(len(regret_UCB["Hoeffding"][f"partition_{J}"])):
        axes[nb_plot].plot(regret_UCB["Hoeffding"][f"partition_{J}"][reg],
                        c = graph_colors[2], 
                        alpha = 0.5)
    axes[nb_plot].set_title(r"$J =$" + str(J))    
    axes[nb_plot].legend(custom_lines, ["Sequential halving (Bernstein)", "UCB (Hoeffding)"])
for ax in axes.flatten():
    ax.set_xlabel("round $t$")
    ax.set_ylabel("Regret")
    ax.grid(axis="y", alpha = 0.5)
    ax.set_ylim(0, 100)
plt.suptitle(r"Comparison of regret for $\tau =$" + str(tau) + r" and $\sigma =$" + str(sigma))
plt.tight_layout()
plt.savefig("../figures/regret_best.pdf")
plt.close()

# ==========================================================================================
# Zooming sequential halving
# ==========================================================================================
"""
Illustration zooming sequential halving algorithm
"""

with open("results/zooming_results.pkl", "rb") as file:
	zooming_results = pkl.load(file)
tau = 0.6
sigma = 0.02
rndm.seed(123)
partition = np.linspace(0, 1, 501)
list_mu = zooming_results["illustration"]["expected_rewards"]
list_zoom = zooming_results["illustration"]["zooming"]
plt.plot(partition, [0] + list_mu, label = r"$\tau =$" + str(tau) + r"; $\sigma=$" + str(sigma), c = graph_colors[1])
plt.grid(axis = "y")
for i,elm in enumerate(list_zoom):
    plt.vlines(elm[0], ymin= 0, ymax=0.35, colors= "red", alpha = ((i+1)/len(list_zoom))**2)
    plt.vlines(elm[1], ymin= 0, ymax=0.35, colors= "red", alpha = ((i+1)/len(list_zoom))**2)
plt.title(r"Searching for the best range of bids for $\tau=$" + str(tau) + r" and $\sigma=$"+ str(sigma))
plt.savefig("../figures/zooming_sha.pdf")
plt.close()
"""
Performances zooming sequential halving algorithm
"""
fig, ax = plt.subplots(1, figsize = (10,5))
custom_lines = [Line2D([0], [0], color = green, lw=4),
                Line2D([0], [0], color = blue, lw=4, alpha = 0.7)]
label_zoom = []
for i, res in enumerate(zooming_results["results"]):
     ax.scatter([i], res["expected_reward_true"], c = green, s = 50)
     ax.scatter([i] * 30 , res["expected_reward_approx"], c = blue, alpha = 0.5, s = 30)
     label_zoom.append(r"$tau =$" + str(res["tau"]) + "\n" + r"$\sigma =$" + str(res["sigma"]))
ax.set_title(r"Expected rewards of the subintervals chosen by the Zooming Sequential Halving algorithm")
ax.set_xticks(np.arange(0, 9), label_zoom)
ax.legend(custom_lines, 
          ["Expected reward of the optimal bid",
           "Expected reward of the subinterval selected"])
plt.savefig("../figures/expected_zoom.pdf")
plt.close()
     