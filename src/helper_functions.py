# Functions used in algorithms and plots

# Module importation
from scipy.stats import norm
from scipy.optimize import fmin
import numpy as np

# Functions

def reward(bid, threshold, bsup = 1):
    """
    Compute the reward for a given bid and threshold
    :param bid: bid made by an arm
    :type bid: float between 0 and bsup
    :param threshold: threshold (unknown) sampled by the threshold distribution
    :type threshold: float
    :param bsup: upper bound of the bids
    :type bsup: float positive
    :returns: indicator that the bid is above the threshold
    :rtype: boolean
    """
    return (bsup - bid) * (threshold <= bid)

def get_mu(a0, a1, tau, sigma, bsup = 1):
    """
    Compute the expected reward for a given uniform distribution of bids and normal threshold distribution
    :param a0: lower bound of the uniform bid distribution
    :type a0: float between 0 and bsup
    :param a1: upper bound of the uniform bid distribution
    :type a1: float between 0 and bsup (greater than a0)
    :param tau: mean of the normal threshold distribution
    :type tau: float positive
    :param sigma: standard deviation of the threshold distribution
    :type sigma: float positive
    :param bsup: upper bound of the bids
    :type bsup: float positive
    :returns: expectaction of the reward distribution
    :rtype: float
    """
    a0_tilde = (a0 - tau) / sigma
    a1_tilde = (a1 - tau) / sigma
    return sigma/(2 * (a1 - a0)) * (norm.cdf(a1_tilde) * (sigma - a1_tilde * (tau - 2 * bsup + a1)) - norm.cdf(a0_tilde) * (sigma - a0_tilde * (tau - 2 * bsup + a0)) + norm.pdf(a1_tilde) * (2 * (bsup - tau) - sigma * a1_tilde) - norm.pdf(a0_tilde) * (2 * (bsup - tau) - sigma * a0_tilde))

def get_optimal_mu(partition, tau, sigma, unif_range = False):
    """
    Find the maximum expectation given a partition of uniform bid distributions and threshold ditribution parameters
    :param partition: bounds of the bid distributions
    :type partition: numpy.ndarray
    :param tau: mean of the normal threshold distribution
    :type tau: float positive
    :param sigma: standard deviation of the threshold distribution
    :type sigma: float positive
    :param unif_range: indicator that the optimal range of bid is returned
    :type unif_range: boolean
    :returns: expectaction of the reward distribution and bound of the optimal bid distribution in the partition (if unif_range)
    :rtype: float or tupple 
    """
    bsup = partition[-1]
    x0 = bsup / 2 # initial value for the optimisation
    increment = partition[1] - partition[0] # size of the bid ranges
    # Two potential lower bounds of the optimal arm distribution
    a0_star = round(fmin(lambda a0: - get_mu(a0, a0 + increment, tau, sigma),
           x0,
           xtol = increment / 10,
           disp = 0)[0] * 1 / increment) * increment
    a1_star = a0_star + increment
    # Test which is the best lower bound
    if get_mu(a0_star, a1_star, tau, sigma, bsup) < get_mu(a1_star, a1_star + increment, tau, sigma, bsup):
        a_star = a1_star
    else:
        a_star = a0_star
    if unif_range:
        return get_mu(a_star, a_star + increment, tau, sigma, bsup), (a_star, a_star + increment)
    else:
        return get_mu(a_star, a_star + increment, tau, sigma, bsup)

# Find variance
def get_variance(a0, a1, tau, sigma, bsup = 1):
    """
    Compute the variance of the reward distribution
    :param a0: lower bound of the uniform bid distribution
    :type a0: float between 0 and bsup
    :param a1: upper bound of the uniform bid distribution
    :type a1: float between 0 and bsup (greater than a0)
    :param tau: mean of the normal threshold distribution
    :type tau: float positive
    :param sigma: standard deviation of the threshold distribution
    :type sigma: float positive
    :param bsup: upper bound of the bids
    :type bsup: float positive
    :returns: variance of the reward distribution
    :rtype: float
    """
    def get_second_moment(a0, a1, tau, sigma, bsup = 1):
        """
        Compute the second moment of the reward distribution
        :returns: second moment of the reward distribution
        :rtype: float
        """
        a0_tilde = (a0 - tau) / sigma
        a1_tilde = (a1 - tau) / sigma
        sigma2 = sigma ** 2
        sigma3 = sigma ** 3
        bsup_tau = bsup - tau
        bsup_tau2 = bsup_tau ** 2
        bsup_tau3 = bsup_tau ** 3
        return 1 / (3 * (a1 - a0)) * (norm.cdf(a1_tilde) * (bsup_tau3 + 3 * sigma2 * bsup_tau - (bsup - a1)**3) - norm.cdf(a0_tilde) * (bsup_tau3 + 3 * sigma2 * bsup_tau - (bsup - a0)**3) + norm.pdf(a1_tilde) * (sigma3 * (a1_tilde ** 2 + 2) + 3 * sigma * bsup_tau2 - 3 * sigma2 * bsup_tau * a1_tilde) - norm.pdf(a0_tilde) * (sigma3 * (a0_tilde ** 2 + 2) + 3 * sigma * bsup_tau2 - 3 * sigma2 * bsup_tau * a0_tilde))
    return get_second_moment(a0, a1, tau, sigma, bsup) - get_mu(a0, a1, tau, sigma, bsup)**2

def n_hoeffding(partition, tau, sigma, delta):
    """
    Compute the number of iterations needed to select the optimal arm given a partition with confidence parameter delta based on the Hoeffding's inequality    :param partition: bounds of the bid distributions
    :param partition: bounds of the bid distributions
    :type partition: numpy.ndarray
    :param tau: mean of the normal threshold distribution
    :type tau: float positive
    :param sigma: standard deviation of the threshold distribution
    :type sigma: float positive
    :param delta: array of confidence parameters
    :type delta: numpy.ndarray of float between 0 and 1
    :returns: number of iterations needed to select the optimal arm
    :rtype: int
    """
    J = len(partition) - 1 # arm number
    mu_opt, bounds = get_optimal_mu(partition, tau, sigma, unif_range=True) # optimal expected reward
    j = np.where(partition == bounds[0])[0][0] # best arm
    i_1 = j + 1 # potential second best arm
    i_2 = j - 1 # potential second best arm
    diff_1 = mu_opt - get_mu(partition[i_1], partition[i_1 + 1], tau, sigma) # distance between optimal expected reward
    diff_2 = mu_opt - get_mu(partition[i_2], partition[i_2 + 1], tau, sigma) # distance between optimal expected reward
    n_hoeff = np.zeros(len(delta))
    for k, d in enumerate(delta):
        n_hoeff[k] = int(2 * np.log(1 / d) * max(((1 - ((j + i_1) / 2 - 1) / J)/ diff_1) ** 2, ((1 - ((j + i_2) / 2 - 1) / J)/ diff_2) ** 2)) + 1
    return n_hoeff
def n_bernstein(partition, tau, sigma, delta):
    """
    Compute the number of iterations needed to select the optimal arm given a partition with confidence parameter delta based on the Bernstein's inequality
    :param partition: bounds of the bid distributions
    :type partition: numpy.ndarray
    :param tau: mean of the normal threshold distribution
    :type tau: float positive
    :param sigma: standard deviation of the threshold distribution
    :type sigma: float positive
    :param delta: array of confidence parameters
    :type delta: numpy.ndarray of float between 0 and 1
    :returns: number of iterations needed to select the optimal arm
    :rtype: int
    """
    J = len(partition) - 1 # arm number
    mu_opt, bounds = get_optimal_mu(partition, tau, sigma, unif_range=True) # optimal expected reward
    j = np.where(partition == bounds[0])[0][0] # best arm
    i_1 = j + 1 # potential second best arm
    i_2 = j - 1 # potential second best arm
    diff_1 = mu_opt - get_mu(partition[i_1], partition[i_1 + 1], tau, sigma) # distance between optimal expected reward
    diff_2 = mu_opt - get_mu(partition[i_2], partition[i_2 + 1], tau, sigma) # distance between optimal expected reward
    var_j = get_variance(bounds[0], bounds[1], tau, sigma) # variance of the best arm
    var_i_1 = get_variance(partition[i_1], partition[i_1 + 1], tau, sigma) # variance of the second potential best arm
    var_i_2 = get_variance(partition[i_2], partition[i_2 + 1], tau, sigma) # variance of the second potential best arm
    sigma_ij_1 = (np.sqrt(var_i_1) + np.sqrt(var_j)) / 2
    sigma_ij_2 = (np.sqrt(var_i_2) + np.sqrt(var_j)) / 2
    n_bern = np.zeros(len(delta))
    for k, d in enumerate(delta):
        n_bern[k] = int(2 * np.log(1 / d) * max((sigma_ij_1 + np.sqrt(sigma_ij_1 ** 2 + 3 * diff_1 * (1 - ((j + i_1) / 2 - 1) / J)) / diff_1) ** 2, (sigma_ij_2 + np.sqrt(sigma_ij_2 ** 2 + 3 * diff_2 * (1 - ((j + i_2) / 2 - 1) / J)) / diff_2) ** 2)) + 1
    return n_bern