# Script of the algorithms used in this thesis

# Module importation
import numpy as np
import numpy.random as rndm
import sys
import os
sys.path.append(os.path.abspath("../src"))
import helper_functions as func

# Epsilon-Greedy algorithm
def eps_greedy(tau, sigma, partition, seed = None, itermax = 5e4):
    """
    Epsilon-Greedy algorithm
    :param tau: mean of the threshold distribution
    :type tau: float positive
    :param sigma: standard deviation of the threshold distribution
    :type sigma: float positive
    :param partition: bounds of the bids
    :type partition: numpy.ndarray
    :param seed: random seed
    :type seed: int
    :param itermax: maximum number of rounds
    :type itermax: int
    :returns: list of selected arms, array of empirical average rewards for each arm, 
    array of counts for each arm, arm with the greatest empirical average reward
    """
    def epsilon_t(t, J):
        """
        Exploration / exploitation strategy
        :param t: round
        :type t: int
        :param J: arm number
        :type J: int
        :returns: exploration / exploitation ratio
        :rtype: float
        """
        return t**(-1/3)*(J * np.log(t))**(1/3)
    rndm.seed(seed)
    J = len(partition) - 1 # am number
    avg_arm_reward = np.zeros(J) # array of empirical average rewards
    arm_count = np.zeros(J) # array of counts for each arm
    selected_arm = [] # selected arm at round t
    t = J 
    # Initialisation (play each arm once)
    for j in range(J):
        # Unknown threshold
        threshold = sigma * rndm.randn() + tau
        # Arm j makes a bid
        bid = partition[j] + (partition[j + 1] - partition[j]) * rndm.rand()
        arm_count[j] += 1
        avg_arm_reward[j] = ((arm_count[j] - 1) * avg_arm_reward[j] + func.reward(bid, threshold)) / arm_count[j]
        selected_arm.append(j)
    # Exploration/Exploitation:
    while t < itermax:
        threshold = sigma * rndm.randn() + tau
        # Exploration or exploitation
        if rndm.rand() < epsilon_t(t, J): # Exploration
            arm_index = rndm.randint(0, J)
        else: # Exploitation
            arm_index = np.argmax(avg_arm_reward) 
        bid = partition[arm_index] + (partition[arm_index + 1] - partition[arm_index]) * rndm.rand()
        arm_count[arm_index] += 1
        avg_arm_reward[arm_index] = ((arm_count[arm_index] - 1) * avg_arm_reward[arm_index] + func.reward(bid, tau)) / arm_count[arm_index]
        selected_arm.append(arm_index)
        t += 1
    return selected_arm, avg_arm_reward, arm_count, np.argmax(avg_arm_reward)
# UCB algorithm
def UCB_algorithm(tau, sigma, partition, delta, seed = None, itermax = 5e4, bernstein = True, oracle = False):
    """
    UCB algorithm
    :param tau: mean of the threshold distribution
    :type tau: float positive
    :param sigma: standard deviation of the threshold distribution
    :type sigma: float positive
    :param partition: bounds of the bids
    :type partition: numpy.ndarray
    :param delta: confidence parameter
    :type delta: float between 0 and 1
    :param seed: random seed
    :type seed: int
    :param itermax: maximum number of rounds
    :type itermax: int
    :param bernstein: indicator Bernstein's bounds (True) or Hoeffding's bounds (False)
    :type bernstein: boolean
    :param oracle: indicator compute the theoritical variance (True) 
    or the empirical variance (False) when Bernstein = True
    :type oracle: boolean
    :returns: list of selected arms, array of empirical average rewards for each arm, 
    array of counts for each arm, list of UCB for each arm, arm with the greatest UCB, 
    arm with the greatest empirical average reward 
    """
    rndm.seed(seed)
    J = len(partition) - 1
    avg_arm_reward = np.zeros(J)
    arm_count = np.zeros(J)
    dict_rewards = {}
    arm_ucb = np.zeros(J)
    selected_arm = []
    t = J
    bound_reward = np.array([1 - j/J for j in range(J)])
    constant_bernstein_1 = 2 * np.log(3/delta)
    constant_bernstein_2 = 3 * np.log(3/delta)
    constant_hoeffding = np.log(2/delta)/2
    # Initialisation
    for j in range(J):
        threshold = sigma * rndm.randn() + tau
        bid = partition[j] + (partition[j + 1] - partition[j]) * rndm.rand()
        arm_count[j] += 1
        rew = func.reward(bid, threshold)
        avg_arm_reward[j] = ((arm_count[j] - 1) * avg_arm_reward[j] + rew) / arm_count[j]
        dict_rewards[f"arm_{j}"] = [rew]
        selected_arm.append(j)
        if bernstein:
            arm_ucb[j] = avg_arm_reward[j] + bound_reward[j] * constant_bernstein_2
        else:
            arm_ucb[j] = avg_arm_reward[j] + bound_reward[j] * np.sqrt(constant_hoeffding)
    while t < itermax:
        arm_index = np.argmax(arm_ucb)
        threshold = sigma * rndm.randn() + tau
        bid = partition[arm_index] + (partition[arm_index + 1] - partition[arm_index]) * rndm.rand()
        arm_count[arm_index] += 1
        rew = func.reward(bid, threshold)
        avg_arm_reward[arm_index] = ((arm_count[arm_index] - 1) * avg_arm_reward[arm_index] + rew) / arm_count[arm_index]
        dict_rewards[f"arm_{arm_index}"] = [rew]
        selected_arm.append(arm_index)
        if bernstein:
            if not oracle:
                arm_std = np.array(dict_rewards[f"arm_{arm_index}"]).std()
            else:
                arm_std = func.get_variance(partition[arm_index], partition[arm_index + 1], tau, sigma)
            arm_std = np.array(dict_rewards[f"arm_{arm_index}"]).std()
            arm_ucb[arm_index] = avg_arm_reward[arm_index] + arm_std * np.sqrt(constant_bernstein_1 / arm_count[arm_index]) + bound_reward[arm_index] * constant_bernstein_2 / arm_count[arm_index]
        else:
            arm_ucb[arm_index] = avg_arm_reward[arm_index] + bound_reward[arm_index] * np.sqrt(constant_hoeffding / arm_count[arm_index])
        t+=1
    return selected_arm, avg_arm_reward, arm_count, arm_ucb, np.argmax(arm_ucb), np.argmax(avg_arm_reward)

# Sequential halving algorithm
def sequential_halving_algorithm(tau, sigma, partition, delta, seed = None, itermax = 5e4, bernstein = True, oracle = False):
    """
    Sequential Halving algorithm
    :param tau: mean of the threshold distribution
    :type tau: float positive
    :param sigma: standard deviation of the threshold distribution
    :type sigma: float positive
    :param partition: bounds of the bids
    :type partition: numpy.ndarray
    :param delta: confidence parameter
    :type delta: float between 0 and 1
    :param seed: random seed
    :type seed: int
    :param itermax: maximum number of rounds
    :type itermax: int
    :param bernstein: indicator Bernstein's bounds (True) or Hoeffding's bounds (False)
    :type bernstein: boolean
    :param oracle: indicator compute the theoritical variance (True) 
    or the empirical variance (False) when Bernstein = True
    :type oracle: boolean
    :returns: list of the selected arms, array of empirical average rewards for each arm, 
    array of counts for each arm, tupple of three terminal arms (left, middle, right), 
    best arm at the end of the algorithm, number of rounds played
    """
    J = len(partition) - 1
    left_index = 0
    mid_index = int((J)/2)
    right_index = J - 1
    avg_arm_reward = np.zeros(J)
    arm_count = np.zeros(J)
    dict_rewards = {}
    if seed != None:
        rndm.seed(seed)
    t = 0
    selected_arm = []
    bound_reward = np.array([1 - j/J for j in range(J)])
    constant_bernstein_1 = 2 * np.log(1/delta)
    constant_bernstein_2 = 3 * np.log(1/delta)
    constant_hoeffding = np.log(1/delta)/2
    consecutive = 0
    while 1:
        # Play three arms
        for ind in [left_index, mid_index, right_index]:
            threshold = sigma * rndm.randn() + tau
            bid = partition[ind] + (partition[ind + 1] - partition[ind]) * rndm.rand()
            rew = func.reward(bid, threshold)
            arm_count[ind] += 1
            avg_arm_reward[ind] = ((arm_count[ind] - 1) * avg_arm_reward[ind] + rew) / arm_count[ind]
            if not (f"arm_{ind}" in dict_rewards):
                dict_rewards[f"arm_{ind}"] = []
            dict_rewards[f"arm_{ind}"].append(rew)
            selected_arm.append(ind)
            t += 1
            if t == itermax:
                break
        if t == itermax:
            break
        ascending = (avg_arm_reward[left_index] <= avg_arm_reward[mid_index]) and (avg_arm_reward[mid_index] <= avg_arm_reward[right_index])
        descending = (avg_arm_reward[left_index] > avg_arm_reward[mid_index]) and (avg_arm_reward[mid_index] > avg_arm_reward[right_index])
        peak = (avg_arm_reward[left_index] < avg_arm_reward[mid_index]) and (avg_arm_reward[mid_index] > avg_arm_reward[right_index])
        if consecutive:    
            if peak:
                if bernstein:
                    if not oracle:
                        left_std = np.array(dict_rewards[f"arm_{left_index}"]).std()
                        mid_std = np.array(dict_rewards[f"arm_{mid_index}"]).std()
                        right_std = np.array(dict_rewards[f"arm_{right_index}"]).std()
                    else:
                        left_std = func.get_variance(partition[left_index], partition[left_index + 1], tau, sigma)
                        mid_std = func.get_variance(partition[mid_index], partition[mid_index + 1], tau, sigma)
                        right_std = func.get_variance(partition[right_index], partition[right_index + 1], tau, sigma)
                    left_confidence_bound = avg_arm_reward[left_index] + left_std * np.sqrt(constant_bernstein_1 / arm_count[left_index]) + bound_reward[left_index] * constant_bernstein_2 / arm_count[left_index]
                    mid_confidence_bound = avg_arm_reward[mid_index] - mid_std * np.sqrt(constant_bernstein_1 / arm_count[mid_index]) - bound_reward[mid_index] * constant_bernstein_2 / arm_count[mid_index]
                    right_confidence_bound = avg_arm_reward[right_index] + right_std * np.sqrt(constant_bernstein_1 / arm_count[right_index]) + bound_reward[right_index] * constant_bernstein_2 / arm_count[right_index]
                else:
                    left_confidence_bound = avg_arm_reward[left_index] + bound_reward[left_index] * np.sqrt(constant_hoeffding / arm_count[left_index])
                    mid_confidence_bound = avg_arm_reward[mid_index] - bound_reward[left_index] * np.sqrt(constant_hoeffding / arm_count[mid_index])
                    right_confidence_bound = avg_arm_reward[right_index] + bound_reward[left_index] * np.sqrt(constant_hoeffding / arm_count[right_index])
                if (left_confidence_bound < mid_confidence_bound) and (right_confidence_bound < mid_confidence_bound):
                    return selected_arm, avg_arm_reward, arm_count, (left_index, mid_index, right_index), mid_index, t
            elif ascending and right_index == J-1:
                if bernstein:
                    if not oracle:
                        mid_std = np.array(dict_rewards[f"arm_{mid_index}"]).std()
                        right_std = np.array(dict_rewards[f"arm_{right_index}"]).std()
                    else:
                        mid_std = func.get_variance(partition[mid_index], partition[mid_index + 1], tau, sigma)
                        right_std = func.get_variance(partition[right_index], partition[right_index + 1], tau, sigma)
                    mid_confidence_bound = avg_arm_reward[mid_index] - mid_std * np.sqrt(constant_bernstein_1 / arm_count[mid_index]) - bound_reward[mid_index] * constant_bernstein_2 / arm_count[mid_index]
                    right_confidence_bound = avg_arm_reward[right_index] + right_std * np.sqrt(constant_bernstein_1 / arm_count[right_index]) + bound_reward[right_index] * constant_bernstein_2 / arm_count[right_index]
                else:
                    mid_confidence_bound = avg_arm_reward[mid_index] - bound_reward[left_index] * np.sqrt(constant_hoeffding / arm_count[mid_index])
                    right_confidence_bound = avg_arm_reward[right_index] + bound_reward[left_index] * np.sqrt(constant_hoeffding / arm_count[right_index])
                if (mid_confidence_bound < right_confidence_bound) and (right_confidence_bound < mid_confidence_bound):
                    return selected_arm, avg_arm_reward, arm_count, (left_index, mid_index, right_index), right_index, t
            elif descending and left_index == 0:
                if bernstein:
                    if not oracle:
                        left_std = np.array(dict_rewards[f"arm_{left_index}"]).std()
                        mid_std = np.array(dict_rewards[f"arm_{mid_index}"]).std()
                    else:
                        left_std = func.get_variance(partition[left_index], partition[left_index + 1], tau, sigma)
                        mid_std = func.get_variance(partition[mid_index], partition[mid_index + 1], tau, sigma)   
                    left_confidence_bound = avg_arm_reward[left_index] + left_std * np.sqrt(constant_bernstein_1 / arm_count[left_index]) + bound_reward[left_index] * constant_bernstein_2 / arm_count[left_index]
                    mid_confidence_bound = avg_arm_reward[mid_index] - mid_std * np.sqrt(constant_bernstein_1 / arm_count[mid_index]) - bound_reward[mid_index] * constant_bernstein_2 / arm_count[mid_index]
                else:
                    left_confidence_bound = avg_arm_reward[left_index] + bound_reward[left_index] * np.sqrt(constant_hoeffding / arm_count[left_index])
                    mid_confidence_bound = avg_arm_reward[mid_index] - bound_reward[left_index] * np.sqrt(constant_hoeffding / arm_count[mid_index])
                if (mid_confidence_bound < left_confidence_bound) and (right_confidence_bound < mid_confidence_bound):
                    return selected_arm, avg_arm_reward, arm_count, (left_index, mid_index, right_index), left_index, t
            peak = (avg_arm_reward[left_index] <= avg_arm_reward[mid_index]) and (avg_arm_reward[mid_index] >= avg_arm_reward[right_index])
        if ascending:
            right_new = min(J - 1, 2 * right_index - mid_index)
            left_index = min(right_new - 2, mid_index)
            mid_index = min(right_new - 1, right_index)
            right_index = right_new
        elif descending:
            left_new = max(0, 2 * left_index - mid_index)
            right_index = max(left_new + 2, mid_index)
            mid_index = max(left_new + 1, left_index)
            left_index = left_new
        elif peak:
            left_index = min(int((left_index + mid_index) / 2), mid_index - 1)
            right_index = max(int((right_index + mid_index) / 2), mid_index + 1)
            if (right_index - left_index == 2):
                consecutive = 1
    print("Max_iter")
    if peak:
        return selected_arm, avg_arm_reward, arm_count, (left_index, mid_index, right_index), mid_index, t
    if ascending:
        return selected_arm, avg_arm_reward, arm_count, (left_index, mid_index, right_index), right_index, t
    if descending:
        return selected_arm, avg_arm_reward, arm_count, (left_index, mid_index, right_index), left_index, t
