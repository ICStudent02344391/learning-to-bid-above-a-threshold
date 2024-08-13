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
    selected_arm = [] # selected arms up to round t
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
    J = len(partition) - 1 # number of arms
    avg_arm_reward = np.zeros(J) # array of empirical average rewards
    arm_count = np.zeros(J) # array of counts
    if bernstein:
        dict_rewards = {} # dictionary to save all the rewards of each arm for empirical standard deviation computation
    arm_ucb = np.zeros(J) # array of UCB
    selected_arm = [] # selected arms up to rount t
    t = J
    bound_reward = np.array([1 - j/J for j in range(J)]) # upper bounds of the reward for each arm
    # Compute constants in bound's formulas
    constant_bernstein_1 = 2 * np.log(1/delta)
    constant_bernstein_2 = 3 * np.log(1/delta)
    constant_hoeffding = np.log(1/delta)/2
    # Initialisation
    for j in range(J):
        threshold = sigma * rndm.randn() + tau # new threshold
        bid = partition[j] + (partition[j + 1] - partition[j]) * rndm.rand() # bid of arm j
        arm_count[j] += 1 # update the count of arm j
        rew = func.reward(bid, threshold) # get the reward
        avg_arm_reward[j] = ((arm_count[j] - 1) * avg_arm_reward[j] + rew) / arm_count[j] # update empirical average reward of arm j
        dict_rewards[f"arm_{j}"] = [rew] # update the rewards of arm j
        selected_arm.append(j) # arm j selected at round j
        if bernstein:
            arm_ucb[j] = avg_arm_reward[j] + bound_reward[j] * constant_bernstein_2 # Compute the Bernstein's bound for arm j
        else:
            arm_ucb[j] = avg_arm_reward[j] + bound_reward[j] * np.sqrt(constant_hoeffding) # Compute the Hoeffding's bound for arm j
    # Exploitation of the arm with the highest UCB
    while t < itermax:
        arm_index = np.argmax(arm_ucb) # get the arm with the highest UCB
        threshold = sigma * rndm.randn() + tau # new threshold
        bid = partition[arm_index] + (partition[arm_index + 1] - partition[arm_index]) * rndm.rand() # new bid of the arm
        arm_count[arm_index] += 1 # update the count of the arm
        rew = func.reward(bid, threshold) # get the reward
        avg_arm_reward[arm_index] = ((arm_count[arm_index] - 1) * avg_arm_reward[arm_index] + rew) / arm_count[arm_index] # update the empirical average reward
        dict_rewards[f"arm_{arm_index}"] = [rew] # update the rewards of the arm
        selected_arm.append(arm_index) # the arm is selected at round t
        if bernstein:
            if not oracle:
                arm_std = np.array(dict_rewards[f"arm_{arm_index}"]).std() # get the empirical standard deviation
            else:
                arm_std = np.sqrt(func.get_variance(partition[arm_index], partition[arm_index + 1], tau, sigma)) # get the theoritical standard deviation
            arm_ucb[arm_index] = avg_arm_reward[arm_index] + arm_std * np.sqrt(constant_bernstein_1 / arm_count[arm_index]) + bound_reward[arm_index] * constant_bernstein_2 / arm_count[arm_index] # compute the Bernstein's UCB
        else:
            arm_ucb[arm_index] = avg_arm_reward[arm_index] + bound_reward[arm_index] * np.sqrt(constant_hoeffding / arm_count[arm_index]) # compute the Hoeffding's UCB
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
    rndm.seed(seed)
    J = len(partition) - 1 # number of arms
    avg_arm_reward = np.zeros(J) # array of empirical average rewards
    arm_count = np.zeros(J) # array of counts
    if bernstein:
        dict_rewards = {} # dictionary to save all the rewards of each arm for empirical standard deviation computation
    selected_arm = [] # selected arms up to rount t

    bound_reward = np.array([1 - j/J for j in range(J)]) # upper bounds of the reward for each arm
    # Compute constants in bound's formulas
    constant_bernstein_1 = 2 * np.log(1/delta)
    constant_bernstein_2 = 3 * np.log(1/delta)
    constant_hoeffding = np.log(1/delta)/2
    # Left, middle and right arm initialisation
    left_index = 0
    mid_index = int((J)/2)
    right_index = J - 1
    consecutive = 0 # Indicator that the three arms are consecutive
    t = 0
    while 1:
        # Play three arms
        for ind in [left_index, mid_index, right_index]:
            threshold = sigma * rndm.randn() + tau # new threshold
            bid = partition[ind] + (partition[ind + 1] - partition[ind]) * rndm.rand() # new bid
            rew = func.reward(bid, threshold) # get the reward
            arm_count[ind] += 1 # update the count
            avg_arm_reward[ind] = ((arm_count[ind] - 1) * avg_arm_reward[ind] + rew) / arm_count[ind] # update the empirical average reward
            if bernstein: # update the rewards if bernstein
                if not (f"arm_{ind}" in dict_rewards):
                    dict_rewards[f"arm_{ind}"] = [] 
                dict_rewards[f"arm_{ind}"].append(rew)
            selected_arm.append(ind) # selected arm at round t
            t += 1 # new round
             # stop the while loop if round itermax
            if t == itermax:
                break
        if t == itermax:
            break
        # Get the configuration of the three arms
        ascending = (avg_arm_reward[left_index] <= avg_arm_reward[mid_index]) and (avg_arm_reward[mid_index] <= avg_arm_reward[right_index])
        descending = (avg_arm_reward[left_index] > avg_arm_reward[mid_index]) and (avg_arm_reward[mid_index] > avg_arm_reward[right_index])
        peak = (avg_arm_reward[left_index] < avg_arm_reward[mid_index]) and (avg_arm_reward[mid_index] > avg_arm_reward[right_index])
        # if the arms are consecutive, UCBs and LCBs need to be computed
        if consecutive:    
            if peak: # peak configuration
                if bernstein: 
                    if not oracle:
                        # empirical standard deviations of the three arms
                        left_std = np.array(dict_rewards[f"arm_{left_index}"]).std() 
                        mid_std = np.array(dict_rewards[f"arm_{mid_index}"]).std()
                        right_std = np.array(dict_rewards[f"arm_{right_index}"]).std()
                    else:
                        # theoretical standard deviations of the three arms
                        left_std = np.sqrt(func.get_variance(partition[left_index], partition[left_index + 1], tau, sigma))
                        mid_std = np.sqrt(func.get_variance(partition[mid_index], partition[mid_index + 1], tau, sigma))
                        right_std = np.sqrt(func.get_variance(partition[right_index], partition[right_index + 1], tau, sigma))
                    # Compute the UCBs of the left and right arms et the LCB of the middle arm (Bernstein)
                    left_confidence_bound = avg_arm_reward[left_index] + left_std * np.sqrt(constant_bernstein_1 / arm_count[left_index]) + bound_reward[left_index] * constant_bernstein_2 / arm_count[left_index]
                    mid_confidence_bound = avg_arm_reward[mid_index] - mid_std * np.sqrt(constant_bernstein_1 / arm_count[mid_index]) - bound_reward[mid_index] * constant_bernstein_2 / arm_count[mid_index]
                    right_confidence_bound = avg_arm_reward[right_index] + right_std * np.sqrt(constant_bernstein_1 / arm_count[right_index]) + bound_reward[right_index] * constant_bernstein_2 / arm_count[right_index]
                else:
                    # compute the UCBs of the left and right arms et the LCB of the middle arm (Hoeffding)
                    left_confidence_bound = avg_arm_reward[left_index] + bound_reward[left_index] * np.sqrt(constant_hoeffding / arm_count[left_index])
                    mid_confidence_bound = avg_arm_reward[mid_index] - bound_reward[left_index] * np.sqrt(constant_hoeffding / arm_count[mid_index])
                    right_confidence_bound = avg_arm_reward[right_index] + bound_reward[left_index] * np.sqrt(constant_hoeffding / arm_count[right_index])
                # check if the middle arm is the optimal arm given the bounds
                if (left_confidence_bound < mid_confidence_bound) and (right_confidence_bound < mid_confidence_bound):
                    return selected_arm, avg_arm_reward, arm_count, (left_index, mid_index, right_index), mid_index, t
            elif ascending and right_index == J-1: # ascending configuration at the far right
                if bernstein:
                    if not oracle:
                        # compute the empirical standard deviations
                        mid_std = np.array(dict_rewards[f"arm_{mid_index}"]).std()
                        right_std = np.array(dict_rewards[f"arm_{right_index}"]).std()
                    else:
                        # compute the theoritical standard deviations
                        mid_std = np.sqrt(func.get_variance(partition[mid_index], partition[mid_index + 1], tau, sigma))
                        right_std = np.sqrt(func.get_variance(partition[right_index], partition[right_index + 1], tau, sigma))
                    # Compute the UCB of the middle arm et the LCB of the right arm (Bernstein)
                    mid_confidence_bound = avg_arm_reward[mid_index] + mid_std * np.sqrt(constant_bernstein_1 / arm_count[mid_index]) + bound_reward[mid_index] * constant_bernstein_2 / arm_count[mid_index]
                    right_confidence_bound = avg_arm_reward[right_index] - right_std * np.sqrt(constant_bernstein_1 / arm_count[right_index]) - bound_reward[right_index] * constant_bernstein_2 / arm_count[right_index]
                else:
                    # Compute the UCB of the middle arm et the LCB of the right arm (Hoeffding)
                    mid_confidence_bound = avg_arm_reward[mid_index] + bound_reward[left_index] * np.sqrt(constant_hoeffding / arm_count[mid_index])
                    right_confidence_bound = avg_arm_reward[right_index] - bound_reward[left_index] * np.sqrt(constant_hoeffding / arm_count[right_index])
                # check if the right arm is the optimal arm given the bounds
                if (mid_confidence_bound < right_confidence_bound) and (right_confidence_bound < mid_confidence_bound):
                    return selected_arm, avg_arm_reward, arm_count, (left_index, mid_index, right_index), right_index, t
            elif descending and left_index == 0:  # descending configuration at the far left
                if bernstein:
                    if not oracle:
                        # compute the empirical standard deviations
                        left_std = np.array(dict_rewards[f"arm_{left_index}"]).std()
                        mid_std = np.array(dict_rewards[f"arm_{mid_index}"]).std()
                    else:
                        # compute the theoritical standard deviations
                        left_std = np.sqrt(func.get_variance(partition[left_index], partition[left_index + 1], tau, sigma))
                        mid_std = np.sqrt(func.get_variance(partition[mid_index], partition[mid_index + 1], tau, sigma))
                    # compute the UCB of the middle arm et the LCB of the left arm (Bernstein)
                    left_confidence_bound = avg_arm_reward[left_index] - left_std * np.sqrt(constant_bernstein_1 / arm_count[left_index]) - bound_reward[left_index] * constant_bernstein_2 / arm_count[left_index]
                    mid_confidence_bound = avg_arm_reward[mid_index] + mid_std * np.sqrt(constant_bernstein_1 / arm_count[mid_index]) + bound_reward[mid_index] * constant_bernstein_2 / arm_count[mid_index]
                else:
                    # compute the UCB of the middle arm et the LCB of the left arm (Hoeffding)
                    left_confidence_bound = avg_arm_reward[left_index] - bound_reward[left_index] * np.sqrt(constant_hoeffding / arm_count[left_index])
                    mid_confidence_bound = avg_arm_reward[mid_index] + bound_reward[left_index] * np.sqrt(constant_hoeffding / arm_count[mid_index])
                # check if the left arm is the optimal arm given the bounds
                if (mid_confidence_bound < left_confidence_bound) and (right_confidence_bound < mid_confidence_bound):
                    return selected_arm, avg_arm_reward, arm_count, (left_index, mid_index, right_index), left_index, t
            peak = (avg_arm_reward[left_index] <= avg_arm_reward[mid_index]) and (avg_arm_reward[mid_index] >= avg_arm_reward[right_index])
        # new left, middle and right arms given the configuration
        if ascending:
            # right shift
            right_new = min(J - 1, 2 * right_index - mid_index)
            left_index = min(right_new - 2, mid_index)
            mid_index = min(right_new - 1, right_index)
            right_index = right_new
        elif descending:
            # left shift
            left_new = max(0, 2 * left_index - mid_index)
            right_index = max(left_new + 2, mid_index)
            mid_index = max(left_new + 1, left_index)
            left_index = left_new
        elif peak:
            # halving
            left_index = min(int((left_index + mid_index) / 2), mid_index - 1)
            right_index = max(int((right_index + mid_index) / 2), mid_index + 1)
            # check if the three arm are consecutive
            if (right_index - left_index == 2):
                consecutive = 1
    print("Max_iter")
    # Return all the results and the best arm given the configuration
    if peak:
        return selected_arm, avg_arm_reward, arm_count, (left_index, mid_index, right_index), mid_index, t
    if ascending:
        return selected_arm, avg_arm_reward, arm_count, (left_index, mid_index, right_index), right_index, t
    if descending:
        return selected_arm, avg_arm_reward, arm_count, (left_index, mid_index, right_index), left_index, t
