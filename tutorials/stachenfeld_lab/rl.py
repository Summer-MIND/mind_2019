import numpy as np
import util

def as_one_hot(ind, n):
    """Turn ind into onehot."""
    vec = np.zeros(n)
    vec[ind] = 1
    return vec

def step(state_ind, transmat, atol=1e-7):
    """Take a random step in the task specified by transition matrix."""
    pvals = np.round(transmat[state_ind] / atol) * atol
    pvals = pvals / np.sum(pvals)
    next_state_onehot = np.random.multinomial(1, pvals=pvals)
    next_state = np.where(next_state_onehot)[0][0]
    return next_state

def value_update(reward, value, current_state, next_state, learning_rate, discount, features=None):
    """Update a value function in the supplied feature space.

    Args:
        reward: scalar reward received at current state
        value: value vector of len n_state
        current_state: state index for next state
        next_state: state index for next state
        learning_rate: scalar learning rate
        discount: scalar discount factor
        features: feature representation of each state
    
    Returns:
        value: sum of expected future reward from each state
        value_weights: weights of value function in feature space
    """
    n = len(value)
    value_shape = value.shape
    value = value.reshape(-1, 1)
    if features is None:
        features = np.eye(n)
    features_pinv = np.linalg.pinv(features)

    value_weights = np.dot(features_pinv, value)

    # prediction error
    # expected = value[current_state]
    # actual = reward + discount * value[next_state]
    expected = np.dot(features[current_state].reshape(1, -1), value_weights)
    actual = reward + discount * np.dot(features[next_state].reshape(1, -1), value_weights)
    pred_err = actual - expected
    
    feature_scale = features[current_state].reshape(-1, 1)
    value_weights += learning_rate * pred_err * feature_scale
    value = np.dot(features, value_weights)
    value = value.reshape(*value_shape)
    return value, value_weights


def reward_amt(value, reward_vec, adj, softmax_inv_temp, discount, start_prob=None):
    """Estimate discounted amount reward received given softmax policy over value.

    A way to assess how well your value function is doing.

    Args:
        value: value vector of len n_state
        reward_vec: reward vector of len n_state
        adj: adjacency matrix
        softmax_inv_temp: inverse temperature parameter for softmax policy
        discount: discount factor
        start_prob: probability of starting in each state (len n_state), default
            None (evaluates to uniform)
    
    Returns:
        sum of expected future reward from each state
    """
    n = len(reward_vec)
    softmax_value = np.exp(softmax_inv_temp *  value) / np.sum(np.exp(softmax_inv_temp*value))
    policy = adj * softmax_value.reshape(1, -1)
    policy = util.l1_normalize_rows(policy)
    sr = np.linalg.pinv(np.eye(n) - discount * policy)
    value = np.dot(sr, reward_vec.reshape(-1, 1)).reshape(-1)
    if start_prob is None:
        start_prob = np.ones(n)*1. / n
    else:
        start_prob = start_prob.reshape(n)
    return np.sum(value * start_prob)

