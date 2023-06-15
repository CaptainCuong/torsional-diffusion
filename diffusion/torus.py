import numpy as np
import tqdm
import os

# Page 4 (Formula 3), 17
def p(x, sigma, N=10):
    # p_.shape: (5001, 5001)
    p_ = 0
    for i in tqdm.trange(-N, N + 1):
        p_ += np.exp(-(x + 2 * np.pi * i) ** 2 / 2 / sigma ** 2)
    return p_


# Page 4, 17
def grad(x, sigma, N=10):
    # x.shape: (5001,)
    # sigma.shape: (5001, 1)
    # p_: row: sigma, col: x
    # p_.shape: (5001, 5001)
    p_ = 0
    for i in tqdm.trange(-N, N + 1):
        p_ += (x + 2 * np.pi * i) / sigma ** 2 * np.exp(-(x + 2 * np.pi * i) ** 2 / 2 / sigma ** 2)
    return p_


X_MIN, X_N = 1e-5, 5000  # relative to pi
SIGMA_MIN, SIGMA_MAX, SIGMA_N = 3e-3, 2, 5000  # relative to pi

# log(x) & log(sigma) are uniformly distributed
x = 10 ** np.linspace(np.log10(X_MIN), 0, X_N + 1) * np.pi # 1e-5 * np.pi < x < np.pi
sigma = 10 ** np.linspace(np.log10(SIGMA_MIN), np.log10(SIGMA_MAX), SIGMA_N + 1) * np.pi # 3e-3 * np.pi < sigma < 2 * np.pi

# Precomputed score. Page 4
'''
2 dimension
    dim 0: uniformly distributed by log(sigma)
    dim 1: uniformly distributed by log(x)
'''
if os.path.exists('.p.npy'):
    p_ = np.load('.p.npy')
    score_ = np.load('.score.npy')
else:
    p_ = p(x, sigma[:, None], N=100)
    np.save('.p.npy', p_)

    '''
    p = log(sum())
    --> score = d_sum()/sum()
    '''
    score_ = grad(x, sigma[:, None], N=100) / p_
    np.save('.score.npy', score_)


def score(x, sigma): # 0 <= x <= 2*pi
    '''
    Convert to index of (sigma) and (x) in score_
    '''
    x = (x + np.pi) % (2 * np.pi) - np.pi # -pi < x < pi
    sign = np.sign(x)
    x = np.log(np.abs(x) / np.pi) # -22.7 < x < -2
    x = (x - np.log(X_MIN)) / (0 - np.log(X_MIN)) * X_N # -4840 < X < 5000
    x = np.round(np.clip(x, 0, X_N)).astype(int) # 0 < x < 5000
    sigma = np.log(sigma / np.pi) # -6 < sigma < 0.7
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N # 0 < sigma < 5000
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int) # 0 <= sigma <= 5000
    return -sign * score_[sigma, x]


def p(x, sigma):
    x = (x + np.pi) % (2 * np.pi) - np.pi
    x = np.log(np.abs(x) / np.pi)
    x = (x - np.log(X_MIN)) / (0 - np.log(X_MIN)) * X_N
    x = np.round(np.clip(x, 0, X_N)).astype(int)
    sigma = np.log(sigma / np.pi)
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return p_[sigma, x]


def sample(sigma):
    out = sigma * np.random.randn(*sigma.shape)
    out = (out + np.pi) % (2 * np.pi) - np.pi
    return out


score_norm_ = score(
    sample(sigma[None].repeat(10000, 0).flatten()), # repeat 10000 times in dimension 0, then flattening into 1 dimension
    sigma[None].repeat(10000, 0).flatten()
).reshape(10000, -1)
score_norm_ = (score_norm_ ** 2).mean(0) # Expected score^2 for each sigma


def score_norm(sigma):
    sigma = np.log(sigma / np.pi)
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return score_norm_[sigma]
