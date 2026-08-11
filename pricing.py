import numpy as np
from scipy.special import ndtr

def _norm_cdf(x):
    return ndtr(x)

def _norm_pdf(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

def _compute_d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def black_scholes_call(S, K, T, r, sigma):
    d1, d2 = _compute_d1_d2(S, K, T, r, sigma)
    call_px = S * _norm_cdf(d1) - K * np.exp(-r * T) * _norm_cdf(d2)
    return call_px

def black_scholes_put(S, K, T, r, sigma):
    d1, d2 = _compute_d1_d2(S, K, T, r, sigma)
    put_px = K * np.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
    return put_px

def delta_call(S, K, T, r, sigma):
    d1, _ = _compute_d1_d2(S, K, T, r, sigma)
    return _norm_cdf(d1)

def delta_put(S, K, T, r, sigma):
    d1, _ = _compute_d1_d2(S, K, T, r, sigma)
    return _norm_cdf(d1) - 1

def gamma_bs(S, K, T, r, sigma):
    d1, _ = _compute_d1_d2(S, K, T, r, sigma)
    return _norm_pdf(d1) / (S * sigma * np.sqrt(T))