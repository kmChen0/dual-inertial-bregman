"""Numerical routines extracted from the recorded experiment implementation."""
import math
import numpy as np

def theta(A, lam):
    t = (math.sqrt(lam) + math.sqrt(lam + 4 * A)) / 2
    return math.sqrt(lam) / t, t * t


def dexpo(first, second):
    d = np.asarray(first) - np.asarray(second)
    return float(np.sum(np.exp(second) * (np.expm1(d) - d)))


def kl(a, u):
    a = np.asarray(a)
    pos = a > 0
    result = float(np.sum(np.exp(u) - a))
    result += float(np.sum(a[pos] * (np.log(a[pos]) - u[pos])))
    return result


def correction(u, s, p):
    return dexpo(s, u) - dexpo(u + p * (s - u), u) / p**2
