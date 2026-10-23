"""Numerical routines extracted from the recorded experiment implementation."""
import numpy as np

def instance(seed, m=192, n=64, tau=0.1):
    rng = np.random.default_rng(seed)
    A = rng.uniform(0, 1, (m, n))
    A /= A.sum(axis=0)
    xs = np.exp(rng.uniform(np.log(0.1), np.log(3.0), n))
    vs = A @ xs
    # grad KL(Ax,b) at xs equals -tau*1, so xs is exactly optimal.
    b = np.exp(tau) * vs
    return A, b, xs, vs, tau


def gradient(A, b, x, tau):
    return A.T @ np.log((A @ x) / b) + tau


def gap(A, vs, x):
    """F(x)-F(xs) = KL(Ax, Axs), no approximate reference or clipping."""
    r = (A @ x - vs) / vs
    phi = (1 + r) * np.log1p(r) - r
    small = np.abs(r) < 1e-4
    z = r[small]
    phi[small] = z*z*(0.5 + z*(-1/6 + z*(1/12 + z*(-1/20 + z/30))))
    return float(vs @ phi)


def schedule(count):
    theta = np.empty(count)
    t = 1.0
    for k in range(count):
        theta[k] = 1/t
        t = (1 + np.sqrt(1 + 4*t*t))/2
    return theta


def step(method, A, b, tau, x, aux, theta, lam):
    if method == "BPG":
        y = x
        nxt = np.exp(np.log(y) - lam*gradient(A, b, y, tau))
        return nxt, aux, 0.0
    if method == "AEPG":
        y = (1-theta)*x + theta*aux
        raw = np.exp(np.log(y) - lam*gradient(A, b, y, tau))
        floor = (1-theta)*x
        nxt = np.maximum(raw, floor)  # Required by their algorithm, not a safeguard.
        new_aux = (nxt-floor)/theta
        return nxt, new_aux, 0.0
    if method == "DI-BFB":
        logx = np.log(x)
        logy = (1-theta)*logx + theta*aux
        lognext = logy - lam*gradient(A, b, np.exp(logy), tau)
        # aux stores log(z); never clip, project, or restart this algorithm.
        return np.exp(lognext), logx+(lognext-logx)/theta, 0.0
    raise ValueError(method)
