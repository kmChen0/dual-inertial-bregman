"""Numerical routines extracted from the recorded experiment implementation."""
import numpy as np

def gain_theta(gain, previous_gain, previous_theta):
    """Stable positive root of a theta² + theta - 1 = 0."""
    a = gain / (previous_gain * previous_theta**2)
    return 2 / (1 + np.sqrt(1 + 4*a))
