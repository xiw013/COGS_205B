import scipy.integrate
from scipy.special import comb
import numpy as np
import numbers

class BayesFactor:

    def __init__(self, n, k):
        # checking for n and k

        for value in [n, k]:
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("All input must be a non-negative integer.")
        if not (k <= n):
            raise ValueError("K cannot be greater than N.")

        self.n = n
        self.k = k

    def likelihood(self, theta):
        # checking for theta
        if not isinstance(theta, numbers.Real) or isinstance(theta, bool):
            raise ValueError("The input theta need to be a number.")
        if not (0 <= theta <= 1):
            raise ValueError("The input theta has to be between 0 and 1 (inclusive).")

        l = comb(self.n, self.k, exact=True) * (theta**(self.k)) * ((1-theta)**(self.n - self.k))
        return l
    
    def evidence_slab(self):
        # assume slab prior = Uniform(0,1)
        integrand = lambda theta: self.likelihood(theta)
        result, _ = scipy.integrate.quad(integrand, 0, 1)
        return result

    def evidence_spike(self):
        theta = 0.5
        result = self.likelihood(theta)
        return result
     
    def bayes_factor(self):
        slab = self.evidence_slab()
        spike = self.evidence_spike()

        if spike == 0: # I think it is impossible to get spike = 0 since theta = 0.5
            raise ZeroDivisionError("Evidence under spike model is zero.")

        return slab / spike

