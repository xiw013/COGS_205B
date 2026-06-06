import math
from scipy.special import comb
from scipy.integrate import quad

class BayesFactor:
    def __init__(self, n, k):
        if not isinstance(n, int) or isinstance(n, bool) or not isinstance(k, int) or isinstance(k, bool):
            raise ValueError("All input must be a non-negative integer.")
        if n < 0 or k < 0:
            raise ValueError("All input must be a non-negative integer.")
        if k > n:
            raise ValueError("K cannot be greater than N.")
        
        self.n = n
        self.k = k

    def likelihood(self, theta):
        if not isinstance(theta, (int, float)):
            raise ValueError("The input theta need to be a number.")
        if not (0 <= theta <= 1):
            raise ValueError("The input theta has to be between 0 and 1 (inclusive).")
        
        # Binomial likelihood: choose(n, k) * theta^k * (1-theta)^(n-k)
        return comb(self.n, self.k) * (theta**self.k) * ((1 - theta)**(self.n - self.k))

    def evidence_slab(self):
        """
        Marginal likelihood under theta ~ Uniform(0, 1).
        Integral of likelihood(theta) * 1 d(theta) from 0 to 1.
        Evidence = 1 / (n + 1).
        """
        try:
            return 1.0 / (self.n + 1)
        except ZeroDivisionError:
            return 0.0

    def evidence_spike(self):
        """
        Marginal likelihood under theta ~ Uniform(0.47, 0.53).
        Integral of likelihood(theta) * (1 / (0.53 - 0.47)) d(theta) from 0.47 to 0.53.
        """
        a, b = 0.47, 0.53
        prior_pdf = 1.0 / (b - a)
        
        # Integration of likelihood function over the spike interval
        # We use scipy.integrate.quad for numerical integration
        result, error = quad(lambda theta: self.likelihood(theta) * prior_pdf, a, b)
        return result

    def bayes_factor(self):
        """
        B = evidence_spike / evidence_slab
        """
        slab = self.evidence_slab()
        spike = self.evidence_spike()
        
        if slab == 0:
            raise ZeroDivisionError("Evidence slab is zero, cannot compute Bayes Factor.")
            
        return spike / slab