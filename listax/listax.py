import jax
import numpy as np
import jax.numpy as jnp
import tskit, msprime
import matplotlib.pyplot as plt

class LiStephens():
    def __init__(self, h, m, gamma, N_est, mu):

        self.L, self.N = h.shape # Number of sites, Number of haplotypes

        self.h = h # Haplotype matrix (L, N)
        self.mu = mu # Mutation rate (scalar, assumed constant across sites)
        self.gamma = gamma # scalar parameter for rho calculation
        self.m = m # Vector specifying the recombination distance between variants in centimorgans (L-1,)
        self.N_est = N_est # Effective population size (scalar)

        # Define N_est = 4 Ne/N where Ne is the effective diploid population size 
        # (ie half of the haploid effective population size).

        # recombination probability vector
        N_e = 4 * self.N_est / self.N
        # raise numpy array to power:

        self.rho = 1 - jnp.expm1(- N_e * jnp.power(self.m, self.gamma))

        # Transition probability matrix (N, N)
        self.pi = (1 - jnp.eye(self.N, self.N)) / (self.N - 1)

    def _get_emission_kernel(self, l):
            theta = jnp.where(
                self.h[l, :][:, None] == self.h[l, :][:, None].T,
                1 - self.mu,
                self.mu
            )

            return theta

    def forward(self, target_l):
        # alpha: (L, N)
        # Forward recursion

        def body(l, alpha):

            # compute the emission kernel
            theta = self._get_emission_kernel(l)

            alpha_normalized = alpha / jnp.sum(alpha, axis=0, keepdims=True)

            part1 = (1 - self.rho[l-1]) * alpha_normalized
            
            part2 = self.rho[l-1] * self.pi
            
            alpha = (part1 + part2)
            alpha = theta @ alpha


            return alpha

        # Initialize alpha for the first site
        alpha = self._get_emission_kernel(l=0) * self.pi

        # Loop over all sites (except the first one)
        # from 1 -> target_l
        alpha = jax.lax.fori_loop(1, target_l+1, body, alpha)

        return alpha

    def backward(self, target_l):
        # Backward recursion

        def body(l, beta):
            l = self.L - l + 1

            theta = self._get_emission_kernel(l)

            beta = (1 - self.rho[l]) * (beta @ theta) / jnp.sum(beta @ theta * self.pi, axis=0) + self.rho[l]
            
            return beta

        # Initialize beta for the last site
        beta = jnp.ones((self.N,self.N))

        # Loop over all sites (except the last one)
        # from L -> target_l + 1
        beta = jax.lax.fori_loop(1, self.L - target_l + 1, body, beta)

        return beta

    def compute_posterior_prob(self, alpha, beta):

        # Compute alpha_l * beta_l
        p = alpha * beta

        # Compute the normalization constant
        normalization_factor = jnp.sum(p, axis=0)

        normalization_factor = jnp.where(normalization_factor == 0, jnp.finfo(float).smallest_normal, normalization_factor)

        p /= normalization_factor

        return p

    def calculate_local_distance(self, p):
        # Calculate local distance matrix from posterior probabilities

        # Smallest observable posterior probability
        epsilon = jnp.finfo(float).eps

        # Calculate local distance matrix
        d = -0.5 * (jnp.log(np.maximum(p, epsilon)) + jnp.log(jnp.maximum(p.T, epsilon)))

        # Set diagonal entries to 0
        d = d * (1 - jnp.eye(self.N, self.N))

        return d

    def run(self, target_l):

        # Run forward and backward recursions
        alpha = self.forward(target_l)
        beta = self.backward(target_l)

        # Calculate posterior probabilities
        p = self.compute_posterior_prob(alpha, beta)

        # Calculate local distance matrix
        d = self.calculate_local_distance(p)

        return p, d
