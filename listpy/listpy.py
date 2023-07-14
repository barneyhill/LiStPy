import torch

class LiStephens(torch.nn.Module):
    def __init__(self, h, m, gamma, N_est, mu):

        self.L, self.N = h.shape # Number of sites, Number of haplotypes

        self.h = h # Haplotype matrix (L, N)
        self.mu = mu # Mutation rate (scalar, assumed constant across sites)
        self.gamma = gamma # scalar parameter for rho calculation
        self.m = m # Vector specifying the recombination distance between variants in centimorgans (L-1,)
        self.N_est = N_est # Effective population size (scalar)
        self.epslion = torch.finfo(float).eps # smallest possible float

        # Define N_est = 4 Ne/N where Ne is the effective diploid population size 
        # (ie half of the haploid effective population size).

        N_e = 4 * self.N_est / self.N

        # recombination probability vector

        self.rho = - torch.expm1(- N_e * torch.pow(self.m, self.gamma))

        # Transition probability matrix (N, N)
        self.pi = (1 - torch.eye(self.N, self.N)) / (self.N - 1)

    def get_emission_kernel(self, l):

        # if h_ij == h_ji then theta_ij = 1 - mu else theta_ij = mu
        
        theta = torch.where(
            self.h[l, :][:, None] == self.h[l, :][:, None].T,
            1 - self.mu,
            self.mu
        )

        return theta

    def forward(self, target_l):
        # alpha: (N, N)
        # Forward recursion

        # Initialize alpha for the first site
        alpha = self.get_emission_kernel(l=0) * self.pi

        # Loop over all sites (except the first one)
        # from 1 -> target_l
        for l in range(1, target_l+1):

            # compute the emission kernel
            theta = self.get_emission_kernel(l)

            alpha_normalized = alpha / torch.sum(alpha, axis=0)

            alpha = (1 - self.rho[l-1]) * alpha_normalized + self.rho[l-1] * self.pi

            alpha = theta * alpha

            alpha = alpha.fill_diagonal_(0)


        return alpha

    def backward(self, target_l):
        # beta: (N, N)
        # Backward recursion

        # Initialize beta for the last site
        beta = torch.ones(self.N, self.N)

        # Loop over all sites (except the last one)
        # from L -> target_l
        for l in range(1, self.L - target_l):

            l = self.L - l - 1

            theta = self.get_emission_kernel(l+1)

            beta = theta * beta

            beta = (1 - self.rho[l]) * beta / torch.sum(beta * self.pi, axis=0) + self.rho[l]
            
            # Set diagonal entries to 0
            beta = beta.fill_diagonal_(0)


        return beta

    def compute_posterior_prob(self, alpha, beta):

        alpha, beta = torch.clamp(alpha, min=0, max=1), torch.clamp(beta, min=0, max=1)

        # Compute alpha_l * beta_l
        p = alpha * beta

        # Compute the normalization constant
        normalization_factor = torch.sum(p, axis=0)

        p /= normalization_factor

        return p

    def calculate_local_distance(self, p):
        # Calculate local distance matrix from posterior probabilities

        # Calculate local distance matrix
        p_clamped = torch.clamp(p, min=self.epslion)
        d = -0.5 * (torch.log(p_clamped) + torch.log(p_clamped.T))

        # Set diagonal entries to 0
        d = d * (1 - torch.eye(self.N, self.N))

        return d

    def run(self, target_l):

        # Run forward and backward recursions
        alpha = self.forward(target_l)
        beta = self.backward(target_l)

        # Calculate posterior probabilities
        p = self.compute_posterior_prob(alpha, beta)

        # Calculate local distance matrix
        d = self.calculate_local_distance(p)

        return p, d, alpha, beta
