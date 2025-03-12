import numpy as np
import scipy.stats as stats


class ABC:
    def __init__(self, simulator, priors, observed_data, epsilon, proposal_std, T, initial_conditions):
        """
        Approximate Bayesian Computation (ABC) using Markov Chain Monte Carlo (MCMC).
        
        Parameters
        ----------
        simulator : class
            The simulator class (not an instance) used for generating synthetic data.
        priors : dict
            A dictionary where keys are parameter names and values are scipy.stats distributions.
        observed_data : ndarray
            The observed dataset to compare with simulated data.
        epsilon : float
            The acceptance threshold for similarity between observed and simulated data.
        proposal_std : dict
            A dictionary specifying the standard deviation for Gaussian proposal distributions for each parameter.
        T : int
            The number of days to simulate.
        initial_conditions : list
            The initial conditions for the simulation.
        """
        self.simulator = simulator
        self.priors = priors
        self.observed_data = observed_data
        self.epsilon = epsilon
        self.proposal_std = proposal_std
        self.T = T
        self.initial_conditions = initial_conditions

    def distance(self, x_obs, x_sim):
        """Compute the distance between observed and simulated data."""
        return np.linalg.norm(x_obs - x_sim)  # Euclidean distance

    def propose_new_params(self, current_params):
        """Propose new parameters using Gaussian perturbations."""
        new_params = {}
        for key in current_params:
            new_params[key] = np.random.normal(current_params[key], self.proposal_std[key])
        return new_params

    def acceptance_ratio(self, theta_current, theta_proposed):
        """Compute the acceptance ratio."""
        prior_current = np.prod([self.priors[k].pdf(theta_current[k]) for k in self.priors])
        prior_proposed = np.prod([self.priors[k].pdf(theta_proposed[k]) for k in self.priors])

        q_current_given_proposed = np.prod([
            stats.norm(theta_proposed[k], self.proposal_std[k]).pdf(theta_current[k])
            for k in self.priors
        ])
        q_proposed_given_current = np.prod([
            stats.norm(theta_current[k], self.proposal_std[k]).pdf(theta_proposed[k])
            for k in self.priors
        ])

        return min(1, (prior_proposed * q_current_given_proposed) / (prior_current * q_proposed_given_current))

    def run(self, n_iterations):
        """Run the ABC-MCMC algorithm for n_iterations."""
        # Initialize the first parameter set by sampling from priors
        theta_current = {key: prior.rvs() for key, prior in self.priors.items()}
        accepted_params = [theta_current]

        for i in range(n_iterations):
            # Step M2: Propose new parameters
            theta_proposed = self.propose_new_params(theta_current)

            # Step M3: Simulate dataset
            # X_sim, _ = create_dataset(self.simulator, {k: stats.uniform(v, 0.01) for k, v in theta_proposed.items()}, 1, self.T, self.initial_conditions)

            # Step M4: Compute distance and check threshold
            if self.distance(self.observed_data.flatten(), X_sim.flatten()) <= self.epsilon:
                # Step M5: Accept with probability alpha
                alpha = self.acceptance_ratio(theta_current, theta_proposed)
                if np.random.rand() < alpha:
                    theta_current = theta_proposed  # Accept new parameters

            # Store the accepted parameters
            accepted_params.append(theta_current)

        return accepted_params