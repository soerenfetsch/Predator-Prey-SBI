import numpy as np
import scipy.stats as stats
import logging
import os
import json
import pandas as pd

from tqdm import tqdm


def create_array_dataset(simulator, priors, n_samples, T, num_steps, randomize=True):
    """
    Create a dataset for SBI using parameter samples from priors and corresponding simulated outputs.
    Output is formatted into numpy arrays for usage in ABC algorithms.
    
    Parameters
    ----------
    simulator : class
        The simulator class (not an instance) to be used for generating the data.
    priors : dict
        Dictionary where keys are parameter names and values are sampling functions.
    n_samples : int
        Number of simulation runs to perform.
    T : int
        Number of days to simulate.
    num_steps : int
        Number of steps to simulate.
    randomize : bool, optional
        Whether to add noise to the simulation outputs and autocorrelation randomization
        to the growth rates. Default is True.

    Returns
    -------
    X_data : list
        The observables with time of shape (n_samples, (num_observables + 1), T).
        Time is included as the first row of the observables.
    y_data : ndarray
        The target data (parameters) of shape (n_samples, len(priors)).
    """
    X_data, y_data = [], []
    t_range = (0, T, num_steps)
    
    while len(X_data) < n_samples:
        # Sample parameters from the prior distributions
        params = {key: prior.rvs() for key, prior in priors.items()}

        try:
            # Instantiate and run the simulator
            sim_instance = simulator(params, t_range, randomize=randomize)
            sol = sim_instance.simulate()

            # Extract time and observables
            time_values = sol.t  # Shape (T,)
            observables = np.array(sol.y)  # Shape (num_observables, T)

            # Stack time as the first row
            data_array = np.vstack((time_values, observables))  # Shape (num_observables + 1, T)

        except Exception as e:
            logging.error(f"Error during simulation: {e}")
            logging.info(f"Parameters: {params}")
            continue

        X_data.append(data_array)
        y_data.append(list(params.values()))
    
    return np.array(X_data), np.array(y_data)


def create_array_simulation(simulator, params, T, num_steps, randomize=True):
    """
    Create a single simulation for SBI using the given parameters and simulator.
    Output is formatted into numpy arrays for usage in ABC algorithms.
    
    Parameters
    ----------
    simulator : class
        The simulator class (not an instance) to be used for generating the data.
    params : dict
        Dictionary where keys are parameter names and values are parameter values.
    T : int
        Number of days to simulate.
    num_steps : int
        Number of steps to simulate.
    randomize : bool, optional
        Whether to add noise to the simulation outputs and autocorrelation randomization
        to the growth rates. Default is True.
    
    Returns
    -------
    X_data : ndarray
        The observables with time of shape ((num_observables + 1), T).
        Time is included as the first row of the observables.
    """
    t_range = (0, T, num_steps)
    max_iterations = 50
    
    data_array = None
    iterations = 0
    while data_array is None and iterations < max_iterations:
        try:
            # Instantiate and run the simulator
            sim_instance = simulator(params, t_range, randomize=randomize)
            sol = sim_instance.simulate()

            # Extract time and observables
            time_values = sol.t  # Shape (T,)
            observables = np.array(sol.y)  # Shape (num_observables, T)

            # Stack time as the first row
            data_array = np.vstack((time_values, observables))  # Shape (num_observables + 1, T)

        except Exception as e:
            logging.error(f"Error during simulation: {e}")
            logging.info(f"Parameters: {params}")
            iterations += 1
            if iterations >= max_iterations:
                logging.error("Maximum iterations reached.")
            continue
    
    return data_array


class ABC:
    def __init__(self, simulator, priors, observed_data, epsilon, proposal_std, T, num_steps, randomize=True, redraw_iterations=500, save_dir=None):
        """
        Approximate Bayesian Computation (ABC) using Markov Chain Monte Carlo (MCMC).
        
        Parameters
        ----------
        simulator : class
            The simulator class (not an instance) used for generating synthetic data.
        priors : dict
            A dictionary where keys are parameter names and values are scipy.stats distributions.
        observed_data : list
            The observed dataset to compare with simulated data.
            Experiments have different time points, each experiment is of shape (num_observables + 1, T_obs).
            First row is time, remaining rows are observables.
        epsilon : float
            The acceptance threshold for similarity between observed and simulated data.
        proposal_std : dict
            A dictionary specifying the standard deviation for Gaussian proposal distributions for each parameter.
        T : int
            The number of days to simulate.
        num_steps : int
            The number of steps to simulate.
        randomize : bool, optional
            Whether to add noise to the simulation outputs and autocorrelation randomization
            to the growth rates. Default is True.
        redraw_iterations : int, optional
            Number of iterations after which the parameters are redrawn from the prior if no distance below epsilon is found.
            Default is 500.
        save_dir : str, optional
            Directory to save the results. Is created if it does not exist. Default is None.
        """
        self.simulator = simulator
        self.priors = priors
        self.observed_data = observed_data
        self.epsilon = epsilon
        self.proposal_std = proposal_std
        self.T = T
        self.num_steps = num_steps
        self.randomize = randomize
        self.redraw_iterations = redraw_iterations

        if save_dir is not None:
            self.save_dir = save_dir
            self.initialize_save()

    def initialize_save(self):
        """Initialize save directory and log file."""
        os.makedirs(self.save_dir, exist_ok=True)

        # Reset any existing logging handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Set up new logging configuration
        log_file = os.path.join(self.save_dir, "abc_mcmc.log")
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        logging.info("ABC-MCMC initialized. Logging started.")

        # Create empty files for results
        self.accepted_params_file = os.path.join(self.save_dir, "accepted_params.csv")
        self.distances_file = os.path.join(self.save_dir, "distances.csv")
        self.simulated_data_file = os.path.join(self.save_dir, "simulated_data.npy")

    def single_distance(self, x_obs, x_sim):
        """
        Compute the distance between observed and simulated data by 
        finding the closest time points in the simulation for each observed data point.

        Parameters
        ----------
        x_obs : ndarray
            Observed data of shape (num_observables + 1, T_obs).
            First row is time, remaining rows are observables.
        x_sim : ndarray
            Simulated data of shape (num_observables + 1, T_sim).
            First row is time, remaining rows are observables.

        Returns
        -------
        float
            Euclidean distance between the best-matched time points.
        """
        time_obs = x_obs[0, :]  # Observed time points
        time_sim = x_sim[0, :]  # Simulated time points
        
        # Find the closest time indices in the simulation data for each observed time point
        closest_indices = np.array([np.argmin(np.abs(time_sim - t)) for t in time_obs])

        # Extract the corresponding observables
        obs_values = x_obs[1:, :]  # Shape: (num_observables, T_obs)
        sim_values = x_sim[1:, closest_indices]  # Shape: (num_observables, T_obs)

        # Compute Euclidean distance over observables at matched time points
        distance = np.linalg.norm(obs_values - sim_values)

        return distance
    
    def save_progress(self, accepted_params, distances, simulated_data):
        """Save the current state of the MCMC process to files."""
        if not self.save_dir:
            return  # Skip saving if no directory is specified

        # Save accepted parameters
        resulting_params = {key: np.array([params[key] for params in accepted_params])
                            for key in accepted_params[0]}
        pd.DataFrame(resulting_params).to_csv(self.accepted_params_file, index=False)

        # Save distances
        pd.DataFrame({"Distance": distances}).to_csv(self.distances_file, index=False)

        # Save simulated data
        np.save(self.simulated_data_file, np.array(simulated_data))

        logging.info(f"Progress saved: {len(accepted_params)} samples stored.")
    
    def distance(self, observed_data, X_sim):
        """
        Compute the distance between observed and simulated data for all experiments.

        Parameters
        ----------
        observed_data : list
            List of observed data of shape (num_observables + 1, T_obs).
            Each experiment has different time points.
        X_sim : ndarray
            Simulated data of shape (n_samples, (num_observables + 1), T_sim).
            Time is included as the first row of the observables.

        Returns
        -------
        float
            Euclidean distance as sum of distances for each experiment.
        """
        return sum([self.single_distance(X_obs, X_sim) for X_obs in observed_data])

    def propose_new_params(self, current_params):
        """Propose new parameters using Gaussian perturbations."""
        new_params = {}
        for key, prior in self.priors.items():
            proposed_value = np.random.normal(current_params[key], self.proposal_std[key])

            # Bound checking
            if isinstance(prior.dist, stats._continuous_distns.uniform_gen):  # Uniform prior
                lower, upper = prior.args[0], prior.args[0] + prior.args[1]
                proposed_value = np.clip(proposed_value, lower, upper)  # Clip to valid range
            
            elif isinstance(prior.dist, stats._continuous_distns.gamma_gen):  # Gamma prior (must be positive)
                while proposed_value <= 0:  # Ensure valid sample
                    proposed_value = np.random.normal(current_params[key], self.proposal_std[key])

            new_params[key] = proposed_value

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
        """
        Run the ABC-MCMC algorithm for n_iterations.
        Returns a dictionary of the accepted parameters at each iteration.

        Parameters
        ----------
            n_iterations: int, number of iterations to run the algorithm.

        Returns
        -------
            dict: Dictionary of the accepted parameters at each iteration.
            list: List of distances between observed and simulated data at each iteration.
            list: List of simulated data at each iteration.
        """
        # Initialize the first parameter set by sampling from priors
        theta_current = {key: prior.rvs() for key, prior in self.priors.items()}
        logging.info(f"Starting ABC-MCMC with initial parameters: {theta_current}")
        accepted_params = []

        distances = []
        simulated_data = []

        redraw_counter = 0

        for i in tqdm(range(n_iterations), desc="ABC-MCMC"):
            # Step M2: Propose new parameters
            theta_proposed = self.propose_new_params(theta_current)

            # Step M3: Simulate dataset
            X_sim = create_array_simulation(self.simulator, theta_proposed, self.T, self.num_steps, self.randomize)

            # Sometimes the parameters cause the simulation to fail as the values get to large for the poisson distribution
            if X_sim is None:
                # Therefore skip this iteration and propose new parameters
                continue

            # Step M4: Compute distance and check threshold
            distance = self.distance(self.observed_data, X_sim)
            if distance <= self.epsilon:
                # Step M5: Accept with probability alpha
                alpha = self.acceptance_ratio(theta_current, theta_proposed)
                logging.info(f"Distance below epsilon: {distance}, alpha: {alpha}")
                if np.random.rand() < alpha:
                    logging.info(f"Accepted new parameters: {theta_proposed}")
                    theta_current = theta_proposed  # Accept new parameters

            # Store the accepted parameters
            accepted_params.append(theta_current)
            # Store the interesting developmental parameters
            distances.append(distance)
            simulated_data.append(X_sim)

            if self.save_dir and i % 100 == 0:  # Save progress every 100 iterations
                self.save_progress(accepted_params, distances, simulated_data)

            # If no distance below epsilon is found after a certain number of iterations, redraw parameters
            redraw_counter += 1
            if redraw_counter >= self.redraw_iterations and not any([d <= self.epsilon for d in distances]):
                logging.info("Redrawing parameters...")
                theta_current = {key: prior.rvs() for key, prior in self.priors.items()}
                redraw_counter = 0

        # Final save
        self.save_progress(accepted_params, distances, simulated_data)

        resulting_params = {key: np.array([params[key] for params in accepted_params]) for key in accepted_params[0]}

        return resulting_params, distances, simulated_data