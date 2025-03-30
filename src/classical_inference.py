import numpy as np
import scipy.stats as stats
import logging
import os
import h5py
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
        The observables with time of shape ((num_observables + 1), num_steps).
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
            observables = np.array(sol.y)  # Shape (num_observables, n_steps)

            # Stack time as the first row
            data_array = np.vstack((time_values, observables))  # Shape (num_observables + 1, n_steps)

        except Exception as e:
            logging.error(f"Error during simulation: {repr(e)}")
            logging.info(f"Parameters: {params}")
            iterations += 1
            if iterations >= max_iterations:
                logging.error("Maximum iterations reached.")
            continue

    # Fill array with negative numbers if simulation failed
    if data_array is None:
        data_array = np.full((6, num_steps), -1)

    # Replace NaN values with inf
    data_array = np.nan_to_num(data_array, nan=np.inf)
    
    return data_array

def single_distance(x_obs, x_sim):
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

def calc_distance(observed_data, X_sim):
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
    return sum([single_distance(X_obs, X_sim) for X_obs in observed_data])

def append_to_csv(new_data, file_path):
    """Append new data to a CSV file."""
    df = pd.DataFrame(new_data)  # Convert NumPy array to DataFrame
    df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)

def append_memmap_npy(file_path, new_data):
    """Appends a NumPy array to a .npy file using memory mapping (efficient for large data)."""
    shape = new_data.shape  # (num_samples, num_features)

    if not os.path.exists(file_path):
        # If file doesn't exist, create a new memmap file
        fp = np.memmap(file_path, dtype=new_data.dtype, mode='w+', shape=shape)
        fp[:] = new_data  # Write new data
    else:
        # Load existing file to get its shape
        existing_data = np.load(file_path, mmap_mode='r')
        new_shape = (existing_data.shape[0] + shape[0],) + existing_data.shape[1:]
        
        # Create new memory-mapped file with expanded shape
        fp = np.memmap(file_path, dtype=new_data.dtype, mode='w+', shape=new_shape)
        fp[:existing_data.shape[0]] = existing_data  # Copy old data
        fp[existing_data.shape[0]:] = new_data  # Append new data

    del fp  # Ensure changes are flushed to disk

def append_h5py(file_path, new_data, key="data"):
    """Append new data to an HDF5 file."""
    new_data = np.asarray(new_data)
    with h5py.File(file_path, "a") as f:
        if key in f:
            dset = f[key]
            # Ensure consistency with existing dataset
            if new_data.shape != dset.shape[1:]:
                raise ValueError(f"Shape mismatch: cannot append {new_data.shape} to {dset.shape}")

            dset.resize(dset.shape[0] + 1, axis=0)  # Expand first axis (num_simulations)
            dset[-1] = new_data  # Append new simulation
        else:
            # Create dataset with expandable first axis
            f.create_dataset(key, data=new_data[np.newaxis, ...], maxshape=(None, 6, 4000))


class ABCMCMC:
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
        self.logger = logging.getLogger(__name__)

        if save_dir is not None:
            self.save_dir = save_dir
            self.initialize_save()

    def initialize_save(self):
        """Initialize save directory and log file."""
        os.makedirs(self.save_dir, exist_ok=True)

        # Set up new logging configuration
        log_file = os.path.join(self.save_dir, "abc_mcmc.log")
        self.logger.handlers.clear()

        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(file_handler)

        self.logger.info("ABC-MCMC initialized. Logging started.")

        # Create empty files for results
        self.accepted_params_file = os.path.join(self.save_dir, "accepted_params.csv")
        self.distances_file = os.path.join(self.save_dir, "distances.csv")
        self.simulated_data_file = os.path.join(self.save_dir, "simulated_data.h5")
    
    def save_progress(self, accepted_params, distances, new_simulated_data):
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
        append_h5py(self.simulated_data_file, np.array(new_simulated_data))

        self.logger.info(f"Progress saved: {len(accepted_params)} samples stored.")

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
        """
        # Initialize the first parameter set by sampling from priors
        theta_current = {key: prior.rvs() for key, prior in self.priors.items()}
        self.logger.info(f"Starting ABC-MCMC with initial parameters: {theta_current}")
        accepted_params = []

        distances = []

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
            distance = calc_distance(self.observed_data, X_sim)
            if distance <= self.epsilon:
                # Step M5: Accept with probability alpha
                alpha = self.acceptance_ratio(theta_current, theta_proposed)
                self.logger.info(f"Distance below epsilon: {distance}, alpha: {alpha}")
                if np.random.rand() < alpha:
                    self.logger.info(f"Accepted new parameters: {theta_proposed}")
                    theta_current = theta_proposed  # Accept new parameters

            # Store the accepted parameters
            accepted_params.append(theta_current)
            # Store the interesting developmental parameters
            distances.append(distance)

            if self.save_dir: # Save progress
                self.save_progress(accepted_params, distances, X_sim)

            # If no distance below epsilon is found after a certain number of iterations, redraw parameters
            redraw_counter += 1
            if redraw_counter >= self.redraw_iterations and not any([d <= self.epsilon for d in distances]):
                self.logger.info("Redrawing parameters...")
                theta_current = {key: prior.rvs() for key, prior in self.priors.items()}
                redraw_counter = 0

        resulting_params = {key: np.array([params[key] for params in accepted_params]) for key in accepted_params[0]}

        return resulting_params, distances
    

class ABCSMC:
    def __init__(self, simulator, priors, observed_data, epsilons, perturbation_std, T, num_particles, num_steps, randomize=True, kernel_type="gaussian", save_dir=None):
        """
        Approximate Bayesian Computation using Sequential Monte Carlo (ABC-SMC).

        Parameters
        ----------
        simulator : class
            The simulator class used for generating synthetic data.
        priors : dict
            Dictionary where keys are parameter names and values are scipy.stats distributions.
        observed_data : list
            Observed dataset to compare with simulated data.
            Experiments have different time points, each experiment is of shape (num_observables + 1, T_obs).
            First row is time, remaining rows are observables.
        epsilons : list
            List of tolerance values for each iteration (ε1 > ε2 > ... > εT <= 0).
        perturbation_std : dict
            Standard deviation for perturbation kernels per parameter.
        T : int
            The number of days to simulate.
        num_particles : int
            Number of particles in each population.
        num_steps : int
            Number of simulation steps.
        randomize : bool, optional
            Whether to add noise to the simulation outputs and autocorrelation randomization
            to the growth rates. Default is True.
        kernel_type : str, optional
            Type of perturbation kernel ("gaussian" or "uniform"), default is "gaussian".
        save_dir : str, optional
            Directory to save results, default is None.
        """
        self.simulator = simulator
        self.priors = priors
        self.observed_data = observed_data
        self.epsilons = epsilons
        self.perturbation_std = perturbation_std
        self.T = T
        self.num_particles = num_particles
        self.num_steps = num_steps
        self.randomize = randomize
        self.kernel_type = kernel_type
        self.save_dir = save_dir
        self.logger = logging.getLogger(__name__)

        if save_dir:
            self.initialize_save()

    def initialize_save(self):
        """Initialize save directory and log file."""
        os.makedirs(self.save_dir, exist_ok=True)

        log_file = os.path.join(self.save_dir, "abc_smc.log")
        # Reset any existing logging handlers
        self.logger.handlers.clear()

        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(file_handler)
        self.logger.info("ABC-SMC initialized.")

        self.accepted_params_file = os.path.join(self.save_dir, "accepted_params.csv")
        self.distances_file = os.path.join(self.save_dir, "distances.csv")
        self.weights_file = os.path.join(self.save_dir, "weights.csv")
        self.simulated_data_file = os.path.join(self.save_dir, "simulated_data.h5")

    def perturb(self, theta):
        """Apply perturbation kernel to a parameter set."""
        new_theta = {}
        for key, value in theta.items():
            if self.kernel_type == "gaussian":
                new_value = np.random.normal(value, self.perturbation_std[key])
            elif self.kernel_type == "uniform":
                new_value = value + np.random.uniform(-self.perturbation_std[key], self.perturbation_std[key])
            else:
                raise ValueError("Unsupported kernel type.")
            
            # Enforce prior constraints
            prior = self.priors[key]
            if isinstance(prior.dist, stats._continuous_distns.uniform_gen):  
                lower, upper = prior.args[0], prior.args[0] + prior.args[1]
                new_value = np.clip(new_value, lower, upper)
            elif isinstance(prior.dist, stats._continuous_distns.gamma_gen):  
                while new_value <= 0:
                    new_value = np.random.normal(value, self.perturbation_std[key])

            new_theta[key] = new_value

        return new_theta

    def run(self, max_iterations_per_particle=10000):
        """
        Run the ABC-SMC algorithm.

        Parameters
        ----------
        max_iterations_per_particle : int, optional
            Maximum number of iterations per particle, default is 5000.
            If maximum is reached, the algorithm will throw an exception.

        Returns
        -------
        list : List of dictionaries containing accepted parameters at each iteration.
        list : List of weights for accepted particles.
        list : List of distances.
        """
        theta_t = []
        weights_t = []
        distances_t = []

        for t, epsilon in enumerate(self.epsilons):
            self.logger.info(f"Iteration {t+1}/{len(self.epsilons)} with epsilon {epsilon}")

            particles = []
            weights = np.ones(self.num_particles) if t == 0 else np.zeros(self.num_particles)
            distances = []
            simulations = []

            for i in tqdm(range(self.num_particles), desc=f"Iteration {t+1}/{len(self.epsilons)}"):
                iteration = 0
                while True:
                    iteration += 1
                    if iteration > max_iterations_per_particle:
                        raise RuntimeError(f"Maximum iterations reached for particle {i} at iteration {t}.")
                    if t == 0:
                        theta_star = {key: prior.rvs() for key, prior in self.priors.items()}
                    else:
                        index = np.random.choice(self.num_particles, p=weights_t[t-1])
                        theta_star = self.perturb(theta_t[t-1][index])

                    # Sometimes the parameters cause the simulation to fail as the values get to large for the poisson distribution
                    X_sim = create_array_simulation(self.simulator, theta_star, self.T, self.num_steps, self.randomize)
                    if X_sim is None:
                        continue

                    dist = calc_distance(self.observed_data, X_sim)
                    if dist < epsilon:
                        break

                particles.append(theta_star)
                distances.append(dist)
                simulations.append(X_sim)

                if t > 0:
                    prior_prob = np.prod([self.priors[k].pdf(theta_star[k]) for k in self.priors])
                    kernel_prob = np.sum([
                        weights_t[t-1][j] * stats.norm(theta_t[t-1][j][k], self.perturbation_std[k]).pdf(theta_star[k])
                        for j in range(self.num_particles)
                        for k in self.priors
                    ])
                    weights[i] = prior_prob / kernel_prob

            # Normalize weights
            weights /= np.sum(weights)
            self.logger.info(f"Iteration {t+1} completed: {len(particles)} particles accepted.")

            theta_t.append(particles)
            weights_t.append(weights)
            distances_t.append(distances)

            if self.save_dir:
                self.save_progress(theta_t, distances_t, weights_t, simulations)

        return theta_t, weights_t, distances_t

    def save_progress(self, theta_t, distances_t, weights_t, new_simulations):
        """Save accepted parameters, distances, and weights."""
        if not self.save_dir:
            return

        # Save accepted parameters for all iterations and the index of the epsilon
        param_dict = {key: np.concatenate([[theta[key] for theta in t] for t in theta_t])
                      for key in theta_t[-1][0]}
        param_dict["Index"] = np.concatenate([[i] * len(t) for i, t in enumerate(theta_t)])
        pd.DataFrame(param_dict).to_csv(self.accepted_params_file, index=False)

        # Save distances
        pd.DataFrame({"Distance": [d for distances in distances_t for d in distances],
                      "Index": [i for (i, distances) in enumerate(distances_t) for _ in distances]}
                    ).to_csv(self.distances_file, index=False)

        # Save weights
        pd.DataFrame({"Weight": [w for weights in weights_t for w in weights],
                      "Index": [i for (i, weights) in enumerate(weights_t) for _ in weights]}
                    ).to_csv(self.weights_file, index=False)
        
        # Save simulated data
        for sim in new_simulations:
            append_h5py(self.simulated_data_file, np.array(sim))

        self.logger.info(f"Progress saved: {sum(len(t) for t in theta_t)} particles stored.")