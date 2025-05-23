import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import logging
import seaborn as sns

from collections import namedtuple, deque
from typing import override

Solution = namedtuple("Solution", ["t", "y"])


class Simulator:
    """Base class for predator-prey simulations."""
    
    def __init__(self, params, initial_conditions, t_range):
        """
        Initialize the simulator.
        
        Args:
        - params (dict): Model parameters.
        - initial_conditions (tuple): (A0, R0) starting values for algae & rotifers.
        - t_range (tuple): (t_start, t_end, num_points) for simulation time.
        """
        self.params = params
        self.initial_conditions = initial_conditions
        self.t_range = np.linspace(*t_range, endpoint=False)
        
    def model(self, t, y):
        """Defines the ODE system (to be implemented in subclasses)."""
        raise NotImplementedError("Subclasses must implement model equations.")

    def simulate(self):
        """Runs the numerical solver and returns results."""
        sol = scipy.integrate.solve_ivp(self.model, 
                                        [self.t_range[0], self.t_range[-1]], 
                                        self.initial_conditions, 
                                        t_eval=self.t_range)
        return sol

    def plot_results(self, sol, title="Predator-Prey Simulation", save_dir=None):
        """Plots algae and rotifer populations over time."""
        plt.figure(figsize=(10, 5))
        plt.plot(sol.t, sol.y[0], label="Prey", color="green")
        plt.plot(sol.t, sol.y[1], label="Predators", color="red")
        plt.xlabel("Time (days)")
        plt.ylabel("Population")
        plt.title(title)
        plt.legend()

        if save_dir:
            png_title = title.lower().replace(" ", "-")
            save_file = os.path.join(save_dir, f"{png_title}.png")
            plt.savefig(save_file)

        plt.show()


class LotkaVolterraSimulator(Simulator):
    """Lotka-Volterra predator-prey model."""

    @override
    def model(self, t, y):
        A, R = y  # Algae (prey) and Rotifers (predators)
        r, alpha, beta, d = self.params["r"], self.params["alpha"], self.params["beta"], self.params["d"]

        dA_dt = r * A - alpha * A * R
        dR_dt = beta * A * R - d * R

        return [dA_dt, dR_dt]
    

class RosenzweigMacArthurSimulator(Simulator):
    """Rosenzweig-MacArthur predator-prey model."""

    @override
    def model(self, t, y):
        A, R = y  # Algae (prey) and Rotifers (predators)
        r = self.params["r"]
        K = self.params["K"]
        alpha = self.params["alpha"]
        H = self.params["H"]
        beta = self.params["beta"]
        d = self.params["d"]

        dA_dt = r * A * (1 - A / K) - (alpha * A * R) / (A + H)
        dR_dt = (beta * alpha * A * R) / (A + H) - d * R

        return [dA_dt, dR_dt]


class HollingSimulator(Simulator):
    """Holling predator-prey model."""

    @override
    def model(self, t, y):
        A, R = y  # Algae (prey) and Rotifers (predators)
        r = self.params["r"]
        K = self.params["K"]
        alpha = self.params["alpha"]
        H = self.params["H"]
        beta = self.params["beta"]
        d = self.params["d"]

        dA_dt = r * A * (1 - A / K) - (alpha * A**2 * R) / (A**2 + H)
        dR_dt = R * ((beta * A**2 * alpha) / (A**2 + H**2) - d)

        return [dA_dt, dR_dt]
    

class StochasticLotkaVolterraSimulator(Simulator):
    """Stochastic Lotka-Volterra predator-prey model"""

    @override
    def simulate(self):
        A, R = self.initial_conditions
        A_vals, R_vals = [A], [R]

        for t in self.t_range[1:]:
            A = max(A + self.model(t, [A, R])[0], 0)
            R = max(R + self.model(t, [A, R])[1], 0)
            A_vals.append(A)
            R_vals.append(R)

        return Solution(self.t_range, np.array([A_vals, R_vals]))

    @override
    def model(self, t, y):
        A, R = y
        r = self.params["r"]
        alpha = self.params["alpha"]
        beta = self.params["beta"]
        d = self.params["d"]
        sigma_A = self.params["sigma_A"]
        sigma_R = self.params["sigma_R"]

        dt = self.t_range[1] - self.t_range[0]

        dA = (r * A - alpha * A * R) * dt + sigma_A * A * np.random.normal()
        dR = (beta * A * R - d * R) * dt + sigma_R * R * np.random.normal()

        return [dA, dR]
    

class FullPredatorPreyModel(Simulator):
    def __init__(self, params, t_range, randomize=True):
        self.params = params
        self.initial_conditions = [
            params["N_in"],
            params["nu_p"] * params["P0"],
            params["nu_b"] * params["E0"] / params["beta"],
            params["nu_b"] * params["J0"] / params["beta"],
            params["nu_b"] * params["A0"],
            0
        ]
        self.t_range = np.linspace(*t_range, endpoint=False)
        self.R_E_history = deque(maxlen=1000)  # Store past R_E values
        self.R_J_history = deque(maxlen=1000)  # Store past R_J values

        # Store noise for autocorrelated randomness
        self.s = 0

        # Perform randomization
        self.randomization = randomize
    
    def get_past_value(self, history, delay, dt):
        """Retrieve an approximate past value based on the delay."""
        index = int(delay / dt)
        if index < len(history):
            return history[-index]
        # If delay is longer than history, there is no recruitment
        return 0
    
    def perform_growth_randomization(self, r_P, r_B):
        r_P *= 1 + self.s
        r_B *= 1 + self.s
        a = self.params["a"]
        sigma = self.params["sigma"]
        self.s = a * self.s + sigma * np.random.normal(0, np.sqrt(1 - a**2))

        return r_P, r_B
    
    def structure_solution(self, sol):
        # Structures the solution into the output format from the experiments
        # nu parameters have to be given in units of 10**(-3)
        results = np.array([
            # Algae
            sol.y[1] / self.params['nu_p'],
            # Rotifers
            (self.params['beta'] * sol.y[3] + sol.y[4]) / self.params['nu_b'],
            # Eggs
            self.params['beta'] * sol.y[2] / self.params['nu_b'],
            # Egg ratio
            self.params['beta'] * sol.y[2] / (self.params['beta'] * (sol.y[3] + sol.y[2]) + sol.y[4] + 1e-6),
            # Dead rotifers
            sol.y[5] / self.params['nu_b']
        ])
        return Solution(sol.t, results)

    
    @override
    def model(self, t, y):
        """Defines the full predator-prey model with stochasticity and delays."""
        N, P, E, J, A, D = y
        
        # Unpack parameters
        delta = self.params['delta']
        N_in = self.params['N_in']
        r_P = self.params['r_P']
        r_B = self.params['r_B']
        K_P = self.params['K_P']
        K_B = self.params['K_B']
        kappa = self.params['kappa']
        epsilon = self.params['epsilon']
        m = self.params['m']
        beta = self.params['beta']
        theta = self.params['theta']
        tau = self.params['tau']
        dt = self.t_range[1] - self.t_range[0]

        # Perform growth randomization if enabled and get growth rates
        if self.randomization:
            r_P, r_B = self.perform_growth_randomization(r_P, r_B)
        
        # Functional responses
        F_P = r_P * N / (K_P + N)
        F_B = r_B * (P**kappa) / (K_B**kappa + P**kappa)
        
        # Recruitment rates with delays
        R_E = F_B * A
        R_J = self.get_past_value(self.R_E_history, theta, dt) * np.exp(-delta * theta)
        R_A = self.get_past_value(self.R_J_history, tau, dt) * np.exp(-delta * tau)
        
        # Store current recruitment rates
        self.R_E_history.append(R_E)
        self.R_J_history.append(R_J)
        
        # Differential equations
        dN_dt = delta * (N_in - N) - F_P * P
        dP_dt = F_P * P - (F_B * (beta * J + A) / epsilon) - delta * P
        dE_dt = R_E - R_J - delta * E
        dJ_dt = R_J - R_A - (m + delta) * J
        dA_dt = beta * R_A - (m + delta) * A
        dD_dt = m * (J + A) - delta * D
        
        return [dN_dt, dP_dt, dE_dt, dJ_dt, dA_dt, dD_dt]
    
    @override
    def simulate(self):
        """Runs the numerical solver with stochastic growth rates."""
        y = self.initial_conditions
        N, P, E, J, A, D = y
        N_vals, P_vals, E_vals, J_vals, A_vals, D_vals = [N], [P], [E], [J], [A], [D]
        dt = self.t_range[1] - self.t_range[0]

        for t in self.t_range[1:]:
            dN_dt, dP_dt, dE_dt, dJ_dt, dA_dt, dD_dt = self.model(t, y)
            N = max(N + dN_dt * dt, 0)
            P = max(P + dP_dt * dt, 0)
            E = max(E + dE_dt * dt, 0)
            J = max(J + dJ_dt * dt, 0)
            A = max(A + dA_dt * dt, 0)
            D = max(D + dD_dt * dt, 0)

            assert not any(np.isnan([N, P, E, J, A, D])), f"NaN values at t={t}!"

            if self.randomization:
                try:
                    N_measured = max(np.random.poisson(N), 0)
                    P_measured = max(np.random.poisson(P), 0)
                    E_measured = max(np.random.poisson(E), 0)
                    J_measured = max(np.random.poisson(J), 0)
                    A_measured = max(np.random.poisson(A), 0)
                    D_measured = max(np.random.poisson(D), 0)
                except ValueError as e:
                    print(f"Value error at t={t}: {e}")
                    print(f"Values: {N}, {P}, {E}, {J}, {A}, {D}")
                    raise
            else:
                N_measured, P_measured, E_measured, J_measured, A_measured, D_measured = N, P, E, J, A, D

            # Update the y value
            y = [N, P, E, J, A, D]

            # Append the measurements
            N_vals.append(N_measured)
            P_vals.append(P_measured)
            E_vals.append(E_measured)
            J_vals.append(J_measured)
            A_vals.append(A_measured)
            D_vals.append(D_measured)

        solution = Solution(self.t_range, np.array([N_vals, P_vals, E_vals, J_vals, A_vals, D_vals]))
        return self.structure_solution(solution)
    
    def plot_results(self, sol, time_range=None, title="Full Predator-Prey Simulation", save_file=None):
        """Plots all state variables over time with enhanced aesthetics."""
        
        labels = ["Algae", "Rotifers", "Eggs", "Egg Ratio", "Dead Rotifers"]
        colors = sns.color_palette("Set1", len(labels))  # More visually distinct colors
        linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]  # Different styles for clarity

        time = sol.t.copy()
        values = sol.y.copy()

        if time_range is not None:
            mask = (time >= time_range[0]) & (time <= time_range[1])
            time = time[mask]
            values = values[:, mask]

        plt.figure(figsize=(12, 7))
        for i in range(len(labels)):
            plt.plot(time, values[i], label=labels[i], color=colors[i], linestyle=linestyles[i], linewidth=2)

        plt.xlabel("Time (days)", fontsize=16, fontweight="bold")
        plt.ylabel("Population", fontsize=16, fontweight="bold")
        plt.title(title, fontsize=18, fontweight="bold", pad=15)
        
        plt.legend(fontsize=13, frameon=True, fancybox=True, shadow=True)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        if save_file:
            plt.savefig(save_file, dpi=300)
        
        plt.show()

    def discretize_solution(self, sol, new_dt=1):
        """
        Make the solution more discrete into bigger time steps
        that are a multiple of the old time step
        """
        time = sol.t.copy()
        values =sol.y.copy()

        old_dt = time[1] - time[0]
        assert new_dt >= old_dt, f"New time step {new_dt} is smaller than old time step {old_dt}"
        assert (new_dt / old_dt) % 1 < 1e-6, f"New time step {new_dt} is not multiple of old {old_dt}"

        indeces = range(0, len(time), int(new_dt / old_dt))

        time = time[indeces]
        values = values[:, indeces]

        return Solution(time, values)

    def plot_normalized_results(self, sol, time_range=None, title="Full Predator-Prey Simulation"):
        """
        Plots all normalized state variables over time.
        Additionally, a different normalizing constant for each state variable
        makes the values spread out over time more easily.
        """
        labels = ["Algae", "Rotifers", "Eggs", "Egg Ratio", "Dead Rotifers"]
        colors = ["green", "red", "black", "blue", "yellow"]
        normalizing_constants = [1.0, 0.8, 0.6, 1.0, 0.2]

        time = sol.t.copy()
        values = sol.y.copy()

        if time_range is not None:
            mask = (time >= time_range[0]) & (time <= time_range[1])
            time = time[mask]
            values = values[:, mask]
        
        plt.figure(figsize=(12, 6))
        for i in range(len(labels)):
            normalized_result = normalizing_constants[i] * values[i] / values[i].max()
            plt.plot(time, normalized_result, label=labels[i], color=colors[i], linestyle='-')
        
        plt.xlabel("Time (days)", fontsize=18)
        plt.ylabel("Abundance", fontsize=18)
        plt.title(title, fontsize=20)
        plt.legend()
        plt.show()


def create_dataset(simulator, priors, n_samples, T, reporting_interval=1, randomize=True):
    """
    Create a dataset for SBI using parameter samples from priors and corresponding simulated outputs.
    
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
    reporting_interval : float
        The time interval at which to report observables (discretizes the solution).
    randomize : bool
        Whether to randomize the simulator. True by default.

    Returns
    -------
    X_data : list
        The input data (observables) of shape (n_samples, (num_observables + 1) * T).
    y_data : ndarray
        The target data (parameters) of shape (n_samples, len(priors)).
    """
    X_data, y_data = [], []
    t_range = (0, T, int(10 * T / reporting_interval))
    
    while len(X_data) < n_samples:
        # Sample parameters from the prior distributions
        params = {key: prior.rvs() for key, prior in priors.items()}

        try:
            # Instantiate and run the simulator
            sim_instance = simulator(params, t_range, randomize=randomize)
            sol = sim_instance.simulate()
            sol = sim_instance.discretize_solution(sol, new_dt=reporting_interval)

            columns=["Time", "Algae", "Rotifers", "Eggs", "Egg_Ratio", "Dead_Rotifers"]

            num_variables = len(sol.y)
            

            observables_dict = {columns[i+1]: sol.y[i] for i in range(num_variables)}
            observables_dict["Time"] = sol.t

        except Exception as e:
            logging.error(f"Error during simulation: {e}")
            logging.info(f"Parameters: {params}")
            continue

        X_data.append(observables_dict)
        y_data.append(list(params.values()))
    
    return X_data, np.array(y_data)

def create_and_save_simulated_data(simulator, priors, n_samples, T, X_dir, y_dir, randomize=True):
    """
    Simulate data using create_dataset and save it in CSV format, 
    with parameters stored in a JSON file.

    Parameters
    ----------
    simulator : class
        The simulator class used to generate the data.
    priors : dict
        The prior distributions for the parameters.
    n_samples: int
        The number of simulation runs to perform.
    T : int
        The number of days to simulate.
    X_dir : str
        Directory where CSV files should be saved.
    y_dir : str
        Directory where the params file should be saved.
    randomize : bool
        Whether to randomize the simulator. True by default.
    """

    os.makedirs(X_dir, exist_ok=True)  # Ensure output directory exists

    # Get class name of the simulator
    simulator_name = simulator.__name__

    # Generate dataset
    X_data, y_data = create_dataset(simulator, priors, n_samples, T, randomize=randomize)

    param_records = {}

    for i in range(n_samples):

        X_sample = X_data[i]
        # Save simulation results to CSV
        basename = f"{simulator_name}_sim_{i}"
        if randomize:
            basename = f"{basename}_randomized"
        filename = f"{basename}.csv"
        filepath = os.path.join(X_dir, filename)
        df = pd.DataFrame(X_sample)
        df.to_csv(filepath, index=False)

        # Store parameters used for this simulation
        param_records[filename] = {key: float(y_data[i, j]) for j, key in enumerate(priors.keys())}

    # Save all parameters to a JSON file
    param_base = f"{simulator_name}_params"
    if randomize:
        param_base = f"{param_base}_randomized"
    param_file = f"{param_base}.json"
    y_path = os.path.join(y_dir, param_file)
    with open(y_path, "w") as f:
        json.dump(param_records, f, indent=4)

    print(f"Saved {n_samples} simulations in {X_dir}, parameters stored in {y_dir}.")

