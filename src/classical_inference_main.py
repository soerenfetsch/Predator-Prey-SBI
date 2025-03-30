from classical_inference import ABCMCMC, ABCSMC, create_array_simulation
from simulation import FullPredatorPreyModel
import numpy as np
import pandas as pd
import pyabc

from pathlib import Path
from os import makedirs
from scipy.stats import uniform, norm, gamma
from argparse import ArgumentParser


SRC_DIR = Path(__file__).parent
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR / 'data' / 'processed'
EXP_DIR = ROOT_DIR / 'experiments' / 'classical'


# Class for constrained prior in pyABC SMC2
class ConstrainedPrior(pyabc.DistributionBase):
    def __init__(self):
        self.delta = pyabc.RV("uniform", 0.4, 0.3)
        self.N_in = pyabc.RV("norm", 80, 10)
        self.r_P = pyabc.RV("gamma", 3, scale=1)
        self.r_B = pyabc.RV("gamma", 2, scale=1)
        self.K_P = pyabc.RV("uniform", 3, 3)
        self.K_B = pyabc.RV("uniform", 10, 10)
        self.kappa = pyabc.RV("uniform", 1, 0.5)
        self.epsilon = pyabc.RV("uniform", 0.1, 0.4)
        self.m = pyabc.RV("uniform", 0.1, 0.2)
        self.beta = pyabc.RV("uniform", 3, 4)
        self.theta = pyabc.RV("uniform", 0.3, 0.6)
        self.tau = pyabc.RV("uniform", 1, 2)
        self.a = pyabc.RV("uniform", 0, 1)
        self.sigma = pyabc.RV("uniform", 0.01, 0.1)
        self.nu_p = pyabc.RV("uniform", 20, 15)
        self.nu_b = pyabc.RV("uniform", 0.4, 0.5)
        self.P0 = pyabc.RV("gamma", 1, scale=1)
        self.E0 = pyabc.RV("gamma", 0.1, scale=0.1)
        self.J0 = pyabc.RV("gamma", 0.1, scale=0.1)
        self.A0 = pyabc.RV("gamma", 1, scale=1)

    def rvs(self, *args, **kwargs):
        name_val_dict = {}
        # Sample for uniform prior
        for param_name in ('delta', 'K_P', 'K_B', 'kappa', 'epsilon', 'm',
                           'beta', 'theta', 'tau', 'a', 'sigma', 'nu_p', 'nu_b'):
            param = getattr(self, param_name)
            value = param.rvs()
            lower, upper = param.distribution.args[0], param.distribution.args[1]
            value = np.clip(value, lower, upper)
            name_val_dict[param_name] = value
        # Sample for normal prior
        for param_name in ("N_in",):
            param = getattr(self, param_name)
            value = param.rvs()
            name_val_dict[param_name] = value
        # Sample for gamma prior
        for param_name in ('r_P', 'r_B', 'P0', 'E0', 'J0', 'A0'):
            param = getattr(self, param_name)
            value = param.rvs()
            while value < 0:
                value = param.rvs()
            name_val_dict[param_name] = value
        return pyabc.Parameter(**name_val_dict)
    
    def pdf(self, x):
        return np.prod([param.pdf(x[param_name]) for param_name, param in self.__dict__.items()
                        if isinstance(param, pyabc.RV)])
    
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
    # Convert to np array if dict
    if isinstance(x_obs, dict):
        x_obs = x_obs["data"]
    if isinstance(x_sim, dict):
        x_sim = x_sim["data"]

    time_obs = x_obs[0, :]  # Observed time points
    time_sim = x_sim[0, :]  # Simulated time points
    
    # Find the closest time indices in the simulation data for each observed time point
    closest_indices = np.array([np.argmin(np.abs(time_sim - t)) for t in time_obs])

    # Extract the corresponding observables
    obs_values = x_obs[1:, :]  # Shape: (num_observables, T_obs)
    sim_values = x_sim[1:, closest_indices]  # Shape: (num_observables, T_obs)

    # Compute Euclidean distance over observables at matched time points
    distance = np.linalg.norm(obs_values - sim_values)

    if np.isnan(distance):
        return np.inf  # Use a large value instead of NaN

    return distance

def load_data(selected_experiments=[1, 2, 3, 4, 5]):
    data_files = sorted(DATA_DIR.glob('*.csv'), key=lambda x: int(x.stem.split('_')[0].removeprefix('C')))

    dfs = {f.stem.split('_')[0]: pd.read_csv(f) for f in data_files}

    # Map numeric indices to experiment names
    experiment_map = {i + 1: f'C{i+1}' for i in range(7)}
    selected_experiment_names = [experiment_map[i] for i in selected_experiments if i in experiment_map]

    experiment_data = []

    for name in selected_experiment_names:
        df = dfs[name]

        # Extract time column
        sim_time_values = df["time"].values  # Assuming the first column is time

        # Extract observable values (excluding time and medium column)
        algae_values = df["algae"]
        rotifers_values = df["rotifers"]
        egg_values = df["eggs"]
        egg_ratio_values = df["egg-ratio"]
        dead_animals_values = df["dead animals"]
        sim_observable_values = np.array([algae_values,
                                        rotifers_values,
                                        egg_values,
                                        egg_ratio_values,
                                        dead_animals_values])

        # Combine time and observables into a single array
        structured_data = np.vstack([sim_time_values, sim_observable_values])

        # Store in dictionary
        experiment_data.append(structured_data)

    print('Data loaded successfully')
    print('Data shape:', experiment_data[0].shape)

    return experiment_data

def main(args):
    if args.debug:
        # Set logging level to debug
        import logging
        logging.basicConfig(level=logging.DEBUG)
    # Load data
    selected_experiments = args.experiments
    data = load_data(selected_experiments)

    # Define prior
    prior = {
        "delta": uniform(0.4, 0.3),  # Uniform prior between 0.4 and 0.7
        "N_in": norm(80, 10),        # Normal prior (mean=80, std=10)
        "r_P": gamma(3, scale=1),    # Gamma prior (shape=3, scale=1)
        "r_B": gamma(2, scale=1),    # Gamma prior (shape=2, scale=1)
        "K_P": uniform(3, 3),        # Uniform prior between 3 and 6
        "K_B": uniform(10, 10),      # Uniform prior between 10 and 20
        "kappa": uniform(1, 0.5),    # Uniform prior between 1 and 1.5
        "epsilon": uniform(0.1, 0.4),# Uniform prior between 0.1 and 0.5
        "m": uniform(0.1, 0.2),      # Uniform prior between 0.1 and 0.3
        "beta": uniform(3, 4),       # Uniform prior between 3 and 7
        "theta": uniform(0.3, 0.6),  # Uniform prior between 0.3 and 0.9
        "tau": uniform(1, 2),        # Uniform prior between 1 and 3
        "a": uniform(0, 1),          # Autocorrelation between 0 and 1
        "sigma": uniform(0.01, 0.1), # Noise level between 0.01 and 0.11
        'nu_p': uniform(20, 15),     # Uniform prior between 20 and 35
        'nu_b': uniform(0.4, 0.5),   # Uniform prior between 0.4 and 0.9
        'P0': gamma(1, scale=1),     # Gamma prior (shape=1, scale=1)
        'E0': gamma(0.1, scale=0.1), # Gamma prior (shape=0.1, scale=0.1)
        'J0': gamma(0.1, scale=0.1), # Gamma prior (shape=0.1, scale=0.1)
        'A0': gamma(1, scale=1)      # Gamma prior (shape=1, scale=1)
    }

    proposal_std = {
        "delta": 0.03,
        "N_in": 0.3,
        "r_P": 0.03,
        "r_B": 0.03,
        "K_P": 0.03,
        "K_B": 0.03,
        "kappa": 0.03,
        "epsilon": 0.03,
        "m": 0.03,
        "beta": 0.03,
        "theta": 0.03,
        "tau": 0.03,
        "a": 0.03,
        "sigma": 0.03,
        'nu_p': 0.03,
        'nu_b': 0.03,
        'P0': 0.01,
        'E0': 0.01,
        'J0': 0.01,
        'A0': 0.01
    }

    # Define ABC method
    if args.method == 'mcmc':
        save_dir = EXP_DIR / 'ABC-MCMC' / args.savedir
        abc = ABCMCMC(FullPredatorPreyModel, prior, data, args.epsilon[0],
                      proposal_std, args.n_days, args.n_steps, args.randomize,
                      save_dir=save_dir)
        run_param = 30000 if args.run_param is None else args.run_param
        run_params = {"n_iterations": run_param}
    elif args.method == 'smc':
        save_dir = EXP_DIR / 'ABC-SMC' / args.savedir
        abc = ABCSMC(FullPredatorPreyModel, prior, data, args.epsilon,
                     proposal_std, args.n_days, args.n_particles,
                     args.n_steps, args.randomize, save_dir=save_dir)
        run_param = 500 if args.run_param is None else args.run_param
        run_params = {"max_iterations_per_particle": run_param}
    elif args.method == 'smc2':
        # Concatenate the data for all experiments
        data = np.concatenate(data, axis=1)
        def model(parameters):
            # Convert Parameter of RV to dict
            parameters = {param: value for param, value in parameters.items()}
            sim = create_array_simulation(FullPredatorPreyModel, parameters,
                                          args.n_days, args.n_steps, args.randomize)
            return {"data": sim}
        
        save_dir = EXP_DIR / 'ABC-SMC2' / args.savedir
        print(f'Saving results to {save_dir}')
        if not save_dir.exists():
            raise FileNotFoundError(f"Directory {save_dir} does not exist")
        makedirs(save_dir, exist_ok=True)
        constrained_prior = ConstrainedPrior()
        abc = pyabc.ABCSMC(
            model,
            constrained_prior,
            single_distance,
            population_size=args.n_particles
        )

        # Create database for storing results
        db_file = "pyabc_smc2.db"
        abc.new(pyabc.create_sqlite_db_id(save_dir, db_file), {"data": data})

        # Run params
        run_param = 5 if args.run_param is None else args.run_param
        run_params = {"minimum_epsilon": args.epsilon[0],
                      "max_nr_populations": run_param}

    # Run ABC
    print(f'Running ABC mode {args.method} in save dir {args.savedir}')
    abc.run(**run_params)
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--method', type=str, default='mcmc', choices=['mcmc', 'smc', 'smc2'])
    parser.add_argument('--epsilon', type=float, nargs='+', default=[1200])
    parser.add_argument('--n_days', type=int, default=400, help='Number of days to simulate')
    parser.add_argument('--n_steps', type=int, default=4000, help='Number of steps to simulate')
    parser.add_argument('--n_particles', type=int, default=100, help='Number of particles for SMC')
    parser.add_argument('--randomize', action='store_true', help='Randomize simulation')
    parser.add_argument('--experiments', type=int, nargs='+', choices=range(1, 8), default=[1, 2, 3, 4, 5],
                        help='Select experiment indices (1-7)')
    parser.add_argument('--savedir', type=str, default='Run-0', help='Directory name to save results')
    parser.add_argument('--run_param', type=int, help='Run parameter for ABC method')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    main(args)
