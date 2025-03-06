import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

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
        self.t_range = np.linspace(*t_range)
        
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

    def plot_results(self, sol, title="Predator-Prey Simulation"):
        """Plots algae and rotifer populations over time."""
        plt.figure(figsize=(10, 5))
        plt.plot(sol.t, sol.y[0], label="Algae (Prey)", color="green")
        plt.plot(sol.t, sol.y[1], label="Rotifers (Predators)", color="red")
        plt.xlabel("Time (days)")
        plt.ylabel("Population")
        plt.title(title)
        plt.legend()
        plt.show()


class LotkaVolterraSimulator(Simulator):
    """Lotka-Volterra predator-prey model."""

    def model(self, t, y):
        A, R = y  # Algae (prey) and Rotifers (predators)
        r, alpha, beta, d = self.params["r"], self.params["alpha"], self.params["beta"], self.params["d"]

        dA_dt = r * A - alpha * A * R
        dR_dt = beta * A * R - d * R

        return [dA_dt, dR_dt]
    

class RosenzweigMacArthurSimulator(Simulator):
    """Rosenzweig-MacArthur predator-prey model."""

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
