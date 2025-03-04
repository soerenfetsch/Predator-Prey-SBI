import re
from scipy.signal import find_peaks
import numpy as np

def clean_columns_and_extract_column_info(df):
    """
    Clean column names and extract units from column names.
    Args:
        df: pandas DataFrame
    Returns:
        df: pandas DataFrame with cleaned column names
        column_units: dict with column names as keys and units as values
    """
    column_units = {}
    new_columns = []

    for col in df.columns:
        match = re.match(r"(.+?)\s*\((.+?)\)", col)  # Match "name (unit)"
        if match:
            name, unit = match.groups()
            column_units[name.strip()] = unit.strip()  # Store unit
            new_columns.append(name.strip())  # Clean column name
        else:
            new_columns.append(col.strip())  # Keep name as is

    df.columns = new_columns  # Rename columns
    return df, column_units

def compute_cycle_durations(df):
    """
    Compute average cycle durations for algae and rotifers.
    Args:
        df: pandas DataFrame
    Returns:
        dict: average cycle durations for algae, rotifers, eggs, and dead animals
    """
    # Find peaks in populations
    algae_peaks, _ = find_peaks(df["algae"])
    rotifer_peaks, _ = find_peaks(df["rotifers"])
    eggs_peaks, _ = find_peaks(df["eggs"])
    dead_peaks, _ = find_peaks(df["dead animals"])

    # Compute peak-to-peak time differences
    algae_cycle_lengths = np.diff(df["time"].iloc[algae_peaks])
    rotifer_cycle_lengths = np.diff(df["time"].iloc[rotifer_peaks])
    eggs_cycle_lengths = np.diff(df["time"].iloc[eggs_peaks])
    dead_cycle_lengths = np.diff(df["time"].iloc[dead_peaks])

    return {
        "avg_algae_cycle": np.mean(algae_cycle_lengths) if len(algae_cycle_lengths) > 0 else None,
        "avg_rotifer_cycle": np.mean(rotifer_cycle_lengths) if len(rotifer_cycle_lengths) > 0 else None,
        "avg_eggs_cycle": np.mean(eggs_cycle_lengths) if len(eggs_cycle_lengths) > 0 else None,
        "avg_dead_cycle": np.mean(dead_cycle_lengths) if len(dead_cycle_lengths) > 0 else None
    }

def compute_population_stats(df):
    """
    Compute mean and standard deviation of algae, rotifer, eggs, and
    dead animals populations.
    Args:
        df: pandas DataFrame
    Returns:
        dict: mean and standard deviation of populations
    """
    return {
        "mean_algae": df["algae"].mean(),
        "std_algae": df["algae"].std(),
        "mean_rotifers": df["rotifers"].mean(),
        "std_rotifers": df["rotifers"].std(),
        "mean_eggs": df["eggs"].mean(),
        "std_eggs": df["eggs"].std(),
        "mean_dead_animals": df["dead animals"].mean(),
        "std_dead_animals": df["dead animals"].std()
    }

def compute_lag_between_peaks(df):
    """
    Compute average lag between peaks in algae and rotifer populations.
    Args:
        df: pandas DataFrame
    Returns:
        float: average lag between peaks in algae and rotifer populations
    """
    algae_peaks, _ = find_peaks(df["algae"])
    rotifer_peaks, _ = find_peaks(df["rotifers"])

    if len(algae_peaks) == 0 or len(rotifer_peaks) == 0:
        return None  # No peaks found

    # Compute time differences between closest peaks
    lag_times = []
    for r_peak in rotifer_peaks:
        closest_a_peak = algae_peaks[np.argmin(np.abs(algae_peaks - r_peak))]
        lag_times.append(df["time"].iloc[r_peak] - df["time"].iloc[closest_a_peak])

    return np.mean(lag_times) if len(lag_times) > 0 else None