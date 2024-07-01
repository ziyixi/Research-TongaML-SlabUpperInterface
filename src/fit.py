from typing import Tuple

import numpy as np
import pandas as pd
import pygmt
import xarray as xr
from numpy.polynomial.polynomial import Polynomial
from obspy.geodetics.base import degrees2kilometers
from scipy.interpolate import interp1d
from scipy.optimize import minimize

import src.config as config
from src.logger_config import logger
from src.utils import slab_interp


def determine_slab_interface(
    line_no: int,
    df_raw: pd.DataFrame,
    trench_position: float,
    line: Tuple[Tuple[float, float], Tuple[float, float]],
    length_deg: float = 7,
    slab_model: xr.Dataset = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Determine the slab interface for a given line using polynomial fit.

    Args:
        line_no (int): The line number.
        df_raw (pd.DataFrame): The raw DataFrame containing the events.
        trench_position (float): The position of the trench in km.
        line (Tuple[Tuple[float, float], Tuple[float, float]]): The line coordinates.
        length_deg (float, optional): The length of the line in degrees. Defaults to 7.
        slab_model (xr.Dataset, optional): The slab model dataset. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The x and y values of the slab interface.
    """
    df = df_raw.copy()
    for remove_dist, remove_depth in zip(
        config.EVENTS_REMOVE_DIST, config.EVENTS_REMOVE_DEPTH
    ):
        df = df[~((df["dist"] > remove_dist) & (df["dep"] < remove_depth))]
    df = constrain_using_slab2(df, line, length_deg, slab_model=slab_model)

    # Adjust the distances to set the trench position to zero
    df.loc[:, "adjusted_dist"] = df["dist"].sub(trench_position)
    logger.info(
        f"Line {line_no}: {len(df):04d} events, {len(df_raw):04d} raw events, trench at {trench_position:.2f} km"
    )

    # Include the point (0, 0) to ensure the polynomial passes through the trench
    df = pd.concat(
        [df, pd.DataFrame({"adjusted_dist": [0], "dep": [0]})], ignore_index=True
    )

    # Sort the DataFrame by adjusted distance
    df = df.sort_values(by="adjusted_dist")

    # Initial fit of the 3rd order polynomial
    # coeffs = np.polyfit(df["adjusted_dist"], df["dep"], 5)
    coeffs = biased_polyfit(df["adjusted_dist"], df["dep"], 5, alpha=1.0)
    poly = Polynomial(
        coeffs[::-1]
    )  # Polynomial expects coefficients in the opposite order

    # Iteratively refine the fit to ensure 80% of events are below the curve
    uplow = 1
    iteration = 0
    while uplow > config.CURVE_FIT_RATIO:
        mask = df["dep"] <= poly(df["adjusted_dist"])
        df_filtered = df[mask]
        # append 0,0 if not in the data
        if 0 not in df_filtered["adjusted_dist"].values:
            df_filtered = pd.concat(
                [
                    df_filtered,
                    pd.DataFrame({"adjusted_dist": [0], "dep": [0]}),
                ],
                ignore_index=True,
            )
        df_filtered = df_filtered.sort_values(by="adjusted_dist")

        # coeffs = np.polyfit(df_filtered["adjusted_dist"], df_filtered["dep"], 5)
        coeffs = biased_polyfit(
            df_filtered["adjusted_dist"], df_filtered["dep"], 5, alpha=1.0
        )
        poly = Polynomial(coeffs[::-1])
        uplow = mask.mean()

        logger.info(
            f"I: {iteration:03d}, R: {uplow:.2f}, E: {len(df_filtered):04d}, T: {len(df):04d}, TR: {len(df_raw):04d}"
        )
        iteration += 1
        if iteration > 100:
            break

    # Generate the slab surface curve
    x_vals = np.linspace(0, degrees2kilometers(length_deg) - trench_position, 1000)
    y_vals = poly(x_vals)
    x_vals += trench_position

    return x_vals, y_vals


def constrain_using_slab2(
    df: pd.DataFrame,
    line: Tuple[Tuple[float, float], Tuple[float, float]],
    length_deg: float = 7,
    slab_model: xr.Dataset = None,
):
    start, end = line
    startlat, startlon = start
    endlat, endlon = end

    points = pygmt.project(
        center=[startlon, startlat],
        endpoint=[endlon, endlat],
        generate=0.02,
    )
    lons = points.r
    lons[lons < 0] += 360
    lats = points.s

    slab_deps = slab_interp(slab_model, lons, lats)

    # try to fit using linear interpolation
    f = interp1d(
        np.linspace(0, degrees2kilometers(length_deg), len(slab_deps)),
        slab_deps,
        kind="linear",
        fill_value="extrapolate",
    )

    # remove all df points that are above the slab2
    mask = df["dep"] <= (f(df["dist"]) - 50)
    df_filtered = df[~mask]
    logger.info(f"Filtered {len(df) - len(df_filtered)} points using slab2")

    return df_filtered


def biased_polyfit(x, y, degree, alpha=1.0):
    # Define the objective function with a penalty term for smaller y values
    def objective(coeffs, x, y, alpha=1.0):
        poly = np.poly1d(coeffs)
        residuals = y - poly(x)
        penalty = -alpha * np.sum(poly(x))  # Penalty for lower y values
        return np.sum(residuals**2) + penalty

    # Initial guess for the polynomial coefficients
    initial_guess = np.polyfit(x, y, degree)

    # Minimize the objective function
    result = minimize(objective, initial_guess, args=(x, y, alpha))

    # Get the optimized coefficients
    optimized_coeffs = result.x
    return optimized_coeffs
