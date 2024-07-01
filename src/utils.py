from typing import Tuple

import numpy as np
import pandas as pd
import pygmt
import pyproj
import xarray as xr
from obspy.geodetics.base import degrees2kilometers
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator

import src.config as config


def extend_line(
    start: Tuple[float, float], end: Tuple[float, float], length: float
) -> Tuple[float, float]:
    """Extend the current line to the specified length

    Args:
        start (Tuple[float,float]): the coordinate for the starting position
        end (Tuple[float,float]): the coordinate for the current ending position
        length (float): the expected new line length in degree

    Returns:
        Tuple[float,float]: the new ending position
    """
    startlon, startlat = start
    endlon, endlat = end
    g = pyproj.Geod(ellps="WGS84")
    az, _, _ = g.inv(startlon, startlat, endlon, endlat)
    newlon, newlat, _ = g.fwd(startlon, startlat, az, degrees2kilometers(length) * 1000)
    return newlon, newlat


def find_slab_zero_depth_position(
    slab_data: xr.Dataset,
    line: Tuple[Tuple[float, float], Tuple[float, float]],
    line_length_in_deg: float,
    n_points: int = 1000,
):
    """
    Find the percentile along the given line where the slab depth is closest to zero.

    Parameters:
    slab_data (xarray.Dataset): xarray dataset containing slab depth data.
    line (list): List containing two tuples, the start and end points of the line [(lat1, lon1), (lat2, lon2)].
    n_points (int): Number of evenly spaced points along the line.
    line_length_in_deg (float): Length of the line in degrees.

    Returns:
    float: Percentile along the line where slab depth is closest to zero.
    """
    # Extract start and end points
    start, end = line

    # Generate evenly spaced points between the start and end points
    latitudes = np.linspace(start[0], end[0], n_points)
    longitudes = np.linspace(start[1], end[1], n_points)
    latitudes = xr.DataArray(latitudes, dims="line")
    longitudes = xr.DataArray(longitudes, dims="line")

    # Interpolate depths for each point along the line
    depths = slab_data.interp(x=longitudes, y=latitudes, method="linear")["z"].values

    # interp using scipy
    x = np.linspace(0, degrees2kilometers(line_length_in_deg), n_points)
    y = depths
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    f = interpolate.interp1d(x, y, kind="linear", fill_value="extrapolate")
    depths = f(np.linspace(0, degrees2kilometers(line_length_in_deg), n_points))

    # Find the index where the depth is closest to zero
    closest_zero_depth_index = np.argmin(np.abs(depths))

    return np.linspace(0, degrees2kilometers(line_length_in_deg), n_points)[
        closest_zero_depth_index
    ]


def project_catalog(
    catalog: pd.DataFrame,
    line: Tuple[Tuple[float, float], Tuple[float, float]],
    line_length_in_deg: float,
) -> pd.DataFrame:
    """
    Project the catalog to the given line.

    Parameters:
    catalog (pd.DataFrame): DataFrame containing earthquake catalog.
    line (list): List containing two tuples, the start and end points of the line [(lat1, lon1), (lat2, lon2)].
    line_length_in_deg (float): Length of the line in degrees.

    Returns:
    pd.DataFrame: DataFrame containing projected catalog.
    """

    # change column names from lat,lon,dep to y,x,z
    start, end = line
    startlat, startlon = start
    endlat, endlon = end

    catalog = catalog.rename(
        columns={"latitude (degree)": "y", "longitude (degree)": "x", "depth (km)": "z"}
    )

    catalog = catalog[["x", "y", "z"]]
    catalog = catalog.reindex(columns=["x", "y", "z"])
    # project the catalog to the line
    res = pygmt.project(
        data=catalog,
        center=[startlon, startlat],
        endpoint=[endlon, endlat],
        convention="pz",
        unit=True,
        sort=True,
        length=[0, degrees2kilometers(line_length_in_deg)],
        width=[0, config.EVENTS_PROJECTION_WIDTH],
    )

    # change column names back
    res.columns = ["dist", "dep"]
    return res


def generate_perpendicular_lines(
    sw_point: Tuple[float, float],
    ne_point: Tuple[float, float],
    n_points: int = 100,
    length_deg: float = 7,
) -> pd.DataFrame:
    """
    Generate perpendicular lines from evenly sampled points between SW and NE points.

    Parameters:
    sw_point (tuple): (latitude, longitude) of the southwest point.
    ne_point (tuple): (latitude, longitude) of the northeast point.
    n_points (int): Number of evenly spaced points along the line.
    length_deg (float): Length of the perpendicular lines in degrees.

    Returns:
    pd.DataFrame: DataFrame containing the start and end coordinates of each line.
    """
    # Define the coordinate system (WGS84)
    geod = pyproj.Geod(ellps="WGS84")

    # Calculate azimuth from SW to NE
    azimuth, _, _ = geod.inv(sw_point[1], sw_point[0], ne_point[1], ne_point[0])

    # Generate evenly spaced points between SW and NE points
    latitudes = np.linspace(sw_point[0], ne_point[0], n_points)
    longitudes = np.linspace(sw_point[1], ne_point[1], n_points)

    # Function to calculate the perpendicular line coordinates
    def calculate_perpendicular_line(lat, lon, azimuth, length_deg=20):
        distance_km = degrees2kilometers(length_deg)
        # NW direction perpendicular to the original azimuth
        perpendicular_azimuth = azimuth - 90
        if perpendicular_azimuth < 0:
            perpendicular_azimuth += 360
        end = geod.fwd(lon, lat, perpendicular_azimuth, distance_km * 1000)
        return [(lat, lon), (end[1], end[0])]

    # Calculate perpendicular lines at each sampling point
    perpendicular_lines = []
    for lat, lon in zip(latitudes, longitudes):
        line = calculate_perpendicular_line(lat, lon, azimuth, length_deg)
        perpendicular_lines.append(line)

    # Create a DataFrame to store the results
    df = pd.DataFrame(perpendicular_lines, columns=["Start", "End"])
    return df


def slab_interp(
    to_interp_data: xr.DataArray, lons: np.ndarray, lats: np.ndarray
) -> np.ndarray:
    """generate the depth of the slab interface along a (lons,lats) track

    Args:
        to_interp_data (xr.DataArray): the loaded slab interface
        lons (np.ndarray): the lons track
        lats (np.ndarray): the lats track

    Returns:
        np.ndarray: the depth track
    """
    profile_list = []
    for ilon in range(len(lons)):
        profile_list.append([lons[ilon], lats[ilon]])
    # the names and the transverse might be adjusted, this is the gmt format
    grd_interpolating_function = RegularGridInterpolator(
        (to_interp_data.x.data, to_interp_data.y.data),
        -to_interp_data.z.data.T,
        bounds_error=False,
    )

    grd_interp_result = grd_interpolating_function(profile_list)

    # * return the 1d array
    return grd_interp_result
