# import numpy as np
# import xarray as xr
# from pyproj import Geod
# from scipy.interpolate import griddata

# import src.config as config


# def interpolate_points_along_line(start, end, distances):
#     geod = Geod(ellps="WGS84")
#     az12, az21, total_distance = geod.inv(start[1], start[0], end[1], end[0])

#     points = []
#     for distance in distances:
#         if not np.isnan(distance):
#             lon, lat, _ = geod.fwd(
#                 start[1], start[0], az12, distance * 1000
#             )  # distance in meters
#             if lon < 0:
#                 lon += 360
#             points.append((lat, lon))
#         else:
#             break

#     return points


# def fit_slab_interface_manually(coords_df, depths_df):
#     depth_levels = depths_df["depth"].values
#     num_lines = coords_df.shape[0]

#     latitudes = []
#     longitudes = []
#     depths = []

#     for line_idx in range(num_lines):
#         start = coords_df.iloc[line_idx]["Start"]
#         end = coords_df.iloc[line_idx]["End"]
#         distances = depths_df.iloc[:, line_idx + 1].replace(-1, np.nan).values

#         points = interpolate_points_along_line(start, end, distances)
#         latitudes.extend([p[0] for p in points])
#         longitudes.extend([p[1] for p in points])
#         depths.extend(depth_levels[: len(points)])

#     # Create a grid for interpolation
#     grid_lon = np.arange(
#         config.MINLON, config.MAXLON + config.GRID_STEP, config.GRID_STEP
#     )
#     grid_lat = np.arange(
#         config.MINLAT, config.MAXLAT + config.GRID_STEP, config.GRID_STEP
#     )
#     grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

#     # Perform cubic interpolation
#     grid_depth = griddata(
#         (longitudes, latitudes), depths, (grid_lon, grid_lat), method="cubic"
#     )

#     slab_interface = xr.Dataset(
#         {"z": (["y", "x"], -grid_depth)},
#         coords={"y": grid_lat[:, 0], "x": grid_lon[0, :]},
#     )

#     return slab_interface


import numpy as np
import xarray as xr
from obspy.geodetics import degrees2kilometers
from pyproj import Geod
from scipy.interpolate import PchipInterpolator, UnivariateSpline, griddata, interp1d
from scipy.ndimage import gaussian_filter

import src.config as config


def interpolate_points_along_line(start, end, distances):
    geod = Geod(ellps="WGS84")
    az12, az21, total_distance = geod.inv(start[1], start[0], end[1], end[0])

    points = []
    for distance in distances:
        if not np.isnan(distance):
            lon, lat, _ = geod.fwd(
                start[1], start[0], az12, distance * 1000
            )  # distance in meters
            if lon < 0:
                lon += 360
            points.append((lat, lon))
        else:
            break

    return points


def fit_slab_interface_manually(coords_df, depths_df, extra_coords_df, extra_depth_df):
    all_latitudes = []
    all_longitudes = []
    all_depths = []

    for line_idx in range(coords_df.shape[0]):
        start = coords_df.iloc[line_idx]["Start"]
        end = coords_df.iloc[line_idx]["End"]
        distances = depths_df.iloc[:, line_idx + 1].replace(-1, np.nan).values
        # remove all nan values
        distances = distances[~np.isnan(distances)]

        # Interpolate depths along each line
        f = UnivariateSpline(
            distances, depths_df["depth"].values[: len(distances)], k=1, s=0, ext=1
        )

        dense_distance = np.linspace(0, degrees2kilometers(config.LENGTH_DEG), 100)
        interpolated_depths = f(dense_distance)

        interpolated_depths[interpolated_depths == 0] = np.nan

        dense_points = interpolate_points_along_line(start, end, dense_distance)
        dense_latitudes = [p[0] for p in dense_points]
        dense_longitudes = [p[1] for p in dense_points]

        all_latitudes.extend(dense_latitudes)
        all_longitudes.extend(dense_longitudes)
        all_depths.extend(interpolated_depths)

    for line_idx in range(extra_coords_df.shape[0]):
        start = extra_coords_df.iloc[line_idx]["Start"]
        end = extra_coords_df.iloc[line_idx]["End"]
        distances = extra_depth_df.iloc[:, line_idx + 1].replace(-1, np.nan).values
        # remove all nan values
        depths = extra_depth_df["depth"].values[~np.isnan(distances)]
        distances = distances[~np.isnan(distances)]

        # Interpolate depths along each line
        f = UnivariateSpline(distances, depths, k=1, s=0, ext=1)

        dense_distance = np.linspace(0, degrees2kilometers(config.LENGTH_DEG), 200)
        interpolated_depths = f(dense_distance)

        interpolated_depths[interpolated_depths == 0] = np.nan

        dense_points = interpolate_points_along_line(start, end, dense_distance)
        dense_latitudes = [p[0] for p in dense_points]
        dense_longitudes = [p[1] for p in dense_points]

        all_latitudes.extend(dense_latitudes)
        all_longitudes.extend(dense_longitudes)
        all_depths.extend(interpolated_depths)

    # Create a grid for interpolation
    grid_lon = np.arange(
        config.MINLON, config.MAXLON + config.GRID_STEP, config.GRID_STEP
    )
    grid_lat = np.arange(
        config.MINLAT, config.MAXLAT + config.GRID_STEP, config.GRID_STEP
    )
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

    # Perform nearest interpolation on the grid
    grid_depth = griddata(
        (all_longitudes, all_latitudes),
        all_depths,
        (grid_lon, grid_lat),
        method="nearest",
    )

    # Apply Gaussian smoothing
    grid_depth = gaussian_filter(grid_depth, sigma=0.5)

    # Create xarray DataArray
    # slab_interface = xr.DataArray(
    #     grid_depth,
    #     coords={"latitude": grid_lat[:, 0], "longitude": grid_lon[0, :]},
    #     dims=["latitude", "longitude"],
    #     name="depth",
    # )

    # return slab_interface
    slab_interface = xr.Dataset(
        {"z": (["y", "x"], -grid_depth)},
        coords={"y": grid_lat[:, 0], "x": grid_lon[0, :]},
    )
    slab_interface = slab_interface.where(slab_interface.z > -700)
    # print(slab_interface)

    # def constrain_nans(ds, var_name="z"):
    #     da = ds[var_name]  # Extract the DataArray
    #     values = da.values  # Convert DataArray to numpy array
    #     for i in range(values.shape[0]):  # Iterate over each latitude (y)
    #         x_vals = values[i, :]  # Get all longitudes for the specific latitude

    #         # Find the index of the maximum non-NaN longitude
    #         valid_indices = np.where(~np.isnan(x_vals))[0]
    #         if valid_indices.size == 0:
    #             continue  # Skip if all values are NaN
    #         max_non_nan_index = valid_indices[-1]

    #         # Set all previous values to NaN if they are NaN
    #         if max_non_nan_index < len(x_vals) - 1:
    #             for j in range(max_non_nan_index):
    #                 if np.isnan(x_vals[j]):
    #                     x_vals[: j + 1] = np.nan

    #         values[i, :] = x_vals  # Update the numpy array with constrained values

    #     # Update the Dataset with the constrained values
    #     ds[var_name] = xr.DataArray(
    #         values, coords=da.coords, dims=da.dims, name=da.name
    #     )
    #     return ds

    # slab_interface = constrain_nans(slab_interface)

    # print(slab_interface)

    return slab_interface
