import warnings

import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from tqdm import tqdm

import src.config as config
from src import resource
from src.figure import plot_figure
from src.fit import determine_slab_interface
from src.manual_fit import fit_slab_interface_manually
from src.smooth_slab import smooth_across_sublists
from src.utils import (
    extend_line,
    find_slab_zero_depth_position,
    generate_perpendicular_lines,
    project_catalog,
)

pd.options.mode.copy_on_write = True


def process_row_for_slab_interface(
    i, row, projected_events, zero_depth_positions, slab_model, config_length_deg
):
    line = [list(row["Start"]), list(row["End"])]
    return determine_slab_interface(
        i,
        projected_events[i],
        zero_depth_positions[i],
        line,
        config_length_deg,
        slab_model=slab_model,
    )


def main():
    lines = generate_perpendicular_lines(
        config.SW_POINT,
        config.NE_POINT,
        length_deg=config.LENGTH_DEG,
        n_points=config.TOTAL_LINES,
    )

    zero_depth_positions = []
    slab_data = xr.open_dataset(
        resource(["slab2", "ker_slab2_depth.grd"], normal_path=True)
    )
    for _, row in tqdm(lines.iterrows(), total=len(lines), desc="Finding zero depth"):
        line = [list(row["Start"]), list(row["End"])]
        line[0][1] = (line[0][1] + 360) % 360
        line[1][1] = (line[1][1] + 360) % 360
        zero_depth_positions.append(
            float(
                find_slab_zero_depth_position(
                    slab_data, line, line_length_in_deg=config.LENGTH_DEG
                )
            )
        )

    events = pd.read_csv(resource(["catalog", "catalog.csv"], normal_path=True))

    projected_events = []
    for i, row in tqdm(lines.iterrows(), total=len(lines), desc="Projecting events"):
        line = [list(row["Start"]), list(row["End"])]
        projected_events.append(
            project_catalog(events, line, line_length_in_deg=config.LENGTH_DEG)
        )

    extra_projected_events = []
    extra_lines = []
    for startlon, startlat, endlon, endlat in config.EXTRA_LINES:
        endlon, endlat = extend_line(
            [startlon, startlat], [endlon, endlat], config.LENGTH_DEG
        )
        line = [(startlat, startlon), (endlat, endlon)]
        extra_lines.append(line)
        extra_projected_events.append(
            project_catalog(events, line, line_length_in_deg=config.LENGTH_DEG)
        )
    extra_lines = pd.DataFrame(extra_lines, columns=["Start", "End"])

    if not config.MANUAL_FIT:
        slab_model = xr.open_dataset(
            resource(["slab2", "ker_slab2_depth.grd"], normal_path=True)
        )

        rows = list(lines.iterrows())
        slab_interfaces = Parallel(n_jobs=-1)(
            delayed(process_row_for_slab_interface)(
                i,
                row,
                projected_events,
                zero_depth_positions,
                slab_model,
                config.LENGTH_DEG,
            )
            for i, row in tqdm(rows, total=len(rows), desc="Fitting slab interface")
        )

        # smoothing the slab interface
        smoothed = smooth_across_sublists(
            [slab_interface[1] for slab_interface in slab_interfaces],
            smoothing_sigma=config.SMOOTHING_SIGMA,
        )
        slab_interfaces = [
            [slab_interfaces[i][0], smoothed[i]] for i in range(len(slab_interfaces))
        ]
    else:
        depths_df = pd.read_csv(
            resource(["manual_modify", "slab_fit.csv"], normal_path=True)
        )
        extra_distances_df = pd.read_csv(
            resource(["manual_modify", "extra_fit.csv"], normal_path=True)
        )
        final_fitted_interface = fit_slab_interface_manually(
            lines, depths_df, extra_lines, extra_distances_df
        )
        slab_interfaces = [None for _ in range(len(lines))]

    for i, row in lines.iterrows():
        if i % config.PLOT_STEP == 0:
            line = [list(row["Start"]), list(row["End"])]
            plot_figure(
                i,
                projected_events[i],
                slab_interfaces[i],
                line,
                config.LENGTH_DEG,
                events,
                final_fitted_interface,
            )

    for i, projected_event in enumerate(extra_projected_events):
        startlon, startlat, endlon, endlat = config.EXTRA_LINES[i]
        endlon, endlat = extend_line(
            [startlon, startlat], [endlon, endlat], config.LENGTH_DEG
        )
        line = [(startlat, startlon), (endlat, endlon)]
        plot_figure(
            f"extra_{i}",
            projected_event,
            None,
            line,
            config.LENGTH_DEG,
            events,
            final_fitted_interface,
        )

    # for final_fitted_interface, rename x to longitude, y to latitude, and z to depth
    final_fitted_interface = final_fitted_interface.rename(
        {"x": "longitude", "y": "latitude", "z": "depth"}
    )
    # multiply -1 to depth
    final_fitted_interface["depth"] *= -1

    # save the final fitted interface
    final_fitted_interface.to_netcdf(config.XARRAY_SAVE_NAME)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        main()
