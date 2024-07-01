import warnings

import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from tqdm import tqdm

import src.config as config
from src import resource
from src.figure import plot_figure
from src.fit import determine_slab_interface
from src.smooth_slab import smooth_across_sublists
from src.utils import (
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

    # slab_interfaces = []
    slab_model = xr.open_dataset(
        resource(["slab2", "ker_slab2_depth.grd"], normal_path=True)
    )
    # for i, row in tqdm(
    #     lines.iterrows(), total=len(lines), desc="Fitting slab interface"
    # ):
    #     line = [list(row["Start"]), list(row["End"])]
    #     slab_interfaces.append(
    #         determine_slab_interface(
    #             i,
    #             projected_events[i],
    #             zero_depth_positions[i],
    #             line,
    #             config.LENGTH_DEG,
    #             slab_model=slab_model,
    #         )
    #     )
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
            )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        main()
