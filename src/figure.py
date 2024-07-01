from typing import Tuple

import numpy as np
import pandas as pd
import pygmt
import xarray as xr
from obspy.geodetics.base import degrees2kilometers

from src import resource, save_path
from src.utils import slab_interp


def plot_figure(
    line_no: int,
    catalog_line: pd.DataFrame,
    slab_interface: Tuple[np.ndarray, np.ndarray],
    line: Tuple[Tuple[float, float], Tuple[float, float]],
    line_length_in_deg: float,
    catalog: pd.DataFrame,
):
    start, end = line
    startlat, startlon = start
    endlat, endlon = end

    fig = pygmt.Figure()
    pygmt.config(
        FONT_LABEL="18p",
        MAP_LABEL_OFFSET="12p",
        FONT_ANNOT_PRIMARY="16p",
        MAP_FRAME_TYPE="plain",
        MAP_TITLE_OFFSET="4p",
        FONT_TITLE="15p,black",
        MAP_FRAME_PEN="1p,black",
    )

    # plot the slab
    title = f"Slab Interface along ({startlat:.2f}, {startlon:.2f}) to ({endlat:.2f}, {endlon:.2f})"
    fig.basemap(
        projection="X4i/-4i",
        frame=[f"+t{title}", "xaf+lDistance (km)", "yaf+lDepth (km)"],
        region=[
            0,
            line_length_in_deg * degrees2kilometers(1),
            0,
            700,
        ],
    )
    fig.plot(
        x=catalog_line["dist"],
        y=catalog_line["dep"],
        style="c0.08c",
        pen="0.01c,black",
    )
    fig.plot(
        x=slab_interface[0],
        y=slab_interface[1],
        pen="1.5p,red",
        label="Fitted Slab Interface",
    )

    points = pygmt.project(
        center=[startlon, startlat],
        endpoint=[endlon, endlat],
        generate=0.02,
    )
    lons = points.r
    lons[lons < 0] += 360
    lats = points.s

    # plot slab2
    slab_model = xr.open_dataset(
        resource(["slab2", "ker_slab2_depth.grd"], normal_path=True)
    )
    slab_deps = slab_interp(slab_model, lons, lats)
    fig.plot(
        x=np.linspace(
            0,
            line_length_in_deg * degrees2kilometers(1),
            len(slab_deps),
        ),
        y=slab_deps,
        pen="1.5p,magenta",
        label="Slab2",
    )

    # plot slab1
    slab1_model = xr.open_dataset(
        resource(["slab2", "ker_slab1.0_clip.grd"], normal_path=True), decode_cf=False
    )
    slab1_deps = slab_interp(slab1_model, lons, lats)
    fig.plot(
        x=np.linspace(
            0,
            line_length_in_deg * degrees2kilometers(1),
            len(slab1_deps),
        ),
        y=slab1_deps,
        pen="1.5p,blue",
        label="Slab 1.0",
    )
    with pygmt.config(FONT_ANNOT_PRIMARY="8p"):
        fig.legend(
            position="jBL+o0.2c/0.2c",
            transparency=50,
        )

    # plot the map
    fig.shift_origin(xshift="5i")
    fig.basemap(
        region=[-184, -172, -26, -14],
        projection="M4i",
        frame=[
            "xaf+lLongitude (degree)",
            "yaf+lLatitude (degree)",
        ],
    )
    plot_earth_relief(fig)
    # plot events
    pygmt.makecpt(cmap="jet", series=[0, 700, 1], continuous=True, reverse=True)
    fig.plot(
        x=catalog["longitude (degree)"],
        y=catalog["latitude (degree)"],
        style="c0.08c",
        fill=catalog["depth (km)"],
        cmap=True,
    )
    fig.plot(
        x=[startlon, endlon],
        y=[startlat, endlat],
        pen="3p,black",
    )

    save_path(fig, f"{line_no}")


def plot_earth_relief(fig: pygmt.Figure):
    grd_topo = pygmt.datasets.load_earth_relief(
        resolution="02m", region=[-184, -172, -26, -14], registration="gridline"
    )
    assert type(grd_topo) == xr.DataArray
    # plot 2000m contour of topography, start from -2000m to -10000m
    fig.grdcontour(
        grd_topo,
        levels=1000,
        pen="1p,gray",
        limit="-10000/-7000",
    )
    # plot only -1000m contour
    fig.grdcontour(
        grd_topo,
        levels=1000,
        pen="1.3p,gray",
        limit="-1100/-1000",
    )
    fig.coast(land="gray")
