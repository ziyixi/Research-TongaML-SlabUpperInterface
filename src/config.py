SW_POINT = [-26, -174.5]  # southwest point
NE_POINT = [-16, -171]  # northeast point
LENGTH_DEG = 8  # length of the line in degrees
EVENTS_PROJECTION_WIDTH = 100  # width of the projected events

EVENTS_REMOVE_DEPTH = [100, 250, 300]  # depth of the events to remove
EVENTS_REMOVE_DIST = [250, 400, 600]  # distance of the events to remove

CURVE_FIT_RATIO = 0.2  # ratio of the curve fit

# TOTAL_LINES = 2000  # total number of lines
# PLOT_STEP = 200  # plot step
# SMOOTHING_SIGMA = 10.0  # smoothing sigma
TOTAL_LINES = 10  # total number of lines
PLOT_STEP = 1  # plot step
SMOOTHING_SIGMA = 1.0  # smoothing sigma

MANUAL_FIT = True  # manual fit using data/manual_modify/slab_fit.csv
MINLAT, MAXLAT, MINLON, MAXLON = -26, -14, 176, 188
GRID_STEP = 0.1

XARRAY_SAVE_NAME = "tonga_slab_interface.nc"

# extra plotting lines
EXTRA_LINES = [
    [-177.5, -17, -181.5, -23],
    [179.0, -19, -176.5, -19],
]
