"""
Global constants definition
"""

# Registered feeders. `DataLoader(feeder_name)` looks up its configuration here.
# To point this codebase at your own OpenDSS feeder instead of the bundled
# example, add an entry following the same shape (see README for details):
#   - data_path: folder containing an `opendss/` subfolder (with the master
#     .dss file) and a `solar_capacity_factor.csv` file.
#   - master_filename: the master .dss file to compile, inside `opendss/`.
#   - feeder_root: the bus at which the radial distribution feeder starts
#     (e.g. the substation transformer's low-side bus). The DSS file's own
#     root is not always this bus, so the circuit graph is truncated to the
#     subtree rooted here.
#   - v_root_nom: nominal line-to-neutral voltage at feeder_root, in kV.
#   - v_root_multiplier: multiplier to convert the feeder-head voltage
#     timeseries' native units to kV (e.g. 1e-3 if the source data is in V).
#   - v_root_setpoint_pu (optional, default 1.0): only for a feeder without
#     feeder-head voltage data, the regulated voltage at which
#     `DataLoader.get_feeder_head_data` holds the synthetic feeder head.
FEEDERS = {
    "ckt5": {
        "data_path": "data/example_feeder",
        "master_filename": "Master_ckt5.dss",
        # `63664` is a sub-feeder branch point within Ckt5's primary, chosen (rather
        # than the whole ~3,000-bus circuit) to keep the example close to the scale
        # of the paper's own case study (1,196 buses / 553 customers here vs.
        # 1,045 buses / 583 customers there) — the general LinDistFlow computation
        # is O(N^2) in memory, and the full circuit is impractically large/slow on
        # a laptop. The rest of Ckt5 is still there if you want a bigger example.
        "feeder_root": "63664",
        "v_root_nom": 7.1996,   # kV, line-to-neutral (12.47 kV line-line / sqrt(3))
        "v_root_multiplier": 1.0,
        # A typical substation LTC setpoint. At a 1.0 pu feeder head the subtree's own
        # load takes its lowest bus voltage below 0.95 pu (OpenDSS: 0.943 pu at nominal load).
        "v_root_setpoint_pu": 1.03,
    },
}
DEFAULT_FEEDER_NAME = "ckt5"

# Default candidate PV capacity per site (kW) used when no site-specific
# rooftop/consumption-based limit is supplied.
DEFAULT_CANDIDATE_PV_CAPACITY_KW = 10.0

# Fallback annual load factor (mean load / nameplate rating) used to estimate
# annual consumption for loads with no assigned `yearly` load shape.
DEFAULT_ANNUAL_LOAD_FACTOR = 0.35

# Constants
LARGE = 0       # LARGE = 1e9
VOL_TOL = 0.05
LOCATION = "America/Los_Angeles"


"""Plotting parameters"""
GIS_HEIGHT_SHRINK = 0.65
PLOT_LIM_LINES = False
PLOT_TITLE = False
PLOT_COLORBAR_LABEL = False
GIS_FIGSIZE = (6, 5)
GRID_ALPHA = 0.3

# General plot settings
COLOR_CUSTOMER = '#ff7f0e'  # Orange for customer PV
COLOR_UTILITY = '#1f77b4'   # Blue for utility PV
COLOR_OTHER1 = '#2ca02c'   # Green
COLOR_OTHER2 = '#d62728'   # Red
IEEE_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'mathtext.fontset': 'cm',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 12,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
    'figure.dpi': 600,
}
