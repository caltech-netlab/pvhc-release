"""
Plotting primitives (GIS network maps, timeseries, heatmaps) used by scripts/plot_figures.py.
"""
import os
import shutil
import matplotlib
matplotlib.use("Agg")  # non-interactive backend: save PNGs/PDFs without a display,
                        # and don't block on plt.show() in non-interactive runs.
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from dateutil import parser
import datetime
from tqdm import tqdm

timefmt = mdates.DateFormatter('%I%p')

import pathlib, sys
file = pathlib.Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
from src.constants import *


# Configure matplotlib for IEEE style
plt.rcParams.update(IEEE_STYLE)

blk_red_cmap = LinearSegmentedColormap.from_list(
    "black_to_red",
    ["black", "red"]
)

"""
Helper function code.
"""

def load_node_coordinates(csv_path, node_list=None, lowercase=False):
    """
    Load node coordinates from a CSV file.
    
    Args:
        csv_path: str, path to the CSV file containing node coordinates.
            CSV format: bus_name, x_coord, y_coord (comma or space separated)
        node_list: list of str, optional. If provided, returns an np.array 
            of coordinates in the order specified by node_list.
        lowercase: bool, optional. If True, convert bus names to lowercase.
    
    Returns:
        If node_list is None: dict mapping bus names to (x, y) tuples
        If node_list is provided: np.ndarray of shape (N, 2) where N = len(node_list)
    """
    
    # Read CSV and build coordinate dictionary
    coord_dict = {}
    with open(csv_path, 'r') as f:
        # Parse the CSV
        for line in f:
            parts = line.strip().replace(',', ' ').split()
            if len(parts) >= 3:
                bus_name = parts[0]
                x_coord = float(parts[1])
                y_coord = float(parts[2])
                if lowercase:
                    bus_name = bus_name.lower()
                coord_dict[bus_name] = (x_coord, y_coord)
    
    # Return based on whether node_list is provided
    if node_list is None:
        return coord_dict
    else:
        # Build array in the order specified by node_list
        coords_array = np.zeros((len(node_list), 2))
        for i, bus_name in enumerate(node_list):
            if bus_name in coord_dict:
                coords_array[i] = coord_dict[bus_name]
            else:
                # If bus not found, could raise warning or set to NaN
                coords_array[i] = [np.nan, np.nan]
        return coords_array


def fill_missing_node_coordinates(node_coords, edge_list):
    """
    Fill in missing (NaN) coordinates by propagating a neighbor's coordinate
    along tree edges. Many buses -- e.g. synthetic secondary/customer buses
    behind a service transformer -- have no entry of their own in the OpenDSS
    Buscoords file since they are co-located with a coordinate-bearing
    primary bus; this places them there instead of silently dropping them
    from GIS plots. `node_coords` is modified in place and also returned.
    """
    missing = np.isnan(node_coords).any(axis=1)
    while missing.any():
        changed = False
        for i, j in edge_list:
            if missing[i] and not missing[j]:
                node_coords[i] = node_coords[j]
                missing[i] = False
                changed = True
            elif missing[j] and not missing[i]:
                node_coords[j] = node_coords[i]
                missing[j] = False
                changed = True
        if not changed:
            break  # remaining NaNs are on components with no coordinate at all
    return node_coords


def strptime_np(t, unit="us"):
    """Converts str time to numpy.datetime64[unit], the maximum resolution is us."""
    if t is None:
        return None
    else:
        if type(t) is np.datetime64:
            return t
        elif type(t) is str:
            return np.datetime64(parser.parse(t), unit)
        elif type(t) is datetime.datetime:
            return np.datetime64(t, unit)
        else:
            raise TypeError(f"Input t: {t} has type {type(t)}.")
        

def parse_time(t):
    if t is None:
        return None
    elif type(t) is str:
        return strptime_np(t)
    elif type(t) is datetime.datetime:
        return np.datetime64(t)
    elif type(t) is np.datetime64:
        return t
    else:
        raise TypeError(f"Unsupported type: {type(t)}")


def np_searchsorted(t_column, t, mode="nearest", inclusive=False, safe_clip=True):
    """
    Search a particular timestamp in df.
    :param t_column: pandas.Series or dict of numpy.ndarray
    :param t: numpy.datetime64 or an iterable of numpy.datetime64
    :param mode: str, one of
        - "nearest": find nearest element
        - "nearest_before": find the closest time before t. This is the same
            as the index to insert in order to maintain order.
        - "nearest_after": find the closest time after t. This is the same
            as the index to insert in order to maintain order, minus 1.
    :param inclusive: bool,
        - True: return the index of an exact match. If the match is a repeated entry,
            nearest_before returns the index of the first repeated entry and nearest_after
            returns the index of the last repeated entry.
        - False: strict inequality when comparing.
            nearest_before returns the index of the element strictly before t.
            nearest_after returns the index of the elmenet strictly after t.
    :return: array of int, indices into df
    """
    if (t_column is None) or (not len(t_column)):
        raise ValueError("Cannot search timestamp in empty column")
    is_iter = hasattr(t, "__iter__")
    t = t if hasattr(t, "__iter__") else [t]
    L = len(t_column)
    if mode == "nearest":
        inds = t_column.searchsorted(t, side="left")
        for i in range(len(inds)):
            if inds[i] > 0 and (
                inds[i] == len(t_column)
                or abs(t[i] - t_column[inds[i] - 1]) < abs(t[i] - t_column[inds[i]])
            ):
                inds[i] -= 1
    elif mode == "nearest_before":
        inds = t_column.searchsorted(t, side="left")
        for i in range(len(inds)):
            if (inds[i] == L) or (t_column[inds[i]] != t[i]) or not inclusive:
                inds[i] = max(0, inds[i] - 1) if safe_clip else inds[i] - 1
    elif mode == "nearest_after":
        inds = t_column.searchsorted(t, side="right")
        for i in range(len(inds)):
            if (inds[i] < L) and (t_column[inds[i]] == t[i]) and not inclusive:
                inds[i] += 1
            elif (inds[i] > 0) and (t_column[inds[i] - 1] == t[i]) and inclusive:
                inds[i] -= 1
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return inds if is_iter else inds[0]


"""
Plotting code
"""


def plot_voltage_violations_count(t, voltage_arr, v_min=None, v_max=None, figsize=(6, 4), outfile=None, plot_title=PLOT_TITLE):
    """
    x axis: time, y axis: number of violations at that time
    Assumes voltages are in p.u.
    :param t: np.ndarray of datetime64, shape (T,)
    :param voltage_arr: np.ndarray of voltages in p.u., shape (N, T)
    """
    fig, ax = plt.subplots(figsize=figsize)
    if v_min is not None:
        v_min_violation = voltage_arr < v_min
        num_v_min_violation = v_min_violation.sum(axis=1)
        ax.scatter(t, num_v_min_violation, s=figsize[0]*2, label=f'under voltage ({v_min} p.u.)', color='r', marker='x')
    if v_max is not None:
        v_max_violation = voltage_arr > v_max
        num_v_max_violation = v_max_violation.sum(axis=1)
        ax.scatter(t, num_v_max_violation, s=figsize[0], label=f'over voltage ({v_max} p.u.)', color='b')
    # Stylistic formatting
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of violations')
    ax.legend()
    ylim = ax.get_ylim()
    ax.set_ylim(bottom=0, top=max(ylim[1], 2))
    ax.xaxis.set_major_formatter(timefmt)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    N = voltage_arr.shape[0]
    date = str(t[0])[:10]
    if plot_title:
        ax.set_title(f"Voltage violations for {N} buses on {date}")
    fig.tight_layout()
    if outfile:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        plt.savefig(outfile)
    plt.close(fig)


def plot_values_range_timeseries(t, v_arr, ylabel, title=None, v_min=None, v_max=None, ylim=None, figsize=(6, 2.8), fontsize=12, outfile=None, plot_lim_lines=PLOT_LIM_LINES, plot_title=PLOT_TITLE):
    fig, ax = plt.subplots(figsize=figsize)
    T, N = v_arr.shape
    for i in range(N):
        ax.plot(t, v_arr[:, i], color='black', alpha=0.1)
    if v_min is not None and plot_lim_lines:
        ax.axhline(y=v_min, color='r', linestyle='--', label=f'min voltage ({v_min} p.u.)')
    if v_max is not None and plot_lim_lines:
        ax.axhline(y=v_max, color='b', linestyle='--', label=f'max voltage ({v_max} p.u.)')
    # Stylistic formatting
    ax.set_xlabel('Time', fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.xaxis.set_major_formatter(timefmt)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    # Set custom x-ticks if ylim is (0, 1)
    if ylim == (0, 1):
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0', '0.25', '0.50', '0.75', '1.0'])
    if title and plot_title: ax.set_title(title, fontsize=fontsize)
    if ylim: ax.set_ylim(bottom=ylim[0], top=ylim[1])
    ax.grid(True, linestyle='--', alpha=GRID_ALPHA)
    fig.tight_layout()
    if outfile:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        plt.savefig(outfile)
    plt.close(fig)



def plot_voltage_violation_duration(t, voltage_arr, v_min=None, v_max=None, figsize=(12, 4), outfile=None, plot_title=PLOT_TITLE):
    """
    x axis: bus index, ordered by violation duration
    y axis: violation duration (minutes)
    :param t: np.ndarray of datetime64, shape (T,)
    :param voltage_arr: np.ndarray of voltages in p.u., shape (T, N)
    """
    t_delta = np.median(t[1:] - t[:-1]) / np.timedelta64(1, 'h')  # in hours
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    if v_min is not None:
        v_min_violation = voltage_arr < v_min
        num_v_min_violation = np.sort(v_min_violation.sum(axis=0))[::-1] * t_delta
        axs[0].bar(range(len(num_v_min_violation)), num_v_min_violation, color='r', width=1.0)
        axs[0].set_title('Under voltage')
        axs[0].set_ylabel('Violation duration (hours)')
        ylim = axs[0].get_ylim()
        axs[0].set_ylim(bottom=0, top=max(ylim[1], 15))
        axs[0].set_xticks([])
        axs[0].scatter(
            np.where(num_v_min_violation==0), [0] * (num_v_min_violation==0).sum(), 
            s=figsize[1], color='r', label=f'Under voltage ({v_min} p.u.)'
        )
        axs[0].legend()
    if v_max is not None:
        v_max_violation = voltage_arr > v_max
        num_v_max_violation = np.sort(v_max_violation.sum(axis=0))[::-1] * t_delta
        axs[1].bar(range(len(num_v_max_violation)), num_v_max_violation, color='b', width=1.0)
        axs[1].set_title('Over voltage')
        ylim = axs[1].get_ylim()
        axs[1].set_ylim(bottom=0, top=max(ylim[1], 15))
        axs[1].set_xticks([])
        axs[1].scatter(
            np.where(num_v_max_violation==0), [0] * (num_v_max_violation==0).sum(), 
            s=figsize[1], color='b', label=f'Over voltage ({v_max} p.u.)'
        )
        axs[1].legend()
    fig.text(0.5, 0.04, 'Buses, in order of violation duration', ha='center')
    N = voltage_arr.shape[0]
    date = str(t[0])[:10]
    if plot_title:
        fig.suptitle(f"Voltage violation duration for {N} buses on {date}")
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    if outfile:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        plt.savefig(outfile)
    plt.close(fig)


def plot_voltage_violation_duration_amount(t, voltage_arr, v_min=None, v_max=None, figsize=(12, 4), outfile=None, plot_title=PLOT_TITLE):
    """
    x axis: bus index, ordered by violation duration-amount
    y axis: violation duration-amount (minute-p.u.)
    """
    t_delta = np.median(t[1:] - t[:-1]) / np.timedelta64(1, 'h')  # in hours
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    if v_min is not None:
        num_v_min_violation = np.sort(np.maximum(0, (v_min - voltage_arr)).sum(axis=0))[::-1] * t_delta
        axs[0].bar(range(len(num_v_min_violation)), num_v_min_violation, color='r', width=1.0)
        axs[0].set_title('Under voltage')
        axs[0].set_ylabel('Violation duration-amount (hour-p.u.)')
        axs[0].set_ylim(bottom=0)
        axs[0].set_xticks([])
        axs[0].scatter(
            np.where(num_v_min_violation==0), [0] * (num_v_min_violation==0).sum(), 
            s=figsize[1] / 4, color='r', label=f'Under voltage ({v_min} p.u.)'
        )
        axs[0].legend()
    if v_max is not None:
        num_v_max_violation = np.sort(np.maximum(0, (voltage_arr - v_max)).sum(axis=0))[::-1] * t_delta
        axs[1].bar(range(len(num_v_max_violation)), num_v_max_violation, color='b', width=1.0)
        axs[1].set_title('Over voltage')
        axs[1].set_ylim(bottom=0)
        axs[1].set_xticks([])
        axs[1].scatter(
            np.where(num_v_max_violation==0), [0] * (num_v_max_violation==0).sum(), 
            s=figsize[1] / 4, color='b', label=f'Over voltage ({v_max} p.u.)'
        )
        axs[1].legend()
    fig.text(0.5, 0.04, 'Buses, in order of violation duration', ha='center')
    N = voltage_arr.shape[0]
    date = str(t[0])[:10]
    if plot_title:
        fig.suptitle(f"Voltage violation duration for {N} buses on {date}")
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    if outfile:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        plt.savefig(outfile)
    plt.close(fig)


def plot_pu_min_area_under_curve(t, voltage_arr, node_list, v_min=None, v_max=None, figsize=(6, 4), outfile=None, plot_title=PLOT_TITLE):
    """
    For total voltage violation amount * duration, it is equivalent to area under curve.
    This plot is an illustration of the quantity.
    Pick the bus with the highest violation amount to plot (one bus only).
    """
    fig, ax = plt.subplots(figsize=figsize)
    idx = np.argmax(voltage_arr) // voltage_arr.shape[0]
    v_pu = voltage_arr[:, idx]
    ax.plot(t, v_pu, color='black', alpha=0.8)
    ax.fill_between(t, v_pu, v_max, where=(v_pu > v_max), color='blue', alpha=0.5)
    if v_min is not None:
        ax.axhline(y=v_min, color='r', linestyle='--', label=f'min voltage ({v_min} p.u.)')
    if v_max is not None:
        ax.axhline(y=v_max, color='b', linestyle='--', label=f'max voltage ({v_max} p.u.)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Voltage (p.u.)')
    ax.xaxis.set_major_formatter(timefmt)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
    if plot_title:
        ax.set_title(f"Bus {node_list[idx // 3]} phase {'ABC'[idx % 3]} voltage")
    ax.set_ylim(bottom=0.9, top=1.1)
    plt.tight_layout()
    if outfile:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        plt.savefig(outfile)
    plt.close(fig)


def violation_duration_hist(t, violation_arr, figsize=(6, 4), title="", outfile=None, plot_title=PLOT_TITLE):
    """
    Histogram of violation durations.
    x axis: violation duration (minutes), 
    y axis: empirical probability
    :param t: np.ndarray of datetime64, shape (T,)
    :param violation_arr: np.ndarray of boolean, shape (T, N)
    """
    t_delta = np.median(t[1:] - t[:-1]) / np.timedelta64(1, 'h')  # in hours
    # Calculate violation duration for each violation event
    violation_duration = []
    # Use int diff along time axis: +1 = start of violation, -1 = end of violation
    violation_changes = np.diff(violation_arr.astype(int), axis=0)  # shape (T-1, N)
    T, N = violation_arr.shape
    for n in range(N):
        starts = np.where(violation_changes[:, n] == 1)[0] + 1  # start index (in 0..T-1)
        ends   = np.where(violation_changes[:, n] == -1)[0]     # end index (last True)
        # If violation is already active at t=0, prepend start=0
        if violation_arr[0, n]:
            starts = np.r_[0, starts]
        # If violation is still active at t=T-1, append end=T-1
        if violation_arr[-1, n]:
            ends = np.r_[ends, T - 1]
        # Pair starts/ends and compute durations (inclusive)
        violation_duration.append((ends - starts + 1) * t_delta)

    violation_duration = np.concatenate(violation_duration)
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(
        violation_duration, bins=20, edgecolor='black', color='brown', 
        weights=np.ones(len(violation_duration)) * (1 / len(violation_duration)) if len(violation_duration) else None
    )
    ax.set_xlabel('Violation duration (hours)')
    ax.set_ylabel('Empirical probability')
    if title and plot_title: ax.set_title(title)
    # Plot vertical line at median
    if len(violation_duration):
        median = np.median(violation_duration)
        ax.axvline(x=median, color='black', linestyle='--', label=f'Median ({median:.2f} h)')
        ax.legend()
    fig.tight_layout()
    if outfile:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        plt.savefig(outfile)
    plt.close(fig)
    

def gis_visualize(node_coords, outfile, nodal_values=None, edge_values=None, edge_list=None, cb_label_nodal=None, cb_label_edge=None, cmap_nodal='rainbow', cmap_edge=blk_red_cmap, title=None, background_img=None, scale=None, anchor=None, xlim=None, ylim=None, clim_nodal=(None, None), clim_edge=(None, None), zero_length_edge_threshold=0.5, nonzero_only=False, text=None, text_anchor=(0.05, 0.05), figsize=GIS_FIGSIZE, height_shrink=GIS_HEIGHT_SHRINK, markersize=10, plot_title=PLOT_TITLE, plot_colorbar_label=PLOT_COLORBAR_LABEL, combine_colorbars_if_equal=True):
    """
    Plot nodal values (e.g. voltages, pv capacities) on a geo-spatial map.

    :param node_coords: np.ndarray, shape (n, 2) of bus coordinates.
    :param nodal_values: array-like or np.ndarray, shape (n) of nodal values.
        Not all nodes may have a value. The node_idx below specifies which nodes these values correspond to.
    :param node_idx: iterable of int, indices of nodes corresponding to nodal_values.
    :param outfile: str, filepath where the output image will be saved.
    :param background_img: str, optional filepath to a basemap image used as background.
    :param INSTALLED_OR_HOSTING: bool, toggles title/label wording for worst-case scenario.
    :param cb_label_nodal: str, optional label for colorbar in PV capacity plot.
        e.g. "PV Installation Capacity (kW)", or "Total PV Hosting Capacity (kW)".
    :param title: str, optional title for PV capacity plot.
        e.g. "PV Installation Capacity Map: %d kW" % sum(pv_capacities), or "PV Hosting Capacity Map: %d kW" % sum(pv_capacities).
    :param scale: float, optional scale for the background image.
        e.g. scale = 10680
    :param anchor: tuple of float, optional anchor point (bottom-left) for the background image.
        e.g. anchor = (6520760, 1870350)
    :param xlim: tuple of float, optional x-axis limits for the plot.
        e.g. xlim = (6523500, 6531200)
    :param ylim: tuple of float, optional y-axis limits for the plot.
        e.g. ylim = (1870500, 1876400)
    :param clim_nodal: tuple of float, optional color limits for the plot.
        e.g. clim_nodal = (0.8, 1.2)
    :param nonzero_only: bool, if True, only plot nodes with nonzero nodal_values.
    """
    from PIL import Image

    def create_black_background(node_coords, background_img=background_img, figsize=figsize, scale=scale, anchor=anchor):
        fig, ax = plt.subplots(figsize=figsize)
        plt.tick_params(axis="both", which="both", length=0, labelleft=False, labelbottom=False)
        # Plot connecting lines
        if edge_list is not None:
            for (i, j) in edge_list:
                plt.plot([node_coords[i][0], node_coords[j][0]], [node_coords[i][1], node_coords[j][1]], "k", alpha=0.4)
        # Plot background image if provided
        if background_img:
            img = Image.open(background_img)
            aspect_ratio = img.height / img.width
            plt.imshow(
                img,
                alpha=0.5,
                aspect="equal",
                zorder=0,
                extent=[anchor[0], anchor[0] + scale, anchor[1], anchor[1] + scale * aspect_ratio],
            )
        return fig, ax

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plot_nodal = nodal_values is not None
    plot_edge = edge_values is not None
    if height_shrink is not None:
        node_coords[:, 1] *= height_shrink

    coords_scale = np.max(node_coords, axis=0) - np.min(node_coords, axis=0)
    fig, ax = create_black_background(node_coords, figsize=figsize)
    
    # Plot edge values if provided
    if plot_edge:
        assert edge_list is not None
        vmin = clim_edge[0] if clim_edge[0] is not None else np.min(edge_values)
        vmax = clim_edge[1] if clim_edge[1] is not None else np.max(edge_values)
        # edge_values with no variation (e.g. all-zero sensitivity, meaning no
        # scenario/network constraint ever bound -- see compute_series_resistance_
        # sensitivity's docstring) always normalize to exactly `norm(vmin) == 0`,
        # which the value-threshold gate below would then suppress for every
        # single edge, silently producing a blank plot. Bypass that gate here
        # since "below threshold" is meaningless when nothing varies.
        degenerate_values = vmax <= vmin
        if degenerate_values:
            print(f"[Info] edge_values has no variation (all ~{vmin:.6g}); "
                  f"plotting all edges with this single color.")
            vmax = vmin + 1.0
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(cmap_edge)
        for idx, (i, j) in enumerate(edge_list):
            val = norm(edge_values[idx])
            # Handle zero-length edges (e.g. transformer)
            if degenerate_values or val > zero_length_edge_threshold:
                if (np.abs(node_coords[i] - node_coords[j]) / coords_scale).max() < 0.01:
                    plt.scatter(
                        x=node_coords[i][0], 
                        y=node_coords[i][1], 
                        color=cmap(val), 
                        alpha=1, 
                        s=markersize * 4,
                        marker='x'
                    )
                else:
                    plt.plot(
                        [node_coords[i][0], node_coords[j][0]],
                        [node_coords[i][1], node_coords[j][1]],
                        color=cmap(val),
                        alpha=1,
                        linewidth=2,
                    )
            else:
                continue
    
        sm_edge = plt.cm.ScalarMappable(cmap=cmap_edge, norm=norm)
        sm_edge.set_array([])
    
    # Plot nodal values (color the node, i.e. scatter plot)
    if plot_nodal:
        xlist, ylist, r = [], [], []
        for node_coord, node_value in zip(node_coords, nodal_values):
            if nonzero_only and node_value == 0:
                continue
            xlist.append(node_coord[0])
            ylist.append(node_coord[1])
            r.append(node_value)
        norm_nodal = plt.Normalize(
            vmin=clim_nodal[0] if clim_nodal[0] is not None else np.min(nodal_values),
            vmax=clim_nodal[1] if clim_nodal[1] is not None else np.max(nodal_values)
        )
        cmap_obj = plt.get_cmap(cmap_nodal)
        plt.scatter(x=xlist, y=ylist, c=r, s=markersize, norm=norm_nodal, cmap=cmap_nodal, alpha=0.8)
        sm_nodal = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm_nodal)
        sm_nodal.set_array([])

    # Colorbar
    colorbar_nodal, colorbar_edge = plot_nodal, plot_edge
    if combine_colorbars_if_equal and plot_nodal and plot_edge and (clim_nodal == clim_edge):
        colorbar_edge = False
    
    if colorbar_nodal and colorbar_edge:
        cax_nodal = ax.inset_axes([1.02, 0.05, 0.03, 0.40])
        cax_edge  = ax.inset_axes([1.02, 0.55, 0.03, 0.40])
    elif colorbar_nodal:
        cax_nodal = ax.inset_axes([1.02, 0.1, 0.03, 0.75])
    elif colorbar_edge:
        cax_edge  = ax.inset_axes([1.02, 0.1, 0.03, 0.75])
    if colorbar_nodal:
        cb_nodal = fig.colorbar(sm_nodal, cax=cax_nodal)
        if cb_label_nodal and plot_colorbar_label:
            cb_nodal.set_label(cb_label_nodal)
        # Use scientific notation for colorbar if upper scale is large
        if cb_nodal and cb_nodal.get_ticks()[-1] > 1e4:
            cb_nodal.formatter = ticker.ScalarFormatter(useMathText=True)
            cb_nodal.formatter.set_powerlimits((0, 0))
            cb_nodal.ax.yaxis.set_major_locator(ticker.MaxNLocator())
            cb_nodal.update_ticks()
            cb_nodal.ax.yaxis.get_offset_text().set_x(1.)
            cb_nodal.ax.yaxis.get_offset_text().set_ha('left')
    if colorbar_edge:
        cb_edge = fig.colorbar(sm_edge, cax=cax_edge)
        if cb_label_edge and plot_colorbar_label:
            cb_edge.set_label(cb_label_edge)    
        # Use scientific notation for colorbar if upper scale is large
        if cb_edge and cb_edge.get_ticks()[-1] > 1e4:
            cb_edge.formatter = ticker.ScalarFormatter(useMathText=True)
            cb_edge.formatter.set_powerlimits((0, 0))
            cb_edge.ax.yaxis.set_major_locator(ticker.MaxNLocator())
            cb_edge.update_ticks()
            cb_edge.ax.yaxis.get_offset_text().set_x(1.)
            cb_edge.ax.yaxis.get_offset_text().set_ha('left')
    # Plot text if provided
    if text is not None:
        ax.text(text_anchor[0], text_anchor[1], text, 
                transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # General plot formatting
    if title and plot_title: plt.title(title)
    if xlim and ylim:
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
    ax.set_aspect("equal", "box")
    fig.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def gis_visualize_timeseries(timestamps, node_coords, nodal_values, output_dir, edge_list=None, edge_values=None, cb_label_nodal=None, cb_label_edge=None, cmap_nodal='rainbow', cmap_edge=blk_red_cmap, title=None, background_img=None, scale=None, anchor=None, xlim=None, ylim=None, clim_nodal=(None, None), clim_edge=(None, None), figsize=(10, 10), markersize=12, ext='png', plot_title=PLOT_TITLE, plot_colorbar_label=PLOT_COLORBAR_LABEL):
    """
    Plot nodal values (e.g. bus voltages) on a geo-spatial map over time.

    :param timestamps: np.ndarray, time vector.
    :param node_coords: np.ndarray, shape (n, 2) of bus coordinates.
    :param nodal_values: np.ndarray, shape (T, n) of nodal values over time.
    :param edge_values: np.ndarray, shape (T, m) of edge values over time.
    :param output_dir: str, directory where the output images and GIF will be saved.
    The rest of the parameters are the same as gis_visualize.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    # Call gis_visualize for each time step
    for t_idx, t in tqdm(enumerate(timestamps), total=len(timestamps), desc="Generating GIS visualizations"):
        outfile = os.path.join(output_dir, f"bus_voltages_{t}.{ext}")
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        gis_visualize(
            node_coords=node_coords,
            nodal_values=nodal_values[t_idx, :],
            edge_list=edge_list,
            edge_values=edge_values[t_idx, :] if edge_values is not None else None,
            outfile=outfile,
            cb_label_nodal=cb_label_nodal,
            cb_label_edge=cb_label_edge,
            cmap_nodal=cmap_nodal,
            cmap_edge=cmap_edge,
            title=f"{title}: {t}",
            background_img=background_img,
            scale=scale,
            anchor=anchor,
            xlim=xlim,
            ylim=ylim,
            clim_nodal=clim_nodal,
            clim_edge=clim_edge,
            figsize=figsize,
            markersize=markersize,
            plot_title=plot_title,
            plot_colorbar_label=plot_colorbar_label,
        )

    # Make GIF
    from PIL import Image
    images = [Image.open(os.path.join(output_dir, f)) for f in sorted(os.listdir(output_dir))]
    images[0].save(os.path.join(output_dir, f"bus_voltages_{str(timestamps[0])[:10]}.gif"), save_all=True, append_images=images[1:], duration=250, loop=0)


"""
Plot PVHC distribution results
"""
# HX: I changed this function to handle None inputs and arrays with all zeros. 

def plot_pv_capacity_distribution(capacities_isa_customer, capacities_isa_utility, feeder_name, output_dir, sub_folder=None, figsize=(7.16/3*1.6, 7.16/3*4/5*1.6), ext='png', plot_title=PLOT_TITLE, ylim_max=300, ytick_interval=50, fontsize_for_notation=8, fontsize_for_ticks=10, normalized=False, customer_pv_info=None, utility_pv_info=None):
    """
    Plot PV capacity distribution as a stacked histogram.
    
    Parameters:
    - capacities_isa_customer: np.ndarray or None, customer PV capacities
    - capacities_isa_utility: np.ndarray or None, utility PV capacities
    - feeder_name: str, name of the feeder
    - output_dir: Path, output directory
    - sub_folder: str, optional subfolder name for output
    - figsize: tuple, figure size
    - ext: str, file extension for saving
    - ylim_max: int, maximum y-axis value (default 300 for academic consistency)
    - ytick_interval: int, interval between y-axis ticks (default 50)
    - fontsize_for_notation: int, font size for legends and text annotations (default 9)
    - fontsize_for_ticks: int, font size for tick labels, x-label, and y-label (default 10)
    - normalized: bool, if True normalize capacities by capacity limits (default False)
    - customer_pv_info: list, PV info for customer installations (required if normalized=True)
    - utility_pv_info: list, PV info for utility installations (required if normalized=True)
    """
    # Normalize capacities if requested
    if normalized:
        if customer_pv_info is None and capacities_isa_customer is not None:
            raise ValueError("customer_pv_info is required when normalized=True and customer capacities exist")
        if utility_pv_info is None and capacities_isa_utility is not None:
            raise ValueError("utility_pv_info is required when normalized=True and utility capacities exist")
        
        # Normalize customer capacities (stored in same order as capacities array)
        if capacities_isa_customer is not None and len(capacities_isa_customer) > 0:
            capacities_isa_customer = capacities_isa_customer.copy()
            for i, info in enumerate(customer_pv_info):
                limit = info.get("capacity", 1.0)
                if limit > 0:
                    capacities_isa_customer[i] /= limit
        
        # Normalize utility capacities (stored in same order as capacities array)
        if capacities_isa_utility is not None and len(capacities_isa_utility) > 0:
            capacities_isa_utility = capacities_isa_utility.copy()
            for i, info in enumerate(utility_pv_info):
                limit = info.get("capacity", 1.0)
                if limit > 0:
                    capacities_isa_utility[i] /= limit
    
    # Academic-style colors (easy to customize)
    COLOR_CUSTOMER = '#ff7f0e'  # Orange for customer PV
    COLOR_UTILITY = '#1f77b4'   # Blue for utility PV
    COLOR_OTHER1 = '#2ca02c'   # Green for other PV
    COLOR_OTHER2 = '#d62728'   # Red for other PV
    
    # IEEE single column width is typically 3.5 inches, double column is 7.16 inches
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle None cases and arrays with all zeros
    has_customer = (capacities_isa_customer is not None and 
                    len(capacities_isa_customer) > 0 and 
                    np.any(capacities_isa_customer > 0))
    has_utility = (capacities_isa_utility is not None and 
                   len(capacities_isa_utility) > 0 and 
                   np.any(capacities_isa_utility > 0))
    
    # Prepare data for stacked histogram
    bins = 30
    if normalized:
        # For normalized, use ratio bins
        bin_range = (0, 1.1)
    else:
        # Size the range to the actual data rather than a fixed literal --
        # per-customer capacity ceilings are configurable (Eq. 12 x
        # capacity_multiplier in calculate_capacity_limit_and_store.py) and can
        # exceed the old flat 10 kW default; a hardcoded (0, 11) range would
        # silently drop and clip any installation above it from the histogram.
        present_arrays = [arr for arr, present in
                           [(capacities_isa_customer, has_customer), (capacities_isa_utility, has_utility)]
                           if present]
        max_cap = max((arr.max() for arr in present_arrays), default=11.0)
        bin_range = (0, max(11.0, np.ceil(max_cap / 10.0) * 10.0))
    
    if has_customer and has_utility:
        # Plot stacked histogram: customer on bottom, utility on top
        ax.hist([capacities_isa_customer, capacities_isa_utility], 
                bins=bins, range=bin_range, stacked=True,
                label=['Customer PV', 'Utility PV'],
                color=[COLOR_CUSTOMER, COLOR_UTILITY],
                edgecolor='black', linewidth=0.5, alpha=0.7)
    elif has_customer:
        # Plot only customer PV
        ax.hist(capacities_isa_customer, bins=bins, range=bin_range,
                label='Customer PV', color=COLOR_CUSTOMER,
                edgecolor='black', linewidth=0.5, alpha=0.7)
    elif has_utility:
        # Plot only utility PV
        ax.hist(capacities_isa_utility, bins=bins, range=bin_range,
                label='Utility PV', color=COLOR_UTILITY,
                edgecolor='black', linewidth=0.5, alpha=0.7)
    
    # Set labels and title
    if normalized:
        ax.set_xlabel('Normalized PV Capacity (Capacity / Limit)', fontsize=fontsize_for_ticks)
    else:
        ax.set_xlabel('PV Capacity (kW)', fontsize=fontsize_for_ticks)
    ax.set_ylabel('Number of Installations', fontsize=fontsize_for_ticks)
    if plot_title:
        ax.set_title(f'PV Capacity Distribution for {feeder_name}', fontsize=fontsize_for_ticks)
    
    # Set tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=fontsize_for_ticks)
    
    # Add legend only if there's data to show
    if has_customer or has_utility:
        ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='black', fontsize=fontsize_for_notation)
    
    ax.grid(True, linestyle='--', alpha=GRID_ALPHA)
    
    # Set the range of x-axis based on normalized or not
    if normalized:
        ax.set_xlim(0, 1.1)
        ax.set_xticks(np.arange(0, 1.2, 0.1))
    else:
        ax.set_xlim(0, bin_range[1])
        tick_step = max(1, int(np.ceil(bin_range[1] / 11)))
        ax.set_xticks(np.arange(0, bin_range[1] + tick_step, tick_step))

    # Set consistent y-axis limits and ticks for academic paper figures
    # This ensures all subfigures have the same scale for easy comparison
    ax.set_ylim(bottom=0, top=ylim_max) 
    ax.set_yticks(np.arange(0, ylim_max + ytick_interval, ytick_interval))
    
    # Calculate and display total capacities
    text_lines = []
    total_capacity = 0
    
    if has_customer:
        total_capacity_customer = capacities_isa_customer.sum()
        total_capacity += total_capacity_customer
        if normalized:
            text_lines.append(f'Customer PV: {total_capacity_customer:.2f} (norm)')
        else:
            text_lines.append(f'Customer PV: {total_capacity_customer:.1f} kW')
    
    if has_utility:
        total_capacity_utility = capacities_isa_utility.sum()
        total_capacity += total_capacity_utility
        if normalized:
            text_lines.append(f'Utility PV: {total_capacity_utility:.2f} (norm)')
        else:
            text_lines.append(f'Utility PV: {total_capacity_utility:.1f} kW')
    
    if has_customer or has_utility:
        if normalized:
            text_lines.insert(0, f'Total PV: {total_capacity:.2f} (norm)')
        else:
            text_lines.insert(0, f'Total PV: {total_capacity:.1f} kW')
        text_content = '\n'.join(text_lines)
        ax.text(
            0.95, 0.95, text_content,
            transform=ax.transAxes, fontsize=fontsize_for_notation, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.5)
        )
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Create output directory and save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"pv_distribution_{sub_folder}.{ext}" \
        if sub_folder else output_dir / f"pv_distribution.{ext}"
    
    plt.savefig(output_path, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    

def plot_v_bus_heatmap(t, voltage_profiles, voltage_upper_limits, voltage_lower_limits, output_dir, sub_folder=None, violation_type="over", figsize=(6, 4), ext='png', plot_title=PLOT_TITLE, plot_colorbar_label=PLOT_COLORBAR_LABEL):
    """
    Plot voltage violation heatmap from stored simulation results.
    
    Parameters:
    - feeder_name: Name of the feeder
    - sub_folder: Timestamp folder name containing simulation results
    - voltage_upper_limits: np.ndarray of shape (n_buses, 3) with upper voltage limits per bus/phase
    - voltage_lower_limits: np.ndarray of shape (n_buses, 3) with lower voltage limits per bus/phase
    - violation_type: "over" for overvoltage, "under" for undervoltage
    """
    # Reshape
    T, N = voltage_profiles.shape
    n_buses = N // 3
    voltage_reshaped = voltage_profiles.reshape(T, n_buses, 3)
    
    # Compute voltage violations using actual network limits
    if violation_type == "over":
        # Compare each bus/phase voltage against its specific upper limit
        voltage_upper_limits_reshaped = voltage_upper_limits.reshape(n_buses, 3)
        violations = (voltage_reshaped > voltage_upper_limits_reshaped[np.newaxis, :, :]).astype(float)
        title_str = "Over-Voltage Violation Probability"
        output_name = "heatmap_overvoltage"
    else:
        # Compare each bus/phase voltage against its specific lower limit
        voltage_lower_limits_reshaped = voltage_lower_limits.reshape(n_buses, 3)
        violations = (voltage_reshaped < voltage_lower_limits_reshaped[np.newaxis, :, :]).astype(float)
        title_str = "Under-Voltage Violation Probability"
        output_name = "heatmap_undervoltage"
    
    # Average violations across phases for each bus
    violation_per_bus = violations.mean(axis=2)  # (T, n_buses)
    
    # Build matrix of violation heatmap: shape=(n_buses, 48)
    interval_idx_arr = np.array([(d.hour * 60 + d.minute) // 30 for d in t.tolist()])
    violation_heatmap = np.zeros((n_buses, 48))  # Initialize with zeros
    for j in range(48):
        mask = interval_idx_arr == j
        violation_heatmap[:, j] = np.mean(violation_per_bus[mask, :], axis=0)
            
    # Create figure with academic dimensions
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create custom colormap (white to dark red)
    n_colors = 256
    cmap = np.zeros((n_colors, 3))
    cmap[0, :] = [1, 1, 1]  # White for 0
    for i in range(1, n_colors):
        t = (i-1) / (n_colors-1)
        if violation_type == "over":
            cmap[i, :] = [1, 1-0.9*t, 1-0.9*t]  # White to dark red
        elif violation_type == "under":
            cmap[i, :] = [1-0.9*t, 1-0.9*t, 1]  # White to dark blue
    
    # Plot heatmap
    im = ax.imshow(violation_heatmap, aspect='auto', cmap=plt.matplotlib.colors.ListedColormap(cmap),
                   vmin=0, vmax=1, origin='lower')
    
    # Customize axes
    ax.set_xticks(np.arange(0, 49, 4))
    ax.set_xticklabels(np.arange(0, 25, 2))
    
    # Y-axis: show only first and last bus
    ax.set_yticks([0, n_buses - 1])
    ax.set_yticklabels(['Bus 1', f'Bus {n_buses}'])
    
    # Labels and title
    ax.set_xlabel('Time (h)', fontsize=11)
    ax.set_ylabel('Bus Index', fontsize=11)
    if plot_title:
        ax.set_title(title_str, fontsize=12)
    
    # Add colorbar
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if plot_colorbar_label:
        cb.set_label(f'{violation_type.capitalize()}-Voltage Violation Probability', 
                     rotation=270, labelpad=20, fontsize=10)
    cb.ax.tick_params(labelsize=9)
    cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_name}_{sub_folder}.{ext}" \
        if sub_folder else output_dir / f"{output_name}.{ext}"
    
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')

    plt.show()
    return output_path


def plot_violation_24_365(t, values, upper_limits, lower_limits, outfile, figsize=(6, 3), violation_type="both", clim=(0, 1.0), title='Average Voltage Violation Rate 24/365', plot_title=PLOT_TITLE, plot_colorbar_label=PLOT_COLORBAR_LABEL):
    """
    x axis: time (12 months)
    y axis: time (48 half-hour intervals)
    colorbar: average voltage violation rate across all buses
    :param violation_type: "over", "under", or "both"
    """
    assert violation_type in ('over', 'under', 'both'), violation_type
    
    # Reshape
    T, N = values.shape
    value_reshaped = values.reshape(T, N // 3, 3)
    upper_limits = upper_limits.reshape(N // 3, 3) \
        if len(upper_limits.shape) == 1 else upper_limits
    lower_limits = lower_limits.reshape(N // 3, 3) \
        if len(lower_limits.shape) == 1 else lower_limits
    
    # Compute voltage violations using actual network limits
    over_violations = (value_reshaped > upper_limits[np.newaxis, :, :]).astype(float)
    under_violations = (value_reshaped < lower_limits[np.newaxis, :, :]).astype(float)
    
    # Overlay both over-voltage and under-voltage heatmaps. Over: red, under: blue.
    # Average violations across phases for each bus and phase
    over_violations = over_violations.mean(axis=(1,2))      # (T,)
    under_violations = under_violations.mean(axis=(1,2))    # (T,)

    # Parameters for reshaping: 12 months, 48 half-hour intervals per day
    intervals_per_day = 48
    months = 12

    # Take the mean over over_violations and under_violations for each half-hour interval in each month, the output should be a (48, 12) array
    # Build arrays of month index (1-12) and half-hour interval index (0-47) from timestamps
    t = t.tolist() if type(t) is np.ndarray else t
    months_arr = np.array([d.month for d in t])
    interval_idx_arr = np.array([(d.hour * 60 + d.minute) // 30 for d in t])

    # Initialize arrays (intervals x months)
    under_matrix = np.full((intervals_per_day, months), np.nan)
    over_matrix = np.full((intervals_per_day, months), np.nan)

    # Aggregate means for each (interval, month)
    for m in range(months):
        for i in range(intervals_per_day):
            mask = (months_arr == m + 1) & (interval_idx_arr == i)
            if np.any(mask):
                under_matrix[i, m] = under_violations[mask].mean()
                over_matrix[i, m] = over_violations[mask].mean()

    # Plot the two heatmaps overlaid: under (blue) and over (red)
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    fig, ax = plt.subplots(figsize=figsize)

    # Replace NaNs with zero and notify if any found
    if np.isnan(under_matrix).sum() > 0 or np.isnan(over_matrix).sum() > 0:
        print(f"[WARNING] NaN detected in time-heatmap matrices. Replacing NaN with 0.")
    under_matrix = np.nan_to_num(under_matrix, nan=0.0)
    over_matrix = np.nan_to_num(over_matrix, nan=0.0)

    from matplotlib.colors import LinearSegmentedColormap
    def white_anchored_cmap(base_cmap, n=256):
        colors = base_cmap(np.linspace(0, 1, n))
        colors[0] = np.array([1, 1, 1, 1])  # pure white
        return LinearSegmentedColormap.from_list(
            f"white_{base_cmap.name}", colors
        )
    blue_cmap = white_anchored_cmap(plt.cm.Blues)
    red_cmap  = white_anchored_cmap(plt.cm.Reds)
    
    # Plot under-voltage (blue) first, then over-voltage (red) on top with some alpha
    if violation_type in ('both', 'under'):
        ax.imshow(
            under_matrix,
            aspect='auto',
            origin='lower',
            cmap=blue_cmap,
            vmin=clim[0],
            vmax=clim[1],
            interpolation='nearest',
            zorder=1,
        )
    if violation_type in ('both', 'over'):
        ax.imshow(
            over_matrix,
            aspect='auto',
            origin='lower',
            cmap=red_cmap,
            vmin=clim[0],
            vmax=clim[1],
            interpolation='nearest',
            alpha=0.6,
            zorder=2,
        )

    # X-axis: months
    months_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(np.arange(0, 12))
    ax.set_xticklabels(months_labels, rotation=0, ha='center')

    # Y-axis: half-hour intervals (show every 8 ticks = every 4 hours)
    y_ticks = np.arange(0, intervals_per_day + 8, 8)
    y_labels = []
    for i in y_ticks:
        minutes = i * 30
        hh = minutes // 60
        mm = minutes % 60
        y_labels.append(f"{hh:02d}:{mm:02d}")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_ylim(0, intervals_per_day)

    # ax.set_xlabel('Month')
    ax.set_ylabel('Time of Day (hh:mm)')
    if plot_title:
        ax.set_title(title)

    # Add two small colorbars
    norm = Normalize(vmin=clim[0], vmax=clim[1])
    sm_under = ScalarMappable(norm=norm, cmap=blue_cmap)
    sm_over  = ScalarMappable(norm=norm, cmap=red_cmap)
    sm_under.set_array([])
    sm_over.set_array([])

    # Two explicit colorbar axes with identical height
    # [x0, y0, width, height] in axes coordinates
    colorbar_type = violation_type
    if violation_type == 'both' and (np.max(under_matrix) < 0.05 * clim[1]):
        colorbar_type = 'over'
    if colorbar_type == 'both':
        cax_under = ax.inset_axes([1.02, 0.05, 0.03, 0.40])
        cax_over  = ax.inset_axes([1.02, 0.55, 0.03, 0.40])
        cax_under.set_facecolor('white')
        cax_over.set_facecolor('white')
    elif colorbar_type == 'under':
        cax_under = ax.inset_axes([1.02, 0.1, 0.03, 0.75])
        cax_under.set_facecolor('white')
    elif colorbar_type == 'over':
        cax_over  = ax.inset_axes([1.02, 0.1, 0.03, 0.75])
        cax_over.set_facecolor('white')
    
    if colorbar_type in ('both', 'under'):
        cb_under = fig.colorbar(sm_under, cax=cax_under)
        cb_under.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        if plot_colorbar_label:
            cb_under.set_label(
                'Under Violation Probability' 
                if colorbar_type == 'both' else 'Violation Probability'
            )
    if colorbar_type in ('both', 'over'):
        cb_over = fig.colorbar(sm_over, cax=cax_over)
        cb_over.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        if plot_colorbar_label:
            cb_over.set_label(
                'Over Violation Probability' 
                if colorbar_type == 'both' else 'Violation Probability'
            )

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    plt.tight_layout()

    # Save output
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_net_load(total_net_p, total_load_p, total_pv_p, total_batt_p=None, t=None, outfile=None, figsize=(6, 4), plot_title=PLOT_TITLE):
    """
    Plot net load (sum of all injections) over time.
    Phases are ignored (single-phase and phase-to-phase injections are all aggregated).
    
    :param p_inj: np.ndarray of real power injections (kW), shape (T, M)
    :param q_inj: np.ndarray of reactive power injections (kVAR), shape (T, M)
    :param injection_map: list of dict, see simulator.py:simulate
    :param outfile: str, filepath to save the plot
    """
    
    has_battery = not (total_batt_p is None or len(total_batt_p) == 0 or np.all(total_batt_p == 0))
    
    # Plot the net injection, total load, total pv, and optionally battery
    fig, ax = plt.subplots(figsize=figsize)
    x_axis = t if t is not None else np.arange(len(total_net_p))
    ax.plot(x_axis, total_net_p, label='Net Injection (kW)', color='black', linewidth=2)
    ax.plot(x_axis, total_load_p, label='Load (kW)', color='red', linestyle='--')
    ax.plot(x_axis, total_pv_p, label='PV Generation (kW)', color='green', linestyle='--')
    if has_battery:
        ax.plot(x_axis, total_batt_p, label='Battery (kW)', color='blue', linestyle='-.')
    if t is not None:
        ax.xaxis.set_major_formatter(timefmt)
    ax.axhline(y=0, color='gray', alpha=0.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Power (kW)')
    if plot_title:
        ax.set_title('System Load, PV Generation' + (', & Battery' if has_battery else ''))
    ax.legend()
    plt.tight_layout()
    if outfile:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        plt.savefig(outfile)
    plt.close(fig)


def plot_batt_inj(total_batt_p, total_batt_q, t=None, outfile=None, figsize=(12, 4), plot_title=PLOT_TITLE):
    """
    Plot battery injection over time.
    
    :param total_batt_p: np.ndarray of shape (T,) of total battery injections (kW)
    :param total_batt_q: np.ndarray of shape (T,) of total battery reactive power injections (kVAR)
    :param t: np.ndarray of shape (T,) of timestamps
    :param outfile: str, filepath to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    x_axis = t if t is not None else np.arange(len(total_batt_p))
    timefmt = mdates.DateFormatter('%H') if t is not None else None
    
    # Real power subplot
    ax1.plot(x_axis, total_batt_p, label='Battery Injection (kW)', color='blue', linestyle='-.')
    if t is not None:
        ax1.xaxis.set_major_formatter(timefmt)
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Hour of day')
    ax1.set_ylabel('Real Power (kW)')
    if plot_title:
        ax1.set_title('Battery Real Power Injection Over Time')
    # ax1.legend(loc='upper left')
    
    # Reactive power subplot
    ax2.plot(x_axis, total_batt_q, label='Battery Reactive Injection (kVAR)', color='purple', linestyle='-.')
    if t is not None:
        ax2.xaxis.set_major_formatter(timefmt)
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Hour of day')
    ax2.set_ylabel('Reactive Power (kVAR)')
    if plot_title:
        ax2.set_title('Battery Reactive Power Injection Over Time')
    # ax2.legend(loc='upper left')
    
    plt.tight_layout()
    if outfile:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        plt.savefig(outfile)
    plt.close(fig)

"""
Statistics calculations (not plotting)
"""


def load_dict_calculations(p_inj, q_inj, injection_map, t=None):
    """
    Calculate net load at each bus given injections and injection map.
    Phases are ignored (single-phase and phase-to-phase injections are all aggregated).
    
    :param p_inj: np.ndarray of real power injections (kW), shape (T, M)
    :param q_inj: np.ndarray of reactive power injections (kVAR), shape (T, M)
    :param injection_map: list of dict, see simulator.py:simulate
    :return: 
        inj_dict: dict of dict
            {
                "<bus_name>": {
                    'p_load': np.ndarray of shape (T, ni), real power load at bus_idx (kW)
                    'q_load': np.ndarray of shape (T, ni), reactive power load at bus_idx (kVAR)
                    'injections': list of injections from injection_map at bus_idx, length=ni
                }
            }
        load_dict: dict of dict, same structure as inj_dict but only for load injections
        pv_dict: dict of dict, same structure as inj_dict but only for pv injections
        batt_dict: dict of dict, same structure as inj_dict but only for battery injections
        total_net_*: np.ndarray of shape (T,), total net injections across all buses
        total_battery_*: np.ndarray of shape (T,), total battery injections
    """
    inj_dict, load_dict, pv_dict, batt_dict = {}, {}, {}, {}
    
    for i, inj in enumerate(injection_map):
        if inj['bus'] not in inj_dict:
            inj_dict[inj['bus']] = {
                'p_inj': [p_inj[:, i]],
                'q_inj': [q_inj[:, i]],
                'injections': [inj]
            }
        else:
            inj_dict[inj['bus']]['injections'].append(inj)
            inj_dict[inj['bus']]['p_inj'].append(p_inj[:, i])
            inj_dict[inj['bus']]['q_inj'].append(q_inj[:, i])
        # All non-pv injections are considered load
        if 'type' not in inj:
            raise ValueError(f"Unspecified injection type for: {inj}")
        elif inj['type'] == 'load':
            if inj['bus'] not in load_dict:
                load_dict[inj['bus']] = {
                    'p_load': [p_inj[:, i]],
                    'q_load': [q_inj[:, i]],
                    'injections': [inj]
                }
            else:
                load_dict[inj['bus']]['injections'].append(inj)
                load_dict[inj['bus']]['p_load'].append(p_inj[:, i])
                load_dict[inj['bus']]['q_load'].append(q_inj[:, i])
        elif inj['type'][-2:] == 'pv':
            if inj['bus'] not in pv_dict:
                pv_dict[inj['bus']] = {
                    'p_pv': [p_inj[:, i]],
                    'q_pv': [q_inj[:, i]],
                    'injections': [inj]
                }
            else:
                pv_dict[inj['bus']]['injections'].append(inj)
                pv_dict[inj['bus']]['p_pv'].append(p_inj[:, i])
                pv_dict[inj['bus']]['q_pv'].append(q_inj[:, i])
        elif inj['type'] == 'battery':
            if inj['bus'] not in batt_dict:
                batt_dict[inj['bus']] = {
                    'p_battery': [p_inj[:, i]],
                    'q_battery': [q_inj[:, i]],
                    'injections': [inj]
                }
            else:
                batt_dict[inj['bus']]['injections'].append(inj)
                batt_dict[inj['bus']]['p_battery'].append(p_inj[:, i])
                batt_dict[inj['bus']]['q_battery'].append(q_inj[:, i])
        else:
            raise ValueError(f"Unknown injection type: {inj['type']}")
    
    for payload in inj_dict.values():
        payload['p_inj'] = np.stack(payload['p_inj'], axis=1)  # shape (T, ni)
        payload['q_inj'] = np.stack(payload['q_inj'], axis=1)  # shape (T, ni)
    for payload in load_dict.values():
        payload['p_load'] = np.stack(payload['p_load'], axis=1)  # shape (T, ni)
        payload['q_load'] = np.stack(payload['q_load'], axis=1)  # shape (T, ni)
    for payload in pv_dict.values():
        payload['p_pv'] = np.stack(payload['p_pv'], axis=1)  # shape (T, ni)
        payload['q_pv'] = np.stack(payload['q_pv'], axis=1)  # shape (T, ni)
    for payload in batt_dict.values():
        payload['p_battery'] = np.stack(payload['p_battery'], axis=1)  # shape (T, ni)
        payload['q_battery'] = np.stack(payload['q_battery'], axis=1)  # shape (T, ni)
    
    total_net_p = np.stack([i['p_inj'].sum(axis=1) for i in inj_dict.values()], axis=1).sum(axis=1)
    total_net_q = np.stack([i['q_inj'].sum(axis=1) for i in inj_dict.values()], axis=1).sum(axis=1)
    total_load_p = np.stack([i['p_load'].sum(axis=1) for i in load_dict.values()], axis=1).sum(axis=1)
    total_load_q = np.stack([i['q_load'].sum(axis=1) for i in load_dict.values()], axis=1).sum(axis=1)
    if len(pv_dict) > 0:
        total_pv_p = np.stack([i['p_pv'].sum(axis=1) for i in pv_dict.values()], axis=1).sum(axis=1)
        total_pv_q = np.stack([i['q_pv'].sum(axis=1) for i in pv_dict.values()], axis=1).sum(axis=1)
    else:
        total_pv_p = np.zeros_like(total_net_p)
        total_pv_q = np.zeros_like(total_net_q)
    if len(batt_dict) > 0:
        total_battery_p = np.stack([i['p_battery'].sum(axis=1) for i in batt_dict.values()], axis=1).sum(axis=1)
        total_battery_q = np.stack([i['q_battery'].sum(axis=1) for i in batt_dict.values()], axis=1).sum(axis=1)
    else:
        total_battery_p = np.zeros_like(total_net_p)
        total_battery_q = np.zeros_like(total_net_q)

    if t is not None:
        t_delta = np.median((t[1:] - t[:-1])) / np.timedelta64(3600, 's') # in hours
        print(f"Total net real power injection (kWh): {total_net_p.sum() * t_delta:.2f}")
        print(f"Total net reactive power injection (kVARh): {total_net_q.sum() * t_delta:.2f}")
        print(f"Total load real power (kWh): {total_load_p.sum() * t_delta:.2f}")
        print(f"Total load reactive power (kVARh): {total_load_q.sum() * t_delta:.2f}")
        print(f"Total PV real power generation (kWh): {total_pv_p.sum() * t_delta:.2f}")
        print(f"Total PV reactive power generation (kVARh): {total_pv_q.sum() * t_delta:.2f}")
        if len(batt_dict) > 0:
            print(f"Total battery real power (kWh): {total_battery_p.sum() * t_delta:.2f}")
            print(f"Total battery reactive power (kVARh): {total_battery_q.sum() * t_delta:.2f}")
    
    return (inj_dict, load_dict, pv_dict, batt_dict, 
            total_net_p, total_net_q, 
            total_load_p, total_load_q, 
            total_pv_p, total_pv_q,
            total_battery_p, total_battery_q)
