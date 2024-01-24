import io
import numpy as np
import pandas as pd

import cartopy as ct
import cartopy.crs as ccrs
import cartopy.geodesic as cgeo
import cartopy.io.img_tiles as cimgt

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from PIL import Image
from pathlib import Path
from planar import BoundingBox
from collections.abc import Iterable
from shapely.geometry import LineString
from cartopy.mpl.geoaxes import GeoAxes
from urllib.request import urlopen, Request
from eleventh_hour.navigators import (
    ChannelErrors,
    Correlators,
    StateResults,
    ErrorResults,
    CovarianceResults,
)


def geoplot(lat, lon, tiles="satellite", output_dir: Path = None, **kwargs):
    match tiles:
        case "map":
            cimgt.OSM.get_image = __image_spoof
            img = cimgt.OSM()  # spoofed, downloaded street map

        case "satellite":
            cimgt.QuadtreeTiles.get_image = __image_spoof
            img = cimgt.QuadtreeTiles()  # spoofed, downloaded street map

        case _:
            print("invalid style")

    if isinstance(plt.gca(), GeoAxes):
        ax = plt.gca()
    else:
        plt.close()
        fig = plt.gcf()
        ax = fig.add_subplot(projection=img.crs)

    data_crs = ccrs.PlateCarree()
    extent, radius = __compute_multiple_coordinate_extent(lons=lon, lats=lat)

    # auto-calculate scale
    scale = int(120 / np.log(radius))
    scale = (scale < 20) and scale or 19

    ax.set_extent(extent)  # set extents
    ax.add_image(img, int(scale))  # add OSM with zoom specification

    # add site
    ax.scatter(lon, lat, transform=data_crs, **kwargs)

    gl = ax.gridlines(
        draw_labels=True, crs=data_crs, color="k", lw=0.5, auto_update=True
    )

    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = ct.mpl.gridliner.LONGITUDE_FORMATTER
    gl.yformatter = ct.mpl.gridliner.LATITUDE_FORMATTER

    if output_dir is not None:
        plt.tight_layout()
        plt.savefig(output_dir / "geoplot")

    return ax


def __compute_multiple_coordinate_extent(lons, lats):
    pairs = [(lon, lat) for lon, lat in zip(lons, lats)]
    bounding_box = BoundingBox(pairs)

    buffer = 0.15 * bounding_box.height  # add 15% buffer

    min_y = bounding_box.min_point.y - buffer
    max_y = bounding_box.max_point.y + buffer

    height = max_y - min_y
    geodetic_radius = height / 2
    width = height

    points = np.array(
        [
            [bounding_box.center.x, bounding_box.center.y],
            [bounding_box.center.x, bounding_box.center.y + geodetic_radius],
        ],
    )
    radius_geometry = LineString(points)
    radius = cgeo.Geodesic().geometry_length(geometry=radius_geometry)

    min_x = bounding_box.center.x - width
    max_x = bounding_box.center.x + width

    extent = np.round(
        [
            min_x,
            max_x,
            min_y,
            max_y,
        ],
        decimals=8,
    )

    return extent, radius


def __image_spoof(self, tile):
    """this function reformats web requests from OSM for cartopy
    Heavily based on code by Joshua Hrisko at:
        https://makersportal.com/blog/2020/4/24/geographic-visualizations-in-python-with-cartopy
    """

    url = self._image_url(tile)  # get the url of the street map API
    req = Request(url)  # start request
    req.add_header("User-agent", "Anaconda 3")  # add user agent to request
    fh = urlopen(req)
    im_data = io.BytesIO(fh.read())  # get image
    fh.close()  # close url
    img = Image.open(im_data)  # open image with PIL
    img = img.convert(self.desired_tile_form)  # set image format

    return img, self.tileextent(tile), "lower"  # reformat for cartopy


def skyplot(
    az: np.ndarray, el: np.ndarray, name: str | list = None, deg: bool = True, **kwargs
):
    if isinstance(plt.gca(), plt.PolarAxes):
        ax = plt.gca()
    else:
        plt.close()
        fig = plt.gcf()
        ax = fig.add_subplot(projection="polar")

    if deg:
        az = np.radians(az)
    else:
        el = np.degrees(el)

    # format polar axes
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(91, 1)

    degree_sign = "\N{DEGREE SIGN}"
    r_labels = [
        "0" + degree_sign,
        "",
        "30" + degree_sign,
        "",
        "60" + degree_sign,
        "",
        "90" + degree_sign,
    ]
    ax.set_rgrids(range(1, 106, 15), r_labels, angle=22.5)

    ax.set_axisbelow(True)

    # plot
    ax.scatter(az, el, **kwargs)

    # annotate object names
    if name is not None:
        if not isinstance(name, Iterable):
            name = (name,)

        for obj, n in enumerate(name):
            ax.annotate(
                n,
                (az[obj, 0], el[obj, 0]),
                fontsize="x-small",
                path_effects=[pe.withStroke(linewidth=3, foreground="w")],
            )

    ax.figure.canvas.draw()

    return ax


def plot_states(states: StateResults, output_dir: Path = None):
    # trajectory
    plt.figure()
    plt.plot(states.truth_enu_pos[0], states.truth_enu_pos[1], label="truth")
    plt.plot(states.enu_pos[0], states.enu_pos[1], label="vdfll")
    plt.xlabel("East [m]")
    plt.ylabel("North [m]")
    plt.legend()
    ax = plt.gca()
    ax.axis("equal")
    plt.tight_layout()

    if output_dir is not None:
        plt.savefig(output_dir / "trajectory")

    # position
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    titles = ["East", "North", "Up"]
    for index, (ax, title) in enumerate(zip(axes, titles)):
        ax.plot(states.time, states.truth_enu_pos[index], label="truth")
        ax.plot(states.time, states.enu_pos[index], label="vdfll")
        ax.set_title(title)

    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels)
    fig.supxlabel("Time [s]")
    fig.supylabel("Position [m]")
    plt.tight_layout()

    if output_dir is not None:
        plt.savefig(output_dir / "pos")

    # velocity
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    titles = ["East", "North", "Up"]
    for index, (ax, title) in enumerate(zip(axes, titles)):
        ax.plot(states.time, states.truth_enu_vel[index], label="truth")
        ax.plot(states.time, states.enu_vel[index], label="vdfll")
        ax.set_title(title)

    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels)
    fig.supxlabel("Time [s]")
    fig.supylabel("Velocity [m/s]")
    plt.tight_layout()

    if output_dir is not None:
        plt.savefig(output_dir / "vel")

    # clock bias
    plt.figure()
    plt.plot(states.time, states.truth_clock_bias, label="truth")
    plt.plot(states.time, states.clock_bias, label="vdfll")
    plt.xlabel("Time [s]")
    plt.ylabel("Clock Bias [m]")
    plt.legend()
    plt.tight_layout()

    if output_dir is not None:
        plt.savefig(output_dir / "cb")

    # clock drift
    plt.figure()
    plt.plot(states.time, states.truth_clock_drift, label="truth")
    plt.plot(states.time, states.clock_drift, label="vdfll")
    plt.xlabel("Time [s]")
    plt.ylabel("Clock Drift [m/s]")
    plt.legend()
    plt.tight_layout()

    if output_dir is not None:
        plt.savefig(output_dir / "cd")


def plot_covariances(cov: CovarianceResults, output_dir: Path = None):
    # position
    fig, axes = plt.subplots(nrows=3, ncols=1, sharey=True, sharex=True)
    titles = ["East", "North", "Up"]
    for index, (ax, title) in enumerate(zip(axes, titles)):
        ax.plot(cov.time, 3 * np.sqrt(cov.pos[index]), label="vdfll")
        ax.set_title(title)

    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels)
    fig.supxlabel("Time [s]")
    fig.supylabel("Position 3-$\\sigma$ [m]")
    plt.tight_layout()

    if output_dir is not None:
        plt.savefig(output_dir / "pos_cov")

    # velocity
    fig, axes = plt.subplots(nrows=3, ncols=1, sharey=True, sharex=True)
    titles = ["East", "North", "Up"]
    for index, (ax, title) in enumerate(zip(axes, titles)):
        ax.plot(cov.time, 3 * np.sqrt(cov.vel[index]), label="vdfll")
        ax.set_title(title)

    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels)
    fig.supxlabel("Time [s]")
    fig.supylabel("Velocity 3-$\\sigma$ [m/s]")
    plt.tight_layout()

    if output_dir is not None:
        plt.savefig(output_dir / "vel_cov")

    # clock bias
    plt.figure()
    plt.plot(cov.time, 3 * np.sqrt(cov.clock_bias), label="vdfll")
    plt.xlabel("Time [s]")
    plt.ylabel("Clock Bias 3-$\\sigma$ [m]")
    plt.legend()
    plt.tight_layout()

    if output_dir is not None:
        plt.savefig(output_dir / "cb_cov")

    # clock drift
    plt.figure()
    plt.plot(cov.time, 3 * np.sqrt(cov.clock_drift), label="vdfll")
    plt.xlabel("Time [s]")
    plt.ylabel("Clock Drift 3-$\\sigma$ [m/s]")
    plt.legend()
    plt.tight_layout()

    if output_dir is not None:
        plt.savefig(output_dir / "cd_cov")


def plot_errors(errors: ErrorResults, output_dir: Path = None):
    POS_ERROR_BOUNDS = 150
    VEL_ERROR_BOUNDS = 20

    # position error
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
    titles = ["East", "North", "Up"]
    for index, (ax, title) in enumerate(zip(axes, titles)):
        ax.plot(errors.time, errors.pos[index], label="vdfll")
        if np.max(np.abs(errors.pos)) > POS_ERROR_BOUNDS:
            ax.set_ylim(bottom=-POS_ERROR_BOUNDS, top=POS_ERROR_BOUNDS)
        ax.set_title(title)

    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels)
    fig.supxlabel("Time [s]")
    fig.supylabel("Position Error [m]")
    plt.tight_layout()

    if output_dir is not None:
        plt.savefig(output_dir / "pos_error")

    # velocity error
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
    titles = ["East", "North", "Up"]
    for index, (ax, title) in enumerate(zip(axes, titles)):
        ax.plot(errors.time, errors.vel[index], label="vdfll")
        if np.max(np.abs(errors.vel)) > VEL_ERROR_BOUNDS:
            ax.set_ylim(bottom=-VEL_ERROR_BOUNDS, top=VEL_ERROR_BOUNDS)
        ax.set_title(title)

    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels)
    fig.supxlabel("Time [s]")
    fig.supylabel("Velocity Error [m/s]")
    plt.tight_layout()

    if output_dir is not None:
        plt.savefig(output_dir / "vel_error")

    # clock bias error
    plt.figure()
    plt.plot(errors.time, errors.clock_bias, label="vdfll")
    plt.xlabel("Time [s]")
    plt.ylabel("Clock Bias Error [m]")
    plt.legend()
    plt.tight_layout()

    if output_dir is not None:
        plt.savefig(output_dir / "cb_error")

    # clock drift error
    plt.figure()
    plt.plot(errors.time, errors.clock_drift, label="vdfll")
    plt.xlabel("Time [s]")
    plt.ylabel("Clock Drift Error [m/s]")
    plt.legend()
    plt.tight_layout()

    if output_dir is not None:
        plt.savefig(output_dir / "cd_error")


def plot_channel_errors(
    time: np.ndarray, channel_errors: ChannelErrors, output_dir: Path = None
):
    LEGEND_NEMITTERS_THRESH = 15

    chip_df = pd.DataFrame(
        {key: pd.Series(value) for key, value in channel_errors.chip.items()}
    )
    frequency_df = pd.DataFrame(
        {key: pd.Series(value) for key, value in channel_errors.freq.items()}
    )
    prange_df = pd.DataFrame(
        {key: pd.Series(value) for key, value in channel_errors.prange.items()}
    )
    prange_rate_df = pd.DataFrame(
        {key: pd.Series(value) for key, value in channel_errors.prange_rate.items()}
    )

    nemitters = chip_df.columns.droplevel().size
    is_legend_plotted = nemitters <= LEGEND_NEMITTERS_THRESH

    # chip discriminator
    plt.figure()
    plt.plot(time, chip_df.to_numpy(), label=chip_df.columns.droplevel())
    plt.xlabel("Time [s]")
    plt.ylabel("Chip Error [chips]")
    if is_legend_plotted:
        plt.legend()
    plt.tight_layout()

    if output_dir is not None:
        plt.savefig(output_dir / "chip_disc")

    # frequency discriminator
    plt.figure()
    plt.plot(
        time,
        frequency_df.to_numpy(),
        label=frequency_df.columns.droplevel(),
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency Error [Hz]")
    if is_legend_plotted:
        plt.legend()
    plt.tight_layout()

    if output_dir is not None:
        plt.savefig(output_dir / "freq_disc")

    # pseudorange discriminator
    plt.figure()
    plt.plot(time, prange_df.to_numpy(), label=prange_df.columns.droplevel())
    plt.xlabel("Time [s]")
    plt.ylabel("Pseudorange Error [m]")
    if is_legend_plotted:
        plt.legend()
    plt.tight_layout()

    if output_dir is not None:
        plt.savefig(output_dir / "prange_disc")

    # pseudorange rate discriminator
    plt.figure()
    plt.plot(
        time,
        prange_rate_df.to_numpy(),
        label=prange_rate_df.columns.droplevel(),
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Pseudorange Rate Error [m/s]")
    if is_legend_plotted:
        plt.legend()
    plt.tight_layout()

    if output_dir is not None:
        plt.savefig(output_dir / "prange_rate_disc")


def plot_correlators(
    time: np.ndarray, correlators: Correlators, output_dir: Path = None
):
    LEGEND_NEMITTERS_THRESH = 15

    ip_df = pd.DataFrame(
        {key: pd.Series(value) for key, value in correlators.ip.items()}
    )
    qp_df = pd.DataFrame(
        {key: pd.Series(value) for key, value in correlators.qp.items()}
    )

    nemitters = ip_df.columns.droplevel().size
    is_legend_plotted = nemitters <= LEGEND_NEMITTERS_THRESH

    # post-integration signal power
    signal_power = 10 * np.log10(
        np.sqrt(ip_df.to_numpy() ** 2 + qp_df.to_numpy() ** 2) ** 2
    )

    plt.figure()
    plt.plot(time, signal_power, label=ip_df.columns.droplevel())
    plt.xlabel("Time [s]")
    plt.ylabel("Post-Integration Signal Power [dB]")
    if is_legend_plotted:
        plt.legend()
    plt.tight_layout()

    if output_dir is not None:
        plt.savefig(output_dir / "pdi_power")


def plot_cn0s(time: np.ndarray, cn0s: dict, output_dir: Path = None):
    LEGEND_NEMITTERS_THRESH = 15

    cn0s_df = pd.DataFrame({key: pd.Series(value) for key, value in cn0s.items()})

    nemitters = cn0s_df.columns.droplevel().size
    is_legend_plotted = nemitters <= LEGEND_NEMITTERS_THRESH

    plt.figure()
    plt.plot(time, cn0s_df.to_numpy(), label=cn0s_df.columns.droplevel())
    plt.xlabel("Time [s]")
    plt.ylabel("Estimated $C/N_{0}$ [dB-Hz]")
    if is_legend_plotted:
        plt.legend()
    plt.tight_layout()

    if output_dir is not None:
        plt.savefig(output_dir / "cn0s")
