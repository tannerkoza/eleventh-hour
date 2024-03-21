import io
import numpy as np
import pandas as pd

import cartopy as ct
import cartopy.crs as ccrs
import cartopy.geodesic as cgeo
import cartopy.io.img_tiles as cimgt

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.animation as animation
from matplotlib.ticker import FormatStrFormatter

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

TYPES = [".svg", ".png"]


def save_figs(filename=Path, types=[".png"]):
    fig = plt.gcf()

    for t in types:
        fig.savefig(filename.with_suffix(t))


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
    ax.scatter(
        lon[0], lat[0], transform=data_crs, label="initial position", s=100, marker="*"
    )

    gl = ax.gridlines(
        draw_labels=True, crs=data_crs, color="#C5C9C7", lw=0.1, auto_update=True
    )

    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = ct.mpl.gridliner.LONGITUDE_FORMATTER
    gl.yformatter = ct.mpl.gridliner.LATITUDE_FORMATTER

    plt.legend()

    if output_dir is not None:
        plt.tight_layout()
        save_figs(output_dir / "geoplot", types=TYPES)

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


def plot_states(states: StateResults, output_dir: Path = None, **kwargs):
    # trajectory
    plt.figure()
    plt.plot(states.truth_enu_pos[0], states.truth_enu_pos[1], label="truth")
    plt.plot(states.enu_pos[0], states.enu_pos[1], **kwargs)
    plt.xlabel("East [m]")
    plt.ylabel("North [m]")
    plt.legend()
    ax = plt.gca()
    ax.axis("equal")
    plt.tight_layout()

    if output_dir is not None:
        save_figs(output_dir / "trajectory", types=TYPES)

    # position
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    titles = ["East", "North", "Up"]
    for index, (ax, title) in enumerate(zip(axes, titles)):
        ax.plot(states.time, states.truth_enu_pos[index], label="truth")
        ax.plot(states.time, states.enu_pos[index], **kwargs)
        ax.set_title(title)

    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels)
    fig.supxlabel("Time [s]")
    fig.supylabel("Position [m]")
    plt.tight_layout()

    if output_dir is not None:
        save_figs(output_dir / "pos", types=TYPES)

    # velocity
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    titles = ["East", "North", "Up"]
    for index, (ax, title) in enumerate(zip(axes, titles)):
        ax.plot(states.time, states.truth_enu_vel[index], label="truth")
        ax.plot(states.time, states.enu_vel[index], **kwargs)
        ax.set_title(title)

    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels)
    fig.supxlabel("Time [s]")
    fig.supylabel("Velocity [m/s]")
    plt.tight_layout()

    if output_dir is not None:
        save_figs(output_dir / "vel", types=TYPES)

    # clock bias
    plt.figure()
    plt.plot(states.time, states.truth_clock_bias, label="truth")
    plt.plot(states.time, states.clock_bias, **kwargs)
    plt.xlabel("Time [s]")
    plt.ylabel("Clock Bias [m]")
    plt.legend()
    plt.tight_layout()

    if output_dir is not None:
        save_figs(output_dir / "cb", types=TYPES)

    # clock drift
    plt.figure()
    plt.plot(states.time, states.truth_clock_drift, label="truth")
    plt.plot(states.time, states.clock_drift, **kwargs)
    plt.xlabel("Time [s]")
    plt.ylabel("Clock Drift [m/s]")
    plt.legend()
    plt.tight_layout()

    if output_dir is not None:
        save_figs(output_dir / "cd", types=TYPES)


def plot_covariances(cov: CovarianceResults, output_dir: Path = None, **kwargs):
    # position
    fig, axes = plt.subplots(nrows=3, ncols=1, sharey=True, sharex=True)
    titles = ["East", "North", "Up"]
    for index, (ax, title) in enumerate(zip(axes, titles)):
        ax.plot(cov.time, 3 * np.sqrt(cov.pos[index]), **kwargs)
        ax.set_title(title)

    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels)
    fig.supxlabel("Time [s]")
    fig.supylabel("Position 3-$\\sigma$ [m]")
    plt.tight_layout()

    if output_dir is not None:
        save_figs(output_dir / "pos_cov", types=TYPES)

    # velocity
    fig, axes = plt.subplots(nrows=3, ncols=1, sharey=True, sharex=True)
    titles = ["East", "North", "Up"]
    for index, (ax, title) in enumerate(zip(axes, titles)):
        ax.plot(cov.time, 3 * np.sqrt(cov.vel[index]), **kwargs)
        ax.set_title(title)

    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels)
    fig.supxlabel("Time [s]")
    fig.supylabel("Velocity 3-$\\sigma$ [m/s]")
    plt.tight_layout()

    if output_dir is not None:
        save_figs(output_dir / "vel_cov", types=TYPES)

    # clock bias
    plt.figure()
    plt.plot(cov.time, 3 * np.sqrt(cov.clock_bias), **kwargs)
    plt.xlabel("Time [s]")
    plt.ylabel("Clock Bias 3-$\\sigma$ [m]")
    plt.legend()
    plt.tight_layout()

    if output_dir is not None:
        save_figs(output_dir / "cb_cov", types=TYPES)

    # clock drift
    plt.figure()
    plt.plot(cov.time, 3 * np.sqrt(cov.clock_drift), **kwargs)
    plt.xlabel("Time [s]")
    plt.ylabel("Clock Drift 3-$\\sigma$ [m/s]")
    plt.legend()
    plt.tight_layout()

    if output_dir is not None:
        save_figs(output_dir / "cd_cov", types=TYPES)


def plot_errors(errors: ErrorResults, output_dir: Path = None, **kwargs):
    POS_ERROR_BOUNDS = 150
    VEL_ERROR_BOUNDS = 20

    # position error
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
    titles = ["East", "North", "Up"]
    for index, (ax, title) in enumerate(zip(axes, titles)):
        ax.plot(errors.time, errors.pos[index], **kwargs)
        if np.max(np.abs(errors.pos)) > POS_ERROR_BOUNDS:
            ax.set_ylim(bottom=-POS_ERROR_BOUNDS, top=POS_ERROR_BOUNDS)
        ax.set_title(title)

    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels)
    fig.supxlabel("Time [s]")
    fig.supylabel("Position Error [m]")
    plt.tight_layout()

    if output_dir is not None:
        save_figs(output_dir / "pos_error", types=TYPES)

    # velocity error
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
    titles = ["East", "North", "Up"]
    for index, (ax, title) in enumerate(zip(axes, titles)):
        ax.plot(errors.time, errors.vel[index], **kwargs)
        if np.max(np.abs(errors.vel)) > VEL_ERROR_BOUNDS:
            ax.set_ylim(bottom=-VEL_ERROR_BOUNDS, top=VEL_ERROR_BOUNDS)
        ax.set_title(title)

    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels)
    fig.supxlabel("Time [s]")
    fig.supylabel("Velocity Error [m/s]")
    plt.tight_layout()

    if output_dir is not None:
        save_figs(output_dir / "vel_error", types=TYPES)

    # clock bias error
    plt.figure()
    plt.plot(errors.time, errors.clock_bias, **kwargs)
    plt.xlabel("Time [s]")
    plt.ylabel("Clock Bias Error [m]")
    plt.legend()
    plt.tight_layout()

    if output_dir is not None:
        save_figs(output_dir / "cb_error", types=TYPES)

    # clock drift error
    plt.figure()
    plt.plot(errors.time, errors.clock_drift, **kwargs)
    plt.xlabel("Time [s]")
    plt.ylabel("Clock Drift Error [m/s]")
    plt.legend()
    plt.tight_layout()

    if output_dir is not None:
        save_figs(output_dir / "cd_error", types=TYPES)


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
        save_figs(output_dir / "chip_disc", types=TYPES)

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
        save_figs(output_dir / "freq_disc", types=TYPES)

    # pseudorange discriminator
    plt.figure()
    plt.plot(time, prange_df.to_numpy(), label=prange_df.columns.droplevel())
    plt.xlabel("Time [s]")
    plt.ylabel("Pseudorange Error [m]")
    if is_legend_plotted:
        plt.legend()
    plt.tight_layout()

    if output_dir is not None:
        save_figs(output_dir / "prange_disc", types=TYPES)

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
        save_figs(output_dir / "prange_rate_disc", types=TYPES)


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
        save_figs(output_dir / "pdi_power", types=TYPES)


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
        save_figs(output_dir / "cn0s", types=TYPES)


def pf_animation(
    time: np.ndarray,
    truth: np.ndarray,
    rx: np.ndarray,
    particles: np.ndarray,
    weights: np.ndarray,
    output_dir: Path,
):
    fig, ax = plt.subplots()
    nframes = truth[0].size
    interval = np.mean(np.diff(time)) * 1000
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    t = ax.plot(truth[0], truth[1], "*", c="lime", label="truth")[0]
    r = ax.plot(rx[0], rx[1], "*", c="fuchsia", label="rx")[0]

    p = ax.scatter(
        particles[0, 0], particles[1, 0], c="darkblue", s=10, label="particles"
    )

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")

    x_half_range = np.abs(truth[0].max() - truth[0].min())
    y_half_range = np.abs(truth[1].max() - truth[1].min())

    BUFFER = 1.1
    ax.set_xlim(
        truth[0, 0] - x_half_range * BUFFER, truth[0, 0] + x_half_range * BUFFER
    )
    ax.set_ylim(
        truth[1, 0] - y_half_range * BUFFER, truth[1, 0] + y_half_range * BUFFER
    )

    x_half_range = np.abs(particles[0, 0].max() - particles[0, 0].min()) / 2
    y_half_range = np.abs(particles[1, 0].max() - particles[1, 0].min()) / 2
    half_range = max(x_half_range, y_half_range)

    x1 = rx[0, 0] - half_range * BUFFER
    x2 = rx[0, 0] + half_range * BUFFER
    y1 = rx[1, 0] - half_range * BUFFER
    y2 = rx[1, 0] + half_range * BUFFER

    axins = ax.inset_axes(
        [0.05, 0.6, 0.35, 0.35],
        xlim=(x1, x2),
        ylim=(y1, y2),
        xticklabels=[],
        yticklabels=[],
    )

    pz = axins.scatter(
        particles[0, 0], particles[1, 0], c="darkblue", s=5, label="particles"
    )
    rz = axins.plot(rx[0], rx[1], "*", c="fuchsia", label="rx")[0]
    tz = axins.plot(truth[0], truth[1], "*", c="lime", label="truth")[0]

    ax.indicate_inset_zoom(axins, edgecolor="black")
    ax.axis("equal")
    axins.axis("equal")
    ax.grid(lw=0.5)
    ax.legend(loc="upper right")

    def update(frame):
        for patch in ax.patches:
            patch.remove()

        reast = rx[0, frame]
        rnorth = rx[1, frame]
        peast = particles[0, frame]
        pnorth = particles[1, frame]
        alphas = calculate_alphas(weights=weights[frame])

        pdata = np.stack([peast, pnorth]).T
        p.set_offsets(pdata)
        p.set_alpha(alphas)
        r.set_data([rx[0, :frame]], [rx[1, :frame]])
        t.set_data([truth[0, :frame]], [truth[1, :frame]])

        x_half_range = np.abs(peast.max() - peast.min()) / 2
        y_half_range = np.abs(pnorth.max() - pnorth.min()) / 2
        half_range = max(x_half_range, y_half_range)

        BUFFER = 5

        x1 = reast - half_range * BUFFER
        x2 = reast + half_range * BUFFER
        y1 = rnorth - half_range * BUFFER
        y2 = rnorth + half_range * BUFFER

        pz.set_offsets(pdata)
        pz.set_alpha(alphas)
        rz.set_data([rx[0, :frame]], [rx[1, :frame]])
        tz.set_data([truth[0, :frame]], [truth[1, :frame]])

        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        ax.grid(lw=0.5)

        ax.indicate_inset_zoom(axins, edgecolor="black")

        return p

    def calculate_alphas(weights: np.ndarray):
        if np.all(weights == weights[0]):
            alphas = np.ones_like(weights)
        else:
            alphas = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))

        return alphas

    ani = animation.FuncAnimation(
        fig=fig, func=update, frames=nframes, interval=interval
    )
    plt.show()
    # ani.save(output_dir / "pf.gif")
