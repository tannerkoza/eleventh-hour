import io
import itertools
import numpy as np
import cartopy as ct
import cartopy.geodesic as cgeo
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd

from PIL import Image
from planar import BoundingBox
from shapely.geometry import LineString
from urllib.request import urlopen, Request
from cartopy.mpl.geoaxes import GeoAxes
from collections.abc import Iterable


MARKER = itertools.cycle(("o", "v", "^", "<", ">", "s", "8", "p"))


def geoplot(lat, lon, tiles="satellite", **kwargs):
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


def plot_emitter_dataframe(x_data: np.ndarray, df: pd.DataFrame, **kwargs):
    for constellation, emitter in df:
        y_data = df[constellation][emitter]
        plt.plot(x_data, y_data, marker=next(MARKER), **kwargs)


# def plot(
#     truth_rx_states: ns.ReceiverTruthStates,
#     truth_emitter_states: list,
#     vdfll: VDFLL,
#     dop: np.ndarray,
#     output_dir: Path,
# ):
#     sns.set_context("paper")

#     # unpack data
#     azimuth = defaultdict(lambda: defaultdict(lambda: []))
#     elevation = defaultdict(lambda: defaultdict(lambda: []))

#     for epoch in truth_emitter_states:
#         for emitter in epoch.values():
#             azimuth[emitter.constellation][emitter.id].append(emitter.az)
#             elevation[emitter.constellation][emitter.id].append(emitter.el)


#     vdfll_pos = np.array(
#         pm.ecef2enu(
#             x=vdfll.rx_states.x_pos,
#             y=vdfll.rx_states.y_pos,
#             z=vdfll.rx_states.z_pos,
#             lat0=lla0[0],
#             lon0=lla0[1],
#             h0=lla0[2],
#         )
#     )
#     vdfll_vel = np.array(
#         pm.ecef2enuv(
#             u=vdfll.rx_states.x_vel,
#             v=vdfll.rx_states.y_vel,
#             w=vdfll.rx_states.z_vel,
#             lat0=lla0[0],
#             lon0=lla0[1],
#         )
#     )
#     vdfll_cb = vdfll.rx_states.clock_bias
#     vdfll_cd = vdfll.rx_states.clock_drfit

#     pos_error = truth_pos.T - vdfll_pos.T
#     vel_error = truth_vel.T - vdfll_vel.T
#     cb_error = truth_rx_states.clock_bias - vdfll_cb
#     cd_error = truth_rx_states.clock_drift - vdfll_cd

#     ip = vdfll.pad_log(log=[epoch.ip for epoch in vdfll.correlators])
#     qp = vdfll.pad_log(log=[epoch.qp for epoch in vdfll.correlators])

#     gdop = []
#     pdop = []
#     hdop = []
#     vdop = []
#     for epoch in dop:
#         diag = np.diag(epoch)
#         gdop.append(np.linalg.norm(diag))
#         pdop.append(np.linalg.norm(diag[:3]))
#         hdop.append(np.linalg.norm(diag[:2]))
#         vdop.append(diag[2])

#     gdop = np.array(gdop)
#     pdop = np.array(pdop)
#     hdop = np.array(hdop)
#     vdop = np.array(vdop)

#     # plot
#     geoplot(lat=truth_lla[0], lon=truth_lla[1])
#     plt.tight_layout()
#     plt.savefig(fname=output_dir / "geoplot")

#     for constellation in azimuth.keys():
#         az = np.array(vdfll.pad_log(list(azimuth[constellation].values())))
#         el = np.array(vdfll.pad_log(list(elevation[constellation].values())))
#         names = list(azimuth[constellation].keys())

#         skyplot(
#             az=az,
#             el=el,
#             name=names,
#             label=constellation,
#             deg=False,
#         )
#     plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
#     plt.tight_layout()
#     plt.savefig(fname=output_dir / "skyplot")

#     plt.figure()
#     plt.title("DOP")
#     plt.plot(truth_rx_states.time, gdop, label="gdop")
#     plt.plot(truth_rx_states.time, pdop, label="pdop")
#     plt.plot(truth_rx_states.time, hdop, label="hdop")
#     plt.plot(truth_rx_states.time, vdop, label="vdop")
#     plt.xlabel("Time [s]")
#     plt.ylabel("DOP")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(fname=output_dir / "dop")

#     plt.figure()
#     plt.title("Trajectory")
#     plt.plot(truth_pos[0], truth_pos[1], label="truth", marker="*")
#     plt.plot(vdfll_pos[0], vdfll_pos[1], label="vdfll", marker="*")
#     plt.xlabel("East [m]")
#     plt.ylabel("North [m]")
#     plt.legend()
#     ax = plt.gca()
#     ax.axis("equal")
#     plt.tight_layout()
#     plt.savefig(fname=output_dir / "trajectory")

#     fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
#     fig.supxlabel("Time [s]")
#     fig.supylabel("Position [m]")
#     fig.suptitle("Position: East, North, Up")
#     for index, ax in enumerate(axes):
#         ax.plot(truth_rx_states.time, truth_pos[index], label="truth")
#         ax.plot(truth_rx_states.time, vdfll_pos[index], label="vdfll")
#     handles, labels = plt.gca().get_legend_handles_labels()
#     fig.legend(handles, labels)
#     plt.tight_layout()
#     plt.savefig(fname=output_dir / "position")

#     fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
#     fig.supxlabel("Time [s]")
#     fig.supylabel("Velocity [m/s]")
#     fig.suptitle("Velocity: East, North, Up")
#     for index, ax in enumerate(axes):
#         ax.plot(truth_rx_states.time, truth_vel[index], label="truth")
#         ax.plot(truth_rx_states.time, vdfll_vel[index], label="vdfll")
#     handles, labels = plt.gca().get_legend_handles_labels()
#     fig.legend(handles, labels)
#     plt.tight_layout()
#     plt.savefig(fname=output_dir / "velocity")

#     plt.figure()
#     plt.title("Position Error [m]")
#     plt.plot(truth_rx_states.time, pos_error, label=["east", "north", "up"])
#     plt.xlabel("Time [s]")
#     plt.ylabel("Error [m]")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(fname=output_dir / "position_error")

#     plt.figure()
#     plt.title("Velocity Error [m/s]")
#     plt.plot(truth_rx_states.time, vel_error, label=["east", "north", "up"])
#     plt.xlabel("Time [s]")
#     plt.ylabel("Error [m/s]")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(fname=output_dir / "velocity_error")

#     plt.figure()
#     plt.title("Clock Bias Error [m]")
#     plt.plot(truth_rx_states.time, cb_error)
#     plt.xlabel("Time [s]")
#     plt.ylabel("Error [m]")
#     plt.tight_layout()
#     plt.savefig(fname=output_dir / "cb_error")

#     plt.figure()
#     plt.title("Clock Drift Error [m/s]")
#     plt.plot(truth_rx_states.time, cd_error)
#     plt.xlabel("Time [s]")
#     plt.ylabel("Error [m/s]")
#     plt.tight_layout()
#     plt.savefig(fname=output_dir / "cd_error")

#     plt.figure()
#     plt.title("Code Discriminator [m]")
#     plot_emitter_dataframe(truth_rx_states.time, vdfll.prange_errors)
#     plt.xlabel("Time [s]")
#     plt.ylabel("Error [m]")
#     plt.tight_layout()
#     plt.savefig(fname=output_dir / "code_discriminator")

#     plt.figure()
#     plt.title("Frequency Discriminator [m/s]")
#     plot_emitter_dataframe(truth_rx_states.time, vdfll.prange_rate_errors)
#     plt.xlabel("Time [s]")
#     plt.ylabel("Error [m/s]")
#     plt.tight_layout()
#     plt.savefig(fname=output_dir / "f_discriminator")

#     plt.figure()
#     plt.title("Prompt Correlator Phasor")
#     plt.plot(ip, qp, ".")
#     plt.xlabel("Inphase Power")
#     plt.ylabel("Quadrature Power")
#     ax = plt.gca()
#     ax.axis("equal")
#     plt.tight_layout()
#     plt.savefig(fname=output_dir / "prompt_corr_phasor")
