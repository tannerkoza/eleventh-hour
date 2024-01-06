import io
import numpy as np
import cartopy as ct
import cartopy.geodesic as cgeo
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt

from PIL import Image
from planar import BoundingBox
from shapely.geometry import LineString
from urllib.request import urlopen, Request


def geoplot(lat, lon, style="satellite", ax=None, **kwargs):
    if style == "map":
        cimgt.OSM.get_image = __image_spoof
        img = cimgt.OSM()  # spoofed, downloaded street map
    elif style == "satellite":
        cimgt.QuadtreeTiles.get_image = __image_spoof
        img = cimgt.QuadtreeTiles()  # spoofed, downloaded street map
    else:
        print("invalid style")

    if ax is None:
        ax = plt.axes(projection=img.crs)

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
