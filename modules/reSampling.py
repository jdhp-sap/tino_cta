import numpy as np


# code from http://www.redblobgames.com/grids/hexagons


class Hex:
    """ class that holds the two coordinates of a hexagonal pixel """
    def __init__(self, q, r):
        self.q = q
        self.r = r

    def __str__(self):
        return "({}, {})".format(self.q, self.r)


class Cube:
    """ representation of a hexagon in abstract 3D space """
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return "({}, {}, {})".format(self.x, self.y, self.z)


def cube_to_hex(c):  # axial
    q = c.x
    r = c.z
    return Hex(q, r)


def hex_to_cube(h):  # axial
    x = h.q
    z = h.r
    y = -x-z
    return Cube(x, y, z)


def cube_round(c):
    """
    rounds a cube object with floating-point coordinates to have integer coordinets

    Parameters
    ----------
    c : Cube
        Cube with floating-point coordinates to be rounded

    Returns
    -------
    cube : Cube
        Cube with rounded integer coordinates
    """

    rx = round(c.x)
    ry = round(c.y)
    rz = round(c.z)

    x_diff = abs(rx - c.x)
    y_diff = abs(ry - c.y)
    z_diff = abs(rz - c.z)

    if x_diff > y_diff and x_diff > z_diff:
        rx = -ry-rz
    elif y_diff > z_diff:
        ry = -rx-rz
    else:
        rz = -rx-ry

    return Cube(int(rx), int(ry), int(rz))


def hex_round(h):
    """
    rounds a hexagon object with floating-point coordinates to have integer coordinets

    Parameters
    ----------
    h : Hex
        hexagon with floating-point coordinates to be rounded

    Returns
    -------
    hex : Hex
         hexagon with rounded integer coordinates
    """

    return cube_to_hex(cube_round(hex_to_cube(h)))


def pixel_to_hex(x, y, hex_size):
    """
    given x and y coordinates, determines the hexagon containing x and y

    Parameters
    ----------
    x, y : float
        x and y coordinates on the camera
    hex_size : float
        outer radius of the hexagonal pixel

    Returns
    -------
    hex : Hexagon object
        simple object holding the 2D coordinates of the corresponding hexagon in a
        slanted axial coordinate system

    """
    q = x * 2./3. / hex_size
    r = (-x/3. + y/(3**.5)) / hex_size
    return hex_round(Hex(q, r))


def make_qr_to_pix_id_map(pix_x, pix_y, size):
    qr_to_pix_id_map = {}
    for i, (x, y) in enumerate(zip(pix_x, pix_y)):
        hex = pixel_to_hex(x, y, size)
        qr_to_pix_id_map[(hex.q, hex.r)] = i
    return qr_to_pix_id_map


def make_pix_id_to_qr_map(pix_x, pix_y, size):
    pix_id_to_qr_map = {}
    for i, (x, y) in enumerate(zip(pix_x, pix_y)):
        hex = pixel_to_hex(x, y, size)
        pix_id_to_qr_map[i] = (hex.q, hex.r)
    return pix_id_to_qr_map


def make_pix_id_and_qr_map(pix_x, pix_y, size):
    """
    calculates both pix_id_to_qr_map and qr_to_pix_id_map

    Returns
    -------
    pix_id_to_qr_map, qr_to_pix_id_map
        maps between pixel id and q,r hexagonal coordinates
    """
    pix_id_to_qr_map = {}
    qr_to_pix_id_map = {}
    for i, (x, y) in enumerate(zip(pix_x, pix_y)):
        hex = pixel_to_hex(x, y, size)
        pix_id_to_qr_map[i] = (hex.q, hex.r)
        qr_to_pix_id_map[(hex.q, hex.r)] = i
    return pix_id_to_qr_map, qr_to_pix_id_map

qr_to_pix_id_map_tel_map = {}
def resample_hex_to_rect(img, pix_x=None, pix_y=None, hex_size=None, cam_id=None,
                         geom=None, nx=100, ny=100):
    """
    Resamples a hexagonal image into a square image

    Parameters
    ----------
    img : list
        list of the pixel contents
    pix_x, pix_y : numpy arrays of length quantity
        lists of x and y pixel coordinates on the camera
    hex_size : astropy length quantity
        outer radius of the hexagonal pixel
    cam_id : string
        camera ID
    geom : ctapipe geometry object, optional (default: None)
        alternative to pix_x/y, hex_size, cam_id... give the geometry object
    nx, ny : integer
        number of pixels in x and y direction of the resampled grid

    Returns
    -------
    rect_x, rect_y : numpy arrays
        lists of x and y pixel coordinates of the resampled grid
    rect_img : list
        list of the pixel contents of the resampled grid

    """

    if geom is not None:
        pix_x = geom.pix_x
        pix_y = geom.pix_y
        hex_size = 0.5 * (geom.pix_area[0] * (2/3**0.5)**3)**0.5
        cam_id = geom.cam_id

    min_x = np.min(pix_x)
    max_x = np.max(pix_x)

    min_y = np.min(pix_y)
    max_y = np.max(pix_y)

    qr_to_pix_id_map = qr_to_pix_id_map_tel_map[cam_id]

    rect_x, rect_y = np.meshgrid(np.linspace(min_x, max_x, nx),
                                 np.linspace(min_y, max_y, ny))

    rect_x = rect_x.ravel().value
    rect_y = rect_y.ravel().value

    rect_img = []
    for x, y in zip(rect_x, rect_y):
        hex = pixel_to_hex(x, y, hex_size.value)
        if (hex.q, hex.r) in qr_to_pix_id_map:
            rect_img.append(img[qr_to_pix_id_map[(hex.q, hex.r)]])
        else:
            rect_img.append(0)

    return rect_x, rect_y, np.array(rect_img)
