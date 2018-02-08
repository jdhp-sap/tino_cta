import numpy as np

from scipy import ndimage

from ctapipe.utils.CutFlow import CutFlow

from ctapipe.image.cleaning import tailcuts_clean, dilate
from ctapipe.image.geometry_converter import (convert_geometry_hex1d_to_rect2d,
                                              convert_geometry_rect2d_back_to_hexe1d,
                                              astri_to_2d_array, array_2d_to_astri,
                                              chec_to_2d_array, array_2d_to_chec)

from datapipe.denoising.wavelets_mrfilter import WaveletTransform
from datapipe.denoising import cdf
from datapipe.denoising.inverse_transform_sampling import \
    EmpiricalDistribution as EmpDist


class UnknownMode(ValueError):
    pass


class EdgeEvent(RuntimeError):
    pass


class MissingImplementation(KeyError):
    pass


def kill_isolpix(array, neighbours=None, threshold=.2):
    """
    Return array with isolated islands removed.
    Only keeping the biggest islands (largest surface).

    Parameters
    ----------
    array : 2D array
        the input image you want to keep the "biggest" patch of
    neighbours : 2D array, optional (default: None)
        a mask defining what is considered a neighbour
    threshold : float, optional (default: 0.2)
        ignores pixel with entries below this value

    Returns
    -------
    filtered_array : 2D array
        image with just the largest island remaining
    """

    filtered_array = np.copy(array)

    filtered_array[filtered_array < threshold] = 0
    mask = filtered_array > 0

    label_im, nb_labels = ndimage.label(mask, neighbours)

    sums = ndimage.sum(filtered_array, label_im, range(nb_labels + 1))
    mask_sum = sums < np.max(sums)
    remove_pixel = mask_sum[label_im]

    filtered_array[remove_pixel] = 0

    return filtered_array


def get_edge_pixels(camera, rows=1, n_neigh=None):
    """collects a list of pixel IDs that are considered to be "the edge" of the image.

    Parameters
    ----------
    camera : ctapipe CameraGeometry object
        the camera geometry object
    rows : integer, optional (default: 1)
        width of the edge in pixel-rows
    n_neigh : integer, optional (default: None)
        number of nominal number of neighbours for each pixels
        gets set to 6 for hexagonal and 4 for rectangular pixels
        ... you probably don't need to worry about this ...

    Returns
    -------
        python set of pixel IDs to consider "the edge" of the image
        edge_pixels : set

    Note
    ----
    `edge_pixels` is stored as a member of `camera`, since the same camera geometry will
    show up many times
    """
    if not hasattr(camera, "edge_pixels"):
        # the first row consists of all pixels that have less than the nominal number of
        # neighbours -- which is 6 for hexagonal pixels and 4 for rectangular ones
        n_neigh = n_neigh or (6 if "hex" in camera.pix_type else 4)
        edge_pixels = set(id for id, neigh in zip(camera.pix_id, camera.neighbors)
                          if len(neigh) < n_neigh)

        # add more rows by joining `edge_pixels` with the sets of the neighbours of all
        # pixels in `edge_pixels`
        for i in range(rows-1):
            for j in list(edge_pixels):
                edge_pixels |= set(camera.neighbors[j])

        camera.edge_pixels = np.array(list(edge_pixels), dtype=int)
    return camera.edge_pixels


def reject_edge_event(img, geom, rel_thresh=5., abs_thresh=None, rows=1):
    """determines whether any pixel at the edge (defined by `get_edge_pixels`) has a
    too high signal. The threshold is given either by `abs_thresh` or the highest signal
    divided by `rel_thresh`.

    Parameters
    ----------
    img : ndarray
        the camera image
    geom : ctapipe CameraGeometry object
        the camera geometry object
    rel_thresh : float, optional (default: 5.)
        divide the maximum signal by this number to get the threshold for a "too high
        signal at the edge"
    abs_thresh : float, optional (default: None)
        absolute value to consider a "too high signal at the edge"
        if given, overrides `rel_thresh` method

    Returns
    -------
    reject : bool
        whether or not any of the edge pixels has a signal higher than the threshold
    """
    edge_thresh = abs_thresh or (np.max(img)/rel_thresh)
    edge_pixels = get_edge_pixels(geom, rows=rows)
    return (img[edge_pixels] > edge_thresh).any()


class ImageCleaner:

    hex_neighbours_1ring = np.array([[1, 1, 0],
                                     [1, 1, 1],
                                     [0, 1, 1]])
    hex_neighbours_2ring = np.array([[1, 1, 1, 0, 0],
                                     [1, 1, 1, 1, 0],
                                     [1, 1, 1, 1, 1],
                                     [0, 1, 1, 1, 1],
                                     [0, 0, 1, 1, 1]])

    geom_1d_to_2d = {"ASTRICam": astri_to_2d_array,
                     "CHEC": chec_to_2d_array,
                     "hex": convert_geometry_hex1d_to_rect2d}

    geom_2d_to_1d = {"ASTRICam": array_2d_to_astri,
                     "CHEC": array_2d_to_chec,
                     "hex": convert_geometry_rect2d_back_to_hexe1d}

    def __init__(self, mode="wave", dilate=False, island_cleaning=True,
                 skip_edge_events=True, edge_width=1,
                 cutflow=CutFlow("ImageCleaner"),
                 wavelet_options=None,
                 tmp_files_directory='/dev/shm/', mrfilter_directory=None):
        self.mode = mode
        self.edge_width = edge_width
        self.cutflow = cutflow

        if skip_edge_events:
            self.reject_edge_event = reject_edge_event
        else:
            self.reject_edge_event = lambda *a, **b: False
        self.cutflow.add_cut("edge event", self.reject_edge_event)

        if mode in [None, "none", "None"]:
            self.clean = self.clean_none
        elif mode.startswith("wave"):
            self.clean = self.clean_wave
            self.wavelet_cleaning = \
                lambda *arg, **kwargs: WaveletTransform().clean_image(
                                *arg, **kwargs,
                                kill_isolated_pixels=island_cleaning,
                                tmp_files_directory=tmp_files_directory,
                                mrfilter_directory=mrfilter_directory)
            self.island_threshold = 1.5

            # command line parameters for the mr_filter call
            self.wavelet_options = \
                {"ASTRICam": wavelet_options or "-K -C1 -m3 -s2,2,3,3 -n4",
                 "DigiCam": wavelet_options or "-K -C1 -m3 -s6.274,2.629,7.755,0.076 -n4",
                 "FlashCam": wavelet_options or "-K -C1 -m3 -s4,4,5,4 -n4",
                 "NectarCam": wavelet_options or
                    "-K -C1 -m3 -s13.013,2.549,6.559,1.412 -n4",
                 "LSTCam": wavelet_options or
                    "-K -C1 -m3 -s23.343,2.490,-2.856,-0.719 -n4",
                 # WARNING: DUMMY VALUES
                 "CHEC": wavelet_options or "-K -C1 -m3 -s2,2,3,3 -n4"
                 }
            # camera models for noise injection
            self.noise_model = \
                {"ASTRICam": EmpDist(cdf.ASTRI_CDF_FILE),
                 "DigiCam": EmpDist(cdf.DIGICAM_CDF_FILE),
                 "FlashCam": EmpDist(cdf.FLASHCAM_CDF_FILE),
                 "NectarCam": EmpDist(cdf.NECTARCAM_CDF_FILE),
                 "LSTCam": EmpDist(cdf.LSTCAM_CDF_FILE),
                 # WARNING: DUMMY FILE
                 "CHEC": EmpDist(cdf.DIGICAM_CDF_FILE)}

        elif mode.startswith("tail"):
            self.clean = self.clean_tail
            self.tail_thresholds = \
                {"ASTRICam": (5, 7),  # (5, 10)?
                 "FlashCam": (12, 15),
                 "LSTCam": (5, 10),  # ?? (3, 6) for Abelardo...
                 # ASWG Zeuthen talk by Abelardo Moralejo:
                 "NectarCam": (4, 8),
                 # "FlashCam": (4, 8),  # there is some scaling missing?
                 "DigiCam": (3, 6),
                 "CHEC": (2, 4),
                 "SCTCam": (1.5, 3)}

            self.island_threshold = 1.5
            self.dilate = dilate
        else:
            raise UnknownMode(
                'cleaning mode "{}" not found'.format(mode))

        if island_cleaning:
            self.island_cleaning = kill_isolpix
        else:
            # just a pass-through that does nothing
            # (saves an if-statement in every `clean` call)
            self.island_cleaning = lambda x, *args, **kw: x

    def clean_wave(self, img, cam_geom):
        if cam_geom.pix_type.startswith("hex"):
            new_img, new_geom = self.clean_wave_hex(img, cam_geom)
        elif cam_geom.pix_type.startswith("rect"):
            new_img, new_geom = self.clean_wave_rect(img, cam_geom)
        else:
            raise MissingImplementation("wavelet cleaning not yet implemented"
                                        " for geometry {}".format(cam_geom.cam_id))

        if self.cutflow.cut("edge event",
                            img=new_img, geom=new_geom, rows=self.edge_width):
            raise EdgeEvent

        return new_img, new_geom

    def clean_wave_rect(self, img, cam_geom):
        try:
            array2d_img = self.geom_1d_to_2d[cam_geom.cam_id](img)
        except KeyError:
            raise MissingImplementation("wavelet cleaning not yet implemented"
                                        " for geometry {}".format(cam_geom.cam_id))

        cleaned_img = self.wavelet_cleaning(
                array2d_img, raw_option_string=self.wavelet_options[cam_geom.cam_id],
                noise_distribution=self.noise_model[cam_geom.cam_id])

        self.cutflow.count("wavelet cleaning")

        # wavelet_transform still leaves some isolated pixels; remove them
        cleaned_img = self.island_cleaning(cleaned_img)

        new_img = self.geom_2d_to_1d[cam_geom.cam_id](cleaned_img)
        new_geom = cam_geom

        return new_img, new_geom

    def clean_wave_hex(self, img, cam_geom):
        rot_geom, rot_img = convert_geometry_hex1d_to_rect2d(
                                cam_geom, img, cam_geom.cam_id)

        cleaned_img = self.wavelet_cleaning(
                rot_img, raw_option_string=self.wavelet_options[cam_geom.cam_id],
                noise_distribution=self.noise_model[cam_geom.cam_id])

        self.cutflow.count("wavelet cleaning")

        cleaned_img = self.island_cleaning(cleaned_img,
                                           neighbours=self.hex_neighbours_1ring,
                                           threshold=self.island_threshold)

        unrot_geom, unrot_img = convert_geometry_rect2d_back_to_hexe1d(
            rot_geom, cleaned_img, cam_geom.cam_id)

        return unrot_img, unrot_geom

    def clean_tail(self, img, cam_geom):
        mask = tailcuts_clean(
                cam_geom, img,
                picture_thresh=self.tail_thresholds[cam_geom.cam_id][1],
                boundary_thresh=self.tail_thresholds[cam_geom.cam_id][0])
        if self.dilate:
            dilate(cam_geom, mask)
        img[~mask] = 0

        self.cutflow.count("tailcut cleaning")

        if "ASTRI" in cam_geom.cam_id:
            # turn into 2d to apply island cleaning
            img = astri_to_2d_array(img)

            # if set, remove all signal patches but the biggest one
            new_img = self.island_cleaning(img)

            # turn back into 1d array
            new_img = array_2d_to_astri(new_img)
            new_geom = cam_geom

        else:
            # turn into 2d to apply island cleaning
            rot_geom, rot_img = convert_geometry_hex1d_to_rect2d(
                                    cam_geom, img, cam_geom.cam_id)

            # if set, remove all signal patches but the biggest one
            cleaned_img = self.island_cleaning(rot_img, self.hex_neighbours_1ring)

            # turn back into 1d array
            unrot_geom, unrot_img = convert_geometry_rect2d_back_to_hexe1d(
                rot_geom, cleaned_img, cam_geom.cam_id)

            new_img = unrot_img
            new_geom = unrot_geom

        if self.cutflow.cut("edge event",
                            img=new_img, geom=new_geom, rows=self.edge_width):
            raise EdgeEvent

        return new_img, new_geom

    def clean_none(self, img, cam_geom):
        return img, cam_geom
