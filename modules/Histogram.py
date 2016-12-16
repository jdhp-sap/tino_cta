import numpy as np
from scipy import ndimage, interpolate
# from numba import jit


class nDHistogram:
    def __init__(self, bin_edges, labels=[""], expected_mode=True):

        for label, edge in zip(labels, bin_edges):
            if len(edge) < 2:
                print("need at least two bin edges per dimension -- "+label)
                exit(-1)

        self.dimension = len(bin_edges)
        self.bin_edges = bin_edges
        self.labels    = labels

        self.data = np.zeros([len(x)+1 for x in bin_edges])
        self.norm = np.zeros_like(self.data)

    def __str__(self):
        string = "bin edges:\n"
        for label, edge in zip(self.labels, self.bin_edges):
            string += label
            string += "\t"
            string += str(edge)
            string += "\n"
        return string

    def find_bin(self, value, axis):
        # np.digitize seems to ignore units... convert arg to unit of axis as workaround
        try:
            return np.digitize(value.to(axis.unit), axis)
        except AttributeError:
            return np.digitize(value, axis)

    def find_bins(self, args):
        bins = []
        for ij, (arg, axis) in enumerate( zip(args,self.bin_edges)):
            # np.digitize seems to ignore units... convert arg to unit of axis as workaround
            try:
                bins += [ np.digitize(arg.to(axis.unit), axis) ]
            except AttributeError:
                bins += [ np.digitize(arg, axis) ]
        return bins

    def fill(self, args, value=1):
        bins = self.find_bins(args)
        self.data[bins] += value
        self.norm[bins] += 1

    def evaluate(self, args):
        return self.data[ self.find_bins(args) ]

    def interpolate(self, args, out_of_bounds_value=0., order=3):
        bins_u = np.array(self.find_bins(args))
        bins_l = bins_u - 1

        margs       = np.zeros_like(args)
        bin_u_edges = np.zeros_like(args)
        bin_l_edges = np.zeros_like(args)
        for ij, axis in enumerate(self.bin_edges):
            unit = args[ij].unit
            margs[ij]       = args[ ij ].value
            bin_u_edges[ij] = axis[ bins_u[ij]-1 ].to(unit).value
            bin_l_edges[ij] = axis[ bins_l[ij]-1 ].to(unit).value
        coordinates =  (margs-bin_u_edges) / (bin_u_edges-bin_l_edges) * (bins_u - bins_l) + bins_u

        mcoordinates = []
        for coor in coordinates:
            mcoordinates.append([coor])

        try:
            return ndimage.map_coordinates(self.data, mcoordinates, order=order)[0] * self.data.unit
        except AttributeError:
            return ndimage.map_coordinates(self.data, mcoordinates, order=order)[0]

    def fill_bin(self, value, args):
        self.data[args] += value
        self.norm[args] += 1

    def get_bin_content(self, args, normed=False):
        data = self.data
        norm = self.norm
        for arg in args:
            data = data[arg]
            norm = norm[arg]
        return data / norm if normed else data

    def get_outlier(self):
        data = self.data
        while len(data.shape):
            data = data[1:-1].sum(axis=0)
        return self.data.sum() - data

    def get_overflow(self):
        data = self.data
        while len(data.shape):
            data = data[:-1].sum(axis=0)
        return self.data.sum() - data

    def get_underflow(self):
        data = self.data
        while len(data.shape):
            data = data[1:].sum(axis=0)
        return self.data.sum() - data

    def normalise(self):
        self.data[self.norm != 0] = self.data[self.norm != 0] / self.norm[self.norm != 0]
        return self

    def write(self, filename, normed=False):
        norm = self.norm
        data = self.data
        if normed:
            self.normalise()
        np.savez_compressed(filename,
                            data=data,
                            axes=self.bin_edges,
                            labels=self.labels)

    @classmethod
    def read(cls, filename):
        with np.load(filename) as data:
            histo = cls( data['axes'], data['labels'] )
            histo.data = data['data']
            histo.norm = np.ones_like(histo.data)
        return histo


def get_subplot(hist, x, y=None, clip_outliers=True):
    a = hist.data
    b = hist.norm

    dim = hist.dimension

    # if only one axis is given (i.e. reduce to 1D histogram) set y=x
    if y is None:
        y = x

    # if @clip_outliers is set, the first and last bin are ignored,
    # otherwise they get summed up with the rest
    f = +1 if clip_outliers else 0
    l = -1 if clip_outliers else None

    # remove all axes before either of the two we look at
    for i in range(min(x, y)):
        a = a[f:l, ...].sum(axis=0)
        b = b[f:l, ...].sum(axis=0)

    # remove all axes between the two we look at
    for i in range(abs(x-y)-1):
        a = a[:, f:l, ...].sum(axis=1)
        b = b[:, f:l, ...].sum(axis=1)

    # remove all axes beyond either of the two we look at
    for i in range(dim-max(x, y)-1):
        a = a[..., f:l].sum(axis=-1)
        b = b[..., f:l].sum(axis=-1)

    a = a.astype(np.float)
    b = b.astype(np.float)

    # add a small number to every 0 to prevent division by 0
    b[b == 0] = 0.001

    # normalise: total number of PE by total number of hit pixels
    c = a / b

    if x == y:  return c[f:l]
    elif x < y: return c[f:l, f:l]
    else:       return c[f:l, f:l].T

