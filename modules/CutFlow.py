from astropy.table import Table

from collections import OrderedDict

class UndefinedCutException(Exception):
    pass

class PureCountingCutException(Exception):
    pass


class CutFlow():
    '''
    a class that keeps track of e.g. events/images that passed cuts or other
    events that could reject them '''
    def __init__(self, name="CutFlow"):
        '''
            Parameters:
            -----------
            name : string (default: "CutFlow")
                name for the specific instance
        '''
        self.cuts = OrderedDict()
        self.name = name

    def count(self, cut):
        '''
            counts an event/image at a given stage of the analysis

            Parameters:
            -----------
            cut : string
                name of the cut/stage where you want to count

            Note:
            -----
            If @cut is not yet being tracked, it will simply be added
            Will be an alias to __getitem__
        '''
        if cut not in self.cuts:
            self.cuts[cut] = [None, 1]
        else:
            self.cuts[cut][1] += 1

    def set_cut(self, cut, function):
        '''
            sets a function that selects on whatever you want to count
            sets the counter corresponding to the selection criterion to 0
            that means: it overwrites whatever you counted before under this
            name

            Parameters:
            -----------
            function : function
                a function that is your selection criterion
                should return True if event shall pass and False if event
                shall be rejected
            cut : string
                name of the cut/stage where you want to count

            Note:
            -----
            Will be an alias to add_cut
        '''
        self.cuts[cut] = [function, 0]

    def cut(self, cut, *args, **kwargs):
        '''
            selects the function associated with @cut and hands it all
            additional arguments provided. if the function returns True,
            the event counter is incremented.

            Parameters:
            -----------
            cut : string
                name of the selection criterion
            args, kwargs: additional arguments
                anything you want to hand to the associated function

            Returns:
            --------
            True if the function evaluats to True
            False otherwise

            Raises:
            -------
            UndefinedCutException if @cut is not known
            PureCountingCutException if @cut has no associated function
            (i.e. manual counting mode)
        '''
        if cut not in self.cuts:
            raise UndefinedCutException(
                "unknown cut {} -- only know: {}"
                .format(cut, [a for a in self.cuts.keys()]))
        elif self.cuts[cut][0] is None:
            raise PureCountingCutException(
                "{} has no function associated".format(cut))

        if self.cuts[cut][0](*args, **kwargs):
            self.cuts[cut][1] += 1
            return True
        else:
            return False

    def __call__(self, *args, **kwargs):
        '''
            creates an astropy table of the cut names, counted events and
            selection efficiencies
            prints the instance name and the astropy table

            Parameters:
            -----------
            kwargs : keyword arguments
                arguments to be passed to the get_table function

            Returns:
            --------
            t : astropy.Table
                the table containing the cut names, counted events and
                efficiencies -- sorted in the order the cuts were added if not
                specified otherwise
        '''
        print(self.name)
        t = self.get_table(*args, **kwargs)
        print(t)
        return t

    def get_table(self, base_cut=None, sort_column=None, sort_reverse=False):
        '''
            creates an astropy table of the cut names, counted events and
            selection efficiencies

            Parameters:
            -----------
            base_cut : string (default: None)
                name of the selection criterion that should be taken as 100 %
                in efficiency calculation
                if not given, the criterion with the highest count is used
            sort_column : integer (default: None)
                the index of the column that should be used for sorting the entries
                by default the table is sorted in the order the cuts were added
                (index 0: cut name, index 1: number of passed events, index 2: efficiency)
            sort_reverse : bool (default: False)
                if true, revert the order of the entries

            Returns:
            --------
            t : astropy.Table
                the table containing the cut names, counted events and
                efficiencies -- sorted in the order the cuts were added if not
                specified otherwise
        '''

        if base_cut is None:
            base_value = max([a[1] for a in self.cuts.values()])
        elif base_cut not in self.cuts:
            raise UndefinedCutException(
                "unknown cut {} -- only know: {}"
                .format(base_cut, [a for a in self.cuts.keys()]))
        else:
            base_value = self.cuts[base_cut][1]

        t = Table([[cut for cut in self.cuts.keys()],
                   [self.cuts[cut][1] for cut in self.cuts.keys()],
                   [self.cuts[cut][1]/base_value for cut in self.cuts.keys()]],
                  names=['Cut Name', 'selected Events', 'Efficiency'])

        if sort_column:
            t.sort(t.colnames[sort_column])
            t.reverse()
        if sort_reverse:
            t.reverse()
        return t

    add_cut = set_cut
    __getitem__ = count
