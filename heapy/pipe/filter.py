"""Event-file filtering utilities for high-energy astronomy data.

Provides the ``Filter`` class, which wraps an ``astropy.table.Table`` of
photon events and applies user-supplied boolean expressions to select subsets
of rows.  Filters accumulate in sequence and can be cleared to restore the
original event list.

Typical usage:
    flt = Filter.from_fits('events.fits')
    flt.eval("(base['ENERGY'] > 15) & (base['ENERGY'] < 150)")
    flt.eval("base['TIME'] > 12345.0")
    filtered_table = flt.evt
"""

from astropy import table
from astropy.io import fits



class Filter(object):
    """Apply sequential boolean filters to a photon event table.

    Wraps an ``astropy.table.Table`` and maintains a mutable filtered view
    (``evt``) alongside the original unmodified copy.  Multiple calls to
    ``eval`` narrow the selection cumulatively; ``clear`` resets to the full
    event list.

    Attributes:
        evt: Current filtered event table (``astropy.table.Table``).
        exprs: List of filter expression strings applied so far, in order.

    Example:
        >>> flt = Filter.from_fits('sw00012345678bevshsp_uf.evt')
        >>> flt.eval("base['ENERGY'] > 50")
        >>> len(flt.evt)
        4231
    """

    def __init__(self, event):
        """Initialise with an existing event table.

        Args:
            event: Photon event data as an ``astropy.table.Table``.  A copy is
                taken internally so the original is never modified.

        Raises:
            AssertionError: If ``event`` is not an ``astropy.table.Table``.
        """

        msg = 'evt is not the type of astropy.tabel.Table'
        assert isinstance(event, table.Table), msg

        self._evt = event
        self.evt = self._evt.copy()

        self.exprs = []


    @classmethod
    def from_fits(cls, file, idx=None):
        """Load an event table from a FITS file and return a ``Filter`` instance.

        Reads the HDU specified by ``idx``, or falls back to the ``EVENTS``
        extension by name when ``idx`` is ``None``.

        Args:
            file: Path to the FITS event file.
            idx: HDU index (integer) or name (string) to read.  When ``None``,
                the ``EVENTS`` extension is used.

        Returns:
            A ``Filter`` instance wrapping the loaded event table.

        Raises:
            KeyError: If ``idx`` is ``None`` and no ``EVENTS`` extension exists
                in the file.
        """

        hdu = fits.open(file)

        if idx is not None:
            evt = hdu[idx]
        else:
            try:
                evt = hdu['EVENTS']
            except KeyError:
                raise KeyError('EVENTS extension not found!')

        evt = table.Table.read(evt)
        hdu.close()

        return cls(evt)


    @property
    def tags(self):
        """Return the column names of the original (unfiltered) event table.

        Returns:
            List of column name strings from the source ``astropy.table.Table``.
        """

        return self._evt.colnames


    @property
    def base(self):
        """Return the current filtered event table as a column-keyed dictionary.

        Provides the namespace used inside ``eval`` expressions, mapping each
        tag name to its ``astropy.table.Column`` array.

        Returns:
            Dictionary mapping column name strings to their column arrays from
            ``evt``.
        """

        return {tag: self.evt[tag] for tag in self.tags}


    def info(self, tag=None):
        """Print metadata about the current filtered event table or a single column.

        When called without arguments, prints the ``info`` summary of ``evt``.
        When ``tag`` is given, prints the ``info`` summary for that column.
        Either way, the list of accumulated filter expressions is also printed.

        Args:
            tag: Column name to describe.  Prints table-level info when ``None``.

        Raises:
            AssertionError: If ``tag`` is not a recognised column name.
        """

        if tag is None:
            print(self.evt.info)
        else:
            msg = '%s is not one of tags' % tag
            assert tag in self.tags, msg

            print(self.evt[tag].info)

        print('\n'.join(self.exprs))


    def eval(self, expr):
        """Evaluate a boolean expression and apply it as a row filter.

        The expression is evaluated with Python's built-in ``eval`` using
        ``base`` as the local namespace, so column arrays are referenced as
        ``base['COLNAME']``.  The resulting boolean mask is applied to ``evt``
        in place, and ``expr`` is appended to ``exprs``.  Passing ``None``
        is a no-op.

        Args:
            expr: Python expression string that evaluates to a boolean array
                with the same length as the current ``evt``.  ``None`` is
                silently ignored.

        Example:
            >>> flt.eval("(base['ENERGY'] > 15) & (base['ENERGY'] < 150)")
        """

        if expr is not None:

            flt = eval(expr, {}, self.base)
            self.evt = self.evt[flt]

            self.exprs.append(expr)


    def clear(self):
        """Reset ``evt`` to the original unfiltered event table.

        Restores ``evt`` from the internal copy taken at construction and clears
        the accumulated expression list.
        """

        self.evt = self._evt.copy()

        self.exprs = []
