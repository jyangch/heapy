"""HEALPix-based probability map localization classes for GBM events.

Provides ``HealPix``, a base class for sky-localization probability maps
stored in the HEALPix pixelization scheme, and ``gbmHealPix``, a
GBM-specific subclass that reads from FITS files and applies Earth-
occultation corrections.

Example:
    from heapy.geos.localization import gbmHealPix
    hpx = gbmHealPix('glg_healpix_all_bn210101000_v00.fit')
    ra, dec = hpx.centroid
    prob = hpx.get_probability(ra, dec)
"""

import contextlib
import os
import re

from astropy import units as u
from astropy.coordinates import angular_separation
from astropy.io import fits
import healpy as hp
from matplotlib.figure import Figure
import numpy as np

from .skymap import gbmSkyMap


class HealPix:
    """Base class for HEALPix probability maps.

    Stores a normalized differential probability array (``_prob``) and a
    cumulative significance / credible-level array (``_sig``) in HEALPix
    RING ordering.  All sky-coordinate arguments are in degrees (RA/Dec,
    ICRS / geocentric apparent).

    Attributes:
        prob: Differential probability per pixel (normalized to sum 1).
        sig: Cumulative credible-level array (0 = most probable pixel,
            1 = least probable).
    """

    def __init__(self):

        self._prob = np.array([], dtype=float)
        self._sig = np.array([], dtype=float)

    @property
    def prob(self):

        return self._prob

    @property
    def sig(self):

        return self._sig

    @property
    def npix(self):
        """Return the total number of HEALPix pixels in the map."""

        return len(self.prob)

    @property
    def nside(self):
        """Return the HEALPix ``NSIDE`` resolution parameter."""

        return hp.npix2nside(self.npix)

    @property
    def pixel_area(self):
        """Return the solid angle of one pixel in square degrees."""

        return 4.0 * 180.0**2 / (np.pi * self.npix)

    @property
    def centroid(self):
        """Return the RA and Dec of the peak-probability pixel.

        Returns:
            A tuple ``(ra, dec)`` in degrees for the pixel with the
            highest differential probability.
        """

        pix = np.argmax(self.prob)
        theta, phi = hp.pix2ang(self.nside, pix)

        return self._phi_to_ra(phi), self._theta_to_dec(theta)

    def get_probability(self, ra, dec, density=True):
        r"""Return the interpolated probability at a sky position.

        Uses bilinear HEALPix interpolation (``hp.get_interp_val``).

        Args:
            ra: Right ascension in degrees.
            dec: Declination in degrees.
            density: If ``True``, divide by pixel area to return
                probability density in deg\ :sup:`-2`.  If ``False``,
                return the raw per-pixel probability.

        Returns:
            Interpolated probability (or probability density) at the
            requested position.
        """

        phi = self._ra_to_phi(ra)
        theta = self._dec_to_theta(dec)
        prob = hp.get_interp_val(self.prob, theta, phi)

        if density:
            prob /= self.pixel_area

        return prob

    def get_confidence(self, ra, dec):
        """Return the credible level at a sky position.

        A value of 0 indicates the most probable pixel; 1 indicates the
        least probable.  Equivalent to ``1 - sig`` at the queried pixel.

        Args:
            ra: Right ascension in degrees.
            dec: Declination in degrees.

        Returns:
            Credible level in the range ``[0, 1]``.
        """

        phi = self._ra_to_phi(ra)
        theta = self._dec_to_theta(dec)

        return 1.0 - hp.get_interp_val(self.sig, theta, phi)

    def get_confidence_area(self, level):
        """Return the sky area enclosed within a given credible level.

        Args:
            level: Credible level threshold in the range ``[0, 1]``.
                For example, ``0.9`` returns the 90% credible-region area.

        Returns:
            Sky area in square degrees enclosed within ``level``.
        """

        numpix = np.sum((1.0 - self.sig) <= level)

        return numpix * self.pixel_area

    def get_probability_map(
        self, numpts_ra=360, numpts_dec=180, density=True, use_significance=False
    ):
        r"""Return a gridded probability map on a regular RA/Dec grid.

        Args:
            numpts_ra: Number of grid points along the RA axis.
            numpts_dec: Number of grid points along the Dec axis.
            density: If ``True``, divide probability values by pixel area
                to produce a density in deg\ :sup:`-2`.  Ignored when
                ``use_significance`` is ``True``.
            use_significance: If ``True``, return the cumulative
                significance (``sig``) array instead of ``prob``.

        Returns:
            A tuple ``(prob_grid, ra_axis, dec_axis)`` where
            ``prob_grid`` has shape ``(numpts_dec, numpts_ra)``,
            ``ra_axis`` has length ``numpts_ra`` in degrees, and
            ``dec_axis`` has length ``numpts_dec`` in degrees.
        """

        grid_pix, phi, theta = self._mesh_grid(numpts_ra, numpts_dec)

        if use_significance:
            prob_grid = self.sig[grid_pix]
        else:
            prob_grid = self.prob[grid_pix]
            if density:
                prob_grid /= self.pixel_area

        return prob_grid, self._phi_to_ra(phi), self._theta_to_dec(theta)

    def get_confidence_contours(self, level, numpts_ra=360, numpts_dec=180):
        """Return contour vertex arrays at a given credible level.

        Draws contours on a temporary ``matplotlib`` figure and extracts
        the resulting path vertices.

        Args:
            level: Credible level at which to extract contours, in the
                range ``[0, 1]`` (e.g., ``0.9`` for the 90% contour).
            numpts_ra: Number of grid points along the RA axis used for
                contouring.
            numpts_dec: Number of grid points along the Dec axis used for
                contouring.

        Returns:
            A list of ``(N, 2)`` arrays, one per disconnected contour
            segment.  Each row contains ``[ra, dec]`` in degrees.
        """

        grid_pix, phi, theta = self._mesh_grid(numpts_ra, numpts_dec)

        sig_grid = 1.0 - self.sig[grid_pix]

        ra_axis = self._phi_to_ra(phi)
        dec_axis = self._theta_to_dec(theta)

        fig = Figure()
        ax = fig.add_subplot(111)
        contour_set = ax.contour(ra_axis, dec_axis, sig_grid, levels=[level])

        paths = contour_set.collections[0].get_paths()
        vertices = [path.vertices for path in paths]

        return vertices

    def get_association_probability(self, ra, dec, prior=0.5):
        """Return the Bayesian association probability for a sky position.

        Computes the posterior probability that the event is associated
        with a source at ``(ra, dec)``, comparing the signal hypothesis
        (localization map) against a uniform null hypothesis.

        Args:
            ra: Right ascension of the candidate source in degrees.
            dec: Declination of the candidate source in degrees.
            prior: Prior probability of association, in ``[0, 1]``.
                Defaults to ``0.5``.

        Returns:
            Posterior association probability in ``[0, 1]``.

        Raises:
            ValueError: If ``prior`` is outside ``[0, 1]``.
        """

        if not 0.0 <= prior <= 1.0:
            raise ValueError('Prior probability must be within 0-1, inclusive')

        pixel_solid_angle_sr = hp.nside2resol(self.nside) ** 2
        prob_null_per_pixel = (1.0 / (4.0 * np.pi)) * pixel_solid_angle_sr

        prob_signal_per_pixel = self.get_probability(ra, dec, density=False)

        numerator = prob_signal_per_pixel * prior
        denominator = numerator + (prob_null_per_pixel * (1.0 - prior))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def get_overlap_association_probability(self, other_healpix, prior=0.5):
        """Return the overlap-based Bayesian association probability.

        Computes the posterior probability that this map and
        ``other_healpix`` are associated by comparing the pixel-wise
        overlap integral against a uniform null hypothesis.  Both maps
        are up-graded to the higher ``NSIDE`` before comparison.

        Args:
            other_healpix: Another ``HealPix`` instance to compare
                against.
            prior: Prior probability of association, in ``[0, 1]``.
                Defaults to ``0.5``.

        Returns:
            Posterior association probability in ``[0, 1]``.

        Raises:
            ValueError: If ``prior`` is outside ``[0, 1]``.
        """

        if not 0.0 <= prior <= 1.0:
            raise ValueError('Prior probability must be within 0-1, inclusive')

        prob1 = self.prob
        prob2 = other_healpix.prob

        target_nside = max(self.nside, other_healpix.nside)

        if self.nside < target_nside:
            prob1 = hp.ud_grade(prob1, nside_out=target_nside)
            prob1 = self._assert_prob(prob1)
        if other_healpix.nside < target_nside:
            prob2 = hp.ud_grade(prob2, nside_out=target_nside)
            prob2 = self._assert_prob(prob2)

        pixel_solid_angle_sr = hp.nside2resol(target_nside) ** 2
        prob_uniform = (1.0 / (4.0 * np.pi)) * pixel_solid_angle_sr
        null_hyp = np.sum(prob1 * prob_uniform)

        alt_hyp = np.sum(prob1 * prob2)

        numerator = alt_hyp * prior
        denominator = numerator + (null_hyp * (1.0 - prior))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    @staticmethod
    def _ra_to_phi(ra):

        return np.deg2rad(ra)

    @staticmethod
    def _phi_to_ra(phi):

        return np.rad2deg(phi)

    @staticmethod
    def _dec_to_theta(dec):

        return np.deg2rad(90.0 - dec)

    @staticmethod
    def _theta_to_dec(theta):

        return np.rad2deg(np.pi / 2.0 - theta)

    @staticmethod
    def _get_credible_levels(prob):

        p = np.asarray(prob)
        p_flat = p.ravel()

        sort_idx = np.argsort(p_flat)[::-1]
        credible_values = np.cumsum(p_flat[sort_idx])

        cls_flat = np.empty_like(p_flat)
        cls_flat[sort_idx] = credible_values

        return np.clip(cls_flat.reshape(p.shape), 0.0, 1.0)

    def _assert_prob(self, prob):

        prob[prob < 0.0] = 0.0
        prob /= prob.sum()

        return prob

    def _assert_sig(self, sig):

        if sig is not None:
            sig = np.clip(sig, 0.0, 1.0)

        return sig

    def _mesh_grid(self, num_phi, num_theta):

        theta = np.linspace(np.pi, 0.0, num_theta)
        phi = np.linspace(0.0, 2 * np.pi, num_phi)

        phi_grid, theta_grid = np.meshgrid(phi, theta)

        grid_pix = hp.ang2pix(self.nside, theta_grid, phi_grid)

        return grid_pix, phi, theta


class gbmHealPix(HealPix):
    """GBM HEALPix localization map loaded from a FITS file.

    Extends ``HealPix`` with GBM-specific metadata (trigger time, Sun
    position, Earth position and radius, detector pointings) read from
    the FITS headers.  Earth-occultation corrections can be applied
    analytically before computing association probabilities.

    Attributes:
        file: Path to the source FITS file.
        headers: Dictionary of FITS headers keyed by HDU name.
    """

    def __init__(self, file):
        """Initialize by reading probability and significance maps from a FITS file.

        Args:
            file: Path to the GBM HEALPix FITS file.
        """

        super().__init__()

        self._file = file

        self._read()

    def _read(self):

        with fits.open(self._file, memmap=False) as hdulist:
            self._headers = {hdu.name: hdu.header for hdu in hdulist}

            prob, sig = hp.read_map(self._file, field=(0, 1), memmap=False)
            self._prob = self._assert_prob(prob)
            self._sig = self._assert_sig(sig)

        with contextlib.suppress(KeyError):
            self._set_det_pointing()

    @property
    def file(self):

        return self._file

    @file.setter
    def file(self, new_file):

        self._file = new_file

        self._read()

    @property
    def headers(self):

        return self._headers

    @property
    def trigtime(self):
        """Return the GBM trigger time in Mission Elapsed Time seconds.

        Returns ``None`` if the ``TRIGTIME`` keyword is absent from the
        PRIMARY header.
        """

        try:
            return self.headers['PRIMARY']['TRIGTIME']
        except KeyError:
            return None

    @property
    def sun_location(self):
        """Return the Sun position as ``(ra, dec)`` in degrees.

        Returns ``None`` if the ``SUN_RA`` / ``SUN_DEC`` keywords are
        absent from the HEALPIX header.
        """

        try:
            return (self.headers['HEALPIX']['SUN_RA'], self.headers['HEALPIX']['SUN_DEC'])
        except KeyError:
            return None

    @property
    def geo_location(self):
        """Return the geocenter position as ``(ra, dec)`` in degrees.

        Returns ``None`` if the ``GEO_RA`` / ``GEO_DEC`` keywords are
        absent from the HEALPIX header.
        """

        try:
            return (self.headers['HEALPIX']['GEO_RA'], self.headers['HEALPIX']['GEO_DEC'])
        except KeyError:
            return None

    @property
    def geo_radius(self):
        """Return the Earth angular radius in degrees.

        Falls back to ``67.5`` degrees if the ``GEO_RAD`` keyword is
        absent from the HEALPIX header.
        """

        try:
            return self.headers['HEALPIX']['GEO_RAD']
        except KeyError:
            return 67.5

    @property
    def earth_occulted_probability(self):
        """Return the total probability occulted by the Earth.

        Returns ``None`` if the geocenter location is unknown.
        """

        if self.geo_location is None:
            return None

        mask, geo_mask = self._earth_mask()

        return np.sum(self.prob[mask][geo_mask])

    def to_fits(self, filename):
        """Write the probability map to a FITS file.

        Reorders the ``prob`` and ``sig`` arrays to nested ordering
        before writing, and copies the original PRIMARY and HEALPIX
        headers into the output file.

        Args:
            filename: Output FITS file path.  Overwritten if it already
                exists.
        """

        prob_arr = hp.reorder(self.prob, r2n=True)
        sig_arr = hp.reorder(self.sig, r2n=True)
        column_names = ['PROBABILITY', 'SIGNIFICANCE']

        hp.write_map(
            filename,
            (prob_arr, sig_arr),
            nest=True,
            coord='C',
            overwrite=True,
            column_names=column_names,
            extra_header=self.headers['HEALPIX'].cards,
        )

        with fits.open(filename, mode='update') as hdulist:
            hdulist[0].header.extend(self.headers['PRIMARY'])
            hdulist[1].name = 'HEALPIX'
            hdulist[1].header['TTYPE1'] = ('PROBABILITY', 'Differential probability per pixel')
            hdulist[1].header['TTYPE2'] = ('SIGNIFICANCE', 'Integrated probability')
            hdulist.writeto(filename, overwrite=True, checksum=True)

    def apply_earth_occultation(self):
        """Return a copy of this map with Earth-occulted pixels zeroed.

        Pixels within ``geo_radius`` degrees of the geocenter are set to
        zero probability.  The remaining probability is renormalized and
        a new significance array is recomputed.

        Returns:
            A new ``gbmHealPix`` instance with the occulted pixels
            removed.

        Raises:
            ValueError: If the geocenter location is not available in
                the headers.
        """

        if self.geo_location is None:
            raise ValueError('Location of geocenter is not known. Check headers.')

        mask, geo_mask = self._earth_mask()

        new_prob = np.copy(self.prob)
        temp_prob = new_prob[mask]
        temp_prob[geo_mask] = 0.0
        new_prob[mask] = temp_prob

        new_prob = self._assert_prob(new_prob)
        new_sig = self._assert_sig(1.0 - self._get_credible_levels(new_prob))

        from copy import deepcopy

        new_obj = deepcopy(self)

        new_obj._prob = new_prob
        new_obj._sig = new_sig

        return new_obj

    def get_association_probability(self, ra, dec, prior=0.5):
        """Return the Bayesian association probability, accounting for Earth occultation.

        Positions within ``geo_radius`` degrees of the geocenter return
        ``0.0`` immediately.  For all other positions, the earth-occulted
        map is used as the signal hypothesis.

        Args:
            ra: Right ascension of the candidate source in degrees.
            dec: Declination of the candidate source in degrees.
            prior: Prior probability of association, in ``[0, 1]``.
                Defaults to ``0.5``.

        Returns:
            Posterior association probability in ``[0, 1]``.

        Raises:
            ValueError: If ``prior`` is outside ``[0, 1]``.
        """

        if not 0.0 <= prior <= 1.0:
            raise ValueError('Prior probability must be within 0-1, inclusive')

        if self.geo_location is not None:
            geo_ra, geo_dec = self.geo_location
            angle = angular_separation(
                ra * u.deg, dec * u.deg, geo_ra * u.deg, geo_dec * u.deg
            ).to_value(u.deg)

            if angle < self.geo_radius:
                return 0.0

        pixel_solid_angle_sr = hp.nside2resol(self.nside) ** 2
        prob_null_per_pixel = (1.0 / (4.0 * np.pi)) * pixel_solid_angle_sr

        visible_map = self.apply_earth_occultation()
        prob_signal_per_pixel = visible_map.get_probability(ra, dec, density=False)

        numerator = prob_signal_per_pixel * prior
        denominator = numerator + (prob_null_per_pixel * (1.0 - prior))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def get_overlap_association_probability(self, other_healpix, prior=0.5):
        """Return the overlap-based association probability with Earth occultation applied.

        Identical to ``HealPix.get_overlap_association_probability`` but
        uses the earth-occulted version of this map as the signal
        hypothesis.

        Args:
            other_healpix: Another ``HealPix`` instance to compare
                against.
            prior: Prior probability of association, in ``[0, 1]``.
                Defaults to ``0.5``.

        Returns:
            Posterior association probability in ``[0, 1]``.

        Raises:
            ValueError: If ``prior`` is outside ``[0, 1]``.
        """

        if not 0.0 <= prior <= 1.0:
            raise ValueError('Prior probability must be within 0-1, inclusive')

        visible_map = self.apply_earth_occultation()
        prob1 = visible_map.prob
        prob2 = other_healpix.prob

        target_nside = max(self.nside, other_healpix.nside)

        if self.nside < target_nside:
            prob1 = hp.ud_grade(prob1, nside_out=target_nside)
            prob1 = self._assert_prob(prob1)
        if other_healpix.nside < target_nside:
            prob2 = hp.ud_grade(prob2, nside_out=target_nside)
            prob2 = self._assert_prob(prob2)

        pixel_solid_angle_sr = hp.nside2resol(target_nside) ** 2
        prob_uniform = (1.0 / (4.0 * np.pi)) * pixel_solid_angle_sr
        null_hyp = np.sum(prob1 * prob_uniform)

        alt_hyp = np.sum(prob1 * prob2)

        numerator = alt_hyp * prior
        denominator = numerator + (null_hyp * (1.0 - prior))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def get_observable_fraction(self, other_healpix):
        """Return the fraction of ``other_healpix`` probability visible to GBM.

        Pixels in ``other_healpix`` that are occulted by Earth (within
        ``geo_radius`` degrees of the geocenter) are excluded from the
        sum.

        Args:
            other_healpix: A ``HealPix`` instance whose observable
                fraction is computed relative to this map's geocenter
                position.

        Returns:
            Fraction of ``other_healpix`` total probability that lies
            outside the Earth-occultation cone, in ``[0, 1]``.

        Raises:
            ValueError: If the geocenter location is not available in
                the headers.
        """

        if self.geo_location is None:
            raise ValueError('Location of geocenter is not known. Check headers.')

        prob_arr = other_healpix.prob
        active_mask = np.where(prob_arr > 0.0)[0]

        if active_mask.size == 0:
            return 0.0

        theta, phi = hp.pix2ang(other_healpix.nside, active_mask)
        ra = self._phi_to_ra(phi)
        dec = self._theta_to_dec(theta)

        geo_ra, geo_dec = self.geo_location
        angle = angular_separation(
            ra * u.deg, dec * u.deg, geo_ra * u.deg, geo_dec * u.deg
        ).to_value(u.deg)

        visible_mask = angle > self.geo_radius

        visible_prob = np.sum(prob_arr[active_mask][visible_mask])
        total_prob = np.sum(prob_arr)

        return visible_prob / total_prob

    def extract_skymap(self, savepath='./geometry'):

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        skymap = gbmSkyMap()
        skymap.plot_galactic()
        skymap.add_healpix(self)

        skymap.save(savepath + '/sky_map_healpix.pdf')

    def _set_det_pointing(self):

        keys = list(self.headers['HEALPIX'].keys())
        regex = re.compile('N._RA|B._RA')
        dets = [key.split('_')[0] for key in keys if re.match(regex, key)]

        for det in dets:
            setattr(
                self,
                det.lower() + '_pointing',
                (self.headers['HEALPIX'][det + '_RA'], self.headers['HEALPIX'][det + '_DEC']),
            )

    def _earth_mask(self):

        mask = self.prob > 0.0
        theta, phi = hp.pix2ang(self.nside, np.arange(self.npix))
        ra_mask = self._phi_to_ra(phi)[mask]
        dec_mask = self._theta_to_dec(theta)[mask]

        geo_ra, geo_dec = self.geo_location

        angle = angular_separation(
            geo_ra * u.deg, geo_dec * u.deg, ra_mask * u.deg, dec_mask * u.deg
        ).to_value(u.deg)

        geo_mask = angle <= self.geo_radius

        return mask, geo_mask
