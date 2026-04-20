"""Fermi/GBM detector enumeration with spacecraft body-frame pointing.

Defines ``gbmDetector``, an ``Enum`` that encodes the long name, sequential
index, azimuth, and zenith angle of each Fermi Gamma-ray Burst Monitor (GBM)
detector in the spacecraft body frame.
"""

from enum import Enum



class gbmDetector(Enum):
    """Enumerate all Fermi/GBM detectors with body-frame pointing angles.

    Each member stores the detector's long name, sequential index number, and
    its boresight direction expressed as azimuth and zenith angles (in degrees)
    in the GBM spacecraft body frame.  The 12 NaI (N0–NB) and 2 BGO (B0–B1)
    scintillation detectors are included.

    Attributes:
        long_name: FITS-style long identifier (e.g. ``'NAI_00'``).
        number: Sequential detector index (0–13).
        azimuth: Boresight azimuth angle in the spacecraft frame, in degrees.
        zenith: Boresight zenith angle in the spacecraft frame, in degrees.
    """

    N0 = ('NAI_00', 0, 45.89, 20.58)
    N1 = ('NAI_01', 1, 45.11, 45.31)
    N2 = ('NAI_02', 2, 58.44, 90.21)
    N3 = ('NAI_03', 3, 314.87, 45.24)
    N4 = ('NAI_04', 4, 303.15, 90.27)
    N5 = ('NAI_05', 5, 3.35, 89.79)
    N6 = ('NAI_06', 6, 224.93, 20.43)
    N7 = ('NAI_07', 7, 224.62, 46.18)
    N8 = ('NAI_08', 8, 236.61, 89.97)
    N9 = ('NAI_09', 9, 135.19, 45.55)
    NA = ('NAI_10', 10, 123.73, 90.42)
    NB = ('NAI_11', 11, 183.74, 90.32)
    B0 = ('BGO_00', 12, 0.00, 90.00)
    B1 = ('BGO_01', 13, 180.00, 90.00)

    def __init__(self, long_name, number, azimuth, zenith):
        """Initialize a detector enum member with its pointing metadata.

        Args:
            long_name: FITS-style long detector identifier (e.g. ``'NAI_00'``).
            number: Sequential detector index in the range 0–13.
            azimuth: Boresight azimuth in the spacecraft body frame, in degrees.
            zenith: Boresight zenith in the spacecraft body frame, in degrees.
        """
        
        self.long_name = long_name
        self.number = number
        self.azimuth = azimuth
        self.zenith = zenith


    @classmethod
    def from_name(cls, name):
        """Return the detector member matching a short name string.

        The lookup is case-insensitive and expects the abbreviated form used
        as enum member names (e.g. ``'n0'``, ``'NA'``, ``'b1'``).

        Args:
            name: Short detector name to look up (case-insensitive).

        Returns:
            The matching ``gbmDetector`` member.

        Raises:
            ValueError: If ``name`` does not correspond to any known detector.
        """
        
        if name.upper() in cls.__members__:
            return cls[name.upper()]

        raise ValueError(f"Unknown detector name: {name}")


    @classmethod
    def from_index(cls, index):
        """Return the detector member matching a sequential index number.

        Args:
            index: Detector index in the range 0–13.

        Returns:
            The matching ``gbmDetector`` member.

        Raises:
            ValueError: If ``index`` does not correspond to any known detector.
        """
        
        for d in cls:
            if d.number == index:
                return d

        raise ValueError(f"Unknown detector index: {index}")


    @classmethod
    def get_nai(cls):
        """Return all NaI scintillation detector members.

        Returns:
            A list of ``gbmDetector`` members whose names begin with ``'N'``
            (N0 through NB, 12 detectors in total).
        """
        
        return [d for d in cls if d.name[0] == 'N']


    @classmethod
    def get_bgo(cls):
        """Return all BGO scintillation detector members.

        Returns:
            A list of ``gbmDetector`` members whose names begin with ``'B'``
            (B0 and B1, 2 detectors in total).
        """
        
        return [d for d in cls if d.name[0] == 'B']


    def __repr__(self):

        return "Detector(name='{}', long_name='{}', number={}, azimuth={}, zenith={})".format(
            self.name, self.long_name, self.number, self.azimuth, self.zenith)


    def __str__(self):

        return self.__repr__()
