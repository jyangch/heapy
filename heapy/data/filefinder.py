"""Locate and download data files from local directories or FTP servers.

Provides ``FileFinder``, a utility class that first searches a local
directory for files matching a glob-like feature string and, when no local
match is found, falls back to an FTP source.  An FTP_TLS connection is
reused across multiple calls and reconnected automatically on failure.

Typical usage:
    from heapy.data.filefinder import FileFinder
    ff = FileFinder(local_dir='/data/gbm', ftp_url='ftp://host/path')
    files = ff.find('glg_tte_n0_bn*_v00.fit')
"""

import os
import ftplib
import warnings
from tqdm import tqdm
from urllib.parse import urlparse



class FileFinder(object):
    """Find data files locally or download them from an FTP server.

    Searches the configured local directory first.  If no match is found and
    an FTP URL is set, it queries the FTP server and downloads matching files
    into the local directory before returning their local paths.  The FTP
    connection is kept alive between calls and reconnected as needed.

    Attributes:
        local_files: List of local file paths found during the last
            ``find`` call, or ``None`` if ``find`` has not been called.
        ftp_files: List of FTP file paths found during the last ``find``
            call, or ``None`` if the FTP branch was not reached.
        ftp_connection: Active ``ftplib.FTP_TLS`` connection, or ``None``
            when disconnected.
    """

    def __init__(self, local_dir, ftp_url=None):
        """Initialize FileFinder with a local directory and optional FTP URL.

        Args:
            local_dir: Path to the local directory used as the primary
                search location and download destination.
            ftp_url: Full FTP URL (e.g. ``'ftp://host/remote/path'``), or
                ``None`` to disable FTP fallback.
        """

        self._local_dir = os.path.abspath(local_dir)
        self._ftp_url = urlparse(ftp_url) if ftp_url else None

        self.local_files = None
        self.ftp_files = None
        self.ftp_connection = None


    @property
    def local_dir(self):
        """Absolute path of the local search and download directory."""

        return self._local_dir


    @local_dir.setter
    def local_dir(self, new_local_dir):

        self._local_dir = os.path.abspath(new_local_dir)


    @property
    def ftp_url(self):
        """Parsed FTP URL (``urllib.parse.ParseResult``) or ``None``."""

        return self._ftp_url


    @ftp_url.setter
    def ftp_url(self, new_ftp_url):

        old_ftp_hostname = self._ftp_url.hostname if self._ftp_url else None

        self._ftp_url = urlparse(new_ftp_url) if new_ftp_url else None

        new_ftp_hostname = self._ftp_url.hostname if self._ftp_url else None

        if old_ftp_hostname and old_ftp_hostname != new_ftp_hostname:

            if self.ftp_connection:

                self.ftp_connection.quit()
                self.ftp_connection = None


    def __del__(self):

        if self.ftp_connection:
            self.ftp_connection.quit()


    def find(self, feature):
        """Return local paths of files matching ``feature``.

        Searches the local directory first.  When no local match is found
        and an FTP URL has been configured, queries the FTP server and
        downloads matching files before returning their local paths.

        Args:
            feature: Glob-like pattern string supporting ``*`` as a wildcard
                (e.g. ``'glg_tte_n0_*_v00.fit'``).

        Returns:
            A list of absolute local file paths that match ``feature``, or
            ``None`` when no match is found anywhere.

        Note:
            A ``UserWarning`` is emitted when no match is found after
            searching both local storage and the FTP server.  Failed
            downloads are retried once; files that remain corrupt (< 100
            bytes) after the retry are removed and a warning is issued.
        """

        self.local_files = self._get_files_from_local()
        matching_local_files = self._match_files(self.local_files, feature)

        if matching_local_files:

            return matching_local_files

        if self.ftp_url:
            self.ftp_files = self._get_files_from_ftp()
            matching_ftp_files = self._match_files(self.ftp_files, feature)

            if matching_ftp_files:

                downloaded_files_in_local = []

                pbar = tqdm(matching_ftp_files)

                for ftp_file_to_download in pbar:

                    pbar.set_description(f'Downloading {os.path.basename(ftp_file_to_download)}')

                    local_file_to_write = os.path.join(self.local_dir, os.path.basename(ftp_file_to_download))

                    success = self._download_file_from_ftp(ftp_file_to_download, local_file_to_write)

                    if not success:
                        print(f'Retrying download {os.path.basename(ftp_file_to_download)}')
                        success = self._download_file_from_ftp(ftp_file_to_download, local_file_to_write)

                    if success:
                        downloaded_files_in_local.append(local_file_to_write)
                    else:
                        warnings.warn(f"Failed to download {os.path.basename(ftp_file_to_download)} after retry.", UserWarning)

                return downloaded_files_in_local

        warnings.warn(f"No files found matching the feature: {feature}", UserWarning)
        return None


    def _get_files_from_local(self):

        if not os.path.exists(self.local_dir):
            warnings.warn(f"Directory '{self.local_dir}' does not exist.", UserWarning)
            return []

        return [os.path.join(self.local_dir, f) for f in os.listdir(self.local_dir)
                if os.path.isfile(os.path.join(self.local_dir, f))]


    def _get_files_from_ftp(self):

        self._ensure_ftp_connection()

        ftp_path = self.ftp_url.path

        try:
            return self.ftp_connection.nlst(ftp_path)
        except ftplib.all_errors as e:
            warnings.warn(f"FTP error: {str(e)}", UserWarning)
            return []


    def _download_file_from_ftp(self, ftp_file_path, local_file_path):

        self._ensure_ftp_connection()

        try:
            with open(local_file_path, 'wb') as local_file:
                self.ftp_connection.retrbinary(f"RETR {ftp_file_path}", local_file.write)
        except ftplib.all_errors as e:
            warnings.warn(f"FTP download error: {str(e)}", UserWarning)
            if os.path.exists(local_file_path) and os.path.getsize(local_file_path) < 100:
                warnings.warn(f"Downloaded file may be corrupted: {local_file_path}", UserWarning)
                os.remove(local_file_path)
            return False

        if os.path.exists(local_file_path) and os.path.getsize(local_file_path) < 100:
            warnings.warn(f"Downloaded file may be corrupted: {local_file_path}", UserWarning)
            os.remove(local_file_path)
            return False

        return True


    def _ensure_ftp_connection(self):

        if self.ftp_connection is None:

            ftp_host = self.ftp_url.hostname
            ftp_user = self.ftp_url.username or 'anonymous'
            ftp_pass = self.ftp_url.password or ''

            try:
                self.ftp_connection = ftplib.FTP_TLS(ftp_host)
                self.ftp_connection.login(user=ftp_user, passwd=ftp_pass)
                self.ftp_connection.prot_p()
                print(f"Connected to FTP: {ftp_host}")
            except ftplib.all_errors as e:
                warnings.warn(f"FTP connection error: {str(e)}", UserWarning)
                self.ftp_connection = None
                self._ensure_ftp_connection()

        else:

            if not self._is_ftp_connection_alive():
                print("FTP connection lost, reconnecting...")
                self.ftp_connection = None
                self._ensure_ftp_connection()


    def _is_ftp_connection_alive(self):

        try:
            self.ftp_connection.voidcmd("NOOP")
            return True
        except (ftplib.error_temp, ftplib.error_perm, ftplib.error_proto, OSError):
            return False


    def _match_files(self, files_in_dir, feature):

        if not files_in_dir:
            return []

        feature_list = [f for f in feature.split('*') if f]

        starts_with = feature.startswith('*')
        ends_with = feature.endswith('*')

        matching_files = []

        for file in files_in_dir:
            file_name = os.path.basename(file)

            if not starts_with and not file_name.startswith(feature_list[0]):
                continue

            if not ends_with and not file_name.endswith(feature_list[-1]):
                continue

            match = True
            pos = 0
            for feat in feature_list:
                pos = file_name.find(feat, pos)
                if pos == -1:
                    match = False
                    break
                pos += len(feat)

            if match:
                matching_files.append(file)

        return matching_files
