"""Locate data files locally, over HTTPS, or on FTP servers.

Provides ``FileFinder``, a utility class that first searches a local
directory for files matching a glob-like feature string and, when no local
match is found, tries HTTPS directory listings and downloads before FTP.
Each protocol caches its own directory listings. An FTP_TLS connection is
opened only when needed, reused, and reconnected automatically on failure.

Example:
    from heapy.data.filefinder import FileFinder
    ff = FileFinder(local_dir='/data/gbm', ftp_url='ftp://host/path')
    files = ff.find('glg_tte_n0_bn*_v00.fit')
"""

from contextlib import suppress
import ftplib
from html.parser import HTMLParser
from http.client import HTTPException
import os
import re
import tempfile
from urllib.error import HTTPError
from urllib.parse import quote, unquote, urljoin, urlparse
from urllib.request import Request, urlopen
import warnings

from tqdm import tqdm


class FileFinder:
    """Find data files locally or download them over HTTPS or FTP.

    Searches the configured local directory first.  If no match is found and
    a remote URL is set, queries that server and downloads matching files
    into the local directory. HTTPS is preferred, with FTP as an optional
    fallback. Either protocol can also be used independently.

    Attributes:
        local_files: List of local file paths found during the last
            ``find`` call, or ``None`` if ``find`` has not been called.
        ftp_files: List of FTP file paths found during the last ``find``
            call, or ``None`` if the FTP branch was not reached.
        https_files: List of file paths from the HTTPS directory index, or
            ``None`` if the HTTPS branch was not reached.
        ftp_connection: Active ``ftplib.FTP_TLS`` connection, or ``None``
            when disconnected.
    """

    def __init__(self, local_dir, ftp_url=None, https_url=None):
        """Initialize FileFinder with a local directory and optional remote URLs.

        Args:
            local_dir: Path to the local directory used as the primary
                search location and download destination.
            ftp_url: Full FTP URL (e.g. ``'ftp://host/remote/path'``), or
                ``None`` to disable FTP fallback.
            https_url: Optional HTTPS directory with an HTML file index.
                HTTPS lists and downloads files without needing ``ftp_url``.
                When both are set, FTP is the fallback source. Update both
                URLs when changing directories, or set ``https_url=None``
                for FTP-only retrieval.
        """

        self._local_dir = os.path.abspath(local_dir)
        self._ftp_url = urlparse(ftp_url) if ftp_url else None
        self._https_url = urlparse(https_url) if https_url else None

        self.local_files = None
        self.ftp_files = None
        self.https_files = None
        self.ftp_connection = None

        self._ftp_listing_cache = {}
        self._https_listing_cache = {}

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

        if old_ftp_hostname and old_ftp_hostname != new_ftp_hostname and self.ftp_connection:
            self.ftp_connection.quit()
            self.ftp_connection = None

    @property
    def https_url(self):
        """Parsed HTTPS URL (``urllib.parse.ParseResult``) or ``None``."""

        return self._https_url

    @https_url.setter
    def https_url(self, new_https_url):

        self._https_url = urlparse(new_https_url) if new_https_url else None

    def __del__(self):
        """Close the persistent FTP connection; HTTPS responses close after each transfer."""

        if self.ftp_connection:
            self.ftp_connection.quit()

    def find(self, feature):
        """Find local files, then try HTTPS and finally FTP.

        Each remote protocol lists and downloads its own files. HTTPS can
        operate without an FTP URL. Failed listings, missing matches, or
        failed downloads fall back to FTP when configured; successful HTTPS
        downloads are retained. FTP retries once. HTTPS makes at most five
        attempts, resuming validated partial downloads when supported, and
        stops after two consecutive failures without resumable progress.

        Listings are cached per protocol and URL for this finder. A cached
        listing is refreshed once when matches are missing or the server reports
        that a listed file is no longer available, allowing newly added or
        replaced files to be found.

        Args:
            feature: Pattern supporting ``*`` as a wildcard.

        Returns:
            Absolute paths of successfully downloaded or matching local files,
            sorted by filename (including zero-padded GBM version numbers).
            Returns ``None`` (with a warning) when no remote or local file
            matches, or an empty list when every matching download fails.

        Raises:
            ConnectionError: FTP connection retries are exhausted and no files
                have been downloaded in this call. If some downloads completed,
                stop retrying and return those paths with a warning instead.
        """

        self.https_files = None
        self.ftp_files = None

        self.local_files = self._get_files_from_local()
        matching_local_files = self._match_files(self.local_files, feature)

        if matching_local_files:
            return sorted(matching_local_files)

        sources = [
            (
                self.https_url,
                self._https_listing_cache,
                self._get_files_from_https,
                self._download_file_from_https,
                False,
            ),
            (
                self.ftp_url,
                self._ftp_listing_cache,
                self._get_files_from_ftp,
                self._download_file_from_ftp,
                True,
            ),
        ]

        downloaded_files = {}
        found_remote_match = False

        for url, cache, get_files, download_file, retry_download in sources:
            if url is None:
                continue

            try:
                cached_listing = url in cache
                matching_files = self._match_files(get_files(), feature)
                if not matching_files and cached_listing:
                    matching_files = self._match_files(get_files(refresh=True), feature)
                    cached_listing = False

                attempted_files = set()
                for listing_attempt in range(2):
                    if not matching_files:
                        break

                    found_remote_match = True
                    with tqdm(matching_files) as pbar:
                        for remote_file in pbar:
                            name = os.path.basename(remote_file)
                            if name in downloaded_files or remote_file in attempted_files:
                                continue
                            attempted_files.add(remote_file)
                            pbar.set_description(f'Downloading {name}')
                            local_file = os.path.join(self.local_dir, name)
                            success = download_file(remote_file, local_file)
                            if not success and retry_download:
                                print(f'Retrying download {name}')
                                success = download_file(remote_file, local_file)
                            if success:
                                downloaded_files[name] = local_file
                            else:
                                warnings.warn(
                                    f'Failed to download {name} via {url.scheme.upper()}.',
                                    UserWarning,
                                    stacklevel=2,
                                )

                    if all(os.path.basename(file) in downloaded_files for file in matching_files):
                        return sorted(downloaded_files.values())
                    if listing_attempt or not cached_listing or url in cache:
                        break
                    matching_files = self._match_files(get_files(refresh=True), feature)
            except ConnectionError as e:
                if not downloaded_files:
                    raise
                warnings.warn(
                    f'FTP connection unavailable; returning completed downloads: {e!s}',
                    UserWarning,
                    stacklevel=2,
                )
                break

        if found_remote_match:
            return sorted(downloaded_files.values())
        warnings.warn(f'No files found matching the feature: {feature}', UserWarning, stacklevel=2)
        return None

    def _get_files_from_local(self):

        if not os.path.exists(self.local_dir):
            warnings.warn(
                f"Directory '{self.local_dir}' does not exist.", UserWarning, stacklevel=2
            )
            return []

        return [
            os.path.join(self.local_dir, f)
            for f in os.listdir(self.local_dir)
            if os.path.isfile(os.path.join(self.local_dir, f))
        ]

    def _get_files_from_https(self, refresh=False):
        """Read same-directory file links from an HTTPS HTML index and cache them."""

        if not refresh and self.https_url in self._https_listing_cache:
            self.https_files = self._https_listing_cache[self.https_url]
            return self.https_files

        self.https_files = []
        directory = self.https_url._replace(
            path=self.https_url.path.rstrip('/') + '/', params='', query='', fragment=''
        )

        links = []

        def collect_links(tag, attrs):
            if tag == 'a':
                links.extend(value for name, value in attrs if name == 'href' and value)

        try:
            with urlopen(directory.geturl(), timeout=30) as response:
                parser = HTMLParser()
                parser.handle_starttag = collect_links
                parser.feed(response.read().decode('utf-8'))
            for href in links:
                link = urlparse(urljoin(directory.geturl(), href))
                if (link.scheme, link.netloc) != (directory.scheme, directory.netloc):
                    continue
                if link.query or link.fragment or link.params or link.path.endswith('/'):
                    continue
                if os.path.dirname(link.path) != (directory.path.rstrip('/') or '/'):
                    continue
                name = unquote(os.path.basename(link.path))
                if (
                    not name
                    or name in ('.', '..')
                    or any(char in name for char in ('/', '\\', '\x00'))
                ):
                    continue
                file = os.path.join(unquote(directory.path), name)
                if file not in self.https_files:
                    self.https_files.append(file)
        except (OSError, HTTPException, ValueError) as e:
            warnings.warn(f'HTTPS directory error: {e!s}', UserWarning, stacklevel=2)
            return []

        self._https_listing_cache[self.https_url] = self.https_files
        return self.https_files

    def _get_files_from_ftp(self, refresh=False):
        """Read an FTP directory listing, reusing this finder's cache by default."""

        if not refresh and self.ftp_url in self._ftp_listing_cache:
            self.ftp_files = self._ftp_listing_cache[self.ftp_url]
            return self.ftp_files

        self.ftp_files = []
        self._ensure_ftp_connection()
        try:
            self.ftp_files = self.ftp_connection.nlst(self.ftp_url.path)
        except ftplib.all_errors as e:
            warnings.warn(f'FTP directory error: {e!s}', UserWarning, stacklevel=2)
            return []

        self._ftp_listing_cache[self.ftp_url] = self.ftp_files
        return self.ftp_files

    def _download_file_from_https(self, https_file_path, local_file_path):
        """Download atomically with bounded retries and strong ETag-validated resume."""

        url = self.https_url._replace(
            path=self.https_url.path.rstrip('/') + '/' + quote(os.path.basename(https_file_path)),
            params='',
            query='',
            fragment='',
        ).geturl()
        temporary_path = None
        success = False
        size = 0
        expected_size = None
        validator = None
        max_attempts = 5
        stalled_attempts = 0
        try:
            with (
                tempfile.NamedTemporaryFile(
                    dir=os.path.dirname(local_file_path), suffix='.part', delete=False
                ) as local_file,
                tqdm(
                    unit='B', unit_scale=True, desc=os.path.basename(local_file_path), leave=False
                ) as pbar,
            ):
                temporary_path = local_file.name
                for attempt in range(max_attempts):
                    offset = size if validator else 0
                    headers = {'Accept-Encoding': 'identity'}
                    if offset:
                        headers.update({'Range': f'bytes={offset}-', 'If-Range': validator})
                    try:
                        with urlopen(Request(url, headers=headers), timeout=30) as response:
                            if response.status == 200:
                                local_file.seek(0)
                                local_file.truncate()
                                size = 0
                                length = response.headers.get('Content-Length')
                                expected_size = int(length) if length is not None else None
                                etag = response.headers.get('ETag')
                                validator = etag if etag and not etag.startswith('W/') else None
                                pbar.reset(total=expected_size)
                            elif response.status == 206:
                                match = re.fullmatch(
                                    r'bytes (\d+)-(\d+)/(\d+)',
                                    response.headers.get('Content-Range', ''),
                                )
                                if not offset or match is None:
                                    raise ValueError('Unexpected partial HTTPS response')
                                start, end, total = map(int, match.groups())
                                if (
                                    start != offset
                                    or not start <= end < total
                                    or end != total - 1
                                    or (expected_size is not None and total != expected_size)
                                    or response.headers.get('ETag') != validator
                                ):
                                    raise ValueError('HTTPS resume range or ETag does not match')
                                length = response.headers.get('Content-Length')
                                if length is not None and int(length) != end - start + 1:
                                    raise ValueError('HTTPS range length does not match')
                                expected_size = total
                                pbar.total = total
                            else:
                                raise OSError(f'Unexpected HTTPS status: {response.status}')

                            while True:
                                block = response.read1(64 * 1024)
                                if not block:
                                    break
                                local_file.write(block)
                                size += len(block)
                                pbar.update(len(block))
                            if size < 100 or (expected_size is not None and size != expected_size):
                                raise OSError(
                                    f'Incomplete download: {size} bytes, expected {expected_size}'
                                )
                        success = True
                        break
                    except (OSError, HTTPException, ValueError) as e:
                        warnings.warn(
                            f'HTTPS download error for {os.path.basename(local_file_path)} '
                            f'(attempt {attempt + 1}/{max_attempts}, {size} bytes): {e!s}',
                            UserWarning,
                            stacklevel=2,
                        )
                        if isinstance(e, ValueError):
                            break
                        if isinstance(e, HTTPError):
                            if e.code in (404, 410):
                                self._https_listing_cache.pop(self.https_url, None)
                            if 400 <= e.code < 500 and e.code not in (408, 429):
                                break
                        if validator and size > offset:
                            stalled_attempts = 0
                        else:
                            stalled_attempts += 1
                        if stalled_attempts >= 2 or attempt + 1 == max_attempts:
                            break
                        action = f'Resuming from {size} bytes' if size and validator else 'Retrying'
                        print(f'{action}: {os.path.basename(local_file_path)}')
            if success:
                os.replace(temporary_path, local_file_path)
            return success
        except OSError as e:
            warnings.warn(f'HTTPS download error: {e!s}', UserWarning, stacklevel=2)
            return False
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                os.remove(temporary_path)

    def _download_file_from_ftp(self, ftp_file_path, local_file_path):
        """Download atomically; exhausted connection retries raise ConnectionError."""

        self._ensure_ftp_connection()
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=os.path.dirname(local_file_path), suffix='.part', delete=False
            ) as local_file:
                temporary_path = local_file.name
                self.ftp_connection.retrbinary(f'RETR {ftp_file_path}', local_file.write)
            if os.path.getsize(temporary_path) < 100:
                raise OSError(f'Downloaded file may be corrupted: {local_file_path}')
            os.replace(temporary_path, local_file_path)
            return True
        except ftplib.all_errors as e:
            if isinstance(e, ftplib.error_perm) and str(e).startswith('550'):
                self._ftp_listing_cache.pop(self.ftp_url, None)
            warnings.warn(f'FTP download error: {e!s}', UserWarning, stacklevel=2)
            return False
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                os.remove(temporary_path)

    def _ensure_ftp_connection(self, max_retries=5, timeout=30):
        """Open or refresh the FTP_TLS connection, retrying transient errors.

        Replaces a previous infinite-recursion retry loop with a bounded
        ``max_retries`` attempt counter; on persistent failure raises
        :class:`ConnectionError` so the caller fails fast instead of
        blowing the stack.
        """

        ftp_host = self.ftp_url.hostname
        ftp_user = self.ftp_url.username or 'anonymous'
        ftp_pass = self.ftp_url.password or ''

        for attempt in range(1, max_retries + 1):
            if self.ftp_connection is not None and self._is_ftp_connection_alive():
                return
            if self.ftp_connection is not None:
                with suppress(Exception):
                    self.ftp_connection.close()
                self.ftp_connection = None
                print('FTP connection lost, reconnecting...')

            try:
                self.ftp_connection = ftplib.FTP_TLS(ftp_host, timeout=timeout)
                self.ftp_connection.login(user=ftp_user, passwd=ftp_pass)
                self.ftp_connection.prot_p()
                print(f'Connected to FTP: {ftp_host}')
                return
            except ftplib.all_errors as e:
                self.ftp_connection = None
                warnings.warn(
                    f'FTP connection attempt {attempt}/{max_retries} failed: {e!s}',
                    UserWarning,
                    stacklevel=2,
                )

        raise ConnectionError(
            f'Could not establish FTP_TLS connection to {ftp_host} after {max_retries} attempts'
        )

    def _is_ftp_connection_alive(self):
        """Check if the FTP connection is alive."""

        try:
            self.ftp_connection.voidcmd('NOOP')
            return True
        except (ftplib.error_temp, ftplib.error_perm, ftplib.error_proto, OSError):
            return False

    def _match_files(self, files_in_dir, feature):
        """Match files against a feature pattern."""

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
