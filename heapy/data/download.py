import os
import ftplib
import urllib.error
import urllib.parse
import urllib.request
from tqdm import tqdm


def ftp_download(ftp_url, 
                 save_path, 
                 filenames=None, 
                 namefilter=None):

    tokens = urllib.parse.urlparse(ftp_url)
    serverAddress = tokens.netloc
    directory = tokens.path

    if filenames == None:

        ftp = ftplib.FTP(serverAddress, "anonymous", "", "", timeout=60)

        try:
            ftp.login()

        except:
            try:
                ftp.cwd("/")
            except:
                raise

        ftp.cwd(directory)

        filenames = []
        ftp.retrlines("NLST", filenames.append)

        ftp.close()

    downloaded_files = []

    for i, filename in enumerate(tqdm(filenames)):

        if namefilter != None and filename.find(namefilter) < 0:
            continue

        else:
            local_filename = os.path.join(save_path, filename)

            urllib.request.urlretrieve(
                "ftp://%s/%s/%s" % (serverAddress, directory, filename),
                local_filename)

            urllib.request.urlcleanup()

            downloaded_files.append(local_filename)

    return downloaded_files
