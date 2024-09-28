import os
import ftplib
import urllib.parse
from tqdm import tqdm


def ftp_download(ftp_url, 
                 savepath, 
                 filenames=None, 
                 namefilter=None):

    tokens = urllib.parse.urlparse(ftp_url)
    server_address = tokens.netloc
    server_path = tokens.path

    ftp = ftplib.FTP_TLS(server_address, timeout=60)

    try:
        ftp.login()
        ftp.prot_p()

    except:
        try:
            ftp.cwd("/")
        except:
            raise

    ftp.cwd(server_path)
    
    server_files = ftp.nlst()
    
    if filenames is None:
        filenames = server_files
        
    ftp_info = {
        'downloaded': [], 
        'notfound': [], 
        'existed': [], 
        'filter': []
        }
    
    pbar = tqdm(filenames)
    
    for filename in pbar:
        
        pbar.set_description(f'downloading {filename}')
        
        if namefilter != None and filename.find(namefilter) < 0:
            
            ftp_info['filter'].append(filename)
            continue
        
        else:
            
            local_filename = os.path.join(savepath, filename)
            
            if filename not in server_files:
                
                ftp_info['notfound'].append(filename)
                
            elif os.path.isfile(local_filename):
                
                ftp_info['existed'].append(local_filename)
                
            else:

                with open(local_filename, 'wb') as f_obj:
                    ftp.retrbinary('RETR ' + filename, f_obj.write)
                    
                ftp_info['downloaded'].append(local_filename)

    ftp.close()

    return ftp_info
