import os
import json
import shutil
import warnings
import subprocess
import numpy as np
from astropy.io import fits
import plotly.graph_objs as go
from astropy.table import Table
from .retrieve import epRetrieve
from ..util.data import msg_format, NpEncoder
from ..util.time import ep_met_to_utc, ep_utc_to_met



class epXselect(object):
    
    def __init__(self, 
                 evtfile=None, 
                 regfile=None, 
                 bkregfile=None
                 ):
        
        self.evtfile = evtfile
        self.regfile = regfile
        self.bkregfile = bkregfile


    @classmethod
    def from_rtv(cls, rtv):

        if isinstance(rtv, epRetrieve):
            evtfile = rtv.rtv_res['evt']
            regfile = rtv.rtv_res['reg']
            bkregfile = rtv.rtv_res['bkreg']
            
        elif isinstance(rtv, dict):
            if 'evt' in rtv:
                evtfile = rtv['evt']
            else:
                msg = 'evt is not a key of rvt'
                raise ValueError(msg_format(msg))
            
            if 'reg' in rtv:
                regfile = rtv['reg']
            else:
                msg = 'reg is not a key of rvt'
                raise ValueError(msg_format(msg))
            
            if 'bkreg' in rtv:
                bkregfile = rtv['bkreg']
            else:
                msg = 'bkreg is not a key of rvt'
                raise ValueError(msg_format(msg))
            
        else:
            msg = 'rvt is not the expected format'
            raise ValueError(msg_format(msg))
        
        return cls(evtfile, regfile, bkregfile)
    
    
    @property
    def evtfile(self):
        
        return self._evtfile
    
    
    @evtfile.setter
    def evtfile(self, new_evtfile):
        
        self._evtfile = new_evtfile
        
        hdu = fits.open(self._evtfile)
        tstart = hdu['EVENTS'].header['TSTART']
        hdu.close()
        
        self._timezero = tstart
        
        
    @property
    def regfile(self):
        
        return self._regfile
    
    
    @regfile.setter
    def regfile(self, new_regfile):
        
        self._regfile = new_regfile
        
        self._regtxt = self._regfile + '.txt'
        
        if not os.path.isfile(self._regtxt):
            msg = 'reg text file is not found.'
            warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
            
        else:
            with open(self._regtxt) as f_obj:
                lines = f_obj.readlines()
            self.regarea = float(lines[7].split()[3])
        
        
    @property
    def bkregfile(self):
        
        return self._bkregfile
    
    
    @bkregfile.setter
    def bkregfile(self, new_bkregfile):
        
        self._bkregfile = new_bkregfile
        
        self._bkregtxt = self._bkregfile + '.txt'
        
        if not os.path.isfile(self._bkregtxt):
            msg = 'bkreg text file is not found.'
            warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
            
        else:
            with open(self._bkregtxt) as f_obj:
                lines = f_obj.readlines()
            
            self.bkregarea = float(lines[7].split()[3])
        
        
    @property
    def reginfo(self):
        
        os.system(f'cat {self._regtxt}')
        
        return None
    
    
    @property
    def bkreginfo(self):
        
        os.system(f'cat {self._bkregtxt}')
        
        return None
    
    
    @property
    def regratio(self):
        
        try:
            return self.regarea / self.bkregarea
        except:
            msg = 'no region area information, back to default ratio 1/12'
            warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
            return 1 / 12


    @property
    def timezero(self):
        
        return self._timezero
    
    
    @timezero.setter
    def timezero(self, new_timezero):
        
        if isinstance(new_timezero, float):
            self._timezero = new_timezero
            
        elif isinstance(new_timezero, str):
            self._timezero = ep_utc_to_met(new_timezero)
            
        else:
            msg = 'not expected type for timezero'
            raise ValueError(msg_format(msg))


    @staticmethod
    def _run_xselect(commands):
        
        process = subprocess.Popen('xselect', 
                                   stdin=subprocess.PIPE, 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True)

        stdout, stderr = process.communicate(input='\n'.join(commands) + '\nexit\n')

        return stdout, stderr
    
    
    @staticmethod
    def _run_xselect_(commands, xselect_script, log_file):
        
        with open(xselect_script, 'w') as file:
            for command in commands:
                file.write(command + '\n')

        with open(log_file, 'w') as logfile:
            subprocess.run(['xselect', '@' + xselect_script], 
                           check=True, 
                           stdout=logfile, 
                           stderr=logfile)


    def extract_events(self, std=True):
        
        evt_dir = os.path.dirname(self.evtfile) + '/events'
        
        if os.path.isdir(evt_dir):
            rm = input('events folder exist, remove? > yes[no] ')
            if rm == 'yes' or rm == '':
                shutil.rmtree(evt_dir)
            else:
                os.exit()
                
        os.mkdir(evt_dir)
        
        src_evtfile = evt_dir + '/src.evt'
        bkg_evtfile = evt_dir + '/bkg.evt'
        
        commands = ['xsel', 
                    'read events', 
                    os.path.dirname(self.evtfile), 
                    self.evtfile.split('/')[-1], 
                    'yes', 
                    f'filter region {self.regfile}', 
                    'extract events', 
                    f'save events {src_evtfile}', 
                    'no', 
                    'clear events', 
                    'clear region', 
                    f'filter region {self.bkregfile}', 
                    'extract events', 
                    f'save events {bkg_evtfile}', 
                    'no']
        
        stdout, stderr = self._run_xselect(commands)
        
        if std: 
            print(stdout)
            print(stderr)
            
            
    def extract_curve(self, bin, size=5, std=True):
        
        lc_dir = os.path.dirname(self.evtfile) + '/curve'
        
        if os.path.isdir(lc_dir):
            rm = input('curve folder exist, remove? > yes[no] ')
            if rm == 'yes' or rm == '':
                shutil.rmtree(lc_dir)
            else:
                os.exit()
                
        os.mkdir(lc_dir)
        
        src_lcfile = lc_dir + '/src.lc'
        bkg_lcfile = lc_dir + '/bkg.lc'
        
        scc_start = self.timezero + bin[0]
        scc_stop = self.timezero + bin[1]
        
        commands = ['xsel', 
                    'read events', 
                    os.path.dirname(self.evtfile), 
                    self.evtfile.split('/')[-1], 
                    'yes', 
                    'filter time scc', 
                    f'{scc_start}, {scc_stop}', 
                    'x', 
                    'filter pha_cutoff 50 400', 
                    f'filter region {self.regfile}', 
                    f'set binsize {size}', 
                    'extract curve', 
                    f'save curve {src_lcfile}', 
                    'clear region', 
                    f'filter region {self.bkregfile}', 
                    f'set binsize {size}', 
                    'extract curve', 
                    f'save curve {bkg_lcfile}']
        
        stdout, stderr = self._run_xselect(commands)
        
        if std: 
            print(stdout)
            print(stderr)
            
        src_hdu = fits.open(src_lcfile)
        src_timezero = src_hdu['RATE'].header['TIMEZERO']
        src_lc = Table.read(src_hdu['RATE'])
        src_time = src_lc['TIME'] + src_timezero - self.timezero
        src_rate = src_lc['RATE']
        src_error = src_lc['ERROR']
        src_hdu.close()
        
        bkg_hdu = fits.open(bkg_lcfile)
        bkg_timezero = bkg_hdu['RATE'].header['TIMEZERO']
        bkg_lc = Table.read(bkg_hdu['RATE'])
        bkg_time = bkg_lc['TIME'] + bkg_timezero - self.timezero
        bkg_rate = bkg_lc['RATE'] * self.regratio
        bkg_error = bkg_lc['ERROR'] * self.regratio
        
        self.fig = go.Figure()
        src = go.Scatter(x=src_time, 
                         y=src_rate, 
                         mode='markers', 
                         # line_shape='hvh', 
                         name='src', 
                         showlegend=True, 
                         error_y=dict(
                             type='data',
                             array=src_error,
                             thickness=1.5,
                             width=0), 
                         marker=dict(symbol='circle', size=3))
        bkg = go.Scatter(x=bkg_time, 
                         y=bkg_rate, 
                         mode='markers', 
                         # line_shape='hvh', 
                         name='bkg', 
                         showlegend=True, 
                         error_y=dict(
                             type='data',
                             array=bkg_error,
                             thickness=1.5,
                             width=0), 
                         marker=dict(symbol='circle', size=3))
        net = go.Scatter(x=src_time, 
                         y=src_rate - bkg_rate, 
                         mode='markers', 
                         # line_shape='hvh', 
                         name='net', 
                         showlegend=True, 
                         error_y=dict(
                             type='data',
                             array=np.sqrt(src_error ** 2 + bkg_error ** 2),
                             thickness=1.5,
                             width=0), 
                         marker=dict(symbol='circle', size=3))
        
        self.fig.add_trace(src)
        self.fig.add_trace(bkg)
        self.fig.add_trace(net)
        
        self.fig.update_xaxes(title_text=f'Time since {ep_met_to_utc(self.timezero)} (s)')
        self.fig.update_yaxes(title_text='Counts per second')
        self.fig.update_layout(template='plotly_white', height=600, width=800)
        
        self.fig.write_html(lc_dir + '/lcfig.html')


    def extract_spectrum(self, bins, std=True):
        
        spec_dir = os.path.dirname(self.evtfile) + '/spectrum'
        
        if os.path.isdir(spec_dir):
            rm = input('spectrum folder exist, remove? > yes[no] ')
            if rm == 'yes' or rm == '':
                shutil.rmtree(spec_dir)
            else:
                os.exit()
                
        os.mkdir(spec_dir)
            
        json.dump(self.timezero, open(spec_dir + '/timezero.json', 'w'), 
                  indent=4, cls=NpEncoder)
        
        json.dump(bins, open(spec_dir + '/bins.json', 'w'), 
                  indent=4, cls=NpEncoder)
        
        for i, bin in enumerate(bins):
        
            scc_start = self.timezero + bin[0]
            scc_stop = self.timezero + bin[1]
            
            src_specfile = spec_dir + '/int%02d' % i + '.src'
            bkg_specfile = spec_dir + '/int%02d' % i + '.bkg'
        
            commands = ['xsel', 
                        'read events', 
                        os.path.dirname(self.evtfile), 
                        self.evtfile.split('/')[-1], 
                        'yes', 
                        'filter time scc', 
                        f'{scc_start}, {scc_stop}', 
                        'x', 
                        f'filter region {self.regfile}', 
                        'extract spectrum', 
                        f'save spectrum {src_specfile}', 
                        'clear region', 
                        f'filter region {self.bkregfile}', 
                        'extract spectrum', 
                        f'save spectrum {bkg_specfile}']
        
            stdout, stderr = self._run_xselect(commands)
            
            if std: 
                print(stdout)
                print(stderr)
