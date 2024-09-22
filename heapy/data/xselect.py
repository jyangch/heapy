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
from ..util.time import *
from ..util.data import msg_format, NpEncoder



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
        
        self._regtxt = self._regfile[:-4] + '.txt'
        
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
        
        self._bkregtxt = self._bkregfile[:-4] + '.txt'
        
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


    def extract_events(self, time_slice, std=True, savepath=None):
        
        if savepath is None:
            savepath = os.path.dirname(self.evtfile) + '/events'
        
        if os.path.isdir(savepath):
            rm = input('savepath exist, remove? > yes[no] ')
            if rm == 'yes' or rm == '':
                shutil.rmtree(savepath)
            else:
                os.exit()
                
        os.mkdir(savepath)
        
        src_evtfile = savepath + '/src.evt'
        bkg_evtfile = savepath + '/bkg.evt'
        
        scc_start = self.timezero + time_slice[0]
        scc_stop = self.timezero + time_slice[1]
        
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
        
        self.src_evtfile = src_evtfile
        self.bkg_evtfile = bkg_evtfile
        
        if std: 
            print(stdout)
            print(stderr)
            
            
    def extract_curve(self, time_slice, time_binsize=5, pha_bin=[50, 400], std=True, savepath=None):
        
        if savepath is None:
            savepath = os.path.dirname(self.evtfile) + '/curve'
        
        if os.path.isdir(savepath):
            rm = input('savepath exist, remove? > yes[no] ')
            if rm == 'yes' or rm == '':
                shutil.rmtree(savepath)
            else:
                os.exit()
                
        os.mkdir(savepath)
        
        src_lcfile = savepath + '/src.lc'
        bkg_lcfile = savepath + '/bkg.lc'
        
        scc_start = self.timezero + time_slice[0]
        scc_stop = self.timezero + time_slice[1]
        
        pha_start, pha_stop = [int(pha) for pha in pha_bin]
        
        commands = ['xsel', 
                    'read events', 
                    os.path.dirname(self.evtfile), 
                    self.evtfile.split('/')[-1], 
                    'yes', 
                    'filter time scc', 
                    f'{scc_start}, {scc_stop}', 
                    'x', 
                    f'filter pha_cutoff {pha_start} {pha_stop}', 
                    f'filter region {self.regfile}', 
                    f'set binsize {time_binsize}', 
                    'extract curve', 
                    f'save curve {src_lcfile}', 
                    'clear region', 
                    f'filter region {self.bkregfile}', 
                    f'set binsize {time_binsize}', 
                    'extract curve', 
                    f'save curve {bkg_lcfile}']
        
        stdout, stderr = self._run_xselect(commands)
        
        self.src_lcfile = src_lcfile
        self.bkg_lcfile = bkg_lcfile
        
        if std: 
            print(stdout)
            print(stderr)
            
        src_hdu = fits.open(self.src_lcfile)
        src_timezero = src_hdu['RATE'].header['TIMEZERO']
        src_lc = Table.read(src_hdu['RATE'])
        src_time = src_lc['TIME'] + src_timezero - self.timezero
        src_rate = src_lc['RATE']
        src_error = src_lc['ERROR']
        src_hdu.close()
        
        bkg_hdu = fits.open(self.bkg_lcfile)
        bkg_timezero = bkg_hdu['RATE'].header['TIMEZERO']
        bkg_lc = Table.read(bkg_hdu['RATE'])
        bkg_time = bkg_lc['TIME'] + bkg_timezero - self.timezero
        bkg_rate = bkg_lc['RATE'] * self.regratio
        bkg_error = bkg_lc['ERROR'] * self.regratio
        bkg_hdu.close()
        
        net_rate = src_rate - bkg_rate
        net_cts = net_rate * time_binsize
        net_error = np.sqrt(src_error ** 2 + bkg_error ** 2)
        
        fig = go.Figure()
        src = go.Scatter(x=src_time, 
                         y=src_rate, 
                         mode='markers', 
                         name='src counts rate', 
                         showlegend=True, 
                         error_x=dict(
                             type='data',
                             array=np.ones_like(src_time) * time_binsize / 2, 
                             thickness=1.5,
                             width=0), 
                         error_y=dict(
                             type='data',
                             array=src_error,
                             thickness=1.5,
                             width=0), 
                         marker=dict(symbol='circle', size=3))
        bkg = go.Scatter(x=bkg_time, 
                         y=bkg_rate, 
                         mode='markers', 
                         name='bkg counts rate', 
                         showlegend=True, 
                         error_x=dict(
                             type='data',
                             array=np.ones_like(src_time) * time_binsize / 2, 
                             thickness=1.5,
                             width=0), 
                         error_y=dict(
                             type='data',
                             array=bkg_error,
                             thickness=1.5,
                             width=0), 
                         marker=dict(symbol='circle', size=3))
        net = go.Scatter(x=src_time, 
                         y=net_rate, 
                         mode='markers', 
                         name='net counts rate', 
                         showlegend=True, 
                         error_x=dict(
                             type='data',
                             array=np.ones_like(src_time) * time_binsize / 2, 
                             thickness=1.5,
                             width=0), 
                         error_y=dict(
                             type='data',
                             array=net_error,
                             thickness=1.5,
                             width=0), 
                         marker=dict(symbol='circle', size=3))
        
        fig.add_trace(src)
        fig.add_trace(bkg)
        fig.add_trace(net)

        fig.update_xaxes(title_text=f'Time since {ep_met_to_utc(self.timezero)} (s)')
        fig.update_yaxes(title_text=f'Counts per second (binsize={time_binsize} s)')
        fig.update_layout(template='plotly_white', height=600, width=800)
        fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))

        fig.write_html(savepath + '/lc.html')
        json.dump(fig.to_dict(), open(savepath + '/lc.json', 'w'), indent=4, cls=NpEncoder)

        net_ccts = np.cumsum(net_cts)
        
        fig = go.Figure()
        net = go.Scatter(x=src_time, 
                         y=net_ccts, 
                         mode='lines', 
                         name='net cumulated counts', 
                         showlegend=True)
        
        fig.add_trace(net)
        
        fig.update_xaxes(title_text=f'Time since {ep_met_to_utc(self.timezero)} (s)')
        fig.update_yaxes(title_text=f'Cumulated counts (binsize={time_binsize} s)')
        fig.update_layout(template='plotly_white', height=600, width=800)
        fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))
        
        fig.write_html(savepath + '/cum_lc.html')
        json.dump(fig.to_dict(), open(savepath + '/cum_lc.json', 'w'), indent=4, cls=NpEncoder)


    def extract_spectrum(self, time_slices, std=True, savepath=None):
        
        if savepath is None:
            savepath = os.path.dirname(self.evtfile) + '/spectrum'
        
        if os.path.isdir(savepath):
            rm = input('savepath exist, remove? > yes[no] ')
            if rm == 'yes' or rm == '':
                shutil.rmtree(savepath)
            else:
                os.exit()
                
        os.mkdir(savepath)
            
        json.dump(self.timezero, open(savepath + '/timezero.json', 'w'), indent=4, cls=NpEncoder)
        
        json.dump(time_slices, open(savepath + '/time_slices.json', 'w'), indent=4, cls=NpEncoder)
        
        lslices = np.array(time_slices)[:, 0]
        rslices = np.array(time_slices)[:, 1]
        
        for l, r in zip(lslices, rslices):
            scc_start = self.timezero + l
            scc_stop = self.timezero + r
            
            new_l = '{:+f}'.format(l).replace('-', 'm').replace('.', 'd').replace('+', 'p')
            new_r = '{:+f}'.format(r).replace('-', 'm').replace('.', 'd').replace('+', 'p')
            
            file_name = '-'.join(new_l, new_r)
            
            src_specfile = savepath + f'/{file_name}.src'
            bkg_specfile = savepath + f'/{file_name}.bkg'
        
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


    @property
    def src_ts(self):
        
        try:
            src_hdu = fits.open(self.src_evtfile)
            
        except:
            raise AttributeError('please extract events first')
        
        else:
            src_data = src_hdu['EVENTS'].data
            src_evt = src_data['TIME'] - self.timezero
        
            return src_evt
        
        
    @property
    def bkg_ts(self):
        
        try:
            bkg_hdu = fits.open(self.bkg_evtfile)
            
        except:
            raise AttributeError('please extract events first')
        
        else:
            bkg_data = bkg_hdu['EVENTS'].data
            bkg_evt = bkg_data['TIME'] - self.timezero
        
            return bkg_evt
        