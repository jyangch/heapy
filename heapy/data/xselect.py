import os
import json
import shutil
import warnings
import subprocess
import numpy as np
from astropy.io import fits
import plotly.graph_objs as go
from .retrieve import epRetrieve
from ..temporal.txx import ppTxx
from ..autobs.ppsignal import ppSignal
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
    def from_wxtobs(cls, obsname, srcid):
        
        rtv = epRetrieve.from_wxtobs(obsname, srcid)

        evtfile = rtv.rtv_res['evt']
        regfile = rtv.rtv_res['reg']
        bkregfile = rtv.rtv_res['bkreg']
        
        return cls(evtfile, regfile, bkregfile)
    
    
    @property
    def evtfile(self):
        
        return os.path.abspath(self._evtfile)
    
    
    @evtfile.setter
    def evtfile(self, new_evtfile):
        
        self._evtfile = new_evtfile
        
        hdu = fits.open(self._evtfile)
        tstart = hdu['EVENTS'].header['TSTART']
        hdu.close()
        
        self._timezero = tstart
        
        
    @property
    def regfile(self):
        
        return os.path.abspath(self._regfile)
    
    
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
        
        return os.path.abspath(self._bkregfile)
    
    
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
    
    
    @property
    def src_ts(self):
        
        try:
            src_hdu = fits.open(self.src_evtfile)
            
        except AttributeError:
            raise AttributeError('no src event file')
        
        else:
            src_data = src_hdu['EVENTS'].data
            src_ts = src_data['TIME'] - self.timezero
        
            return src_ts
        
        
    @property
    def bkg_ts(self):
        
        try:
            bkg_hdu = fits.open(self.bkg_evtfile)
            
        except AttributeError:
            raise AttributeError('no bkg event file')
        
        else:
            bkg_data = bkg_hdu['EVENTS'].data
            bkg_ts = bkg_data['TIME'] - self.timezero
        
            return bkg_ts
        
        
    @property
    def lc_p1p2(self):
        
        try:
            self._lc_p1p2
        except AttributeError:
            return [50, 400]
        else:
            if self._lc_p1p2 is not None:
                return self._lc_p1p2
            else:
                return [50, 400]
        
        
    @lc_p1p2.setter
    def lc_p1p2(self, new_lc_p1p2):
        
        if isinstance(new_lc_p1p2, (list, type(None))):
            self._lc_p1p2 = new_lc_p1p2
        else:
            raise ValueError('not expected lc_t1t2 type')
        
        
    @property
    def lc_t1t2(self):
        
        try:
            self._lc_t1t2
        except AttributeError:
            return [np.min(self.src_ts), np.max(self.src_ts)]
        else:
            if self._lc_t1t2 is not None:
                return self._lc_t1t2
            else:
                return [np.min(self.src_ts), np.max(self.src_ts)]
        
        
    @lc_t1t2.setter
    def lc_t1t2(self, new_lc_t1t2):
        
        if isinstance(new_lc_t1t2, (list, type(None))):
            self._lc_t1t2 = new_lc_t1t2
        else:
            raise ValueError('not expected lc_t1t2 type')
    
    
    @property
    def lc_interval(self):
        
        return self.lc_t1t2[1] - self.lc_t1t2[0]
    
    
    @property
    def lc_binsize(self):
        
        try:
            self._lc_binsize
        except AttributeError:
            return self.lc_interval / 300
        else:
            if self._lc_binsize is not None:
                return self._lc_binsize
            else:
                return self.lc_interval / 300
            
            
    @lc_binsize.setter
    def lc_binsize(self, new_lc_binsize):
        
        self._lc_binsize = new_lc_binsize
        
        
    @property
    def lc_bins(self):
        
        return np.arange(self.lc_t1t2[0], self.lc_t1t2[1] + 1e-5, self.lc_binsize)
    
    
    @property
    def lc_bin_list(self):
        
        lbins, rbins = self.lc_bins[:-1], self.lc_bins[1:]
        
        return np.vstack((lbins, rbins)).T
    
    
    @property
    def lc_time(self):
        
        return np.mean(self.lc_bin_list, axis=1)
    
    
    @property
    def lc_time_err(self):
        
        lbins, rbins = self.lc_bins[:-1], self.lc_bins[1:]
        
        return (rbins - lbins) / 2
    
    
    @property
    def src_cts(self):
        
        return np.histogram(self.src_ts, bins=self.lc_bins)[0]
    
    
    @property
    def src_cts_err(self):
        
        return np.sqrt(self.src_cts)
    
    
    @property
    def src_rate(self):
        
        return self.src_cts / self.lc_binsize
    
    
    @property
    def src_rate_err(self):
        
        return self.src_cts_err / self.lc_binsize
    
    
    @property
    def bkg_cts(self):
        
        return np.histogram(self.bkg_ts, bins=self.lc_bins)[0]
    
    
    @property
    def bkg_cts_err(self):
        
        return np.sqrt(self.bkg_cts)
        
        
    @property
    def bkg_rate(self):
        
        return self.bkg_cts / self.lc_binsize * self.regratio
    
    
    @property
    def bkg_rate_err(self):
        
        return self.bkg_cts_err / self.lc_binsize * self.regratio
    
    
    @property
    def net_rate(self):
        
        return self.src_rate - self.bkg_rate
    
    
    @property
    def net_rate_err(self):
        
        return np.sqrt(self.src_rate_err ** 2 + self.bkg_rate_err ** 2)
    
    
    @property
    def net_cts(self):
        
        return self.net_rate * self.lc_binsize
    
    
    @property
    def net_cts_err(self):
        
        return self.net_rate_err * self.lc_binsize
    
    
    @property
    def net_ccts(self):
        
        return np.cumsum(self.net_cts)
    
    
    @property
    def lc_ps(self):
        
        lc_ps = ppSignal(self.src_ts, self.bkg_ts, self.lc_bins, backscale=self.regratio)
        lc_ps.loop(sigma=3)
        
        return lc_ps
            
            
    def extract_curve(self, std=False, savepath='./curve', show=False):
            
        savepath = os.path.abspath(savepath)
        
        if os.path.isdir(savepath):
            shutil.rmtree(savepath)
                
        os.mkdir(savepath)
        
        scc_start = self.timezero + self.lc_t1t2[0]
        scc_stop = self.timezero + self.lc_t1t2[1]
        
        pha_start, pha_stop = [int(pha) for pha in self.lc_p1p2]
        
        src_evtfile = savepath + '/src.evt'
        bkg_evtfile = savepath + '/bkg.evt'
        
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
        
        self.lc_ps.save(savepath=savepath + '/ppsignal')
        
        fig = go.Figure()
        src = go.Scatter(x=self.lc_time, 
                         y=self.src_rate, 
                         mode='markers', 
                         name='src counts rate', 
                         showlegend=True, 
                         error_x=dict(
                             type='data',
                             array=self.lc_time_err, 
                             thickness=1.5,
                             width=0), 
                         error_y=dict(
                             type='data',
                             array=self.src_rate_err,
                             thickness=1.5,
                             width=0), 
                         marker=dict(symbol='circle', size=3))
        bkg = go.Scatter(x=self.lc_time, 
                         y=self.bkg_rate, 
                         mode='markers', 
                         name='bkg counts rate', 
                         showlegend=True, 
                         error_x=dict(
                             type='data',
                             array=self.lc_time_err, 
                             thickness=1.5,
                             width=0), 
                         error_y=dict(
                             type='data',
                             array=self.bkg_rate_err,
                             thickness=1.5,
                             width=0), 
                         marker=dict(symbol='circle', size=3))
        net = go.Scatter(x=self.lc_time, 
                         y=self.net_rate, 
                         mode='markers', 
                         name='net counts rate', 
                         showlegend=True, 
                         error_x=dict(
                             type='data',
                             array=self.lc_time_err, 
                             thickness=1.5,
                             width=0), 
                         error_y=dict(
                             type='data',
                             array=self.net_rate_err,
                             thickness=1.5,
                             width=0), 
                         marker=dict(symbol='circle', size=3))
        
        fig.add_trace(src)
        fig.add_trace(bkg)
        fig.add_trace(net)

        fig.update_xaxes(title_text=f'Time since {ep_met_to_utc(self.timezero)} (s)')
        fig.update_yaxes(title_text=f'Counts per second (binsize={self.lc_binsize} s)')
        fig.update_layout(template='plotly_white', height=600, width=800)
        fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))

        if show: fig.show()
        fig.write_html(savepath + '/lc.html')
        json.dump(fig.to_dict(), open(savepath + '/lc.json', 'w'), indent=4, cls=NpEncoder)
        
        fig = go.Figure()
        net = go.Scatter(x=self.lc_time, 
                         y=self.net_ccts, 
                         mode='lines', 
                         name='net cumulated counts', 
                         showlegend=True)
        
        fig.add_trace(net)
        
        fig.update_xaxes(title_text=f'Time since {ep_met_to_utc(self.timezero)} (s)')
        fig.update_yaxes(title_text=f'Cumulated counts (binsize={self.lc_binsize} s)')
        fig.update_layout(template='plotly_white', height=600, width=800)
        fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))
        
        fig.write_html(savepath + '/cum_lc.html')
        json.dump(fig.to_dict(), open(savepath + '/cum_lc.json', 'w'), indent=4, cls=NpEncoder)
        
        
    def calculate_txx(self, sigma=3, mp=True, xx=0.9, pstart=None, pstop=None, savepath='./curve/duration'):
            
        savepath = os.path.abspath(savepath)
        
        if os.path.isdir(savepath):
            shutil.rmtree(savepath)
                
        os.mkdir(savepath)
        
        txx = ppTxx(self.src_ts, self.bkg_ts, self.lc_bins, self.regratio)
        txx.findpulse(sigma=sigma, mp=mp)
        txx.accumcts(xx=xx, pstart=pstart, pstop=pstop)
        txx.save(savepath=savepath)


    def extract_spectrum(self, spec_slices, std=False, savepath='./spectrum'):
            
        savepath = os.path.abspath(savepath)
        
        if os.path.isdir(savepath):
            shutil.rmtree(savepath)
                
        os.mkdir(savepath)
            
        json.dump(self.timezero, open(savepath + '/timezero.json', 'w'), indent=4, cls=NpEncoder)
        
        json.dump(spec_slices, open(savepath + '/spec_slices.json', 'w'), indent=4, cls=NpEncoder)
        
        lslices = np.array(spec_slices)[:, 0]
        rslices = np.array(spec_slices)[:, 1]
        
        for l, r in zip(lslices, rslices):
            scc_start = self.timezero + l
            scc_stop = self.timezero + r
            
            new_l = '{:+.2f}'.format(l).replace('-', 'm').replace('.', 'd').replace('+', 'p')
            new_r = '{:+.2f}'.format(r).replace('-', 'm').replace('.', 'd').replace('+', 'p')
            
            file_name = '-'.join([new_l, new_r])
            
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
