import os
import shutil
import subprocess
import numpy as np
from astropy import table
from astropy.io import fits
import plotly.graph_objs as go
from plotly.subplots import make_subplots
docs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/docs'
from .filter import Filter
from ..data.retrieve import epRetrieve, swiftRetrieve
from ..temp.txx import ppTxx
from ..auto.ppsignal import ppSignal
from ..util.file import copy
from ..util.data import json_dump, rebin, union
from ..util.time import ep_met_to_utc, ep_utc_to_met
from ..util.time import swift_met_to_utc, swift_utc_to_met



class Image(object):
    
    os.environ["HEADASNOQUERY"] = "1"
    
    def __init__(self, 
                 file, 
                 regfile, 
                 bkregfile
                 ):
        
        self._file = file
        self._regfile = regfile
        self._bkregfile = bkregfile
        
        self.prefix = ''
        
        self._ini_xselect()


    @property
    def file(self):
        
        return os.path.abspath(self._file)


    @file.setter
    def file(self, new_file):
        
        self._file = new_file
        
        self._ini_xselect()
        
        
    @property
    def regfile(self):
        
        return os.path.abspath(self._regfile)
    
    
    @regfile.setter
    def regfile(self, new_regfile):
        
        self._regfile = new_regfile
        
        self._ini_xselect()
        
        
    @property
    def bkregfile(self):
        
        return os.path.abspath(self._bkregfile)
    
    
    @bkregfile.setter
    def bkregfile(self, new_bkregfile):
        
        self._bkregfile = new_bkregfile
        
        self._ini_xselect()
        
        
    @staticmethod
    def _run_xselect(commands):
        
        process = subprocess.Popen('xselect', 
                                   stdin=subprocess.PIPE, 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True)

        stdout, stderr = process.communicate(input='\n'.join(commands) + '\nexit\nno\n')

        return stdout, stderr
    
    
    @staticmethod
    def _run_ximage(commands):
        
        process = subprocess.Popen('ximage', 
                                   stdin=subprocess.PIPE, 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True)

        stdout, stderr = process.communicate(input='\n'.join(commands))

        return stdout, stderr
    
    
    def _run_commands(self, commands):
        
        process = subprocess.Popen(commands, 
                                   stdin=subprocess.PIPE, 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True)

        stdout, stderr = process.communicate()

        return stdout, stderr
        
        
    def _ini_xselect(self):
        
        curve_savepath = os.path.dirname(self.file) + f'/{self.prefix}curve'

        self.src_curvefile = curve_savepath + '/src.lc'
        self.bkg_curvefile = curve_savepath + '/bkg.lc'

        spectra_savepath = os.path.dirname(self.file) + f'/{self.prefix}spectra'
            
        self.src_specfile = spectra_savepath + '/src.pi'
        self.bkg_specfile = spectra_savepath + '/bkg.pi'

        events_savepath = os.path.dirname(self.file) + f'/{self.prefix}events'

        self.src_evtfile = events_savepath + '/src.evt'
        self.bkg_evtfile = events_savepath + '/bkg.evt'

        if os.path.exists(curve_savepath):
            shutil.rmtree(curve_savepath)
        os.makedirs(curve_savepath)
            
        if os.path.exists(spectra_savepath):
            shutil.rmtree(spectra_savepath)
        os.makedirs(spectra_savepath)

        if os.path.exists(events_savepath):
            shutil.rmtree(events_savepath)
        os.makedirs(events_savepath)

        commands = ['xsel', 
                    'read events', 
                    os.path.dirname(self.file), 
                    self.file.split('/')[-1], 
                    'yes', 
                    f'filter region {self.regfile}', 
                    'extract curve',
                    f'save curve {self.src_curvefile}',
                    'extract spectrum', 
                    f'save spectrum {self.src_specfile}', 
                    'extract events', 
                    f'save events {self.src_evtfile}', 
                    'no', 
                    'clear events', 
                    'clear region', 
                    f'filter region {self.bkregfile}', 
                    'extract curve',
                    f'save curve {self.bkg_curvefile}',
                    'extract spectrum', 
                    f'save spectrum {self.bkg_specfile}', 
                    'extract events', 
                    f'save events {self.bkg_evtfile}', 
                    'no']
        
        _, _ = self._run_xselect(commands)
        
        src_hdu = fits.open(self.src_specfile)
        self.src_backscale = src_hdu['SPECTRUM'].header['BACKSCAL']
        src_hdu.close()
        
        bkg_hdu = fits.open(self.bkg_specfile)
        self.bkg_backscale = bkg_hdu['SPECTRUM'].header['BACKSCAL']
        bkg_hdu.close()
        
        self.backscale = self.src_backscale / self.bkg_backscale

        hdu = fits.open(self.file)
        self._event = table.Table.read(hdu['EVENTS'])
        self._gti = table.Table.read(hdu['GTI'])
        self._timezero = hdu['EVENTS'].header['TSTART']
        self._filter = Filter(self._event)
        hdu.close()
        
        src_hdu = fits.open(self.src_evtfile)
        self._src_event = table.Table.read(src_hdu['EVENTS'])
        self._src_filter = Filter(self._src_event)
        src_hdu.close()
        
        bkg_hdu = fits.open(self.bkg_evtfile)
        self._bkg_event = table.Table.read(bkg_hdu['EVENTS'])
        self._bkg_filter = Filter(self._bkg_event)
        bkg_hdu.close()
        
        self._filter_info = {'time': None, 'pi': None, 'tag': None}
        
        
    @property
    def event(self):
        
        return self._filter.evt
        
        
    @property
    def src_event(self):
        
        return self._src_filter.evt
    
    
    @property
    def bkg_event(self):
        
        return self._bkg_filter.evt
    
    
    @property
    def gti(self):

        tstart = self._gti['START'] - self.timezero
        tstop = self._gti['STOP'] - self.timezero

        return np.array(union(np.vstack((tstart, tstop)).T))
        
        
    @property
    def timezero(self):
        
        return self._timezero

    
    @timezero.setter
    def timezero(self, new_timezero):
        
        self._timezero = new_timezero
        
        
    @property
    def timezero_utc(self):
        
        return None
    
    
    @property
    def psf_modelfile(self):
        
        return None
    
    
    @property
    def psf_fitradius(self):
        
        return None
        
        
    def slice_time(self, t1t2):
        
        t1, t2 = t1t2
        
        met_t1 = self.timezero + t1
        met_t2 = self.timezero + t2
        
        met_ts = self._event['TIME']
        flt = (met_ts >= met_t1) & (met_ts <= met_t2)
        self._event = self._event[flt]
        
        src_met_ts = self._src_event['TIME']
        flt = (src_met_ts >= met_t1) & (src_met_ts <= met_t2)
        self._src_event = self._src_event[flt]
        
        bkg_met_ts = self._bkg_event['TIME']
        flt = (bkg_met_ts >= met_t1) & (bkg_met_ts <= met_t2)
        self._bkg_event = self._bkg_event[flt]
        
        self._clear_filter()
        
        
    def filter_time(self, t1t2):
        
        if t1t2 is None:
            expr = None
            
        elif isinstance(t1t2, list):
            t1, t2 = t1t2
            
            met_t1 = self.timezero + t1
            met_t2 = self.timezero + t2
                
            expr = f'(TIME >= {met_t1}) * (TIME <= {met_t2})'
            
        else:
            raise ValueError('t1t2 is extected to be list or None')
        
        self._time_filter = t1t2
        
        self._filter_info['time'] = expr
        
        self._filter_update()
        
        
    def filter_pi(self, p1p2):
        
        if p1p2 is None:
            expr = None
            
        elif isinstance(p1p2, list):
            p1, p2 = p1p2
            expr = f'(PI >= {p1}) * (PI <= {p2})'
            
        else:
            raise ValueError('p1p2 is extected to be list or None')
        
        self._pi_filter = p1p2
        
        self._filter_info['pi'] = expr
        
        self._filter_update()
        
        
    def filter(self, expr):
        
        self._filter_info['tag'] = expr
        
        self._filter_update()
        
        
    def _filter_update(self):
        
        self._clear_filter()
        
        self._filter.eval(self._filter_info['time'])
        self._filter.eval(self._filter_info['pi'])
        self._filter.eval(self._filter_info['tag'])
        
        self._src_filter.eval(self._filter_info['time'])
        self._src_filter.eval(self._filter_info['pi'])
        self._src_filter.eval(self._filter_info['tag'])
        
        self._bkg_filter.eval(self._filter_info['time'])
        self._bkg_filter.eval(self._filter_info['pi'])
        self._bkg_filter.eval(self._filter_info['tag'])
        
        
    def _clear_filter(self):
        
        self._filter.clear()
        self._src_filter.clear()
        self._bkg_filter.clear()
        

    @property
    def filter_info(self):
        
        return self._filter_info
    
    
    @property
    def time_filter(self):
        
        if self._filter_info['time'] is None:
            return [np.floor(np.min(self._event['TIME'])) - self.timezero, 
                    np.ceil(np.max(self._event['TIME'])) - self.timezero]
        else:
            return self._time_filter
        
        
    @property
    def pi_filter(self):
        
        if self._filter_info['pi'] is None:
            return [np.min(self._event['PI']), np.max(self._event['PI'])]
        else:
            return self._pi_filter

        
    def extract_image(self, savepath='./image', show=False, std=False):
        
        savepath = os.path.abspath(savepath)
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        
        self.imagefile = savepath + '/image.img'
        
        if os.path.exists(self.imagefile):
            os.remove(self.imagefile)
        
        scc_start = self.timezero + self.time_filter[0]
        scc_stop = self.timezero + self.time_filter[1]
        
        pi_start = self.pi_filter[0]
        pi_stop = self.pi_filter[1]
    
        commands = ['xsel', 
                    'read events', 
                    os.path.dirname(self.file), 
                    self.file.split('/')[-1], 
                    'yes', 
                    'filter time scc', 
                    f'{scc_start}, {scc_stop}', 
                    'x', 
                    f'filter pha_cutoff {pi_start} {pi_stop}',
                    'extract image', 
                    f'save image {self.imagefile}']
    
        stdout, stderr = self._run_xselect(commands)
        
        if std: 
            print(stdout)
            print(stderr)
        
        H, xedges, yedges = np.histogram2d(self.event['X'], self.event['Y'], bins=128)
        H[H == 0] = 1
        
        fig = go.Figure()
        image = go.Heatmap(x=xedges, 
                           y=yedges,
                           z=np.log10(H.T),
                           colorscale='Jet')
        fig.add_trace(image)
        
        fig.update_layout(template='plotly_white', height=700, width=700)

        if show: fig.show()
        fig.write_html(savepath + '/image.html')
        json_dump(fig.to_dict(), savepath + '/image.json')


    @property
    def src_ts(self):
    
        return np.array(self.src_event['TIME']) - self.timezero


    @property
    def bkg_ts(self):
    
        return np.array(self.bkg_event['TIME']) - self.timezero
    
    
    @property
    def lc_t1t2(self):
        
        try:
            self._lc_t1t2
        except AttributeError:
            return self.time_filter
        else:
            if self._lc_t1t2 is not None:
                return self._lc_t1t2
            else:
                return self.time_filter
        
        
    @lc_t1t2.setter
    def lc_t1t2(self, new_lc_t1t2):
        
        if isinstance(new_lc_t1t2, (list, type(None))):
            self._lc_t1t2 = new_lc_t1t2
        else:
            raise ValueError('lc_t1t2 is extected to be list or None')


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
        
        if isinstance(new_lc_binsize, (int, float, type(None))):
            self._lc_binsize = new_lc_binsize
        else:
            raise ValueError('lc_binsize is extected to be int, float or None')


    @property
    def lc_src_ts(self):
        
        idx = (self.src_ts >= self.lc_t1t2[0]) & (self.src_ts <= self.lc_t1t2[1])
        
        return self.src_ts[idx]
    
    
    @property
    def lc_bkg_ts(self):
        
        idx = (self.bkg_ts >= self.lc_t1t2[0]) & (self.bkg_ts <= self.lc_t1t2[1])
        
        return self.bkg_ts[idx]
    
    
    @property
    def lc_bins(self):
        
        return np.arange(self.lc_t1t2[0], self.lc_t1t2[1] + 1e-5, self.lc_binsize)


    @property
    def lc_bin_list(self):
        
        lbins, rbins = self.lc_bins[:-1], self.lc_bins[1:]
        
        return np.vstack((lbins, rbins)).T


    @property
    def lc_mask(self):

        return np.any((self.lc_bin_list[:, None, 0] >= self.gti[:, 0]) & 
                      (self.lc_bin_list[:, None, 1] <= self.gti[:, 1]), axis=1)
        
        
    @property
    def lc_mask_bin_list(self):
        
        return self.lc_bin_list[self.lc_mask]


    @property
    def lc_exps(self):
        
        return self.lc_binsize
    
    
    @property
    def lc_mask_exps(self):
        
        return self.lc_exps[self.lc_mask]
    
    
    @property
    def lc_time(self):
        
        return np.mean(self.lc_bin_list, axis=1)
    
    
    @property
    def lc_mask_time(self):
        
        return self.lc_time[self.lc_mask]
    
    
    @property
    def lc_time_err(self):
        
        lbins, rbins = self.lc_bins[:-1], self.lc_bins[1:]
        
        return (rbins - lbins) / 2
    
    
    @property
    def lc_mask_time_err(self):
        
        return self.lc_time_err[self.lc_mask]


    @property
    def lc_src_cts(self):
        
        return np.histogram(self.lc_src_ts, bins=self.lc_bins)[0]
    
    
    @property
    def lc_mask_src_cts(self):
        
        return self.lc_src_cts[self.lc_mask]
    
    
    @property
    def lc_src_cts_err(self):
        
        return np.sqrt(self.lc_src_cts)
    
    
    @property
    def lc_mask_src_cts_err(self):

        return self.lc_src_cts_err[self.lc_mask]


    @property
    def lc_src_rate(self):
        
        return self.lc_src_cts / self.lc_exps
    
    
    @property
    def lc_mask_src_rate(self):
        
        return self.lc_src_rate[self.lc_mask]
    
    
    @property
    def lc_src_rate_err(self):
        
        return self.lc_src_cts_err / self.lc_exps
    
    
    @property
    def lc_mask_src_rate_err(self):
        
        return self.lc_src_rate_err[self.lc_mask]
    
    
    @property
    def lc_bkg_cts(self):
        
        return np.histogram(self.lc_bkg_ts, bins=self.lc_bins)[0]
    
    
    @property
    def lc_mask_bkg_cts(self):
        
        return self.lc_bkg_cts[self.lc_mask]
    
    
    @property
    def lc_bkg_cts_err(self):
        
        return np.sqrt(self.lc_bkg_cts)
    
    
    @property
    def lc_mask_bkg_cts_err(self):
        
        return self.lc_bkg_cts_err[self.lc_mask]
        
        
    @property
    def lc_bkg_rate(self):
        
        return self.lc_bkg_cts / self.lc_exps * self.backscale
    
    
    @property
    def lc_mask_bkg_rate(self):
        
        return self.lc_bkg_rate[self.lc_mask]
    
    
    @property
    def lc_bkg_rate_err(self):
        
        return self.lc_bkg_cts_err / self.lc_exps * self.backscale
    
    
    @property
    def lc_mask_bkg_rate_err(self):
        
        return self.lc_bkg_rate_err[self.lc_mask]
    
    
    @property
    def lc_net_rate(self):
        
        return self.lc_src_rate - self.lc_bkg_rate
    
    
    @property
    def lc_mask_net_rate(self):
        
        return self.lc_net_rate[self.lc_mask]
    
    
    @property
    def lc_net_rate_err(self):
        
        return np.sqrt(self.lc_src_rate_err ** 2 + self.lc_bkg_rate_err ** 2)
    
    
    @property
    def lc_mask_net_rate_err(self):
        
        return self.lc_net_rate_err[self.lc_mask]
    
    
    @property
    def lc_net_cts(self):
        
        return self.lc_net_rate * self.lc_exps
    
    
    @property
    def lc_mask_net_cts(self):
        
        return self.lc_net_cts[self.lc_mask]
    
    
    @property
    def lc_net_cts_err(self):
        
        return self.lc_net_rate_err * self.lc_exps
    
    
    @property
    def lc_mask_net_cts_err(self):
        
        return self.lc_net_cts_err[self.lc_mask]
    
    
    @property
    def lc_net_ccts(self):
        
        return np.cumsum(self.lc_net_cts)
    
    
    @property
    def ps_sigma(self):
        
        try:
            self._ps_sigma
        except AttributeError:
            return 3
        else:
            return self._ps_sigma


    @ps_sigma.setter
    def ps_sigma(self, new_ps_sigma):
        
        if isinstance(new_ps_sigma, (int, float)):
            self._ps_sigma = new_ps_sigma
        else:
            raise ValueError('ps_sigma is extected to be int or float')
    
    
    @property
    def lc_ps(self):
        
        lc_ps = ppSignal(self.src_ts, self.bkg_ts, self.lc_bins, backscale=self.backscale)
        lc_ps.loop(sigma=self.ps_sigma)
        
        return lc_ps
        
        
    def extract_curve(self, savepath='./curve', sig=True, show=False):
        
        savepath = os.path.abspath(savepath)
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        
        if sig: self.lc_ps.save(savepath=savepath + '/ppsignal')
        
        fig = go.Figure()
        src = go.Scatter(x=self.lc_mask_time, 
                         y=self.lc_mask_src_rate, 
                         mode='markers', 
                         name='src counts rate', 
                         showlegend=True, 
                         error_x=dict(
                             type='data',
                             array=self.lc_mask_time_err, 
                             thickness=1.5,
                             width=0), 
                         error_y=dict(
                             type='data',
                             array=self.lc_mask_src_rate_err,
                             thickness=1.5,
                             width=0), 
                         marker=dict(symbol='circle', size=3))
        bkg = go.Scatter(x=self.lc_mask_time, 
                         y=self.lc_mask_bkg_rate, 
                         mode='markers', 
                         name='bkg counts rate', 
                         showlegend=True, 
                         error_x=dict(
                             type='data',
                             array=self.lc_mask_time_err,
                             thickness=1.5,
                             width=0), 
                         error_y=dict(
                             type='data',
                             array=self.lc_mask_bkg_rate_err,
                             thickness=1.5,
                             width=0), 
                         marker=dict(symbol='circle', size=3))
        net = go.Scatter(x=self.lc_mask_time, 
                         y=self.lc_mask_net_rate, 
                         mode='markers', 
                         name='net counts rate', 
                         showlegend=True, 
                         error_x=dict(
                             type='data',
                             array=self.lc_mask_time_err, 
                             thickness=1.5,
                             width=0), 
                         error_y=dict(
                             type='data',
                             array=self.lc_mask_net_rate_err,
                             thickness=1.5,
                             width=0), 
                         marker=dict(symbol='circle', size=3))
        
        fig.add_trace(src)
        fig.add_trace(bkg)
        fig.add_trace(net)

        fig.update_xaxes(title_text=f'Time since {self.timezero_utc} (s)', range=self.lc_t1t2)
        fig.update_yaxes(title_text=f'Counts per second (binsize={self.lc_binsize} s)')
        fig.update_layout(template='plotly_white', height=600, width=800)
        fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))

        if show: fig.show()
        fig.write_html(savepath + '/lc.html')
        json_dump(fig.to_dict(), savepath + '/lc.json')
        
        fig = go.Figure()
        net = go.Scatter(x=self.lc_time, 
                         y=self.lc_net_ccts, 
                         mode='lines', 
                         name='net cumulated counts', 
                         showlegend=True)
        
        fig.add_trace(net)
        
        fig.update_xaxes(title_text=f'Time since {self.timezero_utc} (s)', range=self.lc_t1t2)
        fig.update_yaxes(title_text=f'Cumulated counts (binsize={self.lc_binsize} s)')
        fig.update_layout(template='plotly_white', height=600, width=800)
        fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))
        
        fig.write_html(savepath + '/cum_lc.html')
        json_dump(fig.to_dict(), savepath + '/cum_lc.json')

        
    def calculate_txx(self, mp=True, xx=0.9, pstart=None, pstop=None, 
                      lbkg=None, rbkg=None, savepath='./curve/duration'):
            
        savepath = os.path.abspath(savepath)
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        
        txx = ppTxx(self.src_ts, self.bkg_ts, self.lc_bins, self.backscale)
        txx.findpulse(sigma=self.ps_sigma, mp=mp)
        txx.accumcts(xx=xx, pstart=pstart, pstop=pstop, lbkg=lbkg, rbkg=rbkg)
        txx.save(savepath=savepath)
        
        
    def extract_rebin_curve(self, trange=None, min_sigma=None, min_evt=None, max_bin=None, 
                            savepath='./curve', loglog=False, show=False):
        
        savepath = os.path.abspath(savepath)
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)
            
        if trange is not None:
            idx = (self.lc_bin_list[:, 0] >= trange[0]) * (self.lc_bin_list[:, 1] <= trange[1])
        else:
            idx = np.ones(len(self.lc_bin_list), dtype=bool)
        
        self.lc_rebin_list, self.lc_src_rects, self.lc_src_rects_err, \
            self.lc_bkg_rebcts, self.lc_bkg_rebcts_err = \
                rebin(
                    self.lc_bin_list[idx], 
                    'cstat', 
                    self.lc_src_cts[idx], 
                    cts_err=self.lc_src_cts_err[idx],
                    bcts=self.lc_bkg_cts[idx], 
                    bcts_err=self.lc_bkg_cts_err[idx], 
                    min_sigma=min_sigma, 
                    min_evt=min_evt, 
                    max_bin=max_bin,
                    backscale=self.backscale)
                
        self.lc_retime = np.mean(self.lc_rebin_list, axis=1)
        self.lc_rebinsize = self.lc_rebin_list[:, 1] - self.lc_rebin_list[:, 0]
        self.lc_net_rects = self.lc_src_rects - self.lc_bkg_rebcts * self.backscale
        self.lc_net_rects_err = np.sqrt(self.lc_src_rects_err ** 2 + (self.lc_bkg_rebcts_err * self.backscale) ** 2)
        self.lc_net_rerate = self.lc_net_rects / self.lc_rebinsize
        self.lc_net_rerate_err = self.lc_net_rects_err / self.lc_rebinsize
        
        fig = go.Figure()
        net = go.Scatter(x=self.lc_retime, 
                         y=self.lc_net_rerate, 
                         mode='lines+markers', 
                         name='net lightcurve', 
                         showlegend=True, 
                         error_y=dict(
                             type='data',
                             array=self.lc_net_rerate_err,
                             thickness=1.5,
                             width=0), 
                         marker=dict(symbol='cross-thin', size=0))
        fig.add_trace(net)
        
        if loglog: 
            fig.update_xaxes(title_text=f'Time since {self.timezero_utc} (s)', type='log')
            fig.update_yaxes(title_text='Counts per second', type='log')
        else:
            fig.update_xaxes(title_text=f'Time since {self.timezero_utc} (s)')
            fig.update_yaxes(title_text='Counts per second')
        fig.update_layout(template='plotly_white', height=600, width=800)
        fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))
        
        if show: fig.show()
        fig.write_html(savepath + '/rebin_lc.html')
        json_dump(fig.to_dict(), savepath + '/rebin_lc.json')
        
        
    def check_pileup(self, std=False, show=False):
        
        psf_savepath = os.path.dirname(self.file) + f'/psf'

        psf_savepath = os.path.abspath(psf_savepath)

        if not os.path.exists(psf_savepath):
            os.makedirs(psf_savepath)
        
        try:
            imagefile = self.imagefile
        except:
            imagefile = self.file

        qdpfile = psf_savepath + '/psf.qdp'
        
        if os.path.exists(qdpfile):
            os.remove(qdpfile)
            
        modfile = psf_savepath + '/psf.mod'
        
        if os.path.exists(modfile):
            os.remove(modfile)
        
        commands = [f'read {imagefile}', 
                    'cpd /xtk', 
                    'disp', 
                    'back', 
                    'psf/cur',
                    'col off 1 2 3 4 6',
                    f'rescale x {self.psf_fitradius}',
                    f'model {self.psf_modelfile}',
                    '\n',
                    'fit',
                    'rescale',
                    'plot', 
                    f'wdata {qdpfile[:-4]}',
                    f'wmodel {modfile[:-4]}',
                    'exit',
                    'exit']

        work_path = os.getcwd()
        os.chdir(docs_path + '/psf_model')
        stdout, stderr = self._run_ximage(commands)
        if os.path.exists(docs_path + '/psf_model/psf.qdp'):
            os.remove(docs_path + '/psf_model/psf.qdp')
        os.chdir(work_path)
        
        if std: 
            print(stdout)
            print(stderr)
            
        with open(qdpfile, 'r') as f:
            qdp_lines = f.readlines()

        qdp_data_lines = [line for line in qdp_lines if line.strip() 
                          and not line.strip().startswith('!') 
                          and not line.strip().startswith('@') 
                          and not line.strip().upper().startswith('READ')]
        qdp_data = np.loadtxt(qdp_data_lines)
        
        with open(modfile, 'r') as f:
            mod_lines = f.readlines()
            
        mod_param = float(mod_lines[1].split()[0])
        
        radius_arr = np.logspace(np.log10(qdp_data[0, 0]), np.log10(qdp_data[-1, 0]), 100)
        psf_arr = self._psf_model(radius_arr, mod_param)
        
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.5, 0.5], 
            shared_xaxes=True,
            horizontal_spacing=0,
            vertical_spacing=0.05)
        
        eef = go.Scatter(x=qdp_data[:, 0], 
                         y=qdp_data[:, 4], 
                         mode='markers', 
                         name='Encircled energy fraction', 
                         showlegend=True, 
                         error_x=dict(
                             type='data',
                             array=qdp_data[:, 1],
                             thickness=1.5,
                             width=0), 
                         error_y=dict(
                             type='data',
                             array=qdp_data[:, 5],
                             thickness=1.5,
                             width=0),
                         marker=dict(symbol='circle', size=3))
        eef_nominal_model = go.Scatter(x=np.append(qdp_data[:, 0] - qdp_data[:, 1], 
                                                   qdp_data[-1, 0] + qdp_data[-1, 1]),
                                       y=np.append(qdp_data[:, 6], qdp_data[-1, 6]), 
                                       mode='lines',
                                       name='EEF nominal model',
                                       line_shape='hv',
                                       showlegend=True)
        psf = go.Scatter(x=qdp_data[:, 0], 
                         y=qdp_data[:, 7], 
                         mode='markers', 
                         name='Point spread function', 
                         showlegend=True, 
                         error_x=dict(
                             type='data',
                             array=qdp_data[:, 1],
                             thickness=1.5,
                             width=0), 
                         error_y=dict(
                             type='data',
                             array=qdp_data[:, 8],
                             thickness=1.5,
                             width=0),
                         marker=dict(symbol='circle', size=3))
        psf_nominal_model = go.Scatter(x=np.append(qdp_data[:, 0] - qdp_data[:, 1], 
                                                   qdp_data[-1, 0] + qdp_data[-1, 1]),
                                       y=np.append(qdp_data[:, 9], qdp_data[-1, 9]), 
                                       mode='lines',
                                       name='PSF nominal model',
                                       line_shape='hv',
                                       showlegend=True)
        psf_best_model = go.Scatter(x=radius_arr, 
                                    y=psf_arr, 
                                    name='PSF best model', 
                                    showlegend=True, 
                                    mode='lines', 
                                    line=dict(width=2))
        
        fig.add_trace(eef, row=1, col=1)
        fig.add_trace(eef_nominal_model, row=1, col=1)
        fig.add_trace(psf, row=2, col=1)
        fig.add_trace(psf_nominal_model, row=2, col=1)
        fig.add_trace(psf_best_model, row=2, col=1)
        
        fig.update_xaxes(title_text='', row=1, col=1, type='log')
        fig.update_xaxes(title_text='Radius (arcsec)', row=2, col=1, type='log')
        fig.update_yaxes(title_text='Encircled energy fraction', row=1, col=1, type='log')
        fig.update_yaxes(title_text='PSF (ct/sq.arcsec/s)', row=2, col=1, type='log')
        fig.update_layout(template='plotly_white', height=700, width=700)
        fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))
        
        if show: fig.show()
        fig.write_html(psf_savepath + '/psf.html')
        json_dump(fig.to_dict(), psf_savepath + '/psf.json')


    @property
    def spec_slices(self):
        
        try:
            return self._spec_slices
        except AttributeError:
            return [self.time_filter]
        
        
    @spec_slices.setter
    def spec_slices(self, new_spec_slice):
        
        if isinstance(new_spec_slice, list):
            if isinstance(new_spec_slice[0], list):
                self._spec_slices = new_spec_slice
            else:
                raise ValueError('not expected spec_slices type')
        else:
            raise ValueError('not expected spec_slices type')
        

    def extract_spectrum(self, savepath='./spectrum', std=False):
            
        savepath = os.path.abspath(savepath)
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)
            
        json_dump(self.timezero, savepath + '/timezero.json')
        
        json_dump(self.spec_slices, savepath + '/spec_slices.json')
        
        lslices = np.array(self.spec_slices)[:, 0]
        rslices = np.array(self.spec_slices)[:, 1]
        
        for l, r in zip(lslices, rslices):
            scc_start = self.timezero + l
            scc_stop = self.timezero + r
            
            new_l = '{:+.2f}'.format(l).replace('-', 'm').replace('.', 'd').replace('+', 'p')
            new_r = '{:+.2f}'.format(r).replace('-', 'm').replace('.', 'd').replace('+', 'p')
            
            file_name = '_'.join([new_l, new_r])
            
            src_specfile = savepath + f'/{file_name}.src'
            bkg_specfile = savepath + f'/{file_name}.bkg'
            
            if os.path.exists(src_specfile):
                os.remove(src_specfile)
                
            if os.path.exists(bkg_specfile):
                os.remove(bkg_specfile)
        
            commands = ['xsel', 
                        'read events', 
                        os.path.dirname(self.file), 
                        self.file.split('/')[-1], 
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



class epImage(Image):
    
    def __init__(self, 
                 file, 
                 regfile, 
                 bkregfile, 
                 armregfile=None, 
                 arm=False, 
                 rmffile=None, 
                 arffile=None
                 ):
        
        self._file = file
        self._regfile = regfile
        self._bkregfile = bkregfile
        self._armregfile = armregfile
        self._arm = arm
        self._rmffile = rmffile
        self._arffile = arffile
        
        self.prefix = ''
        
        self._ini_xselect()


    @classmethod
    def from_wxtobs(cls, obsid, srcid, datapath=None):
        
        rtv = epRetrieve.from_wxtobs(obsid, srcid, datapath)

        file = rtv.rtv_res['evt']
        regfile = rtv.rtv_res['reg']
        bkregfile = rtv.rtv_res['bkreg']
        armregfile = rtv.rtv_res['armreg']
        rmffile = rtv.rtv_res['rmf']
        arffile = rtv.rtv_res['arf']
        
        return cls(file, regfile, bkregfile, armregfile, False, rmffile, arffile)
    
    
    @classmethod
    def from_fxtobs(cls, obsid, module, datapath=None):
        
        rtv = epRetrieve.from_fxtobs(obsid, module, datapath)

        file = rtv.rtv_res['evt']
        regfile = rtv.rtv_res['reg']
        bkregfile = rtv.rtv_res['bkreg']
        
        return cls(file, regfile, bkregfile)
            
            
    @property
    def armregfile(self):
        
        if self._armregfile is None:
            return None
        else:
            return os.path.abspath(self._armregfile)
    
    
    @armregfile.setter
    def armregfile(self, new_armregfile):
        
        self._armregfile = new_armregfile
        
        self._ini_xselect()
        
        
    @property
    def arm(self):
        
        return self._arm
    
    
    @arm.setter
    def arm(self, new_arm):
        
        self._arm = new_arm
        
        self._ini_xselect()
        
        
    @property
    def bkregfile(self):
        
        if self.arm and self.armregfile:
            return f'"{os.path.abspath(self._bkregfile)} {self.armregfile}"'
        else:
            return f'"{os.path.abspath(self._bkregfile)}"'


    @property
    def rmffile(self):
        
        if self._rmffile is None:
            return None
        else:
            return os.path.abspath(self._rmffile)
    
    
    @rmffile.setter
    def rmffile(self, new_rmffile):
        
        self._rmffile = new_rmffile
        
        
    @property
    def arffile(self):
        
        if self._arffile is None:
            return None
        else:
            return os.path.abspath(self._arffile)
    
    
    @arffile.setter
    def arffile(self, new_arffile):
        
        self._arffile = new_arffile


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
            raise ValueError('not expected type for timezero')


    @property
    def timezero_utc(self):
        
        return ep_met_to_utc(self.timezero)
    
    
    @staticmethod
    def _psf_model(r, p):
        rc = 5.1964
        beta = 1.5792
        sigma = 8.7997
        w = 0.0844

        king_part = (1 + (r / rc)**2) ** (-beta)
        gauss_part = np.exp(-r**2 / (2 * sigma**2))

        psf = p * (king_part + w * gauss_part)
        return psf
    
    
    @property
    def psf_modelfile(self):
        
        return 'fxt_psf.cod'
    
    
    @property
    def psf_fitradius(self):
        
        try:
            self._psf_fitradius
        except AttributeError:
            return 30
        else:
            if self._psf_fitradius is not None:
                return self._psf_fitradius
            else:
                return 30
        
        
    @psf_fitradius.setter
    def psf_fitradius(self, new_psf_fitradius):
        
        if isinstance(new_psf_fitradius, (int, type(None))):
            self._psf_fitradius = new_psf_fitradius
        else:
            raise ValueError('psf_fitradius is extected to be int or None')
        
        
    def extract_response(self, savepath='./spectrum'):
        
        assert self.rmffile is not None, 'rmffile is not set, cannot extract response'
        assert self.arffile is not None, 'arffile is not set, cannot extract response'
        
        savepath = os.path.abspath(savepath)
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)
            
        rsp_rmffile = savepath + '/ep.rmf'
        rsp_arffile = savepath + '/ep.arf'

        copy(self.rmffile, rsp_rmffile)
        copy(self.arffile, rsp_arffile)



class swiftImage(Image):
    
    def __init__(self, 
                 file, 
                 regfile, 
                 bkregfile, 
                 attfile, 
                 xhdfile,
                 mode
                 ):
        
        self._file = file
        self._regfile = regfile
        self._bkregfile = bkregfile
        self._attfile = attfile
        self._xhdfile = xhdfile
        self._mode = mode
        
        self.prefix = f'{mode}_'
        
        self._ini_xselect()


    @classmethod
    def from_xrtobs(cls, obsid, mode, datapath=None):
        
        rtv = swiftRetrieve.from_xrtobs(obsid, mode, datapath)

        file = rtv.rtv_res['evt']
        regfile = rtv.rtv_res['reg']
        bkregfile = rtv.rtv_res['bkreg']
        attfile = rtv.rtv_res['att']
        xhdfile = rtv.rtv_res['xhd']
        mode = rtv.rtv_res['mode']
        
        return cls(file, regfile, bkregfile, attfile, xhdfile, mode)
    
    
    @property
    def attfile(self):
        
        return os.path.abspath(self._attfile)
    
    
    @attfile.setter
    def attfile(self, new_attfile):
        
        self._attfile = new_attfile
        
        
    @property
    def xhdfile(self):

        return os.path.abspath(self._xhdfile)


    @xhdfile.setter
    def xhdfile(self, new_xhdfile):

        self._xhdfile = new_xhdfile
        
    
    @property
    def mode(self):
        
        return self._mode


    @property
    def utcf(self):
        
        hdu = fits.open(self._file)
        return hdu['EVENTS'].header['UTCFINIT']
    
    
    @property
    def timezero(self):
        
        return self._timezero


    @timezero.setter
    def timezero(self, new_timezero):
        
        if isinstance(new_timezero, float):
            self._timezero = new_timezero
            
        elif isinstance(new_timezero, str):
            self._timezero = swift_utc_to_met(new_timezero, self.utcf)
            
        else:
            raise ValueError('not expected type for timezero')


    @property
    def timezero_utc(self):
        
        return swift_met_to_utc(self.timezero, self.utcf)
    
    
    @staticmethod
    def _psf_model(r, p):
        rc = 3.726
        beta = 1.305
        sigma = 7.422
        w = 0.0807

        king_part = (1 + (r / rc)**2) ** (-beta)
        gauss_part = np.exp(-r**2 / (2 * sigma**2))

        psf = p * (king_part + w * gauss_part)
        return psf
    
    
    @property
    def psf_modelfile(self):
        
        return 'xrt_psf.cod'
    
    
    @property
    def psf_fitradius(self):
        
        try:
            self._psf_fitradius
        except AttributeError:
            return 15
        else:
            if self._psf_fitradius is not None:
                return self._psf_fitradius
            else:
                return 15
        
        
    @psf_fitradius.setter
    def psf_fitradius(self, new_psf_fitradius):
        
        if isinstance(new_psf_fitradius, (int, type(None))):
            self._psf_fitradius = new_psf_fitradius
        else:
            raise ValueError('psf_fitradius is extected to be int or None')
            
            
    @property
    def lc_exps(self):

        curve_savepath = os.path.dirname(self.file) + f'/{self.prefix}curve'

        src_corrfile = curve_savepath + '/src.corr'
        
        if not os.path.exists(src_corrfile):
            
            src_corr_curvefile = curve_savepath + '/src_corr.lc'
            src_instrfile = curve_savepath + '/src_srawinstr.img'
            
            commands = ['xrtlccorr',
                        'clobber=yes',
                        'regionfile=None',
                        f'lcfile={self.src_curvefile}',
                        f'outfile={src_corr_curvefile}',
                        f'corrfile={src_corrfile}',
                        f'attfile={self.attfile}',
                        f'outinstrfile={src_instrfile}',
                        f'infile={self.file}',
                        f'hdfile={self.xhdfile}']
        
            stdout, stderr  = self._run_commands(commands)

            if not os.path.exists(src_corrfile):
                print(stdout)
                print(stderr)

        hdu = fits.open(src_corrfile)
        time_10s = np.array(hdu['LCCORRFACT'].data['TIME'])
        factor_10s = np.array(hdu['LCCORRFACT'].data['CORRFACT'])
        
        diff = np.abs(self.lc_time[:, None] - time_10s[None, :])
        self.lc_factor = factor_10s[np.argmin(diff, axis=1)]
        
        return self.lc_binsize / self.lc_factor
        
        
    def extract_response(self, savepath='./spectrum', std=False):
            
        savepath = os.path.abspath(savepath)
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        
        lslices = np.array(self.spec_slices)[:, 0]
        rslices = np.array(self.spec_slices)[:, 1]
        
        for l, r in zip(lslices, rslices):
            scc_start = self.timezero + l
            scc_stop = self.timezero + r
            
            new_l = '{:+.2f}'.format(l).replace('-', 'm').replace('.', 'd').replace('+', 'p')
            new_r = '{:+.2f}'.format(r).replace('-', 'm').replace('.', 'd').replace('+', 'p')
            
            file_name = '_'.join([new_l, new_r])
            
            exposure_savepath = savepath + '/exposure'
            
            if not os.path.exists(exposure_savepath):
                os.makedirs(exposure_savepath)
            
            evtfile = exposure_savepath + f'/{file_name}.evt'
            
            if os.path.exists(evtfile):
                os.remove(evtfile)
                
            commands = ['xsel', 
                        'read event', 
                        os.path.dirname(self.file), 
                        self.file.split('/')[-1], 
                        'yes', 
                        'filter time scc', 
                        f'{scc_start}, {scc_stop}', 
                        'x', 
                        'extract event copyall=yes', 
                        f'save event {evtfile}', 
                        'no']
        
            stdout, stderr = self._run_xselect(commands)
            
            if std: 
                print(stdout)
                print(stderr)

            expfile = exposure_savepath + f'/{file_name}_ex.img'

            commands = ['xrtexpomap', 
                        'clobber=yes', 
                        f'infile={evtfile}',
                        f'attfile={self.attfile}',
                        f'hdfile={self.xhdfile}',
                        f'outdir={exposure_savepath}/',
                        f'stemout={file_name}']
            
            stdout, stderr = self._run_commands(commands)
            
            if std: 
                print(stdout)
                print(stderr)
                
            src_specfile = savepath + f'/{file_name}.src'
            
            if not os.path.exists(src_specfile):
                self.extract_spectrum(savepath=savepath, std=std)

            rsp_arffile = savepath + f'/{file_name}.arf'

            commands = ['xrtmkarf', 
                        'clobber=yes', 
                        f'expofile={expfile}',
                        f'phafile={src_specfile}',
                        'psfflag=yes',
                        f'outfile={rsp_arffile}',
                        'srcx=-1',
                        'srcy=-1']
            
            stdout, stderr = self._run_commands(commands)
            
            if std: 
                print(stdout)
                print(stderr)
                
            rsp_rmffile = savepath + f'/{file_name}.rmf'
            date_time = swift_met_to_utc((scc_start + scc_stop) / 2, self.utcf)
            date = date_time.split('T')[0]
            time = date_time.split('T')[1]

            commands = ['quzcif',
                        'mission=SWIFT',
                        'instrument=XRT',
                        'detector=-',
                        'filter=-',
                        'codename=matrix',
                        f'date={date}',
                        f'time={time}',
                        'expr=datamode.eq.windowed.and.grade.eq.G0:2.and.XRTVSUB.eq.6']
            
            stdout, stderr = self._run_commands(commands)
            
            copy(stdout.split()[0], rsp_rmffile)
            
            if std: 
                print(stdout)
                print(stderr)
