import os
import warnings
import numpy as np
from astropy import units
from astropy import table
from astropy.io import fits
from astropy.time import Time
import plotly.graph_objs as go
from gbm_drm_gen.drmgen_tte import DRMGenTTE
from astropy.units import UnitsWarning
warnings.simplefilter('ignore', UnitsWarning)
from astropy.utils.metadata import MergeConflictWarning
warnings.simplefilter('ignore', MergeConflictWarning)
from .filter import Filter
from ..data.retrieve import gbmRetrieve, gecamRetrieve, gridRetrieve
from ..temp.txx import pgTxx
from ..auto.polybase import PolyBase
from ..util.file import copy, remove
from ..util.data import msg_format, json_dump, rebin
from ..util.time import fermi_met_to_utc, gecam_met_to_utc, grid_met_to_utc



class Event(object):
    
    def __init__(self, file):
        
        self._file = file
        
        self._read()
        
        
    def _read(self):
        
        if self.file is not None:
            hdu = fits.open(self.file)
            self._event = table.Table.read(hdu['EVENTS'])
            self._ebound = table.Table.read(hdu['EBOUNDS'])
            self._gti = table.Table.read(hdu['GTI'])
            
            self._timezero = np.min(self._event['TIME'])
            self._filter = Filter(self._event)
            self._filter_info = {'time': None, 'energy': None, 'tag': None}


    @staticmethod
    def _ch_to_energy(pi, ch, e1, e2):
        
        pi = np.asarray(pi)
        ch = np.asarray(ch)
        e1 = np.asarray(e1)
        e2 = np.asarray(e2)

        ch_to_index = np.zeros(np.max(ch) + 1, dtype=int)
        ch_to_index[ch] = np.arange(len(ch))
        
        indices = ch_to_index[pi]

        e1_selected = e1[indices]
        e2_selected = e2[indices]

        energy = Event._energy_of_ch(len(pi), e1_selected, e2_selected)

        return energy
    
    
    @staticmethod
    def _energy_of_ch(n, e1, e2):
        
        return e1 + (e2 - e1) * np.random.random_sample(n)


    @property
    def file(self):
        
        return self._file
    
    
    @file.setter
    def file(self, new_file):
        
        self._file = new_file
        
        self._read()
    
    
    @property
    def event(self):
        
        return self._filter.evt
    
    
    @property
    def gti(self):
        
        return self._gti
    
    
    @property
    def ebound(self):
        
        return self._ebound
    
    
    @property
    def chantype(self):
        
        return 'None'
    
    
    @property
    def telescope(self):
        
        return 'None'
    
    
    @property
    def instrument(self):
        
        return 'None'
    
    
    @property
    def timezero(self):
        
        return self._timezero
    
    
    @timezero.setter
    def timezero(self, new_timezero):
        
        if isinstance(new_timezero, float):
            self._timezero = new_timezero
        else:
            raise ValueError('not expected type for timezero')
        
        
    @property
    def timezero_utc(self):
        
        return None


    def slice_time(self, t1t2):
        
        t1, t2 = t1t2
        
        met_t1 = self.timezero + t1
        met_t2 = self.timezero + t2
        
        met_ts = self._event['TIME']
        flt = (met_ts >= met_t1) & (met_ts <= met_t2)
        self._event = self._event[flt]
        
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
        
        
    def filter_energy(self, e1e2):
        
        if e1e2 is None:
            expr = None
            
        elif isinstance(e1e2, list):
            e1, e2 = e1e2
            expr = f'(ENERGY >= {e1}) * (ENERGY <= {e2})'
            
        else:
            raise ValueError('e1e2 is extected to be list or None')
        
        self._filter_info['energy'] = expr
        
        self._filter_update()
        
        
    def filter(self, expr):
        
        self._filter_info['tag'] = expr
        
        self._filter_update()
        
        
    def _filter_update(self):
        
        self._clear_filter()
        
        self._filter.eval(self._filter_info['time'])
        self._filter.eval(self._filter_info['energy'])
        self._filter.eval(self._filter_info['tag'])
        
        
    def _clear_filter(self):
        
        self._filter.clear()
        
        
    @property
    def filter_info(self):
        
        return self._filter_info
    
    
    @property
    def time_filter(self):
        
        if self._filter_info['time'] is None:
            return [np.min(self._event['TIME']) - self.timezero, 
                    np.max(self._event['TIME']) - self.timezero]
        else:
            return self._time_filter


    @property
    def _ts(self):
        
        return np.array(self._event['TIME']) - self.timezero


    @property
    def _pha(self):
        
        try:
            return np.array(self._event['PHA']).astype(int)
        except KeyError:
            return np.array(self._event['PI']).astype(int)


    @property
    def _dtime(self):
        
        return np.array(self._event['DEAD_TIME'])
    
    
    @property
    def ts(self):
    
        return np.array(self.event['TIME']) - self.timezero


    @property
    def pha(self):
        
        try:
            return np.array(self.event['PHA']).astype(int)
        except KeyError:
            return np.array(self.event['PI']).astype(int)


    @property
    def channel(self):
        
        return np.array(self.ebound['CHANNEL'], dtype=int)
    
    
    @property
    def channel_emin(self):
        
        return np.array(self.ebound['E_MIN'])
    
    
    @property
    def channel_emax(self):
        
        return np.array(self.ebound['E_MAX'])
    
    
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
    def spec_t1t2(self):
        
        try:
            self._spec_t1t2
        except AttributeError:
            return self.time_filter
        else:
            if self._spec_t1t2 is not None:
                return self._spec_t1t2
            else:
                return self.time_filter


    @spec_t1t2.setter
    def spec_t1t2(self, new_spec_t1t2):
        
        if isinstance(new_spec_t1t2, (list, type(None))):
            self._spec_t1t2 = new_spec_t1t2
        else:
            raise ValueError('spec_t1t2 is extected to be list or None')
        
        
    @property
    def ignore_t1t2(self):
        
        try:
            self._ignore_t1t2
        except AttributeError:
            return None
        else:
            if self._ignore_t1t2 is not None:
                return self._ignore_t1t2
            else:
                return None


    @ignore_t1t2.setter
    def ignore_t1t2(self, new_ignore_t1t2):
        
        if isinstance(new_ignore_t1t2, (list, type(None))):
            self._ignore_t1t2 = new_ignore_t1t2
        else:
            raise ValueError('ignore_t1t2 is extected to be list or None')
    
    
    @property
    def lc_interval(self):
        
        return self.lc_t1t2[1] - self.lc_t1t2[0]
    
    
    @property
    def spec_interval(self):
        
        return self.spec_t1t2[1] - self.spec_t1t2[0]
    
    
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
    def spec_binsize(self):
        
        try:
            self._spec_binsize
        except AttributeError:
            return self.spec_interval / 200
        else:
            if self._spec_binsize is not None:
                return self._spec_binsize
            else:
                return self.spec_interval / 200
            
            
    @spec_binsize.setter
    def spec_binsize(self, new_spec_binsize):
        
        if isinstance(new_spec_binsize, (int, float, type(None))):
            self._spec_binsize = new_spec_binsize
        else:
            raise ValueError('spec_binsize is extected to be int, float or None')
        
        
    @property
    def lc_ts(self):
        
        idx = (self.ts >= self.lc_t1t2[0]) & (self.ts <= self.lc_t1t2[1])
        
        return self.ts[idx]
    
    
    @property
    def spec_ts(self):
        
        idx = (self._ts >= self.spec_t1t2[0]) & (self._ts <= self.spec_t1t2[1])
        
        return self._ts[idx]
    
    
    @property
    def lc_pha(self):
        
        idx = (self.ts >= self.lc_t1t2[0]) & (self.ts <= self.lc_t1t2[1])
        
        return self.pha[idx]
    
    
    @property
    def spec_pha(self):
        
        idx = (self._ts >= self.spec_t1t2[0]) & (self._ts <= self.spec_t1t2[1])
        
        return self._pha[idx]
    
    
    @staticmethod
    def _poisson_binsize(counts, interval, p):
        
        rate = float(counts) / interval

        if rate == 0:
            return interval
        else:
            binsize = np.log(1 - p) / (- rate)
            
            if binsize > interval:
                return interval
            
            elif binsize > 1:
                return np.ceil(binsize)
            
            elif binsize < 1:
                return binsize


    def exposure(self, bin_list, dead=True):
        
        ts = self._ts
        dtime = self._dtime
        
        lbins = np.array(bin_list)[:, 0]
        rbins = np.array(bin_list)[:, 1]
        binsize = rbins - lbins
        
        dead_time = np.zeros_like(binsize, dtype=float)
        
        if dead:
            for i, (l, r) in enumerate(zip(lbins, rbins)):
                dead_time[i] = 1e-6 * np.sum(dtime[(ts >= l) & (ts < r)])

        return binsize - dead_time


    @property
    def lc_bins(self):
        
        return np.arange(self.lc_t1t2[0], self.lc_t1t2[1] + 1e-5, self.lc_binsize)
    
    
    @property
    def lc_bin_list(self):
        
        lbins, rbins = self.lc_bins[:-1], self.lc_bins[1:]
        
        return np.vstack((lbins, rbins)).T
    
    
    @property
    def lc_exps(self):
        
        return self.exposure(self.lc_bin_list, dead=False)
    
    
    @property
    def lc_time(self):
        
        return np.mean(self.lc_bin_list, axis=1)
    

    @property
    def lc_time_err(self):
        
        lbins, rbins = self.lc_bins[:-1], self.lc_bins[1:]
        
        return (rbins - lbins) / 2
    
    
    @property
    def lc_src_cts(self):
        
        return np.histogram(self.lc_ts, bins=self.lc_bins)[0]
    
    
    @property
    def lc_src_cts_err(self):
        
        return np.sqrt(self.lc_src_cts)
    
    
    @property
    def lc_src_rate(self):
        
        return self.lc_src_cts / self.lc_exps
    
    
    @property
    def lc_src_rate_err(self):
        
        return self.lc_src_cts_err / self.lc_exps
    
    
    @property
    def bs_sigma(self):
        
        try:
            self._bs_sigma
        except AttributeError:
            return 3
        else:
            return self._bs_sigma


    @bs_sigma.setter
    def bs_sigma(self, new_bs_sigma):
        
        if isinstance(new_bs_sigma, (int, float)):
            self._bs_sigma = new_bs_sigma
        else:
            raise ValueError('bs_sigma is extected to be int or float')
        
        
    @property
    def bs_deg(self):
        
        try:
            self._bs_deg
        except AttributeError:
            return None
        else:
            return self._bs_deg


    @bs_deg.setter
    def bs_deg(self, new_bs_deg):
        
        if isinstance(new_bs_deg, (int, type(None))):
            self._bs_deg = new_bs_deg
        else:
            raise ValueError('bs_deg is extected to be int or None')
    
    
    @property
    def lc_bs(self):
        
        lc_bs = PolyBase(self.lc_ts, self.lc_bins, self.lc_exps, self.ignore_t1t2)
        lc_bs.loop(sigma=self.bs_sigma, deg=self.bs_deg)
        lc_bs.loop(sigma=self.bs_sigma, deg=self.bs_deg)
        
        return lc_bs
    
    
    @property
    def lc_bkg_rate(self):
        
        return self.lc_bs.poly.val(self.lc_time)[0]
    
    
    @property
    def lc_bkg_rate_err(self):
        
        return self.lc_bs.poly.val(self.lc_time)[1]
    
    
    @property
    def lc_bkg_cts(self):
        
        return self.lc_bkg_rate * self.lc_exps
    
    
    @property
    def lc_bkg_cts_err(self):
        
        return self.lc_bkg_rate_err * self.lc_exps
    
    
    @property
    def lc_net_rate(self):
        
        return self.lc_src_rate - self.lc_bkg_rate
    
    
    @property
    def lc_net_rate_err(self):
        
        return np.sqrt(self.lc_src_rate_err ** 2 + self.lc_bkg_rate_err ** 2)
    
    
    @property
    def lc_net_cts(self):
        
        return self.lc_net_rate * self.lc_exps
    
    
    @property
    def lc_net_cts_err(self):
        
        return self.lc_net_rate_err * self.lc_exps
    

    @property
    def lc_net_ccts(self):
        
        return np.cumsum(self.lc_net_cts)
    
    
    def extract_curve(self, savepath='./curve', autobs=True, show=False):
        
        savepath = os.path.abspath(savepath)
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)
            
        if autobs:
            lc_bs = self.lc_bs
            lc_bs.save(savepath=savepath + '/polybase')
            
            lc_bkg_rate, lc_bkg_rate_err = lc_bs.poly.val(self.lc_time)
        
            lc_net_rate = self.lc_src_rate - lc_bkg_rate
            lc_net_cts = lc_net_rate * self.lc_exps
            
            lc_net_ccts = np.cumsum(lc_net_cts)
        
        fig = go.Figure()
        src = go.Scatter(x=self.lc_time, 
                         y=self.lc_src_rate, 
                         mode='lines+markers', 
                         name='source lightcurve', 
                         showlegend=True, 
                         error_y=dict(
                             type='data',
                             array=self.lc_src_rate_err,
                             thickness=1.5,
                             width=0), 
                         marker=dict(symbol='cross-thin', size=0))
        fig.add_trace(src)
        
        if autobs:
            bkg = go.Scatter(x=self.lc_time, 
                             y=lc_bkg_rate, 
                             mode='lines+markers', 
                             name='background lightcurve', 
                             showlegend=True, 
                             error_y=dict(
                                 type='data',
                                 array=lc_bkg_rate_err,
                                 thickness=1.5,
                                 width=0), 
                             marker=dict(symbol='cross-thin', size=0))
            fig.add_trace(bkg)
        
        fig.update_xaxes(title_text=f'Time since {self.timezero_utc} (s)')
        fig.update_yaxes(title_text=f'Counts per second (binsize={self.lc_binsize} s)')
        fig.update_layout(template='plotly_white', height=600, width=800)
        fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))
        
        if show: fig.show()
        fig.write_html(savepath + '/lc.html')
        json_dump(fig.to_dict(), savepath + '/lc.json')
        
        if autobs:
            fig = go.Figure()
            net = go.Scatter(x=self.lc_time, 
                            y=lc_net_ccts, 
                            mode='lines', 
                            name='net cumulated counts', 
                            showlegend=True)
            
            fig.add_trace(net)
            
            fig.update_xaxes(title_text=f'Time since {self.timezero_utc} (s)')
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
        
        txx = pgTxx(self.lc_ts, self.lc_bins, self.lc_exps, self.ignore_t1t2)
        txx.findpulse(sigma=self.bs_sigma, deg=self.bs_deg, mp=mp)
        txx.accumcts(xx=xx, pstart=pstart, pstop=pstop, lbkg=lbkg, rbkg=rbkg)
        txx.save(savepath=savepath)

        
    def extract_rebin_curve(self, trange=None, stat='pgstat', min_sigma=None, min_evt=None, max_bin=None, 
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
                    stat, 
                    self.lc_src_cts[idx], 
                    cts_err=self.lc_src_cts_err[idx],
                    bcts=self.lc_bkg_cts[idx], 
                    bcts_err=self.lc_bkg_cts_err[idx], 
                    min_sigma=min_sigma, 
                    min_evt=min_evt, 
                    max_bin=max_bin,
                    backscale=1)
                
        self.lc_retime = np.mean(self.lc_rebin_list, axis=1)
        self.lc_reexps = self.exposure(self.lc_rebin_list, dead=False)
        self.lc_net_rects = self.lc_src_rects - self.lc_bkg_rebcts
        self.lc_net_rects_err = np.sqrt(self.lc_src_rects_err ** 2 + self.lc_bkg_rebcts_err ** 2)
        self.lc_net_rerate = self.lc_net_rects / self.lc_reexps
        self.lc_net_rerate_err = self.lc_net_rects_err / self.lc_reexps
        
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
        
        
    def _extract_src_phaii(self, spec_slices):
        
        num_slices = len(spec_slices)
        num_channels = len(self.channel)
        
        phaii = np.zeros([num_slices, num_channels], dtype=float)
        
        lslices = np.array(spec_slices)[:, 0]
        rslices = np.array(spec_slices)[:, 1]
        
        pha_bins = np.arange(np.min(self.channel), np.max(self.channel) + 2, 1)
        
        for i, (l, r) in enumerate(zip(lslices, rslices)):
            pii = self.spec_pha[(self.spec_ts >= l) & (self.spec_ts < r)]
            pha, _ = np.histogram(pii, pha_bins)
            phaii[i, :] = pha
            
        return phaii
    
    
    def extract_src_phaii(self, savepath='./spectrum'):
            
        phaii = self._extract_src_phaii(self.spec_slices)
        
        savepath = os.path.abspath(savepath)
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)
                
        exps = self.exposure(self.spec_slices)
            
        lslices = np.array(self.spec_slices)[:, 0]
        rslices = np.array(self.spec_slices)[:, 1]
    
        for i, (l, r) in enumerate(zip(lslices, rslices)):
            new_l = '{:+.2f}'.format(l).replace('-', 'm').replace('.', 'd').replace('+', 'p')
            new_r = '{:+.2f}'.format(r).replace('-', 'm').replace('.', 'd').replace('+', 'p')
            
            file_name = '_'.join([new_l, new_r]) + '.src'
            
            pha_hdu = self._to_pha_fits(phaii[i], np.sqrt(phaii[i]), exps[i], self.timezero + l, self.timezero + r, file_name)

            if os.path.isfile(savepath + f'/{file_name}'):
                os.remove(savepath + f'/{file_name}')
            pha_hdu.writeto(savepath + f'/{file_name}')
            
            
    def _extract_bkg_phaii(self, spec_slices, show=False):
        
        num_slices = len(spec_slices)
        num_channels = len(self.channel)
        
        phaii = np.zeros([num_slices, num_channels], dtype=float)
        phaii_err = np.zeros([num_slices, num_channels], dtype=float)
        
        lslices = np.array(spec_slices)[:, 0]
        rslices = np.array(spec_slices)[:, 1]
        
        bins = np.arange(self.spec_t1t2[0], self.spec_t1t2[1]+1e-5, self.spec_binsize)
        
        interp_low = self.spec_t1t2[0] + self.spec_interval / 10
        interp_upp = self.spec_t1t2[1] - self.spec_interval / 10
        interp_range = [max([interp_low, np.mean(bins[:2])]), min([interp_upp, np.mean(bins[-2:])])]
        interp_time = np.linspace(interp_range[0], interp_range[-1], 100)
        
        bs = PolyBase(self.spec_ts, bins, ignore=self.ignore_t1t2)
        bs.loop(sigma=self.bs_sigma, deg=self.bs_deg)
        bs.loop(sigma=self.bs_sigma, deg=self.bs_deg)
        
        ignore = bs.ignore
        brate, _ = bs.poly.val(interp_time)
        
        brate_sum = np.zeros_like(brate)
        
        max_binsize = int(self.spec_interval / 10 * 10) / 10
        
        for i, ch in enumerate(self.channel):
            index = (self.spec_pha == ch)
            ts_i = self.spec_ts[index]
            
            binsize_i = Event._poisson_binsize(len(ts_i), self.spec_interval, 0.99)
            
            if binsize_i < 1: binsize_i = 1
            elif binsize_i < max_binsize: binsize_i = int(binsize_i * 10) / 10
            else: binsize_i = max_binsize

            bins_i = np.arange(self.spec_t1t2[0], self.spec_t1t2[1] + 1e-5, binsize_i)
            
            bs_i = PolyBase(ts_i, bins_i, ignore=ignore)
            bs_i.polyfit(deg=self.bs_deg)
            
            brate_i, _ = bs_i.poly.val(interp_time)
            
            brate_sum = brate_sum + brate_i
            
            for j, (l, r) in enumerate(zip(lslices, rslices)):
                
                bins_j = np.linspace(l, r, 100)
                brate_j, brate_err_j = bs_i.poly.val(bins_j)
                
                phaii[j, i] = np.trapz(brate_j, bins_j)
                phaii_err[j, i] = np.sqrt(np.trapz(brate_err_j ** 2, bins_j))
                
        fig = go.Figure()
        src = go.Scatter(x=bs.time, 
                         y=bs.rate, 
                         mode='lines', 
                         name='total lightcurve', 
                         showlegend=True)
        tot = go.Scatter(x=interp_time, 
                         y=brate, 
                         mode='lines', 
                         name='total background', 
                         showlegend=True)
        sum = go.Scatter(x=interp_time, 
                         y=brate_sum, 
                         mode='lines', 
                         name='summing background', 
                         showlegend=True)
        
        fig.add_trace(src)
        fig.add_trace(tot)
        fig.add_trace(sum)
        
        fig.update_xaxes(title_text=f'Time since {self.timezero_utc} (s)')
        fig.update_yaxes(title_text=f'Counts per second')
        fig.update_layout(template='plotly_white', height=600, width=800)
        fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))
        
        if show: fig.show()
                
        return phaii, phaii_err
                
                
    def extract_bkg_phaii(self, savepath='./spectrum', show=False):
        
        phaii, phaii_err = self._extract_bkg_phaii(self.spec_slices, show=show)
        
        savepath = os.path.abspath(savepath)
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)
            
        exps = self.exposure(self.spec_slices)
            
        lslices = np.array(self.spec_slices)[:, 0]
        rslices = np.array(self.spec_slices)[:, 1]
    
        for i, (l, r) in enumerate(zip(lslices, rslices)):
            new_l = '{:+.2f}'.format(l).replace('-', 'm').replace('.', 'd').replace('+', 'p')
            new_r = '{:+.2f}'.format(r).replace('-', 'm').replace('.', 'd').replace('+', 'p')
            
            file_name = '_'.join([new_l, new_r]) + '.bkg'
            
            pha_hdu = self._to_pha_fits(phaii[i], phaii_err[i], exps[i], self.timezero + l, self.timezero + r, file_name)
            
            if os.path.isfile(savepath + f'/{file_name}'):
                os.remove(savepath + f'/{file_name}')
            pha_hdu.writeto(savepath + f'/{file_name}')
                
                
    def extract_spectrum(self, savepath='./spectrum', show=False):
        
        savepath = os.path.abspath(savepath)
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        
        self.extract_src_phaii(savepath=savepath)
        self.extract_bkg_phaii(savepath=savepath, show=show)


    def _to_pha_fits(self, pha, pha_err, exp, tstart, tstop, file_name):
        
        hdr = fits.Header()
        hdr['HDUCLASS'] = ('ogip', 'Format conforms to OGIP/GSFC conventions')
        hdr['HDUVERS'] = ('1.2.1', 'Version of format (OGIP memo OGIP-92-007)')
        hdr['FILENAME'] = (f'{file_name}', 'name of this file')
        hdr['AUTHOR'] = ('Jun Yang', 'author of this file')
        hdr['EMAIL'] = ('jyang@smail.nju.edu.cn', 'email of author')
        hdr['DATE'] = (Time.now().isot, 'file generated date')
        primary_hdu = fits.PrimaryHDU(header=hdr)

        channel_field = fits.Column(name='CHANNEL', array=self.channel, format='I')
        counts_field = fits.Column(name='COUNTS', array=pha, format='E', unit='count')
        stat_err_field = fits.Column(name='STAT_ERR', array=pha_err, format='E', unit='count')
        spectrum_hdu = fits.BinTableHDU.from_columns([channel_field, counts_field, stat_err_field])

        spectrum_hdu.header['TTYPE1'] = ('CHANNEL', 'label for field 1')
        spectrum_hdu.header['TFORM1'] = ('I', 'data format of field: 2-byte INTEGER')
        spectrum_hdu.header['TTYPE2'] = ('COUNTS', 'label for field 2')
        spectrum_hdu.header['TFORM2'] = ('E', 'data format of field: 32-bit SINGLE FLOAT')
        spectrum_hdu.header['TUNIT2'] = ('count', 'physical unit of field')
        spectrum_hdu.header['TTYPE3'] = ('STAT_ERR', 'label for field 3')
        spectrum_hdu.header['TFORM3'] = ('E', 'data format of field: 32-bit SINGLE FLOAT')
        spectrum_hdu.header['TUNIT3'] = ('count', 'physical unit of field')
        spectrum_hdu.header['EXTNAME'] = ('SPECTRUM', 'name of this binary table extension')
        spectrum_hdu.header['TLMIN1'] = (np.min(self.channel), 'Lowest legal channel number')
        spectrum_hdu.header['TLMAX1'] = (np.max(self.channel), 'Highest legal channel number')
        spectrum_hdu.header['POISSERR'] = (False, 'Poissonian errors to be assumed')
        spectrum_hdu.header['STAT_ERR'] = (0, 'no statisical error specified')
        spectrum_hdu.header['SYS_ERR'] = (0, 'no systematic error specified')
        spectrum_hdu.header['QUALITY'] = (0, 'no data quality information specified')
        spectrum_hdu.header['GROUPING'] = (0, 'no grouping data has been specified')
        spectrum_hdu.header['DETCHANS'] = (len(self.channel), 'Total No. of Detector Channels available')

        spectrum_hdu.header['EXPOSURE'] = (round(exp, 5), 'exposure time in second')
        spectrum_hdu.header['AREASCAL'] = (1.0, 'nominal effective area')
        spectrum_hdu.header['CORRSCAL'] = (1.0, 'correlation scale factor')
        spectrum_hdu.header['BACKSCAL'] = (1.0, 'background scale factor')
        spectrum_hdu.header['BACKFILE'] = ('None', 'background FITS file for this object')
        spectrum_hdu.header['CORRFILE'] = ('None', 'correlation FITS file for this object')
        spectrum_hdu.header['RESPFILE'] = ('None', 'redistribution; RSP')
        spectrum_hdu.header['ANCRFILE'] = ('None', 'ancillary response; ARF')
        
        if self.chantype == 'pha':
            spectrum_hdu.header['CHANTYPE'] = (self.chantype, 'No corrections have been applied')
        elif self.chantype == 'pi':
            spectrum_hdu.header['CHANTYPE'] = (self.chantype, 'Channels assigned by detector electronics')
            
        spectrum_hdu.header['TSTART'] = (round(tstart, 5), 'Observation start time')
        spectrum_hdu.header['TSTOP'] = (round(tstop, 5), 'Observation stop time')
        spectrum_hdu.header['TIMEZERO'] = (round(self.timezero, 5), 'zero time in MET format')
            
        spectrum_hdu.header['HDUCLASS'] = ('OGIP', 'format conforms to OGIP standard')
        spectrum_hdu.header['HDUCLAS1'] = ('SPECTRUM', 'PHA dataset (OGIP memo OGIP-92-007)')
        spectrum_hdu.header['HDUVERS'] = ('1.2.1', 'Version of format (OGIP memo OGIP-92-007)')
        spectrum_hdu.header['TELESCOP'] = (self.telescope, 'Telescope (mission) name')
        spectrum_hdu.header['INSTRUME'] = (self.instrument, 'Instrument name')
        spectrum_hdu.header['FILTER'] = ('None', 'Instrument filter in use')

        channel_field = fits.Column(name='CHANNEL', array=self.channel, format='I', unit='channel')
        e_min_field = fits.Column(name='E_MIN', array=self.channel_emin, format='E', unit='keV')
        e_max_field = fits.Column(name='E_MAX', array=self.channel_emax, format='E', unit='keV')
        ebound_hdu = fits.BinTableHDU.from_columns([channel_field, e_min_field, e_max_field])
        
        ebound_hdu.header['EXTNAME'] = ('EBOUNDS', 'name of this binary table extension')
        ebound_hdu.header['TLMIN1'] = (np.min(self.channel), 'Lowest legal channel number')
        ebound_hdu.header['TLMAX1'] = (np.max(self.channel), 'Highest legal channel number')
        ebound_hdu.header['DETCHANS'] = (len(self.channel), 'Total No. of Detector Channels available')
        
        if self.chantype == 'pha':
            ebound_hdu.header['CHANTYPE'] = (self.chantype, 'No corrections have been applied')
        elif self.chantype == 'pi':
            ebound_hdu.header['CHANTYPE'] = (self.chantype, 'Channels assigned by detector electronics')
        
        ebound_hdu.header['HDUCLASS'] = ('OGIP', 'format conforms to OGIP standard')
        ebound_hdu.header['HDUCLAS1'] = ('RESPONSE', 'These are typically found in RMF files')
        ebound_hdu.header['HDUCLAS2'] = ('EBOUNDS', 'From CAL/GEN/92-002')
        ebound_hdu.header['HDUVERS'] = ('1.2.1', 'Version of format (OGIP memo OGIP-92-007)')
        ebound_hdu.header['TELESCOP'] = (self.telescope, 'Telescope (mission) name')
        ebound_hdu.header['INSTRUME'] = (self.instrument, 'Instrument name')
        ebound_hdu.header['FILTER'] = ('None', 'Instrument filter in use')

        pha_hdu = fits.HDUList([primary_hdu, spectrum_hdu, ebound_hdu])
        
        return pha_hdu



class gbmTTE(Event):
    
    det_name_lookup = {
        "NAI_00": "n0",
        "NAI_01": "n1",
        "NAI_02": "n2",
        "NAI_03": "n3",
        "NAI_04": "n4",
        "NAI_05": "n5",
        "NAI_06": "n6",
        "NAI_07": "n7",
        "NAI_08": "n8",
        "NAI_09": "n9",
        "NAI_10": "na",
        "NAI_11": "nb",
        "BGO_00": "b0",
        "BGO_01": "b1"}
        
    def __init__(self, file, posfile=None):
        
        self._file = file
        self._posfile = posfile
        
        self._read()


    @classmethod
    def from_utc(cls, utc, det):
        
        dets = ['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb','b0','b1']
        msg = 'invalid detector: %s' % det
        assert det in dets, msg_format(msg)
        
        rtv = gbmRetrieve.from_utc(utc=utc, t1=-200, t2=500)

        file = rtv.rtv_res['tte'][det]
            
        msg = 'no retrieved tte file'
        assert file != [], msg_format(msg)
        
        posfile = rtv.rtv_res['poshist']
        
        msg = 'no retrieved poshist file'
        assert posfile != [], msg_format(msg)
        
        return cls(file, posfile)
    

    def _read(self):
        
        det_list = list()
        event_list = list()
        gti_list = list()
        
        for i in range(len(self.file)):
            hdu = fits.open(self.file[i])
            det = hdu['PRIMARY'].header['DETNAM']
            event = table.Table.read(hdu['EVENTS'])
            ebound = table.Table.read(hdu['EBOUNDS'])
            gti = table.Table.read(hdu['GTI'])
            hdu.close()
            
            pha = np.array(event['PHA']).astype(int)
            ch = np.array(ebound['CHANNEL']).astype(int)
            emin = np.array(ebound['E_MIN'])
            emax = np.array(ebound['E_MAX'])
            energy = Event._ch_to_energy(pha, ch, emin, emax)
            event['ENERGY'] = energy * units.keV
            event['DEAD_TIME'] = (pha == 127) * 10.0 + (pha < 127) * 2.6
            
            det_list.append(det)
            event_list.append(event)
            gti_list.append(gti)

        assert len(set(det_list)) == 1, 'currently unsupport for data from different detectors'
        self._det = list(set(det_list))[0]
        
        self._event = table.vstack(event_list)
        self._gti = table.vstack(gti_list)
        self._ebound = ebound

        self._event = table.unique(self._event, keys=['TIME', 'PHA'])
        self._gti = table.unique(self._gti, keys=['START', 'STOP'])

        self._event.sort('TIME')
        self._gti.sort('START')
        
        self._timezero = np.min(self._event['TIME'])
        
        self._filter = Filter(self._event)
        self._filter_info = {'time': None, 'energy': None, 'pha': None, 'tag': None}
        
        if self.posfile is None:
            self._pos_t1t2_list = None
        else:
            pos_t1t2_list = []
            for i in range(len(self.posfile)):
                hdu = fits.open(self.posfile[i])
                pos = table.Table.read(hdu[1])
                hdu.close()
                
                pos_time = pos['SCLK_UTC']
                pos_t1t2 = [np.min(pos_time), np.max(pos_time)]
                pos_t1t2_list.append(pos_t1t2)
            
            self._pos_t1t2_list = pos_t1t2_list


    @property
    def posfile(self):
        
        return self._posfile
    
    
    @posfile.setter
    def posfile(self, new_posfile):
        
        self._posfile = new_posfile
        
        self._read()


    @property
    def chantype(self):
        
        return 'pha'
    
    
    @property
    def telescope(self):
        
        return 'GLAST'
    
    
    @property
    def instrument(self):
        
        return 'GBM'
        
    @property
    def det(self):
            
        return gbmTTE.det_name_lookup[self._det]
    
    
    @property
    def pos_t1t2_list(self):
        
        return self._pos_t1t2_list


    @property
    def timezero_utc(self):
        
        return fermi_met_to_utc(self.timezero)
    
    
    def filter_pha(self, p1p2):
        
        if p1p2 is None:
            expr = None
            
        elif isinstance(p1p2, list):
            p1, p2 = p1p2
            expr = f'(PHA >= {p1}) * (PHA <= {p2})'
            
        else:
            raise ValueError('p1p2 is extected to be list or None')
        
        self._filter_info['pha'] = expr
        
        self._filter_update()
        
        
    def _filter_update(self):
        
        self._clear_filter()
        
        self._filter.eval(self._filter_info['time'])
        self._filter.eval(self._filter_info['pha'])
        self._filter.eval(self._filter_info['energy'])
        self._filter.eval(self._filter_info['tag'])
        
        
    def extract_response(self, ra, dec, savepath='./spectrum'):
        
        savepath = os.path.abspath(savepath)
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)
            
        cwd = os.getcwd()
        os.chdir(savepath)
        
        lslices = np.array(self.spec_slices)[:, 0]
        rslices = np.array(self.spec_slices)[:, 1]
    
        for _, (l, r) in enumerate(zip(lslices, rslices)):
            new_l = '{:+.2f}'.format(l).replace('-', 'm').replace('.', 'd').replace('+', 'p')
            new_r = '{:+.2f}'.format(r).replace('-', 'm').replace('.', 'd').replace('+', 'p')
            
            file_name = '_'.join([new_l, new_r])
            
            rsp_met = (l + r) / 2 + self.timezero
            for i, (t1, t2) in enumerate(self.pos_t1t2_list):
                if (rsp_met >= t1) and (rsp_met <= t2):
                    poshist = self.posfile[i]
                    break
        
            drm = DRMGenTTE(
                tte_file=self.file[0], 
                time=(l + r) / 2, 
                poshist=poshist, 
                T0=self.timezero, 
                mat_type=2, 
                occult=False)
        
            drm.to_fits(
                ra=ra, 
                dec=dec, 
                filename=file_name, 
                overwrite=True)
            
            copy(f'{file_name}_{self.det}.rsp', f'{file_name}.rsp')
            remove(f'{file_name}_{self.det}.rsp')
            
        os.chdir(cwd)



class gecamEVT(Event):

    def __init__(self, file, det, gain_type):
        
        self._file = file
        self._det = det
        self._gain_type = gain_type
        
        self._read()


    @classmethod
    def from_utc(cls, utc, det, gain_type):
        
        rtv = gecamRetrieve.from_utc(utc=utc, t1=-200, t2=500)

        file = rtv.rtv_res['grd_evt']
            
        msg = 'no retrieved evt file'
        assert file != [], msg_format(msg)
        
        return cls(file, det, gain_type)


    def _read(self):
        
        event_list = list()
        gti_list = list()
        
        for i in range(len(self.file)):
            hdu = fits.open(self.file[i])
            event = table.Table.read(hdu[f'EVENTS{self.det}'])
            ebound = table.Table.read(hdu['EBOUNDS'])
            gti = table.Table.read(hdu['GTI'])
            hdu.close()
            
            gt = event['GAIN_TYPE']
            if self.gain_type == 'HG':
                event = event[gt == 0]
            else:
                event = event[gt == 1]
            
            pi = np.array(event['PI']).astype(int)
            ch = np.array(ebound['CHANNEL']).astype(int)
            emin = np.array(ebound['E_MIN'])
            emax = np.array(ebound['E_MAX'])
            energy = Event._ch_to_energy(pi, ch, emin, emax)
            event['ENERGY'] = energy * units.keV
            
            event_list.append(event)
            gti_list.append(gti)
            
        self._event = table.vstack(event_list)
        self._gti = table.vstack(gti_list)
        self._ebound = ebound

        self._event = table.unique(self._event, keys=['TIME', 'PI', 'GAIN_TYPE'])
        self._gti = table.unique(self._gti, keys=['START', 'STOP'])

        self._event.sort('TIME')
        self._gti.sort('START')
        
        self._timezero = np.min(self._event['TIME'])
        
        self._filter = Filter(self._event)
        self._filter_info = {'time': None, 'energy': None, 'pi': None, 'tag': None}
        
        
    @property
    def det(self):
        
        return self._det
    
    
    @det.setter
    def det(self, new_det):
        
        dets = ['%02d' % i for i in range(1, 26)]
        msg = 'invalid detector: %s' % new_det
        assert new_det in dets, msg_format(msg)
        
        self._det = new_det


    @property
    def gain_type(self):
        
        return self._gain_type
    
    
    @gain_type.setter
    def gain_type(self, new_gain_type):
        
        gain_types = ['HG', 'LG']
        msg = 'invalid gain type: %s' % new_gain_type
        assert new_gain_type in gain_types, msg_format(msg)
        
        self._gain_type = new_gain_type
        
        
    @property
    def chantype(self):
        
        return 'pi'
    
    
    @property
    def telescope(self):
        
        return 'GECAM'
    
    
    @property
    def instrument(self):
        
        return 'GRD'
        
        
    @property
    def timezero_utc(self):
        
        return gecam_met_to_utc(self.timezero)
    
    
    def filter_pi(self, p1p2):
        
        if p1p2 is None:
            expr = None
            
        elif isinstance(p1p2, list):
            p1, p2 = p1p2
            expr = f'(PI >= {p1}) * (PI <= {p2})'
            
        else:
            raise ValueError('p1p2 is extected to be list or None')
        
        self._filter_info['pi'] = expr
        
        self._filter_update()
        
        
    def _filter_update(self):
        
        self._clear_filter()
        
        self._filter.eval(self._filter_info['time'])
        self._filter.eval(self._filter_info['pi'])
        self._filter.eval(self._filter_info['energy'])
        self._filter.eval(self._filter_info['tag'])



class gridTTE(Event):

    def __init__(self, file, rspfile, det):
        
        self._file = file
        self._rspfile = rspfile
        self._det = det
        
        self._read()


    @classmethod
    def from_utc(cls, utc, det):
        
        rtv = gridRetrieve.from_utc(utc=utc, t1=-200, t2=500, det=det)

        file = rtv.rtv_res['tte']
        rspfile = rtv.rtv_res['rsp']
            
        msg = 'no retrieved tte file'
        assert file != [], msg_format(msg)
        
        msg = 'no retrieved rsp file'
        assert rspfile != [], msg_format(msg)
        
        return cls(file, rspfile, det)


    def _read(self):
        
        event_list = list()
        
        for i in range(len(self.file)):
            tte_hdu = fits.open(self.file[i])
            rsp_hdu = fits.open(self.rspfile[i])
            event = table.Table.read(tte_hdu[f'T_E{self.det}'])
            ebound = table.Table.read(rsp_hdu['EBOUNDS'])
            tte_hdu.close()
            rsp_hdu.close()
            
            event[f't{self.det}'].name = 'TIME'
            event[f'E{self.det}'].name = 'ENERGY'
            event.remove_columns([f'adcv{self.det}', f'adcv_c{self.det}'])
            
            ch = np.array(ebound['CHANNEL']).astype(int)
            emin = np.array(ebound['E_MIN'])
            emax = np.array(ebound['E_MAX'])
            
            energy = np.array(event['ENERGY'])
            pi = ch[np.searchsorted(emin, energy, side='right') - 1]
            event['PI'] = pi
            
            event['DEAD_TIME'] = np.zeros_like(event['TIME'])
            
            event_list.append(event)
            
        self._event = table.vstack(event_list)
        self._gti = None
        self._ebound = ebound

        self._event = table.unique(self._event, keys=['TIME', 'PI'])
        self._event.sort('TIME')
        
        self._timezero = np.min(self._event['TIME'])
        
        self._filter = Filter(self._event)
        self._filter_info = {'time': None, 'energy': None, 'pi': None, 'tag': None}
        
        
    @property
    def rspfile(self):
        
        return self._rspfile
    
    
    @rspfile.setter
    def rspfile(self, new_rspfile):
        
        self._rspfile = new_rspfile
        
        self._read()
        
        
    @property
    def det(self):
        
        return self._det
    
    
    @det.setter
    def det(self, new_det):
        
        dets = ['%d' % i for i in range(0, 4)]
        msg = 'invalid detector: %s' % new_det
        assert new_det in dets, msg_format(msg)
        
        self._det = new_det
        
        
    @property
    def chantype(self):
        
        return 'pi'
    
    
    @property
    def telescope(self):
        
        return 'GRID'
    
    
    @property
    def instrument(self):
        
        return 'GRID'
        
        
    @property
    def timezero_utc(self):
        
        return grid_met_to_utc(self.timezero)
    
    
    def filter_pi(self, p1p2):
        
        if p1p2 is None:
            expr = None
            
        elif isinstance(p1p2, list):
            p1, p2 = p1p2
            expr = f'(PI >= {p1}) * (PI <= {p2})'
            
        else:
            raise ValueError('p1p2 is extected to be list or None')
        
        self._filter_info['pi'] = expr
        
        self._filter_update()
        
        
    def _filter_update(self):
        
        self._clear_filter()
        
        self._filter.eval(self._filter_info['time'])
        self._filter.eval(self._filter_info['pi'])
        self._filter.eval(self._filter_info['energy'])
        self._filter.eval(self._filter_info['tag'])



class gridgroundTTE(Event):

    def __init__(self, file, det):
        
        self._file = file
        self._det = det
        
        self._read()


    def _read(self):
        
        event_list = list()
        
        for i in range(len(self.file)):
            tte_hdu = fits.open(self.file[i])
            event = table.Table.read(tte_hdu[f'EVENTS{self.det}'])
            ebound = table.Table.read(tte_hdu['EBOUNDS'])
            tte_hdu.close()
            
            ebound['Channel'].name = 'CHANNEL'
            
            pi = np.array(event['PI']).astype(int)
            ch = np.array(ebound['CHANNEL']).astype(int)
            emin = np.array(ebound['E_MIN'])
            emax = np.array(ebound['E_MAX'])
            energy = Event._ch_to_energy(pi, ch, emin, emax)
            event['ENERGY'] = energy * units.keV
            
            event_list.append(event)
            
        self._event = table.vstack(event_list)
        self._gti = None
        self._ebound = ebound

        self._event = table.unique(self._event, keys=['TIME', 'PI'])
        self._event.sort('TIME')
        
        self._timezero = np.min(self._event['TIME'])
        
        self._filter = Filter(self._event)
        self._filter_info = {'time': None, 'energy': None, 'pi': None, 'tag': None}
        
        
    @property
    def det(self):
        
        return self._det
    
    
    @det.setter
    def det(self, new_det):
        
        dets = ['%d' % i for i in range(0, 4)]
        msg = 'invalid detector: %s' % new_det
        assert new_det in dets, msg_format(msg)
        
        self._det = new_det
        
        
    @property
    def chantype(self):
        
        return 'pi'
    
    
    @property
    def telescope(self):
        
        return 'GRID'
    
    
    @property
    def instrument(self):
        
        return 'GRID'
        
        
    @property
    def timezero_utc(self):
        
        return grid_met_to_utc(self.timezero)
    
    
    def filter_pi(self, p1p2):
        
        if p1p2 is None:
            expr = None
            
        elif isinstance(p1p2, list):
            p1, p2 = p1p2
            expr = f'(PI >= {p1}) * (PI <= {p2})'
            
        else:
            raise ValueError('p1p2 is extected to be list or None')
        
        self._filter_info['pi'] = expr
        
        self._filter_update()
        
        
    def _filter_update(self):
        
        self._clear_filter()
        
        self._filter.eval(self._filter_info['time'])
        self._filter.eval(self._filter_info['pi'])
        self._filter.eval(self._filter_info['energy'])
        self._filter.eval(self._filter_info['tag'])
