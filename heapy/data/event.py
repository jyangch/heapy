import os
import json
import numpy as np
from astropy.io import fits
from astropy.time import Time
import plotly.graph_objs as go
from .reduction import Reduction
from ..temporal.txx import pgTxx
from ..util.data import NpEncoder
from ..autobs.polybase import PolyBase



class Event(Reduction):
    
    def __init__(self, reduction):
        
        self.reduction = reduction
        
        
    @property
    def reduction(self):
        
        return self._reduction
    
    
    @reduction.setter
    def reduction(self, new_reduction):
        
        if not isinstance(new_reduction, Reduction):
            raise TypeError('expected an instance of Reduction')
        
        self._reduction = new_reduction
        self.__dict__.update(new_reduction.__dict__)
        
    
    @property
    def timezero_utc(self):
        
        return self._reduction.timezero_utc
        
        
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
        
        return np.array(self.ebound['CHANNEL'])
    
    
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
            return [np.min(self.ts), np.max(self.ts)]
        else:
            if self._lc_t1t2 is not None:
                return self._lc_t1t2
            else:
                return [np.min(self.ts), np.max(self.ts)]
        
        
    @lc_t1t2.setter
    def lc_t1t2(self, new_lc_t1t2):
        
        if isinstance(new_lc_t1t2, (list, type(None))):
            self._lc_t1t2 = new_lc_t1t2
        else:
            raise ValueError('not expected lc_t1t2 type')


    @property
    def spec_t1t2(self):
        
        try:
            self._spec_t1t2
        except AttributeError:
            return [np.min(self.ts), np.max(self.ts)]
        else:
            if self._spec_t1t2 is not None:
                return self._spec_t1t2
            else:
                return [np.min(self.ts), np.max(self.ts)]


    @spec_t1t2.setter
    def spec_t1t2(self, new_spec_t1t2):
        
        if isinstance(new_spec_t1t2, (list, type(None))):
            self._spec_t1t2 = new_spec_t1t2
        else:
            raise ValueError('not expected spec_t1t2 type')
    
    
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
        
        self._lc_binsize = new_lc_binsize
        
        
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
        
        self._spec_binsize = new_spec_binsize
    
    
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
    
    
    def exposure(self, bin_list):
        
        ts = self._ts
        dtime = self._dtime
        
        lbins = np.array(bin_list)[:, 0]
        rbins = np.array(bin_list)[:, 1]
        binsize = rbins - lbins
        
        dead_time = np.empty_like(binsize)
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
        
        return self.exposure(self.lc_bin_list)
    
    
    @property
    def lc_time(self):
        
        return np.mean(self.lc_bin_list, axis=1)
    
    
    @property
    def lc_cts(self):
        
        return np.histogram(self.lc_ts, bins=self.lc_bins)[0]
    
    
    @property
    def lc_cts_err(self):
        
        return np.sqrt(self.lc_cts)
    
    
    @property
    def lc_rate(self):
        
        return self.lc_cts / self.lc_exps
    
    
    @property
    def lc_rate_err(self):
        
        return self.lc_cts_err / self.lc_exps
    
    
    @property
    def lc_bs(self):
        
        lc_bs = PolyBase(self.lc_ts, self.lc_bins, self.lc_exps)
        lc_bs.loop(sigma=3, deg=None)
        
        return lc_bs
    
    
    def extract_curve(self, savepath='./curve'):
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)
            
        self.lc_bs.save(savepath=savepath + '/polybase')
        
        lc_brate, lc_brate_err = self.lc_bs.poly.val(self.lc_time)
        
        lc_nrate = self.lc_rate - lc_brate
        lc_ncts = lc_nrate * self.lc_exps
        
        fig = go.Figure()
        src = go.Scatter(x=self.lc_time, 
                         y=self.lc_rate, 
                         mode='lines+markers', 
                         name='source lightcurve', 
                         showlegend=True, 
                         error_y=dict(
                             type='data',
                             array=self.lc_rate_err,
                             thickness=1.5,
                             width=0), 
                         marker=dict(symbol='cross-thin', size=0))
        bkg = go.Scatter(x=self.lc_time, 
                         y=lc_brate, 
                         mode='lines+markers', 
                         name='background lightcurve', 
                         showlegend=True, 
                         error_y=dict(
                             type='data',
                             array=lc_brate_err,
                             thickness=1.5,
                             width=0), 
                         marker=dict(symbol='cross-thin', size=0))
        
        fig.add_trace(src)
        fig.add_trace(bkg)
        
        fig.update_xaxes(title_text=f'Time since {self.timezero_utc} (s)')
        fig.update_yaxes(title_text=f'Counts per second (binsize={self.lc_binsize} s)')
        fig.update_layout(template='plotly_white', height=600, width=800)
        fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))
        
        fig.show()
        fig.write_html(savepath + '/lc.html')
        json.dump(fig.to_dict(), open(savepath + '/lc.json', 'w'), indent=4, cls=NpEncoder)
        
        lc_nccts = np.cumsum(lc_ncts)
        
        fig = go.Figure()
        net = go.Scatter(x=self.lc_time, 
                         y=lc_nccts, 
                         mode='lines', 
                         name='net cumulated counts', 
                         showlegend=True)
        
        fig.add_trace(net)
        
        fig.update_xaxes(title_text=f'Time since {self.timezero_utc} (s)')
        fig.update_yaxes(title_text=f'Cumulated counts (binsize={self.lc_binsize} s)')
        fig.update_layout(template='plotly_white', height=600, width=800)
        fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))
        
        fig.write_html(savepath + '/cum_lc.html')
        json.dump(fig.to_dict(), open(savepath + '/cum_lc.json', 'w'), indent=4, cls=NpEncoder)
        
        
    def calculate_txx(self, savepath='./duration'):
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        
        txx = pgTxx(self.lc_ts, self.lc_bins, self.lc_exps)
        txx.accumcts(self, xx=0.9, mp=True)
        txx.save(savepath=savepath)


    def _extract_src_phaii(self, spec_slices):
        
        num_slices = len(spec_slices)
        num_channels = len(self.channel)
        
        phaii = np.empty([num_slices, num_channels])
        
        lslices = np.array(spec_slices)[:, 0]
        rslices = np.array(spec_slices)[:, 1]
        
        pha_bins = np.arange(np.min(self.channel), np.max(self.channel) + 2, 1)
        
        for i, (l, r) in enumerate(zip(lslices, rslices)):
            pii = self.spec_pha[(self.spec_ts >= l) & (self.spec_ts < r)]
            pha, _ = np.histogram(pii, pha_bins)
            phaii[i, :] = pha
            
        return phaii
    
    
    def extract_src_phaii(self, spec_slices, savepath='./spectrum'):
            
        phaii = self._extract_src_phaii(spec_slices)
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)
                
        exps = self.exposure(spec_slices)
            
        lslices = np.array(spec_slices)[:, 0]
        rslices = np.array(spec_slices)[:, 1]
    
        for i, (l, r) in enumerate(zip(lslices, rslices)):
            new_l = '{:+.2f}'.format(l).replace('-', 'm').replace('.', 'd').replace('+', 'p')
            new_r = '{:+.2f}'.format(r).replace('-', 'm').replace('.', 'd').replace('+', 'p')
            
            file_name = '-'.join([new_l, new_r]) + '.src'
            
            pha_hdu = self._to_pha_fits(phaii[i], np.sqrt(phaii[i]), exps[i], file_name)

            if os.path.isfile(savepath + f'/{file_name}'):
                os.remove(savepath + f'/{file_name}')
            pha_hdu.writeto(savepath + f'/{file_name}')
                
                
    def _extract_bkg_phaii(self, spec_slices):
        
        num_slices = len(spec_slices)
        num_channels = len(self.channel)
        
        phaii = np.empty([num_slices, num_channels])
        phaii_err = np.empty([num_slices, num_channels])
        
        lslices = np.array(spec_slices)[:, 0]
        rslices = np.array(spec_slices)[:, 1]
        
        bins = np.arange(self.spec_t1t2[0], self.spec_t1t2[1]+1e-5, self.spec_binsize)
        
        interp_low = self.spec_t1t2[0] + self.spec_interval / 8
        interp_upp = self.spec_t1t2[1] - self.spec_interval / 8
        interp_range = [max([interp_low, np.mean(bins[:2])]), min([interp_upp, np.mean(bins[-2:])])]
        interp_time = np.linspace(interp_range[0], interp_range[-1], 100)
        
        bs = PolyBase(self.spec_ts, bins)
        bs.loop(sigma=3, deg=None)
        
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
            
            bs_i = PolyBase(ts_i, bins_i)
            bs_i.polyfit(deg=None, ignore=ignore)
            
            brate_i, _ = bs_i.poly.val(interp_time)
            
            brate_sum = brate_sum + brate_i
            
            for j, (l, r) in enumerate(zip(lslices, rslices)):
                
                bins_j = np.linspace(l, r, 100)
                brate_j, brate_err_j = bs_i.poly.val(bins_j)
                
                phaii[j, i] = np.trapz(brate_j, bins_j)
                phaii_err[j, i] = np.trapz(brate_err_j, bins_j)
                
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
        
        fig.show()
                
        return phaii, phaii_err
                
                
    def extract_bkg_phaii(self, spec_slices, savepath='./spectrum'):
        
        phaii, phaii_err = self._extract_bkg_phaii(spec_slices)
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)
            
        exps = self.exposure(spec_slices)
            
        lslices = np.array(spec_slices)[:, 0]
        rslices = np.array(spec_slices)[:, 1]
    
        for i, (l, r) in enumerate(zip(lslices, rslices)):
            new_l = '{:+.2f}'.format(l).replace('-', 'm').replace('.', 'd').replace('+', 'p')
            new_r = '{:+.2f}'.format(r).replace('-', 'm').replace('.', 'd').replace('+', 'p')
            
            file_name = '-'.join([new_l, new_r]) + '.bkg'
            
            pha_hdu = self._to_pha_fits(phaii[i], phaii_err[i], exps[i], file_name)
            
            if os.path.isfile(savepath + f'/{file_name}'):
                os.remove(savepath + f'/{file_name}')
            pha_hdu.writeto(savepath + f'/{file_name}')
                
                
    def extract_spectrum(self, spec_slices, savepath='./spectrum'):
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        
        self.extract_src_phaii(spec_slices, savepath=savepath)
        self.extract_bkg_phaii(spec_slices, savepath=savepath)


    def _to_pha_fits(self, pha, pha_err, exp, file_name):
        
        hdr = fits.Header()
        hdr['FILENAME'] = (f'{file_name}', 'name of this file')
        hdr['AUTHOR'] = ('JunYang', 'author of this file')
        hdr['EMAIL'] = ('jyang@smail.nju.edu.cn', 'email of author')
        primary_hdu = fits.PrimaryHDU(header=hdr)

        channel_field = fits.Column(name='CHANNEL', array=self.channel, format='I')
        counts_field = fits.Column(name='COUNTS', array=pha, format='J', unit='count')
        stat_err_field = fits.Column(name='STAT_ERR', array=pha_err, format='E', unit='count')
        spectrum_hdu = fits.BinTableHDU.from_columns([channel_field, counts_field, stat_err_field])

        spectrum_hdu.header['TTYPE1'] = ('CHANNEL', 'label for field 1')
        spectrum_hdu.header['TFORM1'] = ('I', 'data format of field: 2-byte INTEGER')
        spectrum_hdu.header['TTYPE2'] = ('COUNTS', 'label for field 2')
        spectrum_hdu.header['TFORM2'] = ('J', 'data format of field: 4-byte INTEGER')
        spectrum_hdu.header['TUNIT2'] = ('count', 'physical unit of field')
        spectrum_hdu.header['TTYPE3'] = ('STAT_ERR', 'label for field 3')
        spectrum_hdu.header['TFORM3'] = ('E', 'data format of field: 32-bit SINGLE FLOAT')
        spectrum_hdu.header['TUNIT3'] = ('count', 'physical unit of field')
        spectrum_hdu.header['EXTNAME'] = ('SPECTRUM', 'name of this binary table extension')

        spectrum_hdu.header['EXPOSURE'] = (round(exp, 5), 'exposure time in second')
        spectrum_hdu.header['AREASCAL'] = (1.0, 'nominal effective area')
        spectrum_hdu.header['CORRSCAL'] = (1.0, 'correlation scale factor')
        spectrum_hdu.header['BACKSCAL'] = (1.0, 'background scale factor')
        spectrum_hdu.header['BACKFILE'] = ('NONE', 'background FITS file for this object')
        spectrum_hdu.header['CORRFILE'] = ('NONE', 'correlation FITS file for this object')
        spectrum_hdu.header['RESPFILE'] = ('NONE', 'redistribution; RSP')
        spectrum_hdu.header['ANCRFILE'] = ('NONE', 'ancillary response; ARF')

        spectrum_hdu.header['DATE'] = (Time.now().isot, 'file generated date')
        spectrum_hdu.header['TIMEZERO'] = (round(self.timezero, 5), 'zero time in MET format')

        channel_field = fits.Column(name='CHANNEL', array=self.channel, format='I', unit='channel')
        e_min_field = fits.Column(name='E_MIN', array=self.channel_emin, format='E', unit='keV')
        e_max_field = fits.Column(name='E_MAX', array=self.channel_emax, format='E', unit='keV')
        ebound_hdu = fits.BinTableHDU.from_columns([channel_field, e_min_field, e_max_field])
        
        ebound_hdu.header['EXTNAME'] = ('EBOUNDS', 'name of this binary table extension')

        pha_hdu = fits.HDUList([primary_hdu, spectrum_hdu, ebound_hdu])
        
        return pha_hdu
