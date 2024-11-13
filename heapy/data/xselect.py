import os
import shutil
import subprocess
import numpy as np
from astropy import table
from astropy.io import fits
import plotly.graph_objs as go
from .filter import Filter
from .retrieve import epRetrieve, swiftRetrieve
from ..temporal.txx import ppTxx
from ..util.data import json_dump
from ..autobs.ppsignal import ppSignal
from ..util.time import ep_met_to_utc, ep_utc_to_met
from ..util.time import swift_met_to_utc, swift_utc_to_met



class Xselect(object):
    
    def __init__(self, 
                 evtfile=None, 
                 regfile=None, 
                 bkregfile=None
                 ):
        
        self._evtfile = evtfile
        self._regfile = regfile
        self._bkregfile = bkregfile
        
        self._ini_xselect()


    @property
    def evtfile(self):
        
        return os.path.abspath(self._evtfile)


    @evtfile.setter
    def evtfile(self, new_evtfile):
        
        self._evtfile = new_evtfile
        
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
        
        
    def _ini_xselect(self):
            
        spectrum_savepath = os.path.dirname(self.evtfile) + '/spectrum'
        
        if os.path.isdir(spectrum_savepath):
            shutil.rmtree(spectrum_savepath)
                
        os.mkdir(spectrum_savepath)
            
        src_specfile = spectrum_savepath + '/src.pi'
        bkg_specfile = spectrum_savepath + '/bkg.pi'
        
        events_savepath = os.path.dirname(self.evtfile) + '/events'
        
        if os.path.isdir(events_savepath):
            shutil.rmtree(events_savepath)
                
        os.mkdir(events_savepath)
        
        src_evtfile = events_savepath + '/src.evt'
        bkg_evtfile = events_savepath + '/bkg.evt'
        
        commands = ['xsel', 
                    'read events', 
                    os.path.dirname(self.evtfile), 
                    self.evtfile.split('/')[-1], 
                    'yes', 
                    f'filter region {self.regfile}', 
                    'extract spectrum', 
                    f'save spectrum {src_specfile}', 
                    'extract events', 
                    f'save events {src_evtfile}', 
                    'no', 
                    'clear events', 
                    'clear region', 
                    f'filter region {self.bkregfile}', 
                    'extract spectrum', 
                    f'save spectrum {bkg_specfile}', 
                    'extract events', 
                    f'save events {bkg_evtfile}', 
                    'no']
        
        _, _ = self._run_xselect(commands)
        
        src_hdu = fits.open(src_specfile)
        self.src_backscale = src_hdu['SPECTRUM'].header['BACKSCAL']
        src_hdu.close()
        
        bkg_hdu = fits.open(bkg_specfile)
        self.bkg_backscale = bkg_hdu['SPECTRUM'].header['BACKSCAL']
        bkg_hdu.close()
        
        self.backscale = self.src_backscale / self.bkg_backscale
        
        self.src_evtfile = src_evtfile
        self.bkg_evtfile = bkg_evtfile
        
        hdu = fits.open(self.evtfile)
        self._event = table.Table.read(hdu['EVENTS'])
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
    def timezero(self):
        
        try:
            return self._timezero
        except AttributeError:
            hdu = fits.open(self._evtfile)
            return hdu['EVENTS'].header['TSTART']

    
    @timezero.setter
    def timezero(self, new_timezero):
        
        self._timezero = new_timezero
        
        
    @property
    def timezero_utc(self):
        
        return None
        
        
    def slice_time(self, t1, t2):
        
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
            return [np.min(self._event['TIME']), np.max(self._event['TIME'])]
        else:
            return self._time_filter


    @property
    def event(self):
        
        return self._filter.evt
        
        
    @property
    def src_event(self):
        
        return self._src_filter.evt
    
    
    @property
    def bkg_event(self):
        
        return self._bkg_filter.evt

        
    def extract_image(self, savepath='./image', show=False):
        
        savepath = os.path.abspath(savepath)
        
        if os.path.isdir(savepath):
            shutil.rmtree(savepath)
                
        os.mkdir(savepath)
        
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
    def lc_slice(self):
        
        return self.time_filter


    @property
    def lc_interval(self):
        
        return self.lc_slice[1] - self.lc_slice[0]
    
    
    @property
    def lc_binsize(self):
        
        try:
            return self._lc_binsize
        except AttributeError:
            return self.lc_interval / 300
            
            
    @lc_binsize.setter
    def lc_binsize(self, new_lc_binsize):
        
        if isinstance(new_lc_binsize, (int, float, type(None))):
            self._lc_binsize = new_lc_binsize
        else:
            raise ValueError('lc_binsize is extected to be int, float or None')


    @property
    def lc_bins(self):
        
        return np.arange(self.lc_slice[0], self.lc_slice[1] + 1e-5, self.lc_binsize)


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
    def lc_src_cts(self):
        
        return np.histogram(self.src_ts, bins=self.lc_bins)[0]
    
    
    @property
    def lc_src_cts_err(self):
        
        return np.sqrt(self.lc_src_cts)
    
    
    @property
    def lc_src_rate(self):
        
        return self.lc_src_cts / self.lc_binsize
    
    
    @property
    def lc_src_rate_err(self):
        
        return self.lc_src_cts_err / self.lc_binsize
    
    
    @property
    def lc_bkg_cts(self):
        
        return np.histogram(self.bkg_ts, bins=self.lc_bins)[0]
    
    
    @property
    def lc_bkg_cts_err(self):
        
        return np.sqrt(self.lc_bkg_cts)
        
        
    @property
    def lc_bkg_rate(self):
        
        return self.lc_bkg_cts / self.lc_binsize * self.backscale
    
    
    @property
    def lc_bkg_rate_err(self):
        
        return self.lc_bkg_cts_err / self.lc_binsize * self.backscale
    
    
    @property
    def lc_net_rate(self):
        
        return self.lc_src_rate - self.lc_bkg_rate
    
    
    @property
    def lc_net_rate_err(self):
        
        return np.sqrt(self.lc_src_rate_err ** 2 + self.lc_bkg_rate_err ** 2)
    
    
    @property
    def lc_net_cts(self):
        
        return self.lc_net_rate * self.lc_binsize
    
    
    @property
    def lc_net_cts_err(self):
        
        return self.lc_net_rate_err * self.lc_binsize
    
    
    @property
    def lc_net_ccts(self):
        
        return np.cumsum(self.lc_net_cts)
    
    
    @property
    def lc_ps(self):
        
        lc_ps = ppSignal(self.src_ts, self.bkg_ts, self.lc_bins, backscale=self.backscale)
        lc_ps.loop(sigma=3)
        
        return lc_ps
        
        
    def extract_curve(self, savepath='./curve', show=False):
        
        savepath = os.path.abspath(savepath)
        
        if os.path.isdir(savepath):
            shutil.rmtree(savepath)
                
        os.mkdir(savepath)
        
        self.lc_ps.save(savepath=savepath + '/ppsignal')
        
        fig = go.Figure()
        src = go.Scatter(x=self.lc_time, 
                         y=self.lc_src_rate, 
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
                             array=self.lc_src_rate_err,
                             thickness=1.5,
                             width=0), 
                         marker=dict(symbol='circle', size=3))
        bkg = go.Scatter(x=self.lc_time, 
                         y=self.lc_bkg_rate, 
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
                             array=self.lc_bkg_rate_err,
                             thickness=1.5,
                             width=0), 
                         marker=dict(symbol='circle', size=3))
        net = go.Scatter(x=self.lc_time, 
                         y=self.lc_net_rate, 
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
                             array=self.lc_net_rate_err,
                             thickness=1.5,
                             width=0), 
                         marker=dict(symbol='circle', size=3))
        
        fig.add_trace(src)
        fig.add_trace(bkg)
        fig.add_trace(net)

        fig.update_xaxes(title_text=f'Time since {self.timezero_utc} (s)')
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
        
        fig.update_xaxes(title_text=f'Time since {self.timezero_utc} (s)')
        fig.update_yaxes(title_text=f'Cumulated counts (binsize={self.lc_binsize} s)')
        fig.update_layout(template='plotly_white', height=600, width=800)
        fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))
        
        fig.write_html(savepath + '/cum_lc.html')
        json_dump(fig.to_dict(), savepath + '/cum_lc.json')

        
    def calculate_txx(self, sigma=3, mp=True, xx=0.9, pstart=None, pstop=None, 
                      lbkg=None, rbkg=None, savepath='./curve/duration'):
            
        savepath = os.path.abspath(savepath)
        
        if os.path.isdir(savepath):
            shutil.rmtree(savepath)
                
        os.mkdir(savepath)
        
        txx = ppTxx(self.src_ts, self.bkg_ts, self.lc_bins, self.backscale)
        txx.findpulse(sigma=sigma, mp=mp)
        txx.accumcts(xx=xx, pstart=pstart, pstop=pstop, lbkg=lbkg, rbkg=rbkg)
        txx.save(savepath=savepath)
        
        
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
        
        if os.path.isdir(savepath):
            shutil.rmtree(savepath)
                
        os.mkdir(savepath)
            
        json_dump(self.timezero, savepath + '/timezero.json')
        
        json_dump(self.spec_slices, savepath + '/spec_slices.json')
        
        lslices = np.array(self.spec_slices)[:, 0]
        rslices = np.array(self.spec_slices)[:, 1]
        
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



class epXselect(Xselect):
    
    def __init__(self, 
                 evtfile=None, 
                 regfile=None, 
                 bkregfile=None, 
                 armregfile=None, 
                 arm=False
                 ):
        
        self._evtfile = evtfile
        self._regfile = regfile
        self._bkregfile = bkregfile
        self._armregfile = armregfile
        self._arm = arm
        
        self._ini_xselect()


    @classmethod
    def from_wxtobs(cls, obsid, srcid, datapath=None):
        
        rtv = epRetrieve.from_wxtobs(obsid, srcid, datapath)

        evtfile = rtv.rtv_res['evt']
        regfile = rtv.rtv_res['reg']
        bkregfile = rtv.rtv_res['bkreg']
        armregfile = rtv.rtv_res['armreg']
        
        return cls(evtfile, regfile, bkregfile, armregfile)
    
    
    @classmethod
    def from_fxtobs(cls, obsid, module, datapath=None):
        
        rtv = epRetrieve.from_fxtobs(obsid, module, datapath)

        evtfile = rtv.rtv_res['evt']
        regfile = rtv.rtv_res['reg']
        bkregfile = rtv.rtv_res['bkreg']
        
        return cls(evtfile, regfile, bkregfile)
            
            
    @property
    def armregfile(self):
        
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
    def timezero(self):
        
        try:
            return self._timezero
        except AttributeError:
            hdu = fits.open(self._evtfile)
            return hdu['EVENTS'].header['TSTART']


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



class swiftXselect(Xselect):
    
    def __init__(self, 
                 evtfile=None, 
                 regfile=None, 
                 bkregfile=None
                 ):
        
        self._evtfile = evtfile
        self._regfile = regfile
        self._bkregfile = bkregfile
        
        self._ini_xselect()


    @classmethod
    def from_xrtobs(cls, obsid, datapath=None):
        
        rtv = swiftRetrieve.from_wxtobs(obsid, datapath)

        evtfile = rtv.rtv_res['evt']
        regfile = rtv.rtv_res['reg']
        bkregfile = rtv.rtv_res['bkreg']
        
        return cls(evtfile, regfile, bkregfile)
    
    
    @property
    def utcf(self):
        
        hdu = fits.open(self._evtfile)
        return hdu['EVENTS'].header['UTCFINIT']


    @property
    def timezero(self):
        
        try:
            return self._timezero
        except AttributeError:
            hdu = fits.open(self._evtfile)
            return hdu['EVENTS'].header['TSTART']


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
