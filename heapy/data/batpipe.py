import os
import shutil
import subprocess
import numpy as np
from astropy.io import fits
import plotly.graph_objs as go
from .retrieve import swiftRetrieve
from ..util.data import json_dump, rebin
from ..util.time import swift_met_to_utc, swift_utc_to_met



class batPipe(object):
    
    os.environ["HEADASNOQUERY"] = "1"
    
    def __init__(self, 
                 ufevtfile=None, 
                 caldbfile=None, 
                 detmaskfile=None,
                 attfile=None,
                 auxfile=None
                 ):
        
        self._ufevtfile = ufevtfile
        self._caldbfile = caldbfile
        self._detmaskfile = detmaskfile
        self._attfile = attfile
        self._auxfile = auxfile
        
        self._general_processing()
        
        
    @classmethod
    def from_batobs(cls, obsid, datapath=None):
        
        rtv = swiftRetrieve.from_batobs(obsid, datapath)

        ufevtfile = rtv.rtv_res['ufevt']
        caldbfile = rtv.rtv_res['caldb']
        detmaskfile = rtv.rtv_res['detmask']
        attfile = rtv.rtv_res['att']
        auxfile = rtv.rtv_res['aux']
        
        return cls(ufevtfile, caldbfile, detmaskfile, attfile, auxfile)
    
    
    @property
    def ufevtfile(self):
        
        return os.path.abspath(self._ufevtfile)


    @ufevtfile.setter
    def ufevtfile(self, new_ufevtfile):
        
        self._ufevtfile = new_ufevtfile
        
        self._general_processing()
    
    
    @property
    def caldbfile(self):
        
        return os.path.abspath(self._caldbfile)


    @caldbfile.setter
    def caldbfile(self, new_caldbfile):
        
        self._caldbfile = new_caldbfile
        
        self._general_processing()
        
        
    @property
    def detmaskfile(self):
        
        return os.path.abspath(self._detmaskfile)


    @detmaskfile.setter
    def detmaskfile(self, new_detmaskfile):
        
        self._detmaskfile = new_detmaskfile
        
        self._general_processing()
        
        
    @property
    def attfile(self):
        
        return os.path.abspath(self._attfile)


    @attfile.setter
    def attfile(self, new_attfile):
        
        self._attfile = new_attfile
        
        self._general_processing()
        
        
    @property
    def auxfile(self):
        
        return os.path.abspath(self._auxfile)


    @auxfile.setter
    def auxfile(self, new_auxfile):
        
        self._auxfile = new_auxfile


    def _run_comands(self, commands):
        
        process = subprocess.Popen(commands, 
                                   stdin=subprocess.PIPE, 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True)

        stdout, stderr = process.communicate()

        return stdout, stderr
    
    
    def bateconvert(self, std=False):
        
        commands = ['bateconvert', 
                    f'infile={self.ufevtfile}',
                    f'calfile={self.caldbfile}',
                    'residfile=CALDB',
                    'pulserfile=CALDB',
                    'fltpulserfile=CALDB']
        
        stdout, stderr = self._run_comands(commands)
        
        if std: 
            print(stdout)
            print(stderr)
    
    
    def batbinevt(self, std=False):
        
        self.dpifile = self.general_savepath + '/grb.dpi'
        
        commands = ['batbinevt',
                    'weighted=no',
                    'outunits=counts',
                    f'infile={self.ufevtfile}',
                    f'outfile={self.dpifile}',
                    'outtype=dpi',
                    'timedel=0',
                    'timebinalg=uniform',
                    'energybins=-']
        
        stdout, stderr = self._run_comands(commands)
        
        if std: 
            print(stdout)
            print(stderr)
    
    
    def bathotpix(self, std=False):
        
        self.maskfile = self.general_savepath + '/grb.mask'
        
        commands = ['bathotpix',
                    f'detmask={self.detmaskfile}',
                    f'infile={self.dpifile}',
                    f'outfile={self.maskfile}']
        
        stdout, stderr = self._run_comands(commands)
        
        if std: 
            print(stdout)
            print(stderr)
    
    
    def batmaskwtevt(self, std=False):
        
        ra = fits.open(self.ufevtfile)['EVENTS'].header['RA_OBJ']
        dec = fits.open(self.ufevtfile)['EVENTS'].header['DEC_OBJ']
        
        commands = ['batmaskwtevt',
                    f'detmask={self.maskfile}',
                    f'infile={self.ufevtfile}',
                    f'attitude={self.attfile}',
                    f'ra={ra}', 
                    f'dec={dec}']
        
        stdout, stderr = self._run_comands(commands)
        
        if std: 
            print(stdout)
            print(stderr)
    
    
    def _general_processing(self):
        
        self.general_savepath = os.path.dirname(os.path.dirname(self.ufevtfile)) + '/batpipe'
        
        if os.path.isdir(self.general_savepath):
            shutil.rmtree(self.general_savepath)
                
        os.mkdir(self.general_savepath)
        
        self.bateconvert()
        self.batbinevt()
        self.bathotpix()
        self.batmaskwtevt()
    
    
    def batbinevt_image(self, savepath, std=False):
        
        self.dpi4file = savepath + '/grb_4.dpi'
        
        commands = ['batbinevt', 
                    f'detmask={self.maskfile}', 
                    'ecol=energy',
                    'weighted=no',
                    'outunits=counts',
                    f'infile={self.ufevtfile}',
                    f'outfile={self.dpi4file}',
                    'outtype=dpi',
                    'timedel=0',
                    'timebinalg=uniform',
                    'energybins=15-25, 25-50, 50-100, 100-350']
        
        stdout, stderr = self._run_comands(commands)
        
        if std: 
            print(stdout)
            print(stderr)
    
    
    def batfftimage(self, savepath, std=False):
        
        self.img4file = savepath + '/grb_4.img'
        
        commands = ['batfftimage', 
                    f'detmask={self.maskfile}',
                    f'infile={self.dpi4file}',
                    f'outfile={self.img4file}',
                    f'attitude={self.attfile}']
        
        stdout, stderr = self._run_comands(commands)
        
        if std: 
            print(stdout)
            print(stderr)
            
            
    def extract_image(self, savepath='./image', std=False):
        
        savepath = os.path.abspath(savepath)
        
        if os.path.isdir(savepath):
            shutil.rmtree(savepath)
                
        os.mkdir(savepath)
        
        self.batbinevt_image(savepath=savepath, std=std)
        self.batfftimage(savepath=savepath, std=std)
        
        
    @property
    def utcf(self):
        
        hdu = fits.open(self._ufevtfile)
        val = hdu['EVENTS'].header['UTCFINIT']
        hdu.close()
        
        return val
    
    
    @property
    def trigtime(self):
        
        hdu = fits.open(self._ufevtfile)
        val = hdu['EVENTS'].header['TRIGTIME']
        hdu.close()
        
        return val
    
    
    @property
    def tstart(self):
        
        hdu = fits.open(self._ufevtfile)
        val = hdu['EVENTS'].header['TSTART']
        hdu.close()
        
        return val
    
    
    @property
    def tstop(self):
        
        hdu = fits.open(self._ufevtfile)
        val = hdu['EVENTS'].header['TSTOP']
        hdu.close()
        
        return val


    @property
    def timezero(self):
        
        try:
            return self._timezero
        except AttributeError:
            return self.trigtime


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
    
    
    def filter_time(self, t1t2):
        
        if isinstance(t1t2, (list, type(None))):
            self._time_filter = t1t2
            
        else:
            raise ValueError('t1t2 is extected to be list or None')
    
    
    @property
    def time_filter(self):
        
        try:
            self._time_filter
        except AttributeError:
            return [self.tstart - self.timezero, self.tstop - self.timezero]
        else:
            if self._time_filter is None:
                return [self.tstart - self.timezero, self.tstop - self.timezero]
            else:
                return self._time_filter
            
        
    def filter_energy(self, e1e2):
        
        if isinstance(e1e2, (list, type(None))):
            self._energy_filter = e1e2
            
        else:
            raise ValueError('e1e2 is extected to be list or None')


    @property
    def energy_filter(self):
        
        try:
            self._energy_filter
        except AttributeError:
            return [15, 350]
        else:
            if self._energy_filter is None:
                return [15, 350]
            else:
                return self._energy_filter


    @property
    def lc_binsize(self):
        
        try:
            return self._lc_binsize
        except AttributeError:
            return 1


    @lc_binsize.setter
    def lc_binsize(self, new_lc_binsize):
        
        self._lc_binsize = new_lc_binsize
    

    def batbinevt_curve(self, savepath, std=False):
        
        self.lcfile = savepath + '/grb.lc'
        
        tstart = self.time_filter[0] + self.timezero
        tstop = self.time_filter[1] + self.timezero
        ebin = f'{self.energy_filter[0]}-{self.energy_filter[1]}'
        
        commands = ['batbinevt',
                    f'detmask={self.maskfile}',
                    f'tstart={tstart}',
                    f'tstop={tstop}',
                    f'infile={self.ufevtfile}',
                    f'outfile={self.lcfile}',
                    'outtype=lc',
                    f'timedel={self.lc_binsize}',
                    'timebinalg=uniform',
                    f'energybins={ebin}']
        
        stdout, stderr = self._run_comands(commands)
        
        if std: 
            print(stdout)
            print(stderr)


    def extract_curve(self, savepath='./curve', std=False, show=False):
        
        savepath = os.path.abspath(savepath)
        
        if os.path.isdir(savepath):
            shutil.rmtree(savepath)
                
        os.mkdir(savepath)
        
        self.batbinevt_curve(savepath=savepath, std=std)
        
        lc_hdu = fits.open(self.lcfile)
        self.lc_time = np.array(lc_hdu['RATE'].data['TIME']) - self.timezero
        self.lc_net_rate = np.array(lc_hdu['RATE'].data['RATE'])
        self.lc_net_rate_err = np.array(lc_hdu['RATE'].data['ERROR'])
        self.lc_net_crate = np.cumsum(self.lc_net_rate)
        
        fig = go.Figure()
        net = go.Scatter(x=self.lc_time, 
                         y=self.lc_net_rate, 
                         mode='lines+markers', 
                         name='net counts rate', 
                         showlegend=True, 
                         error_y=dict(
                             type='data',
                             array=self.lc_net_rate_err,
                             thickness=1.5,
                             width=0), 
                         marker=dict(symbol='cross-thin', size=0))
        fig.add_trace(net)

        fig.update_xaxes(title_text=f'Time since {self.timezero_utc} (s)')
        fig.update_yaxes(title_text=f'Counts per second per detector (binsize={self.lc_binsize} s)')
        fig.update_layout(template='plotly_white', height=600, width=800)
        fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))

        if show: fig.show()
        fig.write_html(savepath + '/lc.html')
        json_dump(fig.to_dict(), savepath + '/lc.json')
        
        fig = go.Figure()
        net = go.Scatter(x=self.lc_time, 
                         y=self.lc_net_crate, 
                         mode='lines', 
                         name='net cumulated rate', 
                         showlegend=True)
        fig.add_trace(net)
        
        fig.update_xaxes(title_text=f'Time since {self.timezero_utc} (s)')
        fig.update_yaxes(title_text=f'Cumulated count rate (binsize={self.lc_binsize} s)')
        fig.update_layout(template='plotly_white', height=600, width=800)
        fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))
        
        fig.write_html(savepath + '/cum_lc.html')
        json_dump(fig.to_dict(), savepath + '/cum_lc.json')
        
        
    def extract_rebin_curve(self, min_sigma=None, min_evt=None, max_bin=None, 
                            savepath='./curve', loglog=False, std=False, show=False):
        
        savepath = os.path.abspath(savepath)
        
        if os.path.isdir(savepath):
            shutil.rmtree(savepath)
                
        os.mkdir(savepath)
        
        self.batbinevt_curve(savepath=savepath, std=std)
        
        lc_hdu = fits.open(self.lcfile)
        self.lc_time = np.array(lc_hdu['RATE'].data['TIME']) - self.timezero
        self.lc_net_rate = np.array(lc_hdu['RATE'].data['RATE'])
        self.lc_net_rate_err = np.array(lc_hdu['RATE'].data['ERROR'])
        
        lbins, rbins = self.lc_time - self.lc_binsize / 2, self.lc_time + self.lc_binsize / 2
        self.lc_bin_list = np.vstack((lbins, rbins)).T
        
        self.lc_net_cts = self.lc_net_rate * self.lc_binsize
        self.lc_net_cts_err = self.lc_net_rate_err * self.lc_binsize
        
        self.lc_rebin_list, self.lc_net_rects, self.lc_net_rects_err, _, _ = \
            rebin(
                self.lc_bin_list, 
                'gstat', 
                self.lc_net_cts, 
                cts_err=self.lc_net_cts_err,
                bcts=None, 
                bcts_err=None, 
                min_sigma=min_sigma, 
                min_evt=min_evt, 
                max_bin=max_bin,
                backscale=1)
                
        self.lc_retime = np.mean(self.lc_rebin_list, axis=1)
        self.lc_rebinsize = self.lc_rebin_list[:, 1] - self.lc_rebin_list[:, 0]
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
        
        lslices = np.array(self.spec_slices)[:, 0]
        rslices = np.array(self.spec_slices)[:, 1]
        
        for l, r in zip(lslices, rslices):
            tstart = self.timezero + l
            tstop = self.timezero + r
            
            new_l = '{:+.2f}'.format(l).replace('-', 'm').replace('.', 'd').replace('+', 'p')
            new_r = '{:+.2f}'.format(r).replace('-', 'm').replace('.', 'd').replace('+', 'p')
            
            file_name = '-'.join([new_l, new_r])
            
            specfile = savepath + f'/{file_name}.pha'
        
            commands = ['batbinevt',
                        f'detmask={self.maskfile}',
                        f'tstart={tstart}',
                        f'tstop={tstop}',
                        f'infile={self.ufevtfile}',
                        f'outfile={specfile}',
                        'outtype=pha',
                        'timedel=0',
                        'timebinalg=uniform',
                        'energybins=CALDB:80']
        
            stdout, stderr = self._run_comands(commands)
            
            if std: 
                print(stdout)
                print(stderr)
                
            commands = ['batphasyserr',
                        f'infile={specfile}',
                        'syserrfile=CALDB']
        
            stdout, stderr = self._run_comands(commands)
            
            if std: 
                print(stdout)
                print(stderr)
                
            commands = ['batupdatephakw',
                        f'infile={specfile}',
                        f'auxfile={self.auxfile}']
        
            stdout, stderr = self._run_comands(commands)
            
            if std: 
                print(stdout)
                print(stderr)
                
            respfile = savepath + f'/{file_name}.rsp'
                
            commands = ['batdrmgen',
                        f'infile={specfile}',
                        f'outfile={respfile}']
        
            stdout, stderr = self._run_comands(commands)
            
            if std: 
                print(stdout)
                print(stderr)
