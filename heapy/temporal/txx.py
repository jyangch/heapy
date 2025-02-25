import os
import json
import warnings
import operator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.interpolate import interp1d
from astropy.stats import sigma_clip, mad_std
from ..autobs.ppsignal import ppSignal
from ..autobs.polybase import PolyBase
from ..util.data import asym_gaus_gen, msg_format, json_dump



class pgTxx(PolyBase):

    def __init__(self, ts, bins, exp=None):
        super().__init__(ts, bins, exp=exp)
        self.pulse_res = None
        self.txx_res = None

    
    def findpulse(self, sigma=3, deg=None, mp=True):
        if self.poly_res is None:
            self.loop(sigma=sigma, deg=deg)
            self.loop(sigma=sigma, deg=deg)

        self.ncts = self.cts - self.bcts
        self.net = self.rate - self.bak

        self.re_ncts = self.re_cts - self.re_bcts
        self.re_net = self.re_ncts / self.re_binsize

        start, stop = (0, 0)
        flat, up, down = (1, 2, 3)
        label = []
        nblock = len(self.re_net)
        for i, val in enumerate(self.re_net):
            if i == 0:
                label.append(start)
            elif i == (nblock - 1):
                label.append(stop)
            else:
                if val > self.re_net[i - 1]:
                    label.append(up)
                elif val < self.re_net[i - 1]:
                    label.append(down)
                else:
                    label.append(flat)

        flag = False
        pstart, pstop = [], []
        for i in range(nblock):
            if (self.re_snr[i] > sigma) and (flag is False):
                assert label[i] != down
                pstart.append(self.edges[i])
                flag = True
            elif (self.re_snr[i] <= sigma) and (flag is True):
                assert label[i] != up
                pstop.append(self.edges[i])
                flag = False

        if flag:
            if len(pstart) > 0:
                pstart.pop()
        if len(pstart) > 0:
            if pstart[0] == self.edges[0]:
                pstart = pstart[1:]
                pstop = pstop[1:]

        if (not mp) and (len(pstart) > 0):
            if len(pstart) > 1:
                msg = 'multi-pulse will be combined into one'
                warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
            pstart = [pstart[0]]
            pstop = [pstop[-1]]
        
        self.pstart = np.array(pstart)
        self.pstop = np.array(pstop)

        self.pulse_res = {'pstart': self.pstart, 'pstop': self.pstop}

    
    def accumcts(self, xx=0.9, mp=True, pstart=None, pstop=None, lbkg=None, rbkg=None):
        if self.pulse_res is None: self.findpulse(mp=mp)

        self.xx = xx

        if len(self.pstart) == 0:
            msg = 'there is no pulse'
            warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
            return False
        
        if pstart is not None:
            self.pstart[0] = pstart
            
        if pstop is not None:
            self.pstop[-1] = pstop

        if lbkg is None:
            lbkg = np.inf
            
        if rbkg is None:
            rbkg = np.inf

        tmin_ = self.pstart[0] - lbkg
        tmin = max(tmin_, self.time[0])
        tmax_ = self.pstop[-1] + rbkg
        tmax = min(tmax_, self.time[-1])
        tindex = np.where((self.time >= tmin) & (self.time <= tmax))[0]

        self.ccts = np.cumsum(self.ncts)

        idx = np.argsort(self.pstop - self.pstart)[0]
        if len(np.where((self.time >= self.pstart[idx]) & (self.time <= self.pstop[idx]))[0]) < 1000:
            interp_dt = (self.pstop[idx] - self.pstart[idx]) / 1000
            interp_time = np.arange(tmin, tmax - 1e-5, interp_dt)
            interp = interp1d(self.time, self.ccts, kind='quadratic')
            interp_ccts = interp(interp_time)
        else:
            interp_time = self.time[tindex]
            interp_ccts = self.ccts[tindex]

        self.csf, self.csf1, self.csf2 = [], [], []
        self.csf_err, self.csf1_err, self.csf2_err = [], [], []
        self.txx, self.txx1, self.txx2 = [], [], []
        self.txx_err, self.txx1_err, self.txx2_err = [], [], []

        for l, r in zip(np.append(tmin, self.pstop), np.append(self.pstart, tmax)):
            csf_i = np.mean(interp_ccts[np.where((interp_time >= l) & (interp_time <= r))])
            csf_err_i = np.std(interp_ccts[np.where((interp_time >= l) & (interp_time <= r))])
            self.csf.append(csf_i)
            self.csf_err.append(csf_err_i)
        dcsf = np.array(self.csf[1:]) - np.array(self.csf[:-1])

        for pi, (l, r) in enumerate(zip(np.append(tmin, self.pstop[:-1]), np.append(self.pstart[1:], tmax))):
            nn = (1 - xx) / 2
            dd = dcsf[pi] * nn
            
            csf1_i = self.csf[pi] + dd
            csf2_i = self.csf[pi + 1] - dd

            self.csf1.append(csf1_i)
            self.csf2.append(csf2_i)

            csf1_err_i = np.sqrt((1 - nn) ** 2 * self.csf_err[pi] ** 2 + nn ** 2 * self.csf_err[pi + 1] ** 2)
            csf2_err_i = np.sqrt(nn ** 2 * self.csf_err[pi] ** 2 + (1 - nn) ** 2 * self.csf_err[pi + 1] ** 2)
            self.csf1_err.append(csf1_err_i)
            self.csf2_err.append(csf2_err_i)

            pt = interp_time[np.where((interp_time >= l) & (interp_time <= r))]
            pcts = interp_ccts[np.where((interp_time >= l) & (interp_time <= r))]

            txx_i, txx1_i, txx2_i = self.find_txx(pt, pcts, csf1_i, csf2_i)
            _, txx1_lo_i, txx2_lo_i = self.find_txx(pt, pcts, csf1_i - csf1_err_i, csf2_i - csf2_err_i)
            _, txx1_hi_i, txx2_hi_i = self.find_txx(pt, pcts, csf1_i + csf1_err_i, csf2_i + csf2_err_i)
            txx1_le_i, txx1_he_i = txx1_i - txx1_lo_i, txx1_hi_i - txx1_i
            txx2_le_i, txx2_he_i = txx2_i - txx2_lo_i, txx2_hi_i - txx2_i

            txx1_i_sam = asym_gaus_gen(txx1_i, txx1_le_i, txx1_he_i, 1000)
            txx2_i_sam = asym_gaus_gen(txx2_i, txx2_le_i, txx2_he_i, 1000)
            txx_lo_i, txx_hi_i = np.percentile(txx2_i_sam - txx1_i_sam, [16, 84])
            txx_le_i, txx_he_i = txx_i - txx_lo_i, txx_hi_i - txx_i

            self.txx.append(txx_i)
            self.txx1.append(txx1_i)
            self.txx2.append(txx2_i)

            self.txx_err.append([txx_le_i, txx_he_i])
            self.txx1_err.append([txx1_le_i, txx1_he_i])
            self.txx2_err.append([txx2_le_i, txx2_he_i])

        self.txx_res = {'xx': self.xx, 'txx':self.txx, 'txx1': self.txx1, 'txx2': self.txx2, 
                        'txx_err': self.txx_err, 'txx1_err': self.txx1_err, 'txx2_err': self.txx2_err, 
                        'csf': self.csf, 'csf1': self.csf1, 'csf2': self.csf2, 'csf_err': self.csf_err, 
                        'csf1_err': self.csf1_err, 'csf2_err': self.csf2_err, 'bins': self.bins, 
                        'cts': self.cts, 'bcts': self.bcts, 'ccts': self.ccts}
        
        print('\n+------------------------------------------------+')
        print(' %-5s%-10s%-8s%-8s%-8s%-8s' % ('id#', 'Txx', 'Txx-', 'Txx+', 'Txx1', 'Txx2'))
        print('+-----------------------------------------------+')
        _ = list(map(print, [' %-5d%-10.3f%-8.3f%-8.3f%-8.3f%-8.3f' % (i+1, t, t_err[0], t_err[1], t1, t2) 
                             for i, (t, t_err, t1, t2) in enumerate(zip(self.txx, self.txx_err, self.txx1, self.txx2))]))
        print('+------------------------------------------------+')


    @staticmethod
    def find_txx(time, ccts, csf1, csf2):
        interp_time = np.linspace(time[0], time[-1], 1000)
        interp = interp1d(time, ccts, kind='linear')
        interp_ccts = interp(interp_time)

        txx1, txx2 = 0, 0
        for i in range(1, len(interp_time)):
            if interp_ccts[i] < csf1:
                continue
            elif (interp_ccts[i-1] < csf1) and (interp_ccts[i] >= csf1):
                txx1 = interp_time[i]
                continue
            elif (csf1 <= interp_ccts[i-1] < csf2) and (csf1 < interp_ccts[i] <= csf2):
                continue
            elif (csf1 < interp_ccts[i-1] <= csf2) and (interp_ccts[i] > csf2):
                txx2 = interp_time[i-1]
                break
            else:
                continue

        txx = txx2 - txx1
        return txx, txx1, txx2


    def save(self, savepath, suffix=''):
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        json_dump(self.pulse_res, savepath + '/pulse_res%s.json'%suffix)
        json_dump(self.txx_res, savepath + '/txx_res%s.json'%suffix)

        rcParams['font.family'] = 'serif'
        rcParams['font.sans-serif'] = 'Georgia'
        rcParams['text.usetex'] = True
        rcParams['font.size'] = 12
        rcParams['pdf.fonttype'] = 42

        fig = plt.figure(figsize=(7, 6))
        gs = fig.add_gridspec(2, 1, wspace=0, hspace=0)
        ax1 = fig.add_subplot(gs[:1, 0])
        ax2 = fig.add_subplot(gs[1:, 0], sharex=ax1)

        ax1.plot(self.time, self.rate, color='k', lw=1.0, label='Light Curve')
        ax1.plot(self.time, self.bak, color='r', lw=1.0, label='Background')
        for i in range(len(self.txx1)):
            ax1.axvline(self.txx1[i], color='g', lw=1.0, ls='--')
            ax1.axvline(self.txx2[i], color='g', lw=1.0, ls='--')
        ax1.set_ylabel('Rate')
        ax1.set_xlim([self.time[0], self.time[-1]])
        ax1.minorticks_on()
        ax1.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
        ax1.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
        ax1.tick_params(which='major', width=1.0, length=5)
        ax1.tick_params(which='minor', width=1.0, length=3)
        ax1.xaxis.set_ticks_position('both')
        ax1.yaxis.set_ticks_position('both')
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.legend(frameon=False)

        ax2.plot(self.time, self.ccts, color='k', lw=1.0)
        for i in range(len(self.txx1)):
            ax2.axvline(self.txx1[i], color='g', lw=1.0, ls='--')
            ax2.axvline(self.txx2[i], color='g', lw=1.0, ls='--')
        for csfi in self.csf:
            ax2.axhline(csfi, color='orange', lw=1.0)
        for i in range(len(self.csf1)):
            ax2.axhline(self.csf1[i], color='orange', lw=1.0, ls='--')
            ax2.axhline(self.csf2[i], color='orange', lw=1.0, ls='--')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Accumulated counts')
        ax2.minorticks_on()
        ax2.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
        ax2.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
        ax2.tick_params(which='major', width=1.0, length=5)
        ax2.tick_params(which='minor', width=1.0, length=3)
        ax2.xaxis.set_ticks_position('both')
        ax2.yaxis.set_ticks_position('both')
        fig.savefig(savepath + '/txx%s.pdf'%suffix, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close(fig)



class ppTxx(ppSignal):

    def __init__(self, ts, bts, bins, backscale=1, exp=None):
        super().__init__(ts, bts, bins, backscale=backscale, exp=exp)
        self.pulse_res = None
        self.txx_res = None

    
    def findpulse(self, sigma=3, mp=True):
        if self.sort_res is None:
            self.loop(sigma=sigma)

        self.ncts = self.cts - self.bcts * self.backscale
        self.net = self.rate - self.bak

        self.re_ncts = self.re_cts - self.re_bcts * self.backscale
        self.re_net = self.re_ncts / self.re_binsize

        start, stop = (0, 0)
        flat, up, down = (1, 2, 3)
        label = []
        nblock = len(self.re_net)
        for i, val in enumerate(self.re_net):
            if i == 0:
                label.append(start)
            elif i == (nblock - 1):
                label.append(stop)
            else:
                if val > self.re_net[i - 1]:
                    label.append(up)
                elif val < self.re_net[i - 1]:
                    label.append(down)
                else:
                    label.append(flat)

        flag = False
        pstart, pstop = [], []
        for i in range(nblock):
            if (self.re_snr[i] > sigma) and (flag is False):
                assert label[i] != down
                pstart.append(self.edges[i])
                flag = True
            elif (self.re_snr[i] <= sigma) and (flag is True):
                assert label[i] != up
                pstop.append(self.edges[i])
                flag = False

        if flag:
            if len(pstart) > 0:
                pstart.pop()
        if len(pstart) > 0:
            if pstart[0] == self.edges[0]:
                pstart = pstart[1:]
                pstop = pstop[1:]

        if (not mp) and (len(pstart) > 0):
            if len(pstart) > 1:
                msg = 'multi-pulse will be combined into one'
                warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
            pstart = [pstart[0]]
            pstop = [pstop[-1]]
        
        self.pstart = np.array(pstart)
        self.pstop = np.array(pstop)

        self.pulse_res = {'pstart': self.pstart, 'pstop': self.pstop}
        
        
    def mc_simulation(self, nmc):
        self.nmc = int(nmc)
        self.mc_ncts = [self.ncts]
        for _ in range(self.nmc):
            ppmc = np.random.poisson(lam=self.cts) - np.random.poisson(lam=self.bcts) * self.backscale
            self.mc_ncts.append(ppmc)
            
            
    def accumcts(self, xx=0.9, mp=True, pstart=None, pstop=None, lbkg=None, rbkg=None):
        if self.pulse_res is None: self.findpulse(mp=mp)
        
        self.mc_simulation(1000)

        self.xx = xx

        if len(self.pstart) == 0:
            msg = 'there is no pulse'
            warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
            return False
        
        if pstart is not None:
            self.pstart[0] = pstart
            
        if pstop is not None:
            self.pstop[-1] = pstop

        if lbkg is None:
            lbkg = np.inf
            
        if rbkg is None:
            rbkg = np.inf

        tmin_ = self.pstart[0] - lbkg
        tmin = max(tmin_, self.time[0])
        tmax_ = self.pstop[-1] + rbkg
        tmax = min(tmax_, self.time[-1])
        tindex = np.where((self.time >= tmin) & (self.time <= tmax))[0]
        
        mc_csf, mc_csf1, mc_csf2 = [], [], []
        mc_txx, mc_txx1, mc_txx2 = [], [], []
        
        self.ccts = np.cumsum(self.mc_ncts[0])
        
        for ncts in self.mc_ncts:
            ccts = np.cumsum(ncts)

            idx = np.argsort(self.pstop - self.pstart)[0]
            if len(np.where((self.time >= self.pstart[idx]) & (self.time <= self.pstop[idx]))[0]) < 1000:
                interp_dt = (self.pstop[idx] - self.pstart[idx]) / 1000
                interp_time = np.arange(tmin, tmax - 1e-5, interp_dt)
                interp = interp1d(self.time, ccts, kind='quadratic')
                interp_ccts = interp(interp_time)
            else:
                interp_time = self.time[tindex]
                interp_ccts = ccts[tindex]
                
            csf, csf_err, csf1, csf2 = [], [], [], []
            txx, txx1, txx2 = [], [], []

            for l, r in zip(np.append(tmin, self.pstop), np.append(self.pstart, tmax)):
                csf_i = np.mean(interp_ccts[np.where((interp_time >= l) & (interp_time <= r))])
                csf_err_i = np.std(interp_ccts[np.where((interp_time >= l) & (interp_time <= r))])
                csf.append(csf_i)
                csf_err.append(csf_err_i)
                
            dcsf = np.array(csf[1:]) - np.array(csf[:-1])

            for pi, (l, r) in enumerate(zip(np.append(tmin, self.pstop[:-1]), np.append(self.pstart[1:], tmax))):
                nn = (1 - xx) / 2
                dd = dcsf[pi] * nn

                csf1_i = csf[pi] + dd
                csf2_i = csf[pi + 1] - dd
                    
                csf1.append(csf1_i)
                csf2.append(csf2_i)

                pt = interp_time[np.where((interp_time >= l) & (interp_time <= r))]
                pcts = interp_ccts[np.where((interp_time >= l) & (interp_time <= r))]

                txx_i, txx1_i, txx2_i = self.find_txx(pt, pcts, csf1_i, csf2_i)

                txx.append(txx_i)
                txx1.append(txx1_i)
                txx2.append(txx2_i)
                
            mc_csf.append(csf)
            mc_csf1.append(csf1)
            mc_csf2.append(csf2)
            mc_txx.append(txx)
            mc_txx1.append(txx1)
            mc_txx2.append(txx2)
            
        self.csf = mc_csf[0]
        self.csf1 = mc_csf1[0]
        self.csf2 = mc_csf2[0]
        
        self.txx = mc_txx[0]
        self.txx1 = mc_txx1[0]
        self.txx2 = mc_txx2[0]

        self.txx_err = []
        self.txx1_err = []
        self.txx2_err = []
        mc_txx = np.array(mc_txx)
        mc_txx1 = np.array(mc_txx1)
        mc_txx2 = np.array(mc_txx2)
        for pi in range(len(self.pstart)):
            mask = sigma_clip(mc_txx[1:, pi], sigma=5, maxiters=5, stdfunc=mad_std).mask
            not_mask = list(map(operator.not_, mask))
            mc_txx_filter = np.array(mc_txx[1:, pi])[not_mask]

            txx_lo, txx_hi = np.percentile(mc_txx_filter, [16, 84])
            txx_err = np.diff([txx_lo, mc_txx[0, pi], txx_hi])
            self.txx_err.append([txx_err[0], txx_err[1]])
            
            mask = sigma_clip(mc_txx1[1:, pi], sigma=5, maxiters=5, stdfunc=mad_std).mask
            not_mask = list(map(operator.not_, mask))
            mc_txx1_filter = np.array(mc_txx1[1:, pi])[not_mask]

            txx1_lo, txx1_hi = np.percentile(mc_txx1_filter, [16, 84])
            txx1_err = np.diff([txx1_lo, mc_txx1[0, pi], txx1_hi])
            self.txx1_err.append([txx1_err[0], txx1_err[1]])
            
            mask = sigma_clip(mc_txx2[1:, pi], sigma=5, maxiters=5, stdfunc=mad_std).mask
            not_mask = list(map(operator.not_, mask))
            mc_txx2_filter = np.array(mc_txx2[1:, pi])[not_mask]

            txx2_lo, txx2_hi = np.percentile(mc_txx2_filter, [16, 84])
            txx2_err = np.diff([txx2_lo, mc_txx2[0, pi], txx2_hi])
            self.txx2_err.append([txx2_err[0], txx2_err[1]])

        self.txx_res = {'xx': self.xx, 'txx':self.txx, 'txx1': self.txx1, 'txx2': self.txx2, 
                        'txx_err': self.txx_err, 'txx1_err': self.txx1_err, 'txx2_err': self.txx2_err, 
                        'csf': self.csf, 'csf1': self.csf1, 'csf2': self.csf2, 'bins': self.bins, 
                        'cts': self.cts, 'bcts': self.bcts, 'ccts': self.ccts}
        
        
        print('\n+------------------------------------------------+')
        print(' %-5s%-10s%-8s%-8s%-8s%-8s' % ('id#', 'Txx', 'Txx-', 'Txx+', 'Txx1', 'Txx2'))
        print('+-----------------------------------------------+')
        _ = list(map(print, [' %-5d%-10.3f%-8.3f%-8.3f%-8.3f%-8.3f' % (i+1, t, t_err[0], t_err[1], t1, t2) 
                             for i, (t, t_err, t1, t2) in enumerate(zip(self.txx, self.txx_err, self.txx1, self.txx2))]))
        print('+------------------------------------------------+')


    @staticmethod
    def find_txx(time, ccts, csf1, csf2):
        interp_time = np.linspace(time[0], time[-1], 1000)
        interp = interp1d(time, ccts, kind='linear')
        interp_ccts = interp(interp_time)

        txx1, txx2 = 0, 0
        for i in range(1, len(interp_time)):
            if interp_ccts[i] < csf1:
                continue
            elif (interp_ccts[i-1] < csf1) and (interp_ccts[i] >= csf1):
                txx1 = interp_time[i]
                continue
            elif (csf1 <= interp_ccts[i-1] < csf2) and (csf1 < interp_ccts[i] <= csf2):
                continue
            elif (csf1 < interp_ccts[i-1] <= csf2) and (interp_ccts[i] > csf2):
                txx2 = interp_time[i-1]
                break
            else:
                continue

        txx = txx2 - txx1
        return txx, txx1, txx2
    
    
    def save(self, savepath, suffix=''):
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        json_dump(self.pulse_res, savepath + '/pulse_res%s.json'%suffix)
        json_dump(self.txx_res, savepath + '/txx_res%s.json'%suffix)

        rcParams['font.family'] = 'serif'
        rcParams['font.sans-serif'] = 'Georgia'
        rcParams['text.usetex'] = True
        rcParams['font.size'] = 12
        rcParams['pdf.fonttype'] = 42

        fig = plt.figure(figsize=(7, 6))
        gs = fig.add_gridspec(2, 1, wspace=0, hspace=0)
        ax1 = fig.add_subplot(gs[:1, 0])
        ax2 = fig.add_subplot(gs[1:, 0], sharex=ax1)

        ax1.plot(self.time, self.rate, color='k', lw=1.0, label='Light Curve')
        ax1.plot(self.time, self.bak, color='r', lw=1.0, label='Background')
        for i in range(len(self.txx1)):
            ax1.axvline(self.txx1[i], color='g', lw=1.0, ls='--')
            ax1.axvline(self.txx2[i], color='g', lw=1.0, ls='--')
        ax1.set_ylabel('Rate')
        ax1.set_xlim([self.time[0], self.time[-1]])
        ax1.minorticks_on()
        ax1.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
        ax1.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
        ax1.tick_params(which='major', width=1.0, length=5)
        ax1.tick_params(which='minor', width=1.0, length=3)
        ax1.xaxis.set_ticks_position('both')
        ax1.yaxis.set_ticks_position('both')
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.legend(frameon=False)

        ax2.plot(self.time, self.ccts, color='k', lw=1.0)
        for i in range(len(self.txx1)):
            ax2.axvline(self.txx1[i], color='g', lw=1.0, ls='--')
            ax2.axvline(self.txx2[i], color='g', lw=1.0, ls='--')
        for csfi in self.csf:
            ax2.axhline(csfi, color='orange', lw=1.0)
        for i in range(len(self.csf1)):
            ax2.axhline(self.csf1[i], color='orange', lw=1.0, ls='--')
            ax2.axhline(self.csf2[i], color='orange', lw=1.0, ls='--')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Accumulated counts')
        ax2.minorticks_on()
        ax2.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
        ax2.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
        ax2.tick_params(which='major', width=1.0, length=5)
        ax2.tick_params(which='minor', width=1.0, length=3)
        ax2.xaxis.set_ticks_position('both')
        ax2.yaxis.set_ticks_position('both')
        fig.savefig(savepath + '/txx%s.pdf'%suffix, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close(fig)
        
        

class ggTxx():

    def __init__(self, time, ncts, ncts_err):
        
        self.time = time
        self.ncts = ncts
        self.ncts_err = ncts_err
        
        
    def mc_simulation(self, nmc):
        
        self.nmc = int(nmc)
        self.mc_ncts = [self.ncts]
        for _ in range(self.nmc):
            ppmc = self.ncts_err * np.random.randn(len(self.ncts)) + self.ncts
            self.mc_ncts.append(ppmc)
            
            
    def accumcts(self, xx=0.9, pstart=None, pstop=None, lbkg=None, rbkg=None, mp=True):
        
        self.mc_simulation(1000)

        self.xx = xx
        
        if pstart is None:
            msg = 'there is no pulse'
            warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
            return False
        elif type(pstart) is not list:
            self.pstart = np.array([pstart])
        else:
            self.pstart = np.sort(pstart)
            
        if pstop is None:
            msg = 'there is no pulse'
            warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
            return False
        elif type(pstop) is not list:
            self.pstop = np.array([pstop])
        else:
            self.pstop = np.sort(pstop)

        if lbkg is None:
            lbkg = np.inf
            
        if rbkg is None:
            rbkg = np.inf

        tmin_ = self.pstart[0] - lbkg
        tmin = max(tmin_, self.time[0])
        tmax_ = self.pstop[-1] + rbkg
        tmax = min(tmax_, self.time[-1])
        tindex = np.where((self.time >= tmin) & (self.time <= tmax))[0]
        
        mc_csf, mc_csf1, mc_csf2 = [], [], []
        mc_txx, mc_txx1, mc_txx2 = [], [], []
        
        self.ccts = np.cumsum(self.mc_ncts[0])
        
        for ncts in self.mc_ncts:
            ccts = np.cumsum(ncts)

            idx = np.argsort(self.pstop - self.pstart)[0]
            if len(np.where((self.time >= self.pstart[idx]) & (self.time <= self.pstop[idx]))[0]) < 1000:
                interp_dt = (self.pstop[idx] - self.pstart[idx]) / 1000
                interp_time = np.arange(tmin, tmax - 1e-5, interp_dt)
                interp = interp1d(self.time, ccts, kind='quadratic')
                interp_ccts = interp(interp_time)
            else:
                interp_time = self.time[tindex]
                interp_ccts = ccts[tindex]
                
            csf, csf_err, csf1, csf2 = [], [], [], []
            txx, txx1, txx2 = [], [], []

            for l, r in zip(np.append(tmin, self.pstop), np.append(self.pstart, tmax)):
                csf_i = np.mean(interp_ccts[np.where((interp_time >= l) & (interp_time <= r))])
                csf_err_i = np.std(interp_ccts[np.where((interp_time >= l) & (interp_time <= r))])
                csf.append(csf_i)
                csf_err.append(csf_err_i)
                
            dcsf = np.array(csf[1:]) - np.array(csf[:-1])

            for pi, (l, r) in enumerate(zip(np.append(tmin, self.pstop[:-1]), np.append(self.pstart[1:], tmax))):
                nn = (1 - xx) / 2
                dd = dcsf[pi] * nn

                csf1_i = csf[pi] + dd
                csf2_i = csf[pi + 1] - dd
                    
                csf1.append(csf1_i)
                csf2.append(csf2_i)

                pt = interp_time[np.where((interp_time >= l) & (interp_time <= r))]
                pcts = interp_ccts[np.where((interp_time >= l) & (interp_time <= r))]

                txx_i, txx1_i, txx2_i = self.find_txx(pt, pcts, csf1_i, csf2_i)

                txx.append(txx_i)
                txx1.append(txx1_i)
                txx2.append(txx2_i)
                
            mc_csf.append(csf)
            mc_csf1.append(csf1)
            mc_csf2.append(csf2)
            mc_txx.append(txx)
            mc_txx1.append(txx1)
            mc_txx2.append(txx2)
            
        self.csf = mc_csf[0]
        self.csf1 = mc_csf1[0]
        self.csf2 = mc_csf2[0]
        
        self.txx = mc_txx[0]
        self.txx1 = mc_txx1[0]
        self.txx2 = mc_txx2[0]

        self.txx_err = []
        self.txx1_err = []
        self.txx2_err = []
        mc_txx = np.array(mc_txx)
        mc_txx1 = np.array(mc_txx1)
        mc_txx2 = np.array(mc_txx2)
        for pi in range(len(self.pstart)):
            mask = sigma_clip(mc_txx[1:, pi], sigma=5, maxiters=5, stdfunc=mad_std).mask
            not_mask = list(map(operator.not_, mask))
            mc_txx_filter = np.array(mc_txx[1:, pi])[not_mask]

            txx_lo, txx_hi = np.percentile(mc_txx_filter, [16, 84])
            txx_err = np.diff([txx_lo, mc_txx[0, pi], txx_hi])
            self.txx_err.append([txx_err[0], txx_err[1]])
            
            mask = sigma_clip(mc_txx1[1:, pi], sigma=5, maxiters=5, stdfunc=mad_std).mask
            not_mask = list(map(operator.not_, mask))
            mc_txx1_filter = np.array(mc_txx1[1:, pi])[not_mask]

            txx1_lo, txx1_hi = np.percentile(mc_txx1_filter, [16, 84])
            txx1_err = np.diff([txx1_lo, mc_txx1[0, pi], txx1_hi])
            self.txx1_err.append([txx1_err[0], txx1_err[1]])
            
            mask = sigma_clip(mc_txx2[1:, pi], sigma=5, maxiters=5, stdfunc=mad_std).mask
            not_mask = list(map(operator.not_, mask))
            mc_txx2_filter = np.array(mc_txx2[1:, pi])[not_mask]

            txx2_lo, txx2_hi = np.percentile(mc_txx2_filter, [16, 84])
            txx2_err = np.diff([txx2_lo, mc_txx2[0, pi], txx2_hi])
            self.txx2_err.append([txx2_err[0], txx2_err[1]])

        self.txx_res = {'xx': self.xx, 'txx':self.txx, 'txx1': self.txx1, 'txx2': self.txx2, 
                        'txx_err': self.txx_err, 'txx1_err': self.txx1_err, 'txx2_err': self.txx2_err, 
                        'csf': self.csf, 'csf1': self.csf1, 'csf2': self.csf2, 'ccts': self.ccts}
        
        
        print('\n+------------------------------------------------+')
        print(' %-5s%-10s%-8s%-8s%-8s%-8s' % ('id#', 'Txx', 'Txx-', 'Txx+', 'Txx1', 'Txx2'))
        print('+-----------------------------------------------+')
        _ = list(map(print, [' %-5d%-10.3f%-8.3f%-8.3f%-8.3f%-8.3f' % (i+1, t, t_err[0], t_err[1], t1, t2) 
                             for i, (t, t_err, t1, t2) in enumerate(zip(self.txx, self.txx_err, self.txx1, self.txx2))]))
        print('+------------------------------------------------+')


    @staticmethod
    def find_txx(time, ccts, csf1, csf2):
        interp_time = np.linspace(time[0], time[-1], 1000)
        interp = interp1d(time, ccts, kind='linear')
        interp_ccts = interp(interp_time)

        txx1, txx2 = 0, 0
        for i in range(1, len(interp_time)):
            if interp_ccts[i] < csf1:
                continue
            elif (interp_ccts[i-1] < csf1) and (interp_ccts[i] >= csf1):
                txx1 = interp_time[i]
                continue
            elif (csf1 <= interp_ccts[i-1] < csf2) and (csf1 < interp_ccts[i] <= csf2):
                continue
            elif (csf1 < interp_ccts[i-1] <= csf2) and (interp_ccts[i] > csf2):
                txx2 = interp_time[i-1]
                break
            else:
                continue

        txx = txx2 - txx1
        return txx, txx1, txx2
    
    
    def save(self, savepath, suffix=''):
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        json_dump(self.txx_res, savepath + '/txx_res%s.json'%suffix)

        rcParams['font.family'] = 'serif'
        rcParams['font.sans-serif'] = 'Georgia'
        rcParams['text.usetex'] = True
        rcParams['font.size'] = 12
        rcParams['pdf.fonttype'] = 42

        fig = plt.figure(figsize=(7, 6))
        gs = fig.add_gridspec(2, 1, wspace=0, hspace=0)
        ax1 = fig.add_subplot(gs[:1, 0])
        ax2 = fig.add_subplot(gs[1:, 0], sharex=ax1)

        ax1.plot(self.time, self.ncts, color='k', lw=1.0, label='Light Curve')
        for i in range(len(self.txx1)):
            ax1.axvline(self.txx1[i], color='g', lw=1.0, ls='--')
            ax1.axvline(self.txx2[i], color='g', lw=1.0, ls='--')
        ax1.set_ylabel('Rate')
        ax1.set_xlim([self.time[0], self.time[-1]])
        ax1.minorticks_on()
        ax1.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
        ax1.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
        ax1.tick_params(which='major', width=1.0, length=5)
        ax1.tick_params(which='minor', width=1.0, length=3)
        ax1.xaxis.set_ticks_position('both')
        ax1.yaxis.set_ticks_position('both')
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.legend(frameon=False)

        ax2.plot(self.time, self.ccts, color='k', lw=1.0)
        for i in range(len(self.txx1)):
            ax2.axvline(self.txx1[i], color='g', lw=1.0, ls='--')
            ax2.axvline(self.txx2[i], color='g', lw=1.0, ls='--')
        for csfi in self.csf:
            ax2.axhline(csfi, color='orange', lw=1.0)
        for i in range(len(self.csf1)):
            ax2.axhline(self.csf1[i], color='orange', lw=1.0, ls='--')
            ax2.axhline(self.csf2[i], color='orange', lw=1.0, ls='--')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Accumulated counts')
        ax2.minorticks_on()
        ax2.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
        ax2.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
        ax2.tick_params(which='major', width=1.0, length=5)
        ax2.tick_params(which='minor', width=1.0, length=3)
        ax2.xaxis.set_ticks_position('both')
        ax2.yaxis.set_ticks_position('both')
        fig.savefig(savepath + '/txx%s.pdf'%suffix, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close(fig)
