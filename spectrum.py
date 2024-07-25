import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import *
from model_utils import teff2mass
from model_utils import ca_idx1, ca_idx2, ca_idx3
from astropy.table import QTable


class Spectrum(object):
    """
    Gaia RVS spectrum object that allows standard label determination and 
    spectrum visualization.
    
    Args:
        spectrum_path (str): path to RVS spectrum file with .csv format
    """
    def __init__(self, spectrum_path):

        # load spectrum data from file
        if spectrum_path[-4:]!='.csv':
            print('ERROR: please download a .csv version of the RVS spectrum')
            return None
        spectrum_df = pd.read_csv(spectrum_path)

        # store object source id
        self.source_id = spectrum_df.source_id[0]

        # store object flux, sigma
        self.flux = np.array(spectrum_df['flux'])[20:-20]
        self.sigma = np.array(spectrum_df['flux_error'])[20:-20]

        # remove nans from flux, sigma
        finite_idx = ~np.isnan(self.flux)
        if np.sum(finite_idx) != len(self.flux):
            self.flux = np.interp(wav, wav[finite_idx], flux[finite_idx])
        self.sigma = np.nan_to_num(self.sigma, nan=1)

        # compute best-fit single star, single star model chisq
        self.cannon_labels, self.chisq = fit_single_star(
            self.flux, 
            self.sigma)
        self.model_flux = rvs_model(self.cannon_labels)

        # compute best-fit binary, binary model chisq
        self.binary_cannon_labels, self.binary_chisq = fit_binary(
            self.flux, 
            self.sigma)
        self.binary_model_flux = binary_model(
            self.binary_cannon_labels[:6], # param1
            self.binary_cannon_labels[6:]) # param2 

        # RVS spectrum signal-to-noise
        self.snr = np.mean(self.flux/self.sigma)
        if self.snr<50:
            print('WARNING: low SNR, labels not be reliable')
        # single star fit training density
        self.training_density = training_density(self.cannon_labels)
        # single star fit chisq evaluated at calcium lines
        calcium_resid = ((self.flux - self.model_flux)/self.sigma)[calcium_mask]
        self.calcium_chisq = np.sum(calcium_resid**2)
        # difference in chisq between single model and binary model
        self.delta_chisq = self.chisq - self.binary_chisq

    def data_table(self):
        """
        Output table with Cannon output labels and metrics described in Angelo et al. (2024)
        """

        # compile data for table
        metric_list = [self.chisq, self.training_density, self.calcium_chisq, self.delta_chisq]
        metric_list = [np.log10(i) for i in metric_list]
        tbl_data = np.array(self.cannon_labels.tolist() + metric_list + [self.snr])

        # define column names
        tbl_names = ('Teff (K)', 'logg (dex)', '[Fe/H] (dex)', 
                     '[alpha/Fe] (dex)', 'Vbroad (km/s)',r'log$\chi^2$', 
                     r'log$\rho(l_n)$', r'log$\chi_{\rm Ca}^2$', r'log$\Delta\chi^2$',
                    'SNR')

        # generate astropy Qtable
        tbl = QTable(tbl_data,
                   names=tbl_names,
                   meta={'name': 'Gaia DR3 {}'.format(self.source_id)})

        # format columns to show 2 significant figures
        tbl['Teff (K)'].info.format = '.0f'
        tbl['SNR'].info.format = '.0f'
        for col in tbl.itercols():
            if col.info.dtype.kind == 'f':        
                col.info.format = '.2f'

        # store table as spectrum attribute
        self.data_table = tbl
        return tbl
    
    def spectrum_plot(self):
        """ Outputs a plot with the spectrum, best-fit Cannon model, and 
            metrics plotted over metric distributions for single stars from 
            El-Badry et al. 2018b
        """
        # load single star data
        eb18_singles = pd.read_csv('./data/elbadry2018_singles_metrics.csv')
        # calculate model residuals
        resid = self.flux - self.model_flux

        # colors for plot
        plt.rcParams['figure.dpi']=150
        model_color ='#4f67d3'
        background_color = '#D0D9D0'
        hist_kwargs = {'color':'k','histtype':'step','lw':1}
        vline_kwargs = {'color':model_color, 'lw':2.5}
        spec_tick_kwargs = {'axis':'x', 'length':8, 'direction':'inout'}
        axislabel_fontsize=15

        # strings for plot
        str1 = str(np.round(np.log10(self.chisq),1))
        str2 = str(np.round(np.log10(self.training_density),1))
        str3 = str(np.round(np.log10(self.calcium_chisq),1))
        str4 = str(np.round(np.log10(self.delta_chisq),1))
        metric_str = r'log $\chi^2$='+str1+r', log $\rho(l_{\rm n})$='+\
                    str2+r', log $\chi^2_{\rm Ca}$='+str3+\
                    r', log $\Delta\chi^2$='+str4

        fig = plt.figure(constrained_layout=True, figsize=(13,10))
        gs = fig.add_gridspec(4, 3, wspace = 0, hspace = 0)
        gs.update(hspace=0)

        # 1D histogram: calcium chisq
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(np.log10(eb18_singles.single_fit_ca_resid), 
             bins=np.linspace(2.3,5,12), **hist_kwargs)
        ax1.set_xlabel(r'log $\chi^2_{\rm Ca}$', fontsize=axislabel_fontsize)
        ax1.set_ylabel('number of stars', fontsize=axislabel_fontsize)
        ax1.axvline(np.log10(self.calcium_chisq), **vline_kwargs)

        # 1D histogram: delta chisq
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(np.log10(eb18_singles.delta_chisq), 
             bins=np.arange(0,5,0.25), **hist_kwargs)
        ax2.set_xlabel(r'log $\Delta\chi^2$', fontsize=axislabel_fontsize)
        ax2.set_ylabel('number of stars', fontsize=axislabel_fontsize)
        ax2.axvline(np.log10(self.delta_chisq), **vline_kwargs)

        # 1D histogram: chisq
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(np.log10(eb18_singles.single_fit_chisq), 
                 bins=np.arange(3,5,0.1), **hist_kwargs)
        ax3.set_xlabel(r'log $\chi^2$', fontsize=axislabel_fontsize)
        ax3.set_ylabel('number of stars', fontsize=axislabel_fontsize)
        ax3.set_xticks(np.arange(3,5,0.5))
        ax3.axvline(np.log10(self.chisq), **vline_kwargs)

        # 1D histogram: training density
        ax4 = fig.add_subplot(gs[1,1])
        ax4.hist(np.log10(eb18_singles.single_fit_training_density),
                      bins=np.arange(-8,-1,0.5), **hist_kwargs)
        ax4.set_xlabel(r'log $\rho(l_{\rm n}$)', fontsize=axislabel_fontsize)
        ax4.set_ylabel('number of stars', fontsize=axislabel_fontsize)
        ax4.axvline(np.log10(self.training_density), **vline_kwargs)

        # HR diagram 
        ax5 = fig.add_subplot(gs[0:2,2])
        ax5.plot(rvs_model.training_set_labels[:,0], 
                 rvs_model.training_set_labels[:,1], 
                 'o', color=background_color, ms=5)
        ax5.plot(self.cannon_labels[0], self.cannon_labels[1], 
                 '*', ms=20,mec='k', mew=1.5, color=model_color)
        ax5.invert_xaxis();ax5.invert_yaxis()
        ax5.set_xlabel(r'T$_{\rm eff}$ (K)', fontsize=axislabel_fontsize)
        ax5.set_ylabel('log$g$ (dex)', fontsize=axislabel_fontsize)

        # spectrum + model fit
        ax6 = fig.add_subplot(gs[2:3, :])
        ax6.axvspan(849.5,850.5,color=background_color, zorder=0, alpha=0.5)
        ax6.axvspan(854, 855,color=background_color, zorder=1, alpha=0.5)
        ax6.axvspan(866,867,color=background_color, zorder=2, alpha=0.5)
        ax6.errorbar(wav, self.flux, self.sigma, 
                     color='k', ecolor='#E8E8E8', linewidth=1.75, elinewidth=4, zorder=4)
        ax6.plot(wav, self.model_flux, '-', 
                 color=model_color, linewidth=1.25, zorder=5)
        source_id_str = 'Gaia DR3 {}'.format(str(self.source_id))
        ax6.text(847, 1.2, source_id_str, fontsize=12, color='k')
        ax6.text(861,1.2,metric_str, fontsize=12)
        ax6.set_xlim(wav.min(), wav.max());ax6.set_ylim(0,1.5)
        ax6.set_ylabel('normalized flux', fontsize=axislabel_fontsize)
        ax6.tick_params(labelbottom=False, **spec_tick_kwargs)

        # residuals
        ax7 = fig.add_subplot(gs[3:4, :], sharex=ax6)
        ax7.axvspan(849.5,850.5,color=background_color, zorder=0, alpha=0.5)
        ax7.axvspan(854, 855,color=background_color, zorder=1, alpha=0.5)
        ax7.axvspan(866,867,color=background_color, zorder=2, alpha=0.5)
        ax7.plot(wav, resid, color='k', lw=1.25, zorder=4)
        ax7.set_ylim(resid.min()-0.1,resid.max()+0.1)
        ax7.set_ylabel('residuals', fontsize=axislabel_fontsize)
        ax7.set_xlabel('wavelength (nm)', fontsize=axislabel_fontsize)
        ax7.tick_params(labelbottom=True, **spec_tick_kwargs)

    def activity_plot(self):
        """ Outputs a plot with the spectrum + best-fit Cannon model + 
        Calcium III equivalent widths, along with Calcium chi-squared
            plotted relative to distribution for single stars from 
            El-Badry et al. 2018b
        """
        # relevant data for plot
        eb18_singles = pd.read_csv('./data/elbadry2018_singles_metrics.csv')
        resid = self.flux - self.model_flux
        log_chisq = np.round(np.log10(self.chisq),1)
        log_calcium_chisq = np.round(np.log10(self.calcium_chisq),1)

        # plotting variables
        model_color='#DEB23C'
        calcium_chisq_str = r'log $\chi_{Ca}^2$='+ str(log_calcium_chisq)
        spec_tick_kwargs = {'axis':'x', 'length':8, 'direction':'inout'}
            
        # compute equivalent width of Ca triplet residuals
        equivalent_width_values = []
        for ca_idx in [ca_idx1, ca_idx2, ca_idx3]:
            # define wavelength, flux, continuum for integrand
            line_w = wav[ca_idx]
            line_continuum = np.ones(len(line_w)) 
            line_resid = (self.flux[ca_idx] - self.model_flux[ca_idx]) + 1 # normalize to 1
            line_integrand = 1 - line_resid/line_continuum
            # compute equivalent width
            equivalent_width = np.trapz(line_integrand, line_w)
            equivalent_width_values.append(equivalent_width)
        W1, W2, W3 = [np.round(i,3) for i in equivalent_width_values]

        # create figure
        plt.rcParams['font.size']=15
        plt.rcParams['figure.dpi']=150
        fig = plt.figure(figsize=(20,6))
        gs = fig.add_gridspec(2, 3)
        plt.subplots_adjust(hspace=0)

        # spectrum + model fit
        ax1 = fig.add_subplot(gs[0:1, :2])
        ax1.errorbar(wav, self.flux, yerr=self.sigma, color='k', 
                     ecolor='#E8E8E8', elinewidth=4, zorder=0)
        ax1.plot(wav, self.model_flux, color=model_color, ls=(0,()), lw=2)
        ax1.text(847,1.1,'best-fit single star\nlog $\chi^2$ ={}'.format(log_chisq),
            color=model_color)
        ax1.text(859.7,1.15,'Gaia DR3 {}    S/N={}'.format(self.source_id, int(self.snr)), 
                 color='k', zorder=5)
        ax1.set_ylabel('normalized\nflux')
        ax1.set_ylim(0.2, 1.4)
        ax1.set_xlim(wav.min(), wav.max())
        ax1.tick_params(labelbottom=False, **spec_tick_kwargs)

        # residuals
        ax2 = fig.add_subplot(gs[1:2, :2], sharex=ax1)
        ax2.plot(wav, resid, color=model_color, ls=(0,()), lw=2)
        ax2.text(849.2, resid.min()-0.03, 'W={0:+}'.format(W1), color='k')
        ax2.text(853.6, resid.min()-0.03, 'W={0:+}'.format(W2), color='k')
        ax2.text(865.6, resid.min()-0.03, 'W={0:+}'.format(W3), color='k')
        ax2.set_ylim(resid.min()-0.05, resid.max()+0.05)
        ax2.tick_params(axis='x', direction='inout', length=15)
        ax2.set_ylabel('residuals')
        ax2.tick_params(labelbottom=True, **spec_tick_kwargs)

        # 1D histogram: calcium chisq
        ax3 = fig.add_subplot(gs[:, 2:])
        ax3.hist(np.log10(eb18_singles.single_fit_ca_resid),
             bins=np.linspace(2.7,5,15), histtype='step', color='k')
        ax3.set_xlabel(r'log $\chi^2_{\rm Ca}$')
        ax3.set_ylabel('number of stars')
        ax3.axvline(np.log10(self.calcium_chisq), color=model_color)
        if log_calcium_chisq<4:
            ax3.text(np.log10(self.calcium_chisq)+0.15, 140, calcium_chisq_str, color=model_color)
        else:
            ax3.text(np.log10(self.calcium_chisq)-1, 140, calcium_chisq_str, color=model_color)
        ax3.text(3.68, 10, 'single star sample', color='k')
        plt.tight_layout()
        plt.show()

    def binary_plot(self):
        """ Outputs a plot with the spectrum + best-fit single + binary models, 
        relevant metrics and components of the binary model,
         and delta chi-squared distribution plotted relative to distribution for 
         single stars from El-Badry et al. 2018b
        """
         # load single star data
        eb18_singles = pd.read_csv('./data/elbadry2018_singles_metrics.csv')

        # calcaulte model residuals + metrics
        single_resid = self.model_flux - self.flux
        binary_resid = self.binary_model_flux - self.flux
        log_delta_chisq = np.round(np.log10(self.delta_chisq),1)
        log_chisq = np.round(np.log10(self.chisq),1)
        log_binary_chisq = np.round(np.log10(self.binary_chisq),1)

        # relevant metrics for binary model
        _, primary_model_flux, secondary_model_flux = binary_model(
            self.binary_cannon_labels[:6], # param1
            self.binary_cannon_labels[6:], # param2
            return_components=True) # param2
        mass_ratio = teff2mass(self.binary_cannon_labels[6])/teff2mass(self.binary_cannon_labels[0])
        drv = self.binary_cannon_labels[-1] - self.binary_cannon_labels[5]

         # training density of binary components
        primary_labels = self.binary_cannon_labels[:5].tolist()
        secondary_labels = self.binary_cannon_labels[6:-1].tolist()
        secondary_labels.insert(2, primary_labels[2])
        secondary_labels.insert(3, primary_labels[3])
        primary_density = np.log10(training_density(primary_labels))
        secondary_density = np.log10(training_density(secondary_labels))

        # plotting variables
        single_color ='#DEB23C'
        binary_color='#4f67d3'
        primary_color='#A5C3E4'
        secondary_color='#E36B44'
        spec_tick_kwargs = {'axis':'x', 'length':8, 'direction':'inout'}
        delta_chisq_str = r'log $\Delta\chi^2$='+ str(log_delta_chisq)
        binary_label_str = r'model binary: $\Delta$RV={} km/s, m$_2$/m$_1$={}'.format(
            drv.round(2), mass_ratio.round(2))

        fig = plt.figure(figsize=(13,7))
        fig = plt.figure(constrained_layout=True, figsize=(13,7))
        plt.rcParams['font.size']=13
        gs = fig.add_gridspec(6, 3, wspace = 0, hspace = 0)
        gs.update(hspace=0)

        # spectrum + single star fit + binary fit
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        ax1.errorbar(wav, self.flux, yerr=self.sigma, 
                     color='k', ecolor='#E8E8E8', linewidth=2, elinewidth=4, zorder=0)
        ax1.plot(wav, self.binary_model_flux, color=binary_color, lw=2)
        ax1.plot(wav, self.model_flux, color=single_color, lw=1)
        ax1.text(847, 1.3, 'Gaia DR3 {}'.format(str(self.source_id)), fontsize=12, color='k')
        ax1.text(859.5,1.2,'best-fit single star\nlog $\chi^2={}$'.format(log_chisq),
             color=single_color, fontsize=11)
        ax1.text(865,1.2,'best-fit binary\nlog $\chi^2={}$'.format(log_binary_chisq),
             color=binary_color, fontsize=11)
        ax1.set_xlim(wav.min(), wav.max());ax1.set_ylim(0,1.6)
        ax1.set_ylabel('normalized flux')
        ax1.tick_params(labelbottom=False, **spec_tick_kwargs)

        # single star + binary residuals
        ax2 = fig.add_subplot(gs[2:4, 0:2], sharex=ax1)
        ax2.plot(wav, single_resid, color=single_color, lw=2, zorder=0)
        ax2.plot(wav, binary_resid, color=binary_color, lw=1, zorder=1)
        ax2.set_ylim(single_resid.min() - 0.1, single_resid.max() + 0.1)
        ax2.set_ylabel('residuals')
        ax2.tick_params(labelbottom=False, **spec_tick_kwargs)

        # binary model components
        ax3 = fig.add_subplot(gs[4:6, 0:2], sharex=ax1)
        ax3.plot(wav, self.flux, color='k', linewidth=1.5)
        ax3.plot(wav, primary_model_flux, color=primary_color, linewidth=1.5)
        ax3.plot(wav, secondary_model_flux, color=secondary_color, linewidth=1.5)
        ax3.plot(wav, self.binary_model_flux, color=binary_color, lw=1.5, ls='--')
        ax3.text(847,0.05,r'model primary, log $\rho(l_n)={}$'.format(
            primary_density.round(1)), color=primary_color, fontsize=11)
        ax3.text(847,-0.1,r'model secondary, log $\rho(l_n)={}$'.format(
            secondary_density.round(1)), color=secondary_color, fontsize=11)
        ax3.text(847,1.15, binary_label_str, color=binary_color, fontsize=11)
        ax3.set_ylim(-0.2,1.4)
        ax3.set_xlabel('wavelength (nm)');ax3.set_ylabel('normalized flux')
        ax3.tick_params(labelbottom=True, **spec_tick_kwargs)

        # 1D histogram: delta chisq
        ax4 = fig.add_subplot(gs[0:4, 2:])
        ax4.hist(np.log10(eb18_singles.delta_chisq),
             bins=np.arange(0,5,0.25), histtype='step', color='k')
        ax4.axvline(np.log10(self.delta_chisq), color=binary_color)
        ax4.text(2.6, 15, 'single star sample', color='k')
        if log_delta_chisq<3:
            ax4.text(np.log10(self.delta_chisq)+0.25, 200, delta_chisq_str, color=binary_color)
        else:
            ax4.text(np.log10(self.delta_chisq)-1.7, 200, delta_chisq_str, color=binary_color)
        ax4.set_xlabel(r'log $\Delta\chi^2$');ax4.set_ylabel('number of stars')

