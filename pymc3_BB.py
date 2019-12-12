import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pymc3 as pm
import numpy as np
import theano.tensor as tt
import healpy as hp
import corner
import pandas as pd
from scipy.interpolate import interp1d
import seaborn as sns


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 40})
plt.rcParams.update({'figure.figsize': [20.0, 18.4]})

###############################################################################

# Fiducial CMB power spectra used for all analyses. These correspond to r=0.01 and A_L=1.

cl_BB = np.loadtxt('./fiducial_spectra/cmb_fiducial_r0p01_unlensed.dat', usecols=3)
ell_BB = np.loadtxt('./fiducial_spectra/cmb_fiducial_r0p01_unlensed.dat', usecols=0)
cl_lens = np.loadtxt('./fiducial_spectra/cmb_fiducial_r0_lensed.dat', usecols=3)
ell_lens = np.loadtxt('./fiducial_spectra/cmb_fiducial_r0_lensed.dat', usecols=0)

assert np.all(ell_BB == ell_lens)
fid_ell = np.copy(ell_BB)
cl_BB = 2.0*np.pi*cl_BB/(fid_ell*(fid_ell + 1.0))
cl_lens = 2.0*np.pi*cl_lens/(fid_ell*(fid_ell + 1.0))

###############################################################################


class BB_Likelihood(object):

    def __init__(self, cl_obs, ell, fiducial_BB=cl_BB, fiducial_lens=cl_lens, fiducial_ell=fid_ell, cl_cov=None,
                 r_lower=0.0, r_upper=1.0, AL_lower=0.0, AL_upper=1.0, num_samples=10000, num_tuning=5000,
                 target_accept=0.95, nuts_start='advi', fix_AL=None):

        """

        CMB B-mode likelihood class. This been written with un-binned spectra in mind, but works equally well for binned
        power spectra. These should just be processed before creating a class instance (including the fiducial spectra).

        Attributes
        ----------
        :param cl_obs: Observed BB Cls - numpy.ndarray.
        :param ell: Corresponding multipoles for the observed Cl (this can be bin positions) - numpy.ndarray.
        :param fiducial_BB: Fiducial primordial B-mode power spectrum (binning should be the same as for observed Cl) -
            numpy.ndarray.
        :param fiducial_lens: Fiducial lensing B-mode power spectrum (binning should be the same as for observed Cl) -
            numpy.ndarray
        :param fiducial_ell: Corresponding multipole values for the fiducial power spectra - numpy.ndarray.
        :param cl_cov: BB Cl covariance matrix - numpy.ndarray.
        :param r_lower: Lower limit on tensor-to-scalar ratio in uniform prior - float.
        :param r_upper: Upper limit on tensor-to-scalar ratio in uniform prior - float.
        :param AL_mu: Mean value for lensing amplitude in Gaussian prior - float.
        :param AL_sd: Standard deviation for lensing amplitude in Gaussian prior - float.
        :param num_samples: Number of samples to be drawn from the B-mode posterior - int.
        :param num_tuning: Number of tuning samples to be drawn with NUTS - int.
        :param target_accept: Sampler target acceptance probability - float.
        :param nuts_start: Either 'advi', in which case we use ADVI initiliasation along with tuning of diagonal mass
            matrix elements, or a dictionary of starting values {'r': r_val, 'AL': AL_val} - str or dict.
        :param fix_AL: Set to float value to fix AL at this - float.

        """

        self.cl_obs = cl_obs
        self.ell = ell
        self.fiducial_BB = fiducial_BB
        self.fiducial_lens = fiducial_lens
        self.fid_ell = fiducial_ell
        self.cl_cov = cl_cov
        self.r_lower = r_lower
        self.r_upper = r_upper
        self.AL_lower = AL_lower
        self.AL_upper = AL_upper
        self.num_samples = num_samples
        self.num_tuning = num_tuning
        self.target_accept = target_accept
        self.nuts_start = nuts_start
        self.fix_AL = fix_AL

    def extract_fiducial(self):

        """
        Returns fiducial power spectrum elements that correspond to the observed power spectra
        :return fid_cl_BB: Fiducial primordial B-mode power spectrum values - numpy.ndarray.
        :return fid_cl_lens: Fiducial lensing B-mode power spectrum values - numpy.ndarray.
        """

        max_ell_idx = int(np.amin(np.where(self.fiducial_BB == 0)[0])) - 1
        interp_ell = self.fid_ell[0:max_ell_idx]
        interp_BB = self.fiducial_BB[0:max_ell_idx]
        interp_lens = self.fiducial_lens[0:max_ell_idx]
        fid_BB_spline = interp1d(np.log10(interp_ell), np.log10(interp_BB), kind='cubic')
        fid_lens_spline = interp1d(np.log10(interp_ell), np.log10(interp_lens), kind='cubic')
        fid_cl_BB = 10**fid_BB_spline(np.log10(self.ell))
        fid_cl_lens = 10**fid_lens_spline(np.log10(self.ell))

        return fid_cl_BB, fid_cl_lens

    def samp_wishart(self):

        """
        Method samples the B-mode posterior, assuming a Wishart likelihood.
        :return trace: Trace of samples from the B-mode posterior - pymc3 MultiTrace.
        :return wishart_model: PyMC3 Wishart model instance - pymc3 Model.
        """

        fid_cl_BB, fid_cl_lens = self.extract_fiducial()
        nlBB = np.sqrt(np.diagonal(self.cl_cov))

        wishart_model = pm.Model()

        with wishart_model:

            r = pm.Uniform('r', lower=self.r_lower, upper=self.r_upper)
            if self.fix_AL is None:
                AL = pm.Uniform('AL', lower=self.AL_lower, upper=self.AL_upper)
                cl_theory = r*fid_cl_BB/0.01 + AL*fid_cl_lens
            elif self.fix_AL is not None:
                cl_theory = r*fid_cl_BB/0.01 + self.fix_AL*fid_cl_lens
            likeBB = pm.DensityDist('likeBB', wishart_loglike, observed={'cl_obs': self.cl_obs, 'cl_theory': cl_theory,
                                                                         'nlBB': nlBB, 'ell': self.ell})

            if self.nuts_start == 'advi':

                step = pm.NUTS(target_accept=self.target_accept)
                trace = pm.sample(draws=self.num_samples, tune=self.num_tuning, init='advi+adapt_diag', step=step)

                return trace, wishart_model

            elif type(self.nuts_start) is dict:

                step = pm.NUTS(target_accept=self.target_accept)
                trace = pm.sample(draws=self.num_samples, tune=self.num_tuning, start=self.nuts_start, step=step)

                return trace, wishart_model

            else:

                raise Exception('nuts_start should either be advi or a dictionary of starting values.'
                                'You gave nuts_start = {}'.format(self.nuts_start))

    def samp_diag_gaussian(self):

        """
        Method samples the B-mode posterior, assuming a diagonal Gaussian likelihood.
        :return trace: Trace of samples from the B-mode posterior - pymc3 MultiTrace.
        :return diag_gauss_model: PyMC3 diagonal Gaussian model instance - pymc3 Model.
        """

        fid_cl_BB, fid_cl_lens = self.extract_fiducial()
        print(fid_cl_BB)
        nlBB = np.sqrt(np.diagonal(self.cl_cov))

        diag_gauss_model = pm.Model()

        with diag_gauss_model:

            r = pm.Uniform('r', lower=self.r_lower, upper=self.r_upper)
            if self.fix_AL is None:
                AL = pm.Uniform('AL', lower=self.AL_lower, upper=self.AL_upper)
                cl_theory = r*fid_cl_BB/0.01 + AL*fid_cl_lens
            elif self.fix_AL is not None:
                cl_theory = r*fid_cl_BB/0.01 + self.fix_AL*fid_cl_lens
            likeBB = pm.Normal('likeBB', mu=cl_theory, sd=nlBB, observed=self.cl_obs)

            if self.nuts_start == 'advi':

                step = pm.NUTS(target_accept=self.target_accept)
                trace = pm.sample(draws=self.num_samples, tune=self.num_tuning, init='advi+adapt_diag', step=step)

                return trace, diag_gauss_model

            elif type(self.nuts_start) is dict:

                step = pm.NUTS(target_accept=self.target_accept)
                trace = pm.sample(draws=self.num_samples, tune=self.num_tuning, start=self.nuts_start, step=step)

                return trace, diag_gauss_model

            else:

                raise Exception('nuts_start should either be advi or a dictionary of starting values.'
                                'You gave nuts_start = {}'.format(self.nuts_start))

    def samp_full_cov_gaussian(self):

        """
        Method samples the B-mode posterior, assuming a full-rank Gaussian likelihood.
        :return trace: Trace of samples from the B-mode posterior - pymc3 MultiTrace.
        :return full_cov_gauss_model: PyMC3 full-rank Gaussian model instance - pymc3 Model.
        """

        fid_cl_BB, fid_cl_lens = self.extract_fiducial()

        full_cov_gauss_model = pm.Model()
        
        chol = np.linalg.cholesky(self.cl_cov)

        with full_cov_gauss_model:

            r = pm.Uniform('r', lower=self.r_lower, upper=self.r_upper)
            if self.fix_AL is None:
                AL = pm.Uniform('AL', lower=self.AL_lower, upper=self.AL_upper)
                cl_theory = r*fid_cl_BB/0.01 + AL*fid_cl_lens
            elif self.fix_AL is not None:
                cl_theory = r*fid_cl_BB/0.01 + self.fix_AL*fid_cl_lens
            likeBB = pm.MvNormal('likeBB', mu=cl_theory, chol=chol, observed=self.cl_obs)

            if self.nuts_start == 'advi':

                step = pm.NUTS(target_accept=self.target_accept)
                trace = pm.sample(draws=self.num_samples, tune=self.num_tuning, init='advi+adapt_diag', step=step)

                return trace, full_cov_gauss_model

            elif type(self.nuts_start) is dict:

                step = pm.NUTS(target_accept=self.target_accept)
                trace = pm.sample(draws=self.num_samples, tune=self.num_tuning, start=self.nuts_start, step=step)

                return trace, full_cov_gauss_model

            else:

                raise Exception('nuts_start should either be advi or a dictionary of starting values.'
                                'You gave nuts_start = {}'.format(self.nuts_start))


class PyMC3_Trace_Analyser(object):

    def __init__(self, trace, pymc3_model, cl_obs, ell, fiducial_BB=cl_BB, fiducial_lens=cl_lens, fiducial_ell=fid_ell,
                 cl_cov=None, theory_r=1.0e-3, theory_AL=1.0, trace_summary_out=None,
                 traceplot_out=None, jointplot_out=None, kdeplot_out=None, corner_out=None, num_ppc=1000, ppc_out=None,
                 fix_AL=None):

        """

        Class for analysis of chain output from sampling the B-mode posterior.

        Attributes
        ----------
        :param trace: B-mode posterior trace output - pymc3 MultiTrace.
        :param pymc3_model: PyMC3 model instance corresponding to the trace object - pymc3 Model.
        :param cl_obs: Observed BB Cls - numpy.ndarray.
        :param ell: Corresponding multipoles for the observed Cl (this can be bin positions) - numpy.ndarray.
        :param fiducial_BB: Fiducial primordial B-mode power spectrum (binning should be the same as for observed Cl) -
            numpy.ndarray.
        :param fiducial_lens: Fiducial lensing B-mode power spectrum (binning should be the same as for observed Cl) -
            numpy.ndarray
        :param fiducial_ell: Corresponding multipole values for the fiducial power spectra - numpy.ndarray.
        :param cl_cov: BB Cl covariance matrix - numpy.ndarray.
        :param theory_r: Tensor-to-scalar ratio used in simulations being analysed - float.
        :param theory_AL: Lensing amplitude used in simulations being analysed - float.
        :param trace_summary_out: Filename for the trace summary csv file - str.
        :param traceplot_out: Filename for the traceplot figure - str.
        :param jointplot_out: Filename for the jointplot figure - str.
        :param kdeplot_out: Filename for the kdeplot figure - str.
        :param corner_out: Filename for the corner plot - str.
        :param num_ppc: Number of posterior predictive samples to draw - int.
        :param ppc_out: Filename for the posterior predictive plot - str.
        :param fix_AL: Set to float value to fix AL at this - float.

        """

        self.trace = trace
        self.pymc3_model = pymc3_model
        self.cl_obs = cl_obs
        self.ell = ell
        self.fiducial_BB = fiducial_BB
        self.fiducial_lens = fiducial_lens
        self.fid_ell = fiducial_ell
        self.cl_cov = cl_cov
        self.theory_r = theory_r
        self.theory_AL = theory_AL
        self.trace_summary_out = trace_summary_out
        self.traceplot_out = traceplot_out
        self.jointplot_out = jointplot_out
        self.kdeplot_out = kdeplot_out
        self.corner_out = corner_out
        self.num_ppc = num_ppc
        self.ppc_out = ppc_out
        self.fix_AL = fix_AL
        
    def extract_fiducial(self):

        """
        :return: Returns fiducial power spectrum elements that correspond to the observed power spectra - numpy.ndarray.
        """

        max_ell_idx = int(np.amin(np.where(self.fiducial_BB == 0)[0])) - 1
        interp_ell = self.fid_ell[0:max_ell_idx]
        interp_BB = self.fiducial_BB[0:max_ell_idx]
        interp_lens = self.fiducial_lens[0:max_ell_idx]
        fid_BB_spline = interp1d(np.log10(interp_ell), np.log10(interp_BB), kind='cubic')
        fid_lens_spline = interp1d(np.log10(interp_ell), np.log10(interp_lens), kind='cubic')
        fid_cl_BB = 10**fid_BB_spline(np.log10(self.ell))
        fid_cl_lens = 10**fid_lens_spline(np.log10(self.ell))

        return fid_cl_BB, fid_cl_lens

    def trace_summary(self):

        """
        Method saves a trace summary to a csv file. Contains summary statistics for the sampler output.
        :return df: Trace summary dataframe - pandas dataframe.
        """

        df = pm.summary(self.trace)
        print('Trace summary below, saving to csv ...')
        print(df)
        df.to_csv(self.trace_summary_out)

        return df

    def traceplot(self):

        """
        Function saves a traceplot for the given trace.
        Seems to be broken when using Wishart likelihood - not sure why.
        """

        try:
            plt.figure()
            pm.traceplot(self.trace)
            plt.savefig(self.traceplot_out, dpi=900)
            plt.close()
        except:
            print('Traceplot function seems to be broken when using the inverse Wishart likelihood, not sure why.')

    def kde_plot(self):

        """
        Function for generating a kde plot for r.
        """
        r_samples = self.trace.get_values('r', combine=True)
        sns.kdeplot(r_samples, shade=True, clip=(0, np.amax(r_samples)))
        plt.xlabel(r'$r$')
        plt.ylabel(r'$P(r|\hat{C}_\ell)$')
        plt.savefig(self.kdeplot_out, dpi=900)
        plt.close()
        
    def joint_plot(self):

        if self.fix_AL is None:
            r_samples = self.trace.get_values('r', combine=True)
            AL_samples = self.trace.get_values('AL', combine=True)

            plt.figure()
            sns.jointplot(r_samples, AL_samples, kind='kde').set_axis_labels(r'$r$', r'$A_L$')
            plt.savefig(self.jointplot_out, dpi=900)
            plt.close()
        elif self.fix_AL is not None:
            print('You have fixed the lensing amplitude, not possible to generate a joint plot.')

    def corner_plot(self):

        """
        Function saves a corner plot for the given trace.
        """

        if self.fix_AL is None:
            r_samples = self.trace.get_values('r', combine=True)
            AL_samples = self.trace.get_values('AL', combine=True)
            cn_samples = np.transpose(np.array([r_samples, AL_samples]))

            plt.figure()
            corner.corner(cn_samples, labels=[r'$r$', r'$A_L$'])
            plt.savefig(self.corner_out, dpi=900)
            plt.close()
        elif self.fix_AL is not None:
            print('You have fixed the lensing amplitude, not possible to generate a corner plot.')

    def posterior_predictive(self):

        """
        Function generates posterior predictive samples and plots them against the data and simulated model.
        :return ppc: Posterior predictive samples for the fitted model - dict.
        """

        ppc = pm.sample_posterior_predictive(self.trace, samples=self.num_ppc, model=self.pymc3_model,
                                             vars=self.pymc3_model.unobserved_RVs)
        r_ppc = ppc['r']
        if self.fix_AL is None:
            AL_ppc = ppc['AL']
        fid_cl_BB, fid_cl_lens = self.extract_fiducial()
        nlBB = np.sqrt(np.diagonal(self.cl_cov))
        cosmic_var = 2.0*self.cl_obs**2/(2.0*self.ell + 1.0)

        plt.figure()
        plt.plot(self.ell, self.ell*(self.ell + 1.0)*(self.theory_r*fid_cl_BB/0.01 +
                                                      self.theory_AL*fid_cl_lens)/(2.0*np.pi), label='Theory', color='k')
        for i in range(len(r_ppc)):
            if self.fix_AL is None:
                plt.plot(self.ell, self.ell*(self.ell + 1.0)*(r_ppc[i]*fid_cl_BB/0.01 + AL_ppc[i]*fid_cl_lens)/(2.0*np.pi),
                         alpha=0.01, color='m')
            elif self.fix_AL is not None:
                plt.plot(self.ell, self.ell*(self.ell + 1.0)*(r_ppc[i]*fid_cl_BB/0.01 + self.fix_AL*fid_cl_lens)/(2.0*np.pi),
                         alpha=0.01, color='m')
        plt.errorbar(self.ell, self.ell*(self.ell + 1.0)*self.cl_obs/(2.0*np.pi), yerr=np.sqrt(cosmic_var + nlBB**2),
                     fmt='.', capsize=3, label='Data', color='c')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(loc='lower right')
        plt.savefig(self.ppc_out, dpi=900)
        plt.close()

        return ppc


def wishart_loglike(cl_obs, cl_theory, nlBB, ell):

    """

    Function gives the logp of the full Wishart Cl likelihood (very non-Gaussian at low-l).

    Inputs
    ------
    :param cl_obs: Observed Cl - numpy.ndarray.
    :param cl_theory: Theoretical Cl - numpy.ndarray.
    :param nlBB: Noise power spectrum - numpy.ndarray.
    :param ell: CMB multipole values - numpy.ndarray.

    Returns
    -------
    :return wishart_like: Full log-likelihood for the CMB power spectrum.

    """

    return -tt.sum((2.0*ell + 1.0)*(cl_obs/(cl_theory + nlBB) + tt.log(cl_theory + nlBB) -
                                    (2.0*ell - 1.0)*tt.log(cl_obs)/(2.0*ell + 1.0))/2.0)


def joint_kde_plot(traces, colours, labels, linestyles, out_filename, true_r=5e-3):

    """
    
    Function for plotting multiple r posteriors on top of one another.
    
    Inputs
    ------
    :param traces: List of traces you want plotted - list.
    :param colours: List of colours to use for the kdeplots - list.
    :param labels: List of labels to use for the kdeplots - list.
    :param linestyles: List of linestyles to use for the kdeplots - list.
    :param out_filename: Output figure filename - str.

    """

    plt.figure()
    for i, trace in enumerate(traces):
        r_samples = trace.get_values('r', combine=True)
        sns.kdeplot(r_samples, shade=True, clip=(0, np.amax(r_samples)), color=colours[i], label=labels[i],
                    linestyle=linestyles[i], linewidth=4)
    plt.axvline(x=true_r, color='k', linestyle='--')
    plt.xlim(0)
    plt.xlabel(r'$r$')
    plt.ylabel(r'$\mathrm{KDE}(r)$')
    plt.legend(loc='upper left')
    plt.savefig(out_filename, dpi=900)
    plt.close()
