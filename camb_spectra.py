import camb
from camb import model, initialpower
import numpy as np
import healpy as hp
import pandas as pd


# CAMB requires gcc/5.4.0 compilers - make sure this is loaded before trying to run CAMB.

def cmb_power_spectra(r, out_lmax=2200, camb_lmax=4000, lens_accuracy=2, out_file=None,
                      H0=67.0, ombh2=0.022, omch2=0.12, omk=0.0, neutrino_hierarchy='degenerate',
                      num_massive_neutrinos=1, mnu=0.06, nnu=3.046, YHe=None, meffsterile=0.0,
                      standard_neutrino_neff=3.046, TCMB=2.7255, tau=None, deltazrei=None, bbn_predictor=None,
                      As=2.0e-9, ns=0.96, nrun=0.0, nrunrun=0.0, nt=None, ntrun=0.0, pivot_scalar=0.05,
                      pivot_tensor=0.05):

    """

    Function for generating CMB power spectra using CAMB.
    See CAMB documentation for explanation of the cosmological parameters.
    Default values are those used in the Python wrapper, and correspond to Planck 2015 values.

    Inputs
    ------
    r: The tensor-to-scalar ratio - float.
    out_lmax: Maximum l to output power spectra to - int.
    camb_lmax: Maximum l value for CAMB to calculate to (set high for better B-mode lensing) - int. 
    lens_accuracy: Accuracy of the lensing calculation, set to 1 or higher for accurate lensing calculations - float.
    out_file: Output dat file for the power spectra, specifiy if you want to save to file (optional) - str.
    H0: Hubble parameter (km/s/Mpc).
    ombh2: Physical density in baryons.
    omch2: Physical density in cold dark matter.
    omk: Curvature parameter.
    neutrino_hierarchy: 'degenerate', 'normal' or 'inverted'.
    num_massive_neutrinos: Number of massive neutrinos (ignored unless 'degenerate').
    mnu: Sum on neutrino masses (in eV).
    nnu: Effective relativistic degrees of freedom.
    YHe: Helium mass fraction. If None set from BBN consistency.
    meffsterile: Effective mass of sterile neutrinos.
    standard_neutrino_neff: Default value for N_eff in standard cosmology.
    TCMB: CMB temperature.
    tau: Optical depth.
    deltazrei: Redshift width of reionization.
    bbn_predictor: Used to get YHe from BBN if YHe is None.
    As: Comoving curvature power at pivot scale.
    ns: Scalar spectral index.
    nrun: Running of the scalar spectral index.
    nrunrun: Running of the running of the scalar spectral index.
    nt: Tensor spectral index. If None set using inflation consistency.
    ntrun: Running of the tensor spectral index.
    pivot_scalar: Pivot scale for the scalar spectrum.
    pivot_tensor: Pivot scale for the tensor spectrum.

    Returns
    -------
    cmb_cl: Dictionary of CMB power spectra (total, lens_potential, lensed_scalar, unlensed_scalar,
    unlensed_total, tensor) - dict.

    """

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=omk, neutrino_hierarchy=neutrino_hierarchy, 
                       num_massive_neutrinos=num_massive_neutrinos, mnu=mnu, nnu=nnu, YHe=YHe, meffsterile=meffsterile,
                       standard_neutrino_neff=standard_neutrino_neff, TCMB=TCMB, tau=tau, deltazrei=deltazrei,
                       bbn_predictor=bbn_predictor)
    pars.WantTensors = True
    pars.InitPower.set_params(As=As, ns=ns, nrun=nrun, nrunrun=nrunrun, r=r, nt=nt, ntrun=ntrun,
                              pivot_scalar=pivot_scalar, pivot_tensor=pivot_tensor)
    pars.set_for_lmax(lmax=camb_lmax, lens_potential_accuracy=lens_accuracy)
    
    results = camb.get_results(pars)

    powers = results.get_cmb_power_spectra(lmax=out_lmax, CMB_unit='muK')

    if out_file is not None:
        df = pd.from_dict(powers)
        df.to_csv(out_file)

    return powers


def mk_cmb_map(powers, nside, out_file):

    """

    Function for generating CMB map realisations given a set of CAMB power spectra.

    Inputs
    ------
    powers: Dictionary of CMB power spectra (total, lens_potential, lensed_scalar, unlensed_scalar,
    unlensed_total, tensor) - dict.
    nside: NSIDE of the outpute maps - int.
    out_file: Output filename for the CMB map - str.

    """

    cl = powers['total']
    ell = np.arange(len(cl[:,0]))
    scaling = 2.0*np.pi/(ell*(ell + 1.0))
    scaling[0] = 1.0
    scaling = np.broadcast_to(scaling, np.shape(np.transpose(cl)))
    cl = np.transpose(cl)*scaling

    maps = hp.synfast(cl, nside, new=True)
    hp.write_map(out_file, maps, overwrite=True)
    

def save_fiducial(powers, spectra_key, out_file):

    """
    
    Function for saving a fiducial power spectrum.

    Inputs
    ------
    powers: Dictionary of CMB power spectra (total, lens_potential, lensed_scalar, unlensed_scalar,
    unlensed_total, tensor) - dict.
    spectra_key: CAMB spectrum key for the desired fiducial spectra - str.
    out_file: Output filename for the fiducial power spectrum txt file - str.

    """

    cl = powers[spectra_key]
    cl = cl[2:,:]
    ell = np.arange(2, len(cl[:,0]) + 2)

    np.savetxt(out_file, np.c_[ell, cl])
