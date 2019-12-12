import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors 
from matplotlib.ticker import ScalarFormatter
import numpy as np
import healpy as hp
import configparser
import argparse
import pymaster as nmt
from mk_file_list import mk_file_list


"""

This code estimates E and B power spectra using the NaMaster library. This performs pure-E and pure-B power spectrum
estimation, effectively eliminating issues arising from E to B leakage when dealing with cut-skies. 

"""

########################################################################################################################

print('Reading in basic parameters ...')

parser = argparse.ArgumentParser(description='Code for estimating pseudo-Cl given two CMB maps with uncorrelated '
                                             'noise.')
parser.add_argument('param_file', help='Main parameters file.')

Config = configparser.ConfigParser()
Config.read(parser.parse_args().param_file)

input_Q_map1 = Config.get('Cl Estimation Params', 'input_Q_map1')
input_U_map1 = Config.get('Cl Estimation Params', 'input_U_map1')
input_Q_map2 = Config.get('Cl Estimation Params', 'input_Q_map2')
input_U_map2 = Config.get('Cl Estimation Params', 'input_U_map2')
input_Q_noise1 = Config.get('Cl Estimation Params', 'input_Q_noise1')
input_U_noise1 = Config.get('Cl Estimation Params', 'input_U_noise1')
input_Q_noise2 = Config.get('Cl Estimation Params', 'input_Q_noise2')
input_U_noise2 = Config.get('Cl Estimation Params', 'input_U_noise2')
cmb_Q_posterior_dir = Config.get('Cl Estimation Params', 'cmb_Q_posterior_dir')
cmb_U_posterior_dir = Config.get('Cl Estimation Params', 'cmb_U_posterior_dir')

mask_file = Config.get('Cl Estimation Params', 'mask_file')
beam_file = Config.get('Cl Estimation Params', 'beam_file')

nside = Config.getint('Cl Estimation Params', 'nside')
lmax = Config.getint('Cl Estimation Params', 'lmax')
custom_bpws = Config.getboolean('Cl Estimation Params', 'custom_bpws')
bpws_file = Config.get('Cl Estimation Params', 'bpws_file')
nlb = Config.getint('Cl Estimation Params', 'nlb')
apodize_scale = Config.getfloat('Cl Estimation Params', 'apodize_scale')
apodize_type = Config.get('Cl Estimation Params', 'apodize_type')
purify_e = Config.getboolean('Cl Estimation Params', 'purify_e')
purify_b = Config.getboolean('Cl Estimation Params', 'purify_b')
direct_posterior = Config.getboolean('Cl Estimation Params', 'direct_posterior')
num_monte_carlo = Config.getint('Cl Estimation Params', 'num_monte_carlo')
compute_gaussian_cov = Config.getboolean('Cl Estimation Params', 'compute_gaussian_cov')

leff_out = Config.get('Cl Estimation Params', 'leff_out')
nl_mc_mean_out = Config.get('Cl Estimation Params', 'nl_mc_mean_out')
cl_mc_mean_out = Config.get('Cl Estimation Params', 'cl_mc_mean_out')
cl_BB_mean_out = Config.get('Cl Estimation Params', 'cl_bb_mean_out')
cl_EE_mean_out = Config.get('Cl Estimation Params', 'cl_ee_mean_out')
cl_mc_std_out = Config.get('Cl Estimation Params', 'cl_mc_std_out')
cl_BB_std_out = Config.get('Cl Estimation Params', 'cl_bb_std_out')
cl_EE_std_out = Config.get('Cl Estimation Params', 'cl_ee_std_out')
cl_mc_cov_EE_EE_out = Config.get('Cl Estimation Params', 'cl_mc_cov_EE_EE_out')
cl_mc_cov_BB_BB_out = Config.get('Cl Estimation Params', 'cl_mc_cov_BB_BB_out')
gaussian_cov_out = Config.get('Cl Estimation Params', 'gaussian_cov_out')

########################################################################################################################

print('Reading in the Q/U maps, associated error maps, mask, and beam file (spherical harmonic transform) ...')

Qmap1 = hp.read_map(input_Q_map1)
Umap1 = hp.read_map(input_U_map1)
Qmap2 = hp.read_map(input_Q_map2)
Umap2 = hp.read_map(input_U_map2)

Qnoise1 = hp.read_map(input_Q_noise1)
Unoise1 = hp.read_map(input_U_noise1)
Qnoise2 = hp.read_map(input_Q_noise2)
Unoise2 = hp.read_map(input_U_noise2)

Q_map_list = mk_file_list(cmb_Q_posterior_dir, '.fits', absolute=True)
U_map_list = mk_file_list(cmb_U_posterior_dir, '.fits', absolute=True)

mask = hp.read_map(mask_file)
beam = np.loadtxt(beam_file)

Qmap1[mask == 0] = 0
Umap1[mask == 0] = 0
Qmap2[mask == 0] = 0
Umap2[mask == 0] = 0

########################################################################################################################

print('Estimating pure E and B pseudo-Cls using NaMaster ...')

# Apodize the mask. Key parameters are the apodization scale and type. See the NaMaster docs for details.
mask_apo = nmt.mask_apodization(mask, aposize=apodize_scale, apotype=apodize_type)

# Generate the ell binning scheme.
if custom_bpws:
    bpws = np.loadtxt('bpws_file', usecols=0)
    weights = np.loadtxt('bpws_file', usecols=1)
    b = nmt.NmtBin(nside, bpws=bpws, ells=ell, weights=weights)
    leff = b.get_effective_ells()
elif not custom_bpws:
    b = nmt.NmtBin(nside, nlb=nlb, lmax=lmax)
    leff = b.get_effective_ells()
    nb = nmt.NmtBin(nside, nlb=1)
n_ell = len(leff)


# This function returns NaMaster field objects with our apodized mask and input Q/U maps.
def get_fields(qmap1, umap1, qmap2, umap2):
    fnum1 = nmt.NmtField(mask_apo, [qmap1, umap1], purify_e=purify_e, purify_b=purify_b, beam=beam)
    fnum2 = nmt.NmtField(mask_apo, [qmap2, umap2], purify_e=purify_e, purify_b=purify_b, beam=beam)
    return fnum1, fnum2

def get_noise_fields(qmap1, umap1, qmap2, umap2):
    fnum1 = nmt.NmtField(mask_apo, [qmap1, umap1], purify_e=purify_e, purify_b=purify_b)
    fnum2 = nmt.NmtField(mask_apo, [qmap2, umap2], purify_e=purify_e, purify_b=purify_b)
    return fnum1, fnum2

# Initialise the workspace for the fields (you only have to do this once).
field_num1, field_num2 = get_fields(Qmap1, Umap1, Qmap2, Umap2)
w = nmt.NmtWorkspace()
w.compute_coupling_matrix(field_num1, field_num2, b)

nfield_num1, nfield_num2 = get_noise_fields(np.random.normal(size=len(Qmap1)) * Qnoise1,
                                            np.random.normal(size=len(Umap1)) * Unoise1,
                                            np.random.normal(size=len(Qmap2)) * Qnoise2,
                                            np.random.normal(size=len(Umap2)) * Unoise2)
noise_w = nmt.NmtWorkspace()
noise_w.compute_coupling_matrix(nfield_num1, nfield_num2, b)

# This function does the actual power spectrum calculation.
def compute_master(f_a, f_b, wsp, noise_bias=None):
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    if noise_bias is None:
        cl_decoupled = wsp.decouple_cell(cl_coupled)
    elif noise_bias is not None:
        cl_decoupled = wsp.decouple_cell(cl_coupled, cl_noise=noise_bias)
    return cl_decoupled


# Now we're ready to run over our Monte Carlo simulations.
# Here we estimate the mean Cl, the standard deviation on these, and the full covariance matrix between bandpowers.
data = []
noise_power = []
snl_power = []

field_num1, field_num2 = get_fields(Qmap1, Umap1, Qmap2, Umap2)
cl_mean = compute_master(field_num1, field_num2, w)

fid_r0_lensed_filename = './fiducial_spectra/cmb_fiducial_r0_lensed.dat'
fid_r0p01_unlensed_filename = './fiducial_spectra/cmb_fiducial_r0p01_unlensed.dat'
theory_r = 5e-3
theory_AL = 0.4

theory_ell = np.loadtxt(fid_r0_lensed_filename, usecols=0)
theory_dl_TT = np.loadtxt(fid_r0_lensed_filename, usecols=1)
theory_dl_EE = np.loadtxt(fid_r0_lensed_filename, usecols=2)
theory_dl_BB = np.loadtxt(fid_r0p01_unlensed_filename, usecols=3)
theory_dl_lens = np.loadtxt(fid_r0_lensed_filename, usecols=3)
theory_dl_TE = np.loadtxt(fid_r0_lensed_filename, usecols=4)

theory_cl_TT = 2 * np.pi * theory_dl_TT / (theory_ell * (theory_ell + 1))
theory_cl_EE = 2 * np.pi * theory_dl_EE / (theory_ell * (theory_ell + 1))
theory_cl_BB = 2 * np.pi * (theory_r * theory_dl_BB / 0.01 + theory_AL * theory_dl_lens) / (theory_ell * (theory_ell + 1))
theory_cl_TE = 2 * np.pi * theory_dl_TE / (theory_ell * (theory_ell + 1))

theory_cl_TT = np.insert(theory_cl_TT, 0, np.array([0, 0]))
theory_cl_EE =np.insert(theory_cl_EE, 0, np.array([0, 0]))
theory_cl_BB =np.insert(theory_cl_BB, 0, np.array([0, 0]))
theory_cl_TE =np.insert(theory_cl_TE, 0, np.array([0, 0]))

theory_spectra = (theory_cl_TT, theory_cl_EE, theory_cl_BB, theory_cl_TE)

if direct_posterior:
    num_monte_carlo = len(Q_map_list)

for i in np.arange(num_monte_carlo):
    print('Monte Carlo Sim {}/{}'.format(i + 1, num_monte_carlo))
    if np.all(Qmap1 == Qmap2) and np.all(Umap1 == Umap2):
        cmbI, cmbQ, cmbU = hp.synfast(theory_spectra, nside, new=True, fwhm=np.radians(70.0 / 60.0))
        if not direct_posterior:
            noise_realise_Q1 = np.random.normal(size=len(Qmap1)) * Qnoise1
            noise_realise_U1 = np.random.normal(size=len(Umap1)) * Unoise1
            noise_realise_Q2 = np.copy(noise_realise_Q1)
            noise_realise_U2 = np.copy(noise_realise_U1)
        elif direct_posterior:
            Q_map = hp.read_map(Q_map_list[i])
            U_map = hp.read_map(U_map_list[i])
            noise_realise_Q1 = Q_map - Qmap1
            noise_realise_U1 = U_map - Umap1
            noise_realise_Q2 = np.copy(noise_realise_Q1)
            noise_realise_U2 = np.copy(noise_realise_U1)
        sn_realise_Q1 = noise_realise_Q1 + cmbQ
        sn_realise_U1 = noise_realise_U1 + cmbU
        sn_realise_Q2 = noise_realise_Q2 + cmbQ
        sn_realise_U2 = noise_realise_U2 + cmbU
        noise_realise_Q1[mask == 0] = 0
        noise_realise_U1[mask == 0] = 0
        noise_realise_Q2[mask == 0] = 0
        noise_realise_U2[mask == 0] = 0
        sn_realise_Q1[mask == 0] = 0
        sn_realise_U1[mask == 0] = 0
        sn_realise_Q2[mask == 0] = 0
        sn_realise_U2[mask == 0] = 0
        noise_field_num1, noise_field_num2 = get_noise_fields(noise_realise_Q1, noise_realise_U1, noise_realise_Q2, noise_realise_U2)
        sn_field_num1, sn_field_num2 = get_fields(sn_realise_Q1, sn_realise_U1, sn_realise_Q2, sn_realise_U2)
        nl = compute_master(noise_field_num1, noise_field_num2, noise_w)
        snl = compute_master(sn_field_num1, sn_field_num2, w)
        noise_power.append(nl)
        snl_power.append(snl)
    else:
        noise_realise_Q1 = np.random.normal(size=len(Qmap1)) * Qnoise1
        noise_realise_U1 = np.random.normal(size=len(Umap1)) * Unoise1
        noise_realise_Q2 = np.random.normal(size=len(Qmap2)) * Qnoise2
        noise_realise_U2 = np.random.normal(size=len(Umap2)) * Unoise2
        cmbI, cmbQ, cmbU = hp.synfast(theory_spectra, nside, new=True, fwhm=np.radians(70.0 / 60.0))
        sn_realise_Q1 =	noise_realise_Q1 + cmbQ
        sn_realise_U1 =	noise_realise_U1 + cmbU
        cmbI, cmbQ, cmbU = hp.synfast(theory_spectra, nside, new=True, fwhm=np.radians(70.0 / 60.0))
        sn_realise_Q2 = noise_realise_Q2 + cmbQ
        sn_realise_U2 = noise_realise_U2 + cmbU
        noise_realise_Q1[mask == 0] = 0
        noise_realise_U1[mask == 0] = 0
        noise_realise_Q2[mask == 0] = 0
        noise_realise_U2[mask == 0] = 0
        sn_realise_Q1[mask == 0] = 0
        sn_realise_U1[mask == 0] = 0
        sn_realise_Q2[mask == 0] = 0
        sn_realise_U2[mask == 0] = 0
        noise_field_num1, noise_field_num2 = get_noise_fields(noise_realise_Q1, noise_realise_U1, noise_realise_Q2, noise_realise_U2)
        sn_field_num1, sn_field_num2 = get_fields(sn_realise_Q1, sn_realise_U1, sn_realise_Q2, sn_realise_U2)
        nl = compute_master(noise_field_num1, noise_field_num2, noise_w)
        snl = compute_master(sn_field_num1, sn_field_num2, w)
        noise_power.append(nl)
        snl_power.append(snl)

noise_power = np.array(noise_power)
snl_power = np.array(snl_power)
nl_mean = np.mean(noise_power, axis=0)
cl_mean = cl_mean - nl_mean
cl_std = np.std(snl_power, axis=0)
nl_EE_EE_covar = np.cov(np.transpose(snl_power[:, 0, :]))
nl_BB_BB_covar = np.cov(np.transpose(snl_power[:, 3, :]))

print('Saving Monte Carlo output.')
np.savetxt(leff_out, leff)
np.savetxt(nl_mc_mean_out, nl_mean)
np.savetxt(cl_mc_mean_out, cl_mean)
np.savetxt(cl_BB_mean_out, cl_mean[3])
np.savetxt(cl_EE_mean_out, cl_mean[0])
np.savetxt(cl_mc_std_out, cl_std)
np.savetxt(cl_BB_std_out, cl_std[3])
np.savetxt(cl_EE_std_out, cl_std[0])
np.savetxt(cl_mc_cov_EE_EE_out, nl_EE_EE_covar)
np.savetxt(cl_mc_cov_BB_BB_out, nl_BB_BB_covar)

if compute_gaussian_cov:

    print('Computing Gaussian estimate of the covariance.')

    field_num1, field_num2 = get_fields(Qmap1, Umap1, Qmap2, Umap2)
    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(field_num1, field_num2, b)
    cl_22 = compute_master(field_num1, field_num2, w)[0]
    cl_ee = cl_22[0]
    cl_eb = cl_22[1]
    cl_be = cl_22[2]
    cl_bb = cl_22[3]

    # Generate a covariance workspace for pre-computing the coupling coefficients.
    cw = nmt.NmtCovarianceWorkspace()
    cw.compute_coupling_coefficients(field_num1, field_num1, field_num1, field_num1)

    covar_22_22 = nmt.gaussian_covariance(cw, 2, 2, 2, 2,
                                          [cl_ee, cl_eb, cl_eb, cl_bb],
                                          [cl_ee, cl_eb, cl_eb, cl_bb],
                                          [cl_ee, cl_eb, cl_eb, cl_bb],
                                          [cl_ee, cl_eb, cl_eb, cl_bb],
                                          w, wb=w).reshape([n_ell, 4,
                                                            n_ell, 4])

    np.savetxt(gaussian_cov_out, covar_22_22)

print('Congratulations, you completed the power spectrum estimation!')
