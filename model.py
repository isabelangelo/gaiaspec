# TO DO: think about defining __all__ list here 
# this might be good since I won't need everything
# from model utils

# TO DO : should I stop initializing the model at Teff=8000
# and instead just look at up to 7000?
# I'd have to re-run the code but 8000 is too high...

import numpy as np
from scipy.optimize import leastsq
from model_utils import *

# initial Teff values for binary model optimizer
initial_teff_arr = [(4100,4000), (6000,4000), (8000,4000), 
                   (6100,6000), (8000,6000), (8000,7900),
                   (5000,4000), (7000,4000), (7000,5000), 
                   (7000,6000), (6000,5000), (5000,5000),
                   (7000,7000), (8000,5000), (8000,7000)]

def density_chisq_inflation(param):
	"""Calculate chi-squared value associated with 
	a particular set of model parameters"""
	if False in np.isfinite(param):
		return np.inf # require finite parameters
	else:
		density = training_density(param)
		if density > 1e-7:
			return 1
		else:
			return np.sqrt((1+np.log10(1e-7/density)))

def binary_model(param1, param2, return_components=False):
	"""
	RVS spectrum model of binary system
	note: Fe/H, alpha/Fe of secondary are assumed to be the same as the primary

	Args:
		param1 (list) : [Teff, logg, Fe/H, alpha/Fe, vbroad, RV] of primary
		param2 : [Teff, logg, vbroad, RV] of secondary
	Returns:
		flux_combined (np.array): per-pixel binary flux
	"""

	# store primary, secondary labels
	teff1, logg1, feh1, alpha1, vbroad1, rv1 = param1
	teff2, logg2, vbroad2, rv2 = param2

	# assume same metallicity for both components
	feh2, alpha2 = feh1, alpha1 

	# compute single star models for both components
	flux1 = rvs_model([teff1, logg1, feh1, alpha1, vbroad1])
	flux2 = rvs_model([teff2, logg2, feh2, alpha2, vbroad2])

	# shift flux2 according to drv
	delta_w1 = wav * rv1/speed_of_light_kms
	delta_w2 = wav * rv2/speed_of_light_kms
	flux1_shifted = np.interp(wav, wav + delta_w1, flux1)
	flux2_shifted = np.interp(wav, wav + delta_w2, flux2)

	# compute relative flux based on spectral type
	W1, W2 = flux_weights(teff1, teff2)

	# add weighted spectra together
	flux_combined = W1*flux1_shifted + W2*flux2_shifted

	# return individual components
	if return_components:
		return flux_combined, W1*flux1_shifted, W2*flux2_shifted
	else:
		return flux_combined

def fit_single_star(flux, sigma):
	"""
	Perform single-star model fit to Gaia RVS spectrum.

	Args:
	    flux (np.array): normalized flux data
	    sigma (np.array): flux error data

	Returns:
		fit_cannon_labels (list): model-derived labels 
			[Teff, logg, Fe/H, alpha/Fe, vbroad]
		fit_chisq (float): chi-squared value associated with 
			best-fit model
	"""

	# mask out calcium triplet
	sigma_for_fit = sigma.copy()
	sigma_for_fit[calcium_mask] = np.inf

	def residuals(param):
		"""Calculate the googdness-of-fit associated with 
		a particular single star model"""

		# re-parameterize from log(vbroad) to vbroad for Cannon
		cannon_param = param.copy()
		cannon_param[-1] = 10**cannon_param[-1]
		# compute chisq
		model = rvs_model(cannon_param)
		weights = 1/np.sqrt(sigma_for_fit**2+rvs_model.s2)
		resid = weights * (model - flux)

		# inflate chisq if labels are in low density label space
		density_weight = density_chisq_inflation(cannon_param)
		return resid * density_weight

	# initial labels from cannon model
	initial_labels = rvs_model._fiducials.copy()

	# re-parameterize from vbroad to log(vroad) in optimizer
	initial_labels[-1] = np.log10(initial_labels[-1]) 
	# perform fit
	fit_labels = leastsq(residuals,x0=initial_labels)[0]
	fit_chisq = np.sum(residuals(fit_labels)**2)
	# re-parameterize from log(vbroad) to vbroad
	fit_cannon_labels = fit_labels.copy()
	fit_cannon_labels[-1] = 10**fit_cannon_labels[-1]
	return fit_cannon_labels, fit_chisq


def fit_binary(flux, sigma):
	"""
	Perform binary model fit to Gaia RVS spectrum.
	Asserts that the primary is the brighter star in the fit

	Args:
	    flux (np.array): normalized flux data
	    sigma (np.array): flux error data

	Returns:
		fit_cannon_labels (list): model-derived labels 
			[Teff, logg, Fe/H, alpha/Fe, vbroad, rv] of primary
			and [Teff, logg, vbroad, rv] of secondary
		fit_chisq (float): chi-squared value associated with 
			best-fit model
	"""

	# mask out calcium triplet
	sigma_for_fit = sigma.copy()
	sigma_for_fit[calcium_mask] = np.inf

	def residuals(param):
		# store primary, secondary parameters
		param1 = param[:6].copy()
		param2 = param[6:].copy()

		# re-parameterize from log(vbroad) to vbroad for Cannon
		param1[-2] = 10**param1[-2]
		param2[-2] = 10**param2[-2]
		param2_full = np.concatenate((param2[:2],param1[2:4],param2[2:3]))

		# prevent model from regions where flux ratio can't be interpolated
		if 2450>param1[0] or 34000<param1[0]:
			return np.inf*np.ones(len(flux))
		elif 2450>param2[0] or 34000<param2[0]:
			return np.inf*np.ones(len(flux))
		else:
			# compute chisq
			model = binary_model(param1, param2)
			weights = 1/np.sqrt(sigma_for_fit**2+rvs_model.s2)
			resid = weights * (flux - model)

			# inflate chisq if labels are in low density label space
			primary_density_weight = density_chisq_inflation(param1[:-1])
			secondary_density_weight = density_chisq_inflation(param2_full)
			density_weight = primary_density_weight * secondary_density_weight
			return resid * density_weight

	def optimizer(initial_teff):
		# determine initial labels
		rv_i = 0
		teff1_i, teff2_i = initial_teff
		logg_i, feh_i, alpha_i, vbroad_i = rvs_model._fiducials[1:]
		# re-parameterize from vbroad to log(vroad) for optimizer
		logvbroad_i = np.log10(vbroad_i)
		initial_labels = [teff1_i, logg_i, feh_i, alpha_i, logvbroad_i, rv_i, 
						teff2_i, logg_i, logvbroad_i, rv_i]

		# perform least-sqaures fit
		fit_labels = leastsq(residuals,x0=initial_labels)[0]
		fit_chisq = np.sum(residuals(fit_labels)**2)
		# re-parameterize from log(vbroad) to vbroad
		fit_cannon_labels = fit_labels.copy()
		fit_cannon_labels[4] = 10**fit_cannon_labels[4]
		fit_cannon_labels[8] = 10**fit_cannon_labels[8]
		return fit_cannon_labels, fit_chisq

	# run optimizers, store fit with lowest chi2
	lowest_global_chi2 = np.inf    
	fit_cannon_labels = None

	for initial_teff in initial_teff_arr:
		results = optimizer(initial_teff)
		if results[1] < lowest_global_chi2:
		    lowest_global_chi2 = results[1]
		    fit_cannon_labels = np.array(results[0])

	# assert that the primary is the brighter star
	if fit_cannon_labels[0]<fit_cannon_labels[6]:
		teff2, logg2, feh12, alpha12, vbroad2, rv2 = fit_cannon_labels[:6]
		teff1, logg1, vbroad1, rv1 = fit_cannon_labels[6:]
		fit_cannon_labels = [teff1, logg1, feh12, alpha12, vbroad1, rv1, 
		teff2, logg2, vbroad2, rv2]
	return fit_cannon_labels, lowest_global_chi2


# okay I think I wrote some code for this
# now I need to test it maybe in the jupyter notebook?








