import numpy as np
import pandas as pd
import astropy.constants as c
import astropy.units as u
import thecannon as tc
from astropy.io import fits
from scipy import stats
from scipy.interpolate import interp1d

__all__ = ["speed_of_light_kms", "wav", "rvs_model", "calcium_mask", 
	"training_density", "flux_weights"]

# speed of light for doppler shift calculation
speed_of_light_kms = c.c.to(u.km/u.s).value

# gaia RVS wavelength array
wav = fits.open('./data/gaia_rvs_wavelength.fits')[0].data[20:-20]

# Cannon model object to use for model-fitting
rvs_model = tc.CannonModel.read('./data/gaia_rvs_model_cleaned.model')

# calcium mask for model-fitting
ca_idx1 = np.where((wav>849.5) & (wav<850.5))[0]
ca_idx2 = np.where((wav>854) & (wav<855))[0]
ca_idx3 = np.where((wav>866) & (wav<867))[0]
calcium_mask = np.array(list(ca_idx1) + list(ca_idx2) + list(ca_idx3))

# training density calculation for custom model fitting
training_data = rvs_model.training_set_labels
training_density_kde = stats.gaussian_kde(training_data.T)
def training_density(param):
	"""Calculate training density for a particular set of labels"""
	density = training_density_kde(param)[0]
	return density


# empirical relative flux weights for binary model
# based on values from Pecaut & Mamajek (2013)
pm2013 = pd.read_csv('./data/PecautMamajek_table.csv', 
                    delim_whitespace=True).replace('...',np.nan)
teff_pm2013 = np.array([float(i) for i in pm2013['Teff']])
VminusI_pm2013 = np.array([float(i) for i in pm2013['V-Ic']])
V_pm2013 = np.array([float(i) for i in pm2013['Mv']])
mass_pm2013 = np.array([float(i) for i in pm2013['Msun']])

# interpolate between columns
valid_mass = ~np.isnan(mass_pm2013)
teff2Vmag = interp1d(teff_pm2013[valid_mass], V_pm2013[valid_mass])
teff2VminusI = interp1d(teff_pm2013[valid_mass],VminusI_pm2013[valid_mass])
teff2mass = interp1d(teff_pm2013[valid_mass], mass_pm2013[valid_mass])

def flux_weights(teff1, teff2):
    """Calculate relative flux weights of primary, secondary"""
    # compute relative I band flux > flux ratio
    I1 = teff2Vmag(teff1) - teff2VminusI(teff1)
    I2 = teff2Vmag(teff2) - teff2VminusI(teff2)

    # compute relative flux weights
    F_rel = 10**((I2-I1)/2.5)
    W2 = 1/(1+F_rel)
    W1 = 1-W2
    return(W1, W2)
