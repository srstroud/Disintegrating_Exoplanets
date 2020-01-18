import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import os

outdir = "results_notCorrectedForSine"

# Param stars (accroding to Gaia DR2)
T_s = 6396.07
R_s = 1.59 * u.solRad
R_s = 1.458 * u.solRad  
R_s = R_s.to(u.au)
#u1 =  #Claret 2017 (https://arxiv.org/abs/1804.10295)
#u2 = #Claret 2017 (https://arxiv.org/abs/1804.10295)

import batman
import spiderman as sp
import lightkurve as lk

spider_params = sp.ModelParams(brightness_model="zhang")
spider_params.n_layers= 5

spider_params.t0 = 1119.7207014691717#2458119.7208322426-2457000               # Central time of PRIMARY transit [days]
spider_params.per = 1.274925046407       # Period [days]
spider_params.inc = 89.4            # Inclination [degrees]
spider_params.ecc= 0.000957929              # Eccentricity
spider_params.w = 9.8                 # Argument of periastron
spider_params.rp = 0.1237202921            # Planet to star radius ratio
spider_params.a= 3.821              # Semi-major axis scaled by stellar radius
spider_params.a_abs = (R_s * spider_params.a).value        # The absolute value of the semi-major axis [AU]

spider_params.p_u1= 0.               # Planetary limb darkening parameter
spider_params.p_u2= 0.               # Planetary limb darkening parameter

spider_params.xi = 0.12       # Ratio of radiative to advective timescale
spider_params.T_n = 3000.     # Temperature of nightside
spider_params.delta_T = 500  # Day-night temperature contrast
spider_params.T_s = T_s    # Temperature of the star

spider_params.l1 = 6.e-7       # The starting wavelength in meters
spider_params.l2 = 1e-6       # The ending wavelength in meters

fpMean = -0.#0001248732

batman_params = batman.TransitParams()
batman_params.t0 = spider_params.t0                      #time of inferior conjunction
batman_params.per = spider_params.per                      #orbital period
batman_params.rp = spider_params.rp                     #planet radius (in units of stellar radii)
batman_params.a = spider_params.a                       #semi-major axis (in units of stellar radii)
batman_params.inc = spider_params.inc                     #orbital inclination (in degrees)
batman_params.ecc = spider_params.ecc                      #eccentricity
batman_params.w = spider_params.w                       #longitude of periastron (in degrees)
batman_params.limb_dark = "quadratic"       #limb darkening model

u1 = 0.273
u2 = 0.10
"""
batman_params.t0 = 2458119.7211                      #time of inferior conjunction
batman_params.per = 1.2749251                      #orbital period
batman_params.rp = 0.12627                     #planet radius (in units of stellar radii)
batman_params.a = 3.835                       #semi-major axis (in units of stellar radii)
batman_params.inc = 88.83                    #orbital inclination (in degrees)
batman_params.ecc = 0.0605                      #eccentricity
batman_params.w = 9.7                       #longitude of periastron (in degrees)
batman_params.limb_dark = "quadratic"       #limb darkening model

u1 = 0.376
u2 = 0.176
"""

batman_params.u = [u1, u2]

t = np.linspace(1491.6, 1517., 1000)
spider_lc = spider_params.lightcurve(t)

m = batman.TransitModel(batman_params, t)    #initializes model
batman_lc = m.light_curve(batman_params)          #calculates light curve

amplitudeStar = 1e-6 * 5e2
omegaStar = 1. / 0.5#1.13
phaseStar = 0.

def modelLC(t):
    m = batman.TransitModel(batman_params, t)    #initializes model
    batman_lc = m.light_curve(batman_params)          #calculates light curve
    lc = batman_lc * spider_params.lightcurve(t) + fpMean
    
    return lc

t = np.linspace(1491.6, 1517., 10000)
lc = lk.LightCurve(time=t, flux=modelLC(t)).fold(batman_params.per, batman_params.t0-0.25*batman_params.per)

tData, yData, dyData = np.loadtxt(os.path.join(outdir, "WASP121-lc.dat")).T
data = lk.LightCurve(time=tData, flux=yData, flux_err=dyData).fold(batman_params.per, batman_params.t0-0.25*batman_params.per).bin(75)

plt.figure(figsize=(10,10))
plt.subplot(2, 1, 1)
plt.plot(lc.time ,lc.flux)
plt.plot(data.time, data.flux, '.')
plt.ylabel("Normalised flux")

plt.subplot(2, 1, 2)
#plt.plot(lc.time ,lc.flux)

time2Model = batman_params.t0 + data.time * batman_params.per - 0.25 * batman_params.per
#time2Model = data.time
plt.plot(data.time, 1e6 * (data.flux-modelLC(time2Model)), '.')
plt.ylabel("Residues [ppm]")

plt.xlabel("Phase")

plt.show()
