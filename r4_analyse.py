import corner 
import numpy as np
import matplotlib.pyplot as plt
import batman
import spiderman as sp
import lightkurve as lk
import astropy.units as u
import matplotlib
import os

import utils

save = True

outdir = "results_notCorrectedForSineLong"
errorLevel = 68.27

#plt.rc('text', usetex=True)
#plt.rc('font', family="Times New Roman")
plt.rc('font', size=14)

samples = utils.pickleRead(os.path.join(outdir, "posteriors.pkl"))
#thetaNames = ["fpMean", "sigma_w", "t0", "per", "inc", "ecc", "w", "rp", "a", "xi", "T_n", "delta_T", "u1", "u2", "amplitudeStar", "omegaStar", "phaseStar"]
thetaNames = ["fpMean", "sigma_w", "t0", "per", "inc", "ecc", "w", "rp", "a", "xi", "T_n", "delta_T", "u1", "u2"]
#thetaNames = ["sigma_w", "t0", "per", "inc", "ecc", "w", "rp", "a", "xi", "T_n", "delta_T", "u1", "u2"]

errorMinus = 0.5 - errorLevel / 200.
errorPlus = 0.5 + errorLevel / 200.
medians = {}
uppers = {}
print("Results", "*" * 50)
print("Param\tMedian\t\t\tError+({}%CI)\t\tError-({}%CI)".format(errorLevel, errorLevel))
posteriors = {}
for ii in range(len(thetaNames)):
    median = np.median(samples[:,ii])
    posteriors[thetaNames[ii]] = samples[:,ii]
    # Computes the errorLevel CI
    upper = np.quantile(samples[:,ii], errorPlus) - median
    lower = np.quantile(samples[:,ii], errorMinus) - median
    
    medians[thetaNames[ii]] = median
    uppers[thetaNames[ii]] = lower
    print("{}\t{}\t{}\t{}".format(thetaNames[ii], median, upper, lower))

#----------------------------------------------------------------------------------------------------------------
# Compute derived params
print("Derived results", "*" * 50)
posteriors["b"] = np.abs(np.cos(np.deg2rad(posteriors["inc"])) * posteriors["a"])
posteriors["totalDurationDays"] = np.sqrt((1 + posteriors["rp"])**2 - posteriors["b"]**2) * posteriors["per"] / np.pi / posteriors["a"]
posteriors["fullDurationDays"] = np.sqrt(((1 - posteriors["rp"])**2 - posteriors["b"]**2) / ((1 + posteriors["rp"])**2 - posteriors["b"]**2)) * posteriors["totalDurationDays"]

NewthetaNames = ["b", "totalDurationDays", "fullDurationDays"]
print("Param\tMedian\t\t\tError+({}%CI)\t\tError-({}%CI)".format(errorLevel, errorLevel))
for ii in range(len(NewthetaNames)):
    median = np.median(posteriors[NewthetaNames[ii]])
    upper = np.quantile(posteriors[NewthetaNames[ii]], errorPlus) - median
    lower = np.quantile(posteriors[NewthetaNames[ii]], errorMinus) - median
    
    medians[NewthetaNames[ii]] = median
    uppers[NewthetaNames[ii]] = lower
    print("{}\t{}\t{}\t{}".format(NewthetaNames[ii], median, upper, lower))
#----------------------------------------------------------------------------------------------------------------

compare2 = {
    "t0": 2458119.72117-2457000.,
    "per": 1.27492510,
    "inc": 88.83,
    "ecc": 0.0605,
    "w": 9.7,
    "rp": 0.12627,
    "a": 3.835,
    }

print("*" * 50)
print("Comparison to given values (MCMC - given) / upperLimitMCMC")
for name in thetaNames:
    if name in compare2.keys():
        comp = compare2[name]
        mes = medians[name]
        sigma = uppers[name]
        print(name, (mes - comp)/sigma)
        
#----------------------------------------------------------------------------------------------------------------

T_s = 6396.07
R_s = 1.458 * u.solRad 
R_s = R_s.to(u.au)

spider_params = sp.ModelParams(brightness_model="zhang")
spider_params.n_layers= 5

spider_params.t0 = medians["t0"]             # Central time of PRIMARY transit [days]
spider_params.per = medians["per"]       # Period [days]
spider_params.inc = medians["inc"]            # Inclination [degrees]
spider_params.ecc= medians["ecc"]              # Eccentricity
spider_params.w = medians["w"]                 # Argument of periastron
spider_params.rp = medians["rp"]            # Planet to star radius ratio
spider_params.a = medians["a"]              # Semi-major axis scaled by stellar radius
spider_params.a_abs = (R_s * spider_params.a).value        # The absolute value of the semi-major axis [AU]

spider_params.xi = medians["xi"]       # Ratio of radiative to advective timescale
spider_params.T_n = medians["T_n"]     # Temperature of nightside
spider_params.delta_T = medians["delta_T"]  # Day-night temperature contrast
spider_params.T_s = T_s    # Temperature of the star

spider_params.l1 = 6.e-7       # The starting wavelength in meters
spider_params.l2 = 1e-6       # The ending wavelength in meters

spider_params.p_u1= 0.               # Planetary limb darkening parameter
spider_params.p_u2= 0.               # Planetary limb darkening parameter

fpMean = medians["fpMean"]

batman_params = batman.TransitParams()
batman_params.t0 = spider_params.t0                      #time of inferior conjunction
batman_params.per = spider_params.per                      #orbital period
batman_params.rp = spider_params.rp                     #planet radius (in units of stellar radii)
batman_params.a = spider_params.a                       #semi-major axis (in units of stellar radii)
batman_params.inc = spider_params.inc                     #orbital inclination (in degrees)
batman_params.ecc = spider_params.ecc                      #eccentricity
batman_params.w = spider_params.w                       #longitude of periastron (in degrees)
batman_params.limb_dark = "quadratic"       #limb darkening model

batman_params.u = [medians["u1"], medians["u2"]]

def modelLC(t):
    m = batman.TransitModel(batman_params, t)    #initializes model
    batman_lc = m.light_curve(batman_params)          #calculates light curve
    #lc =  (1. + medians["amplitudeStar"] * np.sin(medians["omegaStar"] * t * 2. * np.pi + medians["phaseStar"])) * batman_lc * spider_params.lightcurve(t) + fpMean
    lc = batman_lc * spider_params.lightcurve(t) #+ fpMean
    
    return lc

phaseCenter = 0.
tData, yData, dyData = np.loadtxt(os.path.join(outdir, "WASP121-lc.dat")).T
yData -= fpMean
data = lk.LightCurve(time=tData, flux=yData, flux_err=dyData).fold(batman_params.per, batman_params.t0 - phaseCenter*batman_params.per)#.bin(75)

t = np.linspace(data.time.min(), data.time.max(), 1000)
lc = lk.LightCurve(time=t, flux=modelLC(t * batman_params.per - phaseCenter * batman_params.per + batman_params.t0))

fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True,
                               gridspec_kw={'height_ratios': [2, 1.5, 1, 1]},
                               figsize=(12, 10.5))
fig.set_tight_layout({'rect': [0, 0, 1, 0.98], 'pad': 1., 'h_pad': 0})

dataLine = axes[0].plot(data.time, data.flux, '.', label="Data", alpha=0.5, c='lightgrey', zorder=0, rasterized=True)
dataBinned = data.bin(100, method="median")
binLine = axes[0].errorbar(dataBinned.time, dataBinned.flux, yerr=dataBinned.flux_err, label="Binned data", fmt='.', c='r', ms=3, zorder=1)
modelLine = axes[0].plot(lc.time ,lc.flux, label="Model", c='k', zorder=2)
axes[0].set_ylabel("Normalized flux")
axes[0].legend(loc=0)

axes[0].get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
axes[0].get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
axes[0].grid(b=True, which='minor', linewidth=.2)
axes[0].grid(b=True, which='major', linewidth=1)

dataLine = axes[1].plot(data.time, data.flux, '.', label="Data", alpha=0.5, c='lightgrey', zorder=0, rasterized=True)
axes[1].axhline(1., ls='--', color='grey', lw=2)
binLine = axes[1].errorbar(dataBinned.time, dataBinned.flux, yerr=dataBinned.flux_err, label="Binned data", fmt='.', c='r', ms=3, zorder=1)
modelLine = axes[1].plot(lc.time ,lc.flux, label="Model", c='k', zorder=2, lw=2)


axes[1].set_ylabel("Normalized flux")

axes[1].get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
axes[1].get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
axes[1].grid(b=True, which='minor', linewidth=.2)
axes[1].grid(b=True, which='major', linewidth=1)
axes[1].set_ylim([0.9995,1.001])

#ax2 = plt.subplot(2, 1, 2, sharex=ax1)
#ax2.set_aspect(10000, adjustable='datalim')
t = data.time
lc = lk.LightCurve(time=t, flux=modelLC(t * batman_params.per - phaseCenter * batman_params.per + batman_params.t0))
axes[2].plot(data.time, 1e6 * (data.flux-lc.flux), '.', alpha=0.5, c='lightgrey', zorder=0, rasterized=True)

t = dataBinned.time
lc = lk.LightCurve(time=t, flux=modelLC(t * batman_params.per - phaseCenter * batman_params.per + batman_params.t0))
axes[2].errorbar(dataBinned.time, 1e6 * (dataBinned.flux-lc.flux), yerr=dataBinned.flux_err*1e6, fmt='.', c='r', ms=3, zorder=1)

axes[2].get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
axes[2].get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
#axes[1].grid(b=True, which='minor', linewidth=.2)
axes[2].grid(b=True, which='major', linewidth=1)

axes[2].set_ylabel("Residuals [ppm]")

t = dataBinned.time
lc = lk.LightCurve(time=t, flux=modelLC(t * batman_params.per - phaseCenter * batman_params.per + batman_params.t0))
axes[3].errorbar(dataBinned.time, 1e6 * (dataBinned.flux-lc.flux), yerr=dataBinned.flux_err*1e6, fmt='.', c='r', ms=3, zorder=1)

axes[3].get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
axes[3].get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
axes[3].grid(b=True, which='minor', linewidth=.2, linestyle=":")
axes[3].grid(b=True, which='major', linewidth=1)

axes[3].set_ylabel("Binned residuals [ppm]")
axes[3].set_xlabel("Phase")
axes[2].set_xlim([-0.51, 0.51])

if save:
    fig.savefig(os.path.join(outdir, "WASP121-MCMC-fit.pdf"))

#----------------------------------------------------------------------------------------------------------------

# Analyse the residuals

tData = np.sort(tData)
lc = lk.LightCurve(time=tData, flux=modelLC(tData))
residuals = lk.LightCurve(time=lc.time, flux=yData-lc.flux, flux_err=dyData)
np.savetxt(os.path.join(outdir, "WASP121-residuals.dat"), np.array([residuals.time, residuals.flux, residuals.flux_err]))

plt.show()

#----------------------------------------------------------------------------------------------------------------

fig = corner.corner(samples, labels=thetaNames, hist2d_kwargs={"rasterized": True})
if save:
    fig.savefig(os.path.join(outdir, "WASP121-triangle.pdf"))

