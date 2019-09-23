import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from lightkurve import LightCurve
import astropy.units as u
import logging
import batman
import matplotlib
from basic_functions import stop

logging.basicConfig(format='%(asctime)s %(levelname)s: %(name)s(%(funcName)s): %(message)s', level=logging.DEBUG)

import utils

dataDir = "data"
downloadDir = "mastDownload/TESS"
outdir = "results_notCorrectedForSineLong"
correctSine = False

###################################################################################################

utils.mkdir(outdir)

targets = Table.read(os.path.join(dataDir, "targetsIds.fits"))

idsDownloaded = np.logical_not(targets["folderName"] == "NA")

targets = targets[idsDownloaded]

q1 = 0.2098861500
q2 = 0.2455166013
u1 = 2 * np.sqrt(q1) * q2
u2 = np.sqrt(q1) * (1. - 2 * q2)
print(u1)
print(u2)
#
params = batman.TransitParams()
params.t0 =  2458119.72117-2457000                      #time of inferior conjunction
params.per = 1.27492510                      #orbital period
params.rp = 0.12627                      #planet radius (in units of stellar radii)
params.a = 3.835                       #semi-major axis (in units of stellar radii)
params.inc = 87.08                     #orbital inclination (in degrees)
params.ecc = 0.0605                      #eccentricity
params.w = 9.7                       #longitude of periastron (in degrees)
params.u = [u1, u2]                #limb darkening coefficients
params.limb_dark = "quadratic"       #limb darkening model

# According to TESS alert
params.t0 = 1491.9986898
params.a = 3.8083
params.rp = 0.1234

q1 = 0.2098861500
q2 = 0.2455166013
u1 = 2 * np.sqrt(q1) * q2
u2 = np.sqrt(q1) * (1. - 2 * q2)
params.u = [u1, u2]

antiphaseCentred = 0.

period = params.per * u.day#1.27492510 * u.day
t0 = params.t0 * u.day #(2458119.72117-2457000) * u.day

def medianDetrend(lc, durationTransitHour, windowFactorDurationTransit, returnMedians=False):
    
    durationTransit = durationTransitHour / 24.
    window = durationTransit * windowFactorDurationTransit
    
    t = lc.time / window
    t -= np.amin(t)
    tMax = int(np.ceil(np.amax(t)))
    
    medians = []
    ts = []
    te = []
    for windowStart in range(0, tMax):
        ts.append(windowStart)
        te.append(windowStart+1.)
        currentIds = np.where(np.logical_and(t >= windowStart, t < windowStart+1.))
        median = np.nanmedian(lc.flux[currentIds])
        medians.append(median)
        lc.flux_err[currentIds] /= median        
        lc.flux[currentIds] /= median

    if returnMedians:
        return lc, medians, ts, te, t
    return lc 

for target in targets:
    
    logging.info("*** {} {}".format("WASP-121", "*" * 80))
    
    lc = None
    folderName = target["folderName"]
    folderName = "tess2019006130736-s0007-0000000022529346-0131-s"
    
    lc, sectors = utils.loadLC(folderName, downloadDir, fluxType="PDCSAP", normalised=True)
    lcOri = lc.copy()
    """
    lcSAP, sectors = utils.loadLC(folderName, downloadDir, fluxType="SAP", normalised=False)
    
    #####################################################################################
    # 1st part
    idPart1 = lcSAP.time < 1504
    lc1 = lcSAP[idPart1]

    lc1 = lc1.remove_outliers(3)
    
    p = np.poly1d(np.polyfit(lc1.time, lc1.flux, 3)) 
    
    plt.figure()
    plt.scatter(lc1.time, lc1.flux, label="lcSAP", marker='.', s=5)
    plt.plot(lc1.time, p(lc1.time), c='r')

    lcSAP.flux[idPart1] /= p(lcSAP.time[idPart1])
    lcSAP.flux_err[idPart1] /= p(lcSAP.time[idPart1])
    
    #lcSAP.flux[idPart1] /= np.median(lcSAP.flux[idPart1])
    #lcSAP.flux_err[idPart1] /= np.median(lcSAP.flux[idPart1])
    
    #####################################################################################
    # 2nd part
    idPart2 = lcSAP.time > 1504
    lc2 = lcSAP[idPart2]

    lc2 = lc2.remove_outliers(3)
    
    p = np.poly1d(np.polyfit(lc2.time, lc2.flux, 3)) 
    
    plt.figure()
    plt.scatter(lc2.time, lc2.flux, label="LC", marker='.', s=5)
    plt.plot(lc2.time, p(lc2.time), c='r')

    lcSAP.flux[idPart2] /= p(lcSAP.time[idPart2])
    lcSAP.flux_err[idPart2] /= p(lcSAP.time[idPart2])
    
    #lcSAP.flux[idPart2] /= np.median(lcSAP.flux[idPart2])
    #lcSAP.flux_err[idPart2] /= np.median(lcSAP.flux[idPart2])
    
    lcSAP = lcSAP.remove_outliers()
    """
    ####################################################################################
    
    if correctSine:
        idPart2 = lc.time > 1502
        lc2 = lc[idPart2]
        est_amp = 0.0003613511968221827
        est_period = 1.3451298903347606
        est_phase = 0.8884181450805606
        sinFit = est_amp*np.sin((lc2.time / est_period + est_phase)* 2. * np.pi) 
        lc.flux[idPart2] -= sinFit


        
    #####################################################################################

    #Save raw light curve, after removing spurious events, outlier correction and normalization

    #lc.flux[np.where(np.logical_and(lc.time < 1511.85, lc.time > 1511.55))] = np.nan
    lc.flux[lc.time < 1491.92] = np.nan
    lc.flux[np.where(np.logical_and(lc.time < 1505., lc.time > 1503.))] = np.nan
    #lc.flux[np.where(np.logical_and(lc.time < 1509., lc.time > 1507.3))] = np.nan
    #lc.flux[np.where(np.logical_and(lc.time < 1511.85, lc.time > 1511.))] = np.nan
    lc = lc.remove_outliers()
    lc = lc.remove_nans().normalize()
    np.savetxt(os.path.join(outdir, "WASP121-lc_raw.dat"), np.array([lc.time, lc.flux, lc.flux_err]).T)
    stop()



    
    lc = medianDetrend(lc, durationTransitHour=24.*1.2749, windowFactorDurationTransit=1.)
    
    
    #####################################################################################

    np.savetxt(os.path.join(outdir, "WASP121-lc.dat"), np.array([lc.time, lc.flux, lc.flux_err]).T)
    
    #####################################################################################
    # Show what it looks like
    plt.figure()
    plt.scatter(lc.time, lc.flux, label="LC", marker='.', s=5)
    lcB = lc.bin(100, method="median")
    plt.scatter(lcB.time, lcB.flux, label="LC", marker='.', s=5)
    #plt.show(); exit()
    #####################################################################################    
    data = Table([lc.time, lc.flux,  lc.flux_err])
    #data.write("WASP121-lc.dat", format="ascii.no_header", overwrite=True)
    
    durations = np.linspace(0.05, 0.2, 15) * u.day
       
    #####################################################################################
    
    modelTime =np.linspace(1491.5, 1516., 5000)#lc.time# 
    m = batman.TransitModel(params, modelTime)    #initializes model
    modelFlux = m.light_curve(params)          #calculates light curve
    lcBatman = LightCurve(time=modelTime, flux=modelFlux)
       
    ###########################################################################
    
    plt.rc('font', family="Times New Roman")
    plt.rc('font', size=14)
    fig, axes = plt.subplots(2, 1, figsize=(6, 6))
    fig.subplots_adjust(hspace=0.3)
    
    # Plot the light curve and best-fit model
    ax = axes[0]
    ax.plot(lc.time, lc.flux, ".", c='lightgrey', ms=3)
    x = np.linspace(lc.time.min(), lc.time.max(), 3*len(lc.time))
    
    lcp = lc.bin(5)
    ax.errorbar(lcp.time, lcp.flux, yerr=lcp.flux_err, fmt='.', c='k', ms=3)
    ax.plot(modelTime, modelFlux, lw=0.75)
    ax.set_xlabel("time [days]")
    ax.set_ylabel("Normalised flux");
    ax.set_xlim([1491, 1516])
    
    # Plot the folded data points
    ax = axes[1]
    x = (lc.time * u.day - t0 + antiphaseCentred*period) % period - 0.5*period
    m = np.abs(x) < 0.5 * period# *u.day
    ax.plot(x[m] / period + 0.5 - antiphaseCentred, lc.flux[m], ".", color="lightgrey", ms=3)
    
    lcp = lc.fold(period.value, t0=t0.value+(antiphaseCentred-0.5)*period.value).bin(50)
    lct = lcp.time + 0.5 - antiphaseCentred# + antiphaseCentred - 0.5
     
    ax.errorbar(lct, lcp.flux, yerr=lcp.flux_err, fmt='.', c='k', ms=3)
    
    # Over-plot the best fit model
    ax.set_xlabel("Phase")
    ax.set_ylabel("Normalised flux")
    
    xM = (modelTime - params.t0 + antiphaseCentred*params.per) % params.per - 0.5*params.per
    m = np.abs(xM) < 0.5 * params.per# *u.day
    idsSort = np.argsort(xM[m])
    ax.plot(xM[m][idsSort] / params.per + 0.5 - antiphaseCentred, modelFlux[m][idsSort], lw=2)
    
    plt.suptitle(target["name"])
    
    ax.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.grid(b=True, which='minor',  linewidth=.2)
    ax.grid(b=True, which='major',  linewidth=1)
    
    #####################################################################################################################
    
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True,
                                   gridspec_kw={'height_ratios': [1, 1]},
                                   figsize=(12, 5.))
    fig.set_tight_layout({'rect': [0, 0, 1, 0.98], 'pad': 1., 'h_pad': 0})
    
    #dataLine = axes[0].plot(lcOri.time, lcOri.flux, '.', c='k', ms=1.5, zorder=0, rasterized=True)
    markers, caps, bars = axes[0].errorbar(lcOri.time, lcOri.flux, lcOri.flux_err, fmt='o', c='k', ms=1., zorder=0, rasterized=True, capthick=0)
    [bar.set_alpha(0.1) for bar in bars]
    axes[0].set_ylabel("Normalized flux")
    
    axes[0].get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    axes[0].get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    axes[0].grid(b=True, which='minor', linewidth=.2)
    axes[0].grid(b=True, which='major', linewidth=1)
    axes[0].text(0.02, 0.005, 'TESS PDC-corrected SAP photometry',
        verticalalignment='bottom', horizontalalignment='left',
        transform=axes[0].transAxes, ) #fontsize=15)
    axes[0].set_ylim([0.978, 1.005])
    
    #dataLine = axes[1].plot(lc.time, lc.flux, '.', c='k', ms=1.5, zorder=0, rasterized=True)
    markers, caps, bars = axes[1].errorbar(lc.time, lc.flux, lc.flux_err, fmt='o', c='k', ms=1., zorder=0, rasterized=True, capthick=0)
    [bar.set_alpha(0.1) for bar in bars]
    axes[1].set_ylabel("Normalized flux")
    
    axes[1].get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    axes[1].get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    axes[1].grid(b=True, which='minor', linewidth=.2)
    axes[1].grid(b=True, which='major', linewidth=1)
    axes[1].set_xlabel("Epoch (BJD$_\mathrm{TDB}$ - 2457000)")
    axes[1].text(0.02, 0.005, 'Corrected flux',
        verticalalignment='bottom', horizontalalignment='left',
        transform=axes[1].transAxes, ) #fontsize=15)
    axes[1].set_xlim([1491.5, 1516.5])
    axes[1].set_ylim([0.978, 1.005])
    fig.savefig(os.path.join(outdir, "WASP121-TESS-lc.pdf"))
    
    ###########################################################################
    # Look at the 1.13 d stellar rotation Period
    
    lcTransitRemoved = lc.copy()
    
    phase = (t0.value % period.value) / period.value
    fold_time = (((lcTransitRemoved.time - phase * params.per) / params.per) % 1)
    fold_time[fold_time > 0.5] -= 1
    idsNan = np.abs(fold_time) < 0.05
    lcTransitRemoved.flux[idsNan] = np.nan
    idsNan = np.logical_and(fold_time < 0.55, fold_time > 0.45)
    lcTransitRemoved.flux[idsNan] = np.nan
    lcTransitRemoved.remove_nans()
    np.savetxt(os.path.join(outdir, "WASP121-lcTransitRemoved.dat"), np.array([lcTransitRemoved.time, lcTransitRemoved.flux, lcTransitRemoved.flux_err]).T)

    periodSec = 1.13#1.274921
    lcTransitRemovedFolded = lcTransitRemoved.fold(periodSec, t0=params.t0+params.per*0.5)
    
    plt.figure()
    plt.scatter(lcTransitRemovedFolded.time, (lcTransitRemovedFolded.flux-1.)*1e6, marker=".", color="lightgrey", s=3)
    lcTransitRemovedFoldedBinned = lcTransitRemovedFolded.bin(500)
        
    plt.errorbar(lcTransitRemovedFoldedBinned.time, (lcTransitRemovedFoldedBinned.flux-1.)*1e6, yerr=lcTransitRemovedFoldedBinned.flux_err*1e6, fmt='.', c='k', ms=3)
    plt.title("Folded at {} day".format(periodSec))
    plt.xlabel("Phase")
    plt.ylabel("Difference to normalised flux [ppm]")
    
    ###########################################################################

    plt.show()

    logging.debug("{}".format("*" * 100))
    logging.debug("{}".format("*" * 100))
    
    
