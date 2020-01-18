from astropy.table import Table
import astropy.units as u
from astropy.io import fits 
import os
from lightkurve import LightCurve
import numpy as np
import requests
import gzip 
import pickle

import logging
logger = logging.getLogger(__name__)

import subprocess
from matplotlib import rc

def set_fancy(textsize=16):
    rc('font',**{'family':'serif','serif':['Palatino'],'size':textsize})
    rc('text', usetex=True)

def savefig(fname, fig, fancy=False, pdf_transparence=True):
    directory=os.path.dirname(os.path.abspath(fname))
    if not os.path.exists(directory):
        os.makedirs(directory)

    fig.savefig(fname+'.png',dpi=300)

    if fancy: 
        fig.savefig(fname+'.pdf',transparent=pdf_transparence)
        command = 'pdfcrop %s.pdf' % fname
        subprocess.check_output(command, shell=True)
        os.system('mv '+fname+'-crop.pdf '+fname+'.pdf')
        
def mkdir(somedir):
    """
    A wrapper around os.makedirs.
    :param somedir: a name or path to a directory which I should make.
    """
    if not os.path.isdir(somedir):
        os.makedirs(somedir)
        
def TESSSectorLookUp(targetName):
    
    url = 'https://heasarc.gsfc.nasa.gov/cgi-bin/tess/webtess/wtv.py?Entry={}'.format(targetName)
    r = requests.get(url)
    txt = r.text.split("<pre>")[0].split("<\pre>")[0].split("\n") #I changed ("<pre>")[1] to ("<pre>")[0]
    
    res = []
    sectorsNb = []
    for sec in txt:
        if "observed in camera" in sec:
            res.append(sec)
            secNb = int(sec.split(" (")[0][7:])
            sectorsNb.append(secNb)
    if len(res) == 0:
        res = ["will not be observed"]
    
    return "\n".join(res), sectorsNb

def loadLC(folderName, downloadDir, errorIfNot2Min=True, dumpHeader=False, delimiter="|", fluxType="PDCSAP", normalised=True):
    """
    Loads multiple and single Tess light curves and creates a Lightkurve object to store them in. 
    Multiple LCs are detected when the folderName string contains a delimiter.
    
    :param folderName: name of the data folder
    :param downloadDir: name of the root data folder
    :param errorIfNot2Min: behaviour if cadence is not 2 min if `True` (default), raises an error, else warning
    :param dumpHeader: if `True` prints the header of the data
    :param delimiter: delimiter chosen to separate the data path, defualt: |
    :param fluxType: SAP or PDCSAP
    :param normalised: if `True` returns the median-normalised flux
    """

    lc = None
    if "|" in folderName:
        folderNames = folderName.split(delimiter)
    else:
        folderNames = [folderName]
        
    for folderName in folderNames:
        imgFname = "{}_lc.fits".format(folderName)
        imgFname = os.path.join(downloadDir, folderName, imgFname)
        head = fits.getheader(imgFname)
        sector = head["sector"]
        if dumpHeader:
            print(repr(head))
        lightCurveData = Table.read(imgFname)
        cadeances = np.nanmedian((lightCurveData["TIME"][1:] - lightCurveData["TIME"][:-1])*24*60)
        if np.abs(cadeances-2.) < 0.5:
            logger.info("Cadence is 2 min for {}".format(imgFname))
        else:
            if errorIfNot2Min:
                raise RuntimeError("Cadence is {:1.1f} min for {}".format(cadeances, imgFname))
            else:
                logger.warning("Cadence is {:1.1f} min for {}".format(cadeances, imgFname))
        
        lightCurveData["TIME"].unit = u.day
        time = lightCurveData["TIME"]
        flux = lightCurveData["{}_FLUX".format(fluxType)] 
        fluxErr = lightCurveData["{}_FLUX_ERR".format(fluxType)]

        meta = {
            "TIME": lightCurveData["TIME"],
            "MOM_CENTR1": lightCurveData["MOM_CENTR1"],
            "MOM_CENTR2": lightCurveData["MOM_CENTR2"],
            "MOM_CENTR1_ERR": lightCurveData["MOM_CENTR1_ERR"],
            "MOM_CENTR2_ERR": lightCurveData["MOM_CENTR2_ERR"],
            "POS_CORR1": lightCurveData["POS_CORR1"],
            "POS_CORR2": lightCurveData["POS_CORR2"],
            }

        lcTemp = LightCurve(time=time, flux=flux, flux_err=fluxErr, meta=meta)
        lcTemp = lcTemp.remove_nans()
        if normalised:
            lcTemp = lcTemp.normalize()
            
        if lc is None:
            lc = lcTemp
            sectors = sector
        else:
            lc = lc.append(lcTemp)
            sectors = "{},{}".format(sectors, sector)
    
    ids = np.argsort(lc.time)
    lc = lc[ids]
    
    return lc, sectors

def pickleWrite(obj, filepath, protocol=-1):
    """
    I write your python object obj into a pickle file at filepath.
    If filepath ends with .gz, I'll use gzip to compress the pickle.

    :param obj: python container you want to compress
    :param filepath: string, path where the pickle will be written
    :param protocol: Leave protocol = -1 : I'll use the latest binary protocol of pickle.

    """
    if os.path.splitext(filepath)[1] == ".gz":
        pkl_file = gzip.open(filepath, 'wb')
    else:
        pkl_file = open(filepath, 'wb')

    pickle.dump(obj, pkl_file, protocol)
    pkl_file.close()
    logger.debug("Wrote %s" % filepath)


def pickleRead(filepath):
    """
    I read a pickle file and return whatever object it contains.
    If the filepath ends with .gz, I'll unzip the pickle file.

    :param filepath: string, path of the pickle to load
    :return: object contained in the pickle
    """
    if os.path.splitext(filepath)[1] == ".gz":
        pkl_file = gzip.open(filepath,'rb')
    else:
        pkl_file = open(filepath, 'rb')
    obj = pickle.load(pkl_file)
    pkl_file.close()
    logger.debug("Read %s" % filepath)
    return obj

