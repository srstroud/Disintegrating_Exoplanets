#from astroquery.mast import Tesscut
from astroquery.mast import Catalogs
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
from astropy.table import Table
import numpy as np
import astropy.units as u
from astroquery.mast import Observations
import os
import logging

from utils import TESSSectorLookUp

logging.basicConfig(format='%(asctime)s %(levelname)s: %(name)s(%(funcName)s): %(message)s', level=logging.INFO)

dataDir = "data"

cheopsListFName = os.path.join(dataDir, "target_names.txt")
cheopsList = Table.read(cheopsListFName, format="ascii.no_header", delimiter="|")
cheopsList['col1'].name = "name"

tics = []
dist2NearestTics = []
obsIDList = []
mags = []
ras = []
decs = []
pathList = []
TessMags = []
sectorList = []

for iTarget, targetName in enumerate(list(cheopsList["name"])):
    
    logging.debug("{}".format("*" * 100))
    logging.debug("{}".format("*" * 100))
    logging.info("*** {} ({}/{}) {}".format(targetName, iTarget+1, len(list(cheopsList["name"])), "*" * 80))
    try:
        cheopsListTics = Catalogs.query_object(targetName, radius=.02, catalog="TIC")
    except :#astroquery.exceptions.ResolverError():
        logging.error("COULD NOT RESOLVE {} TO A SKY POSITION !!!!".format(targetName))
        continue
    where_closest = np.argmin(cheopsListTics['dstArcSec'])
    
    tics.append(cheopsListTics['ID'][where_closest])
    dist2NearestTics.append(cheopsListTics['dstArcSec'][where_closest])
    mags.append(cheopsListTics['Tmag'][where_closest])

    logging.info("Closest TIC ID to %s: TIC %s, separation of %f arcsec. and a TESS mag. of %f"%
          (targetName, cheopsListTics['ID'][where_closest], cheopsListTics['dstArcSec'][where_closest],
           cheopsListTics['Tmag'][where_closest]))

    simbadResults = Simbad.query_object(targetName)
    coord = SkyCoord(simbadResults["RA"][0], simbadResults["DEC"][0], unit=[u.hourangle, u.deg])
    ras.append(coord.ra.value)
    raUnit = coord.ra.unit
    decs.append(coord.dec.value)
    decUnit = coord.dec.unit
    TessMags.append(cheopsListTics['Tmag'][where_closest])

    obsTable = Observations.query_criteria(filters=["TESS"], objectname=targetName, dataproduct_type=["TIMESERIES"],radius="0.01 deg")
    
    proceed = True
    if len(obsTable) == 0:
        proceed = False
        obsIDList.append(np.nan)
        pathList.append("NA")
        
        sectorsTxt, sectorsNb = TESSSectorLookUp(targetName)
        if len(sectorsNb) == 0:
            sectorList.append("")
        elif len(sectorsNb) == 1:
            sectorList.append(sectorsNb[0])
        else:
            sectorList.append(",".join(["{}".format(s) for s in sectorsNb]))
                              
        logging.info("TESS observations for {} will be in\n{}".format(targetName, sectorsTxt))
        
    if proceed == False:
        logging.info("Retrieved info: name={}, obsID={}".format(targetName, "N/A"))
        continue
    
    dataProductsByObservation = Observations.get_product_list(obsTable)
    
    obsID = obsTable["obsid"]
    
    logging.debug("Dumping obsTable info: {}".format(obsTable))
    if len(obsID) > 1:
        logging.info("There is more than 1 obs to download: {}".format(obsID))
        #raise RuntimeError("You'll have to check this one by hand, there are objects close by.")
        sec_ = []
        for obsrow in obsTable["obs_id"]:
            sec_.append(str(int(obsrow.split("-s")[1].split("-")[0])))
        sectorList.append(",".join(sec_))
        pathList.append("|".join(obsTable["obs_id"]))
    elif len(obsID) == 0:
        raise RuntimeError("No observation found --> you should change something")
    else:
        pathList.append(obsTable["obs_id"][0])
        sectorList.append(int(obsTable["obs_id"][0].split("-s")[1].split("-")[0]))

    ### OLD: seems wrong: obsIDList.append(obsID[0])
    obsIDList.append(pathList[-1])    
    logging.info("Retrieved info: name={}, obsID={}".format(targetName, pathList[-1]))
    
    Observations.download_products(obsID,
                                   #dataproduct_type=["TIMESERIES"],
                                   obs_collection=["TESS"],
                                   extension="fits")

cheopsList["RA"] = ras
cheopsList["RA"].unit = raUnit
cheopsList["DEC"] = decs
cheopsList["DEC"].unit = decUnit
cheopsList["TIC"] = tics
cheopsList["TessMag"] = TessMags
cheopsList["dist2TIC"] = dist2NearestTics
cheopsList["dist2TIC"].unit = u.arcsec
cheopsList["TessMag"] = mags
cheopsList["sectorName"] = sectorList
cheopsList["obsID"] = obsIDList
cheopsList["folderName"] = pathList

cheopsList.write(os.path.join(dataDir, "targetsIds.fits"), format="fits", overwrite=True)
