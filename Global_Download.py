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
from basic_functions import stop

from utils import TESSSectorLookUp


#Go to https://exofop.ipac.caltech.edu/tess/view_toi.php
#Download table as text file 







logging.basicConfig(format='%(asctime)s %(levelname)s: %(name)s(%(funcName)s): %(message)s', level=logging.INFO)

dataDir = "data"

StarListFName = os.path.join(dataDir, "target_names.txt")
StarList = Table.read(StarListFName, format="ascii.no_header", delimiter="|")
StarList['col1'].name = "name"

tics = []
dist2NearestTics = []
obsIDList = []
mags = []
ras = []
decs = []
pathList = []
TessMags = []
sectorList = []

for iTarget, targetName in enumerate(list(StarList["name"])):
    print('Searching for target '+targetName)
    
    logging.debug("{}".format("*" * 100))
    logging.debug("{}".format("*" * 100))
    logging.info("*** {} ({}/{}) {}".format(targetName, iTarget+1, len(list(StarList["name"])), "*" * 80))
    
    #Search for target in MAST database, more specifically in the TESS Input Catalog (TIC)
    try:
        StarListTics = Catalogs.query_object(targetName, radius=.02, catalog="TIC")
    except :#astroquery.exceptions.ResolverError():
        logging.error("COULD NOT RESOLVE {} TO A SKY POSITION !!!!".format(targetName))
        continue
    where_closest = np.argmin(StarListTics['dstArcSec'])
    if len(np.array([where_closest]))!=1:stop('More than one or no target found')
    
    tics.append(StarListTics['ID'][where_closest])
    dist2NearestTics.append(StarListTics['dstArcSec'][where_closest])
    mags.append(StarListTics['Tmag'][where_closest])
    TessMags.append(StarListTics['Tmag'][where_closest])
    
    #Store information about target
    print("Closest TIC ID to %s: TIC %s, separation of %f arcsec. and a TESS mag. of %f"%
          (targetName, StarListTics['ID'][where_closest], StarListTics['dstArcSec'][where_closest],
           StarListTics['Tmag'][where_closest]))

    
    #Try finding target in simbad catalog
    simbadResults = Simbad.query_object(targetName,verbose=False)
    if simbadResults is None:
        for nam_typ in ['HIP', 'TYC', 'UCAC', 'TWOMASS', 'SDSS', 'ALLWISE', 'GAIA', 'APASS', 'KIC']:
            simbadResults = Simbad.query_object(StarListTics[nam_typ][where_closest],verbose=False)
            if simbadResults is not None:break
    if simbadResults is not None:
        coord = SkyCoord(simbadResults["RA"][0], simbadResults["DEC"][0], unit=[u.hourangle, u.deg])
        ras.append(coord.ra.value)
        raUnit = coord.ra.unit
        decs.append(coord.dec.value)
        decUnit = coord.dec.unit
    else:
        ras.append(StarListTics["ra"][where_closest])
        decs.append(StarListTics["dec"][where_closest])        

    #Retrieve data
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
    
    print('---------------------------------------------')

StarList["RA"] = ras
StarList["RA"].unit = raUnit
StarList["DEC"] = decs
StarList["DEC"].unit = decUnit
StarList["TIC"] = tics
StarList["TessMag"] = TessMags
StarList["dist2TIC"] = dist2NearestTics
StarList["dist2TIC"].unit = u.arcsec
StarList["TessMag"] = mags
StarList["sectorName"] = sectorList
StarList["obsID"] = obsIDList
StarList["folderName"] = pathList

StarList.write(os.path.join(dataDir, "targetsIds.fits"), format="fits", overwrite=True)
