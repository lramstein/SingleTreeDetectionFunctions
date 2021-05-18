
# -*- coding: utf-8 -*-
"""
Analyses of different single tree detection method combinations
05 March 2021
Laura Ramstein
Swiss Federal Research Institute WSL
"""


# Defining input and output path
# input: CHM, tiff-file, example name structure <CHM>_<LV95>_<Lidar>_<1054000 (plotnumber)>
# input - Poly: polygon of the plot perimeter, shapefile, example name structure <Poly>_<1054000 (plotnumber)>
# input - BDet: detected single trees (output of first script part), csv-file, example name structure <LM1>_<F0>_<Lidar>_<1054000 (plotnumber)>
# input - BRef: reference trees, csv-file, example name structure <BRef>_<EPSG2056>_<1054000>

# output - Filter: filtered CHM, tiff-file
# output - EBD: detected single trees, shapefile and csv-file (x-coordinate, y-coordinate, tree height [m] (Baumhoehe), dbh [mm] (BHD_mm))
# output - ver:

import sys
sys.path.insert(0, r'x')

path_in = r'x'
path_in_Poly = r'x'
path_in_BDet = r'x'
path_in_BRef = r'x'

path_out_Filter = r'x'
path_out_EBD = r'x'
path_out_ver = r'x'
path_out_stat = r'x'


# Importing functions
import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd

# Choosing filter methods
FilterNone = False #F0: Reference
FilterMenk = False #F1: Gaussfilter - Menk
FilterJakubowski = False #F2: Masks - Jakubowski
FilterEysn = False #F3: LM+Filtering - Eysn
FilterOpening = False #F4: Morph opening
FilterClosing = False #F5: Morph closing
FilterKaartinenFGILOCM = False #F6: FGI_LOCM - Kaartinen
FilterKaartinenMetla = False #F7: Metla

# Choosing detection methods
EBDpeaklocalmax = False
EBDlocalmaxima = False
EBDlinearind = False

# Choosing polygons describing the plot perimeter
Poly = True

# defining the minimal tree height for potential support tree
minh = 10.

# calculating all raster files automatically
for root, dirs, files in os.walk(path_in):
    for fname in files:
        if not fname.startswith('.') and os.path.isfile(os.path.join(root, fname)):
            (base, ext) = os.path.splitext(fname) # split base and extension
            if ext in ('.tif'):  # check the extension
                loadname = path_in+'//'+fname

                # for testing
                #loadname = path_in + '//CHM_LV95_Lidar_1054000.tif'
                #base = 'CHM_LV95_Lidar_1054000'

                # defining version
                split = re.split('_', base) #  split filename at symbol _
                daten = split[2]
                plotname = split[3]
                name = daten + '_' + plotname

                # importing CHM
                from EBD_Func_Laura import ReadHeaderAndLoadRaster
                header, chm, raster_proj = ReadHeaderAndLoadRaster(loadname,True, True)
                chm[chm < 0] = 0

                # read cellsize and coordinates of the raster file
                from EBD_Func_Laura import GivePropertiesOfHeader
                cellsize, xmin, xmax, ymin, ymax = GivePropertiesOfHeader(header)
                delta = cellsize[:]


                # calculate x and y coordinates (LV95) of raster files
                image_size = header['nrows'][0], header['ncols'][0]
                x_values = np.arange(xmin + delta / 2., xmax, delta)
                y_values = np.arange(ymax - delta / 2., ymin, -1 * delta)

                ## Filtering of CHM and detection of local maxima (all combinations)
                if FilterNone:
                    from EBD_Func_Laura import FilterNone
                    image, versF = FilterNone(chm, name, header, path_out_Filter)

                    if EBDpeaklocalmax:
                        from EBD_Func_Laura import EBDpeaklocalmax
                        EBDpeaklocalmax(chm, image, versF, delta, raster_proj, x_values, y_values, minh, name, path_out_EBD)

                    if EBDlocalmaxima:
                        from EBD_Func_Laura import EBDlocalmaxima
                        EBDlocalmaxima(chm, image, versF, delta, raster_proj, x_values, y_values, name, path_out_EBD)

                    if EBDlinearind:
                        from EBD_Func_Laura import EBDlinearind
                        EBDlinearind(chm, image, versF, delta, raster_proj, x_values, y_values, name, path_out_EBD)

                if FilterMenk:
                    from EBD_Func_Laura import FilterMenk
                    image, versF = FilterMenk(chm, delta, name, header, path_out_Filter)

                    if EBDpeaklocalmax:
                        from EBD_Func_Laura import EBDpeaklocalmax
                        EBDpeaklocalmax(chm, image, versF, delta, raster_proj, x_values, y_values, minh, name, path_out_EBD)

                    if EBDlocalmaxima:
                        from EBD_Func_Laura import EBDlocalmaxima
                        EBDlocalmaxima(chm, image, versF, delta, raster_proj, x_values, y_values, name, path_out_EBD)

                    if EBDlinearind:
                        from EBD_Func_Laura import EBDlinearind
                        EBDlinearind(chm, image, versF, delta, raster_proj, x_values, y_values, name, path_out_EBD)

                if FilterJakubowski:
                    from EBD_Func_Laura import FilterJakubowski
                    image, versF = FilterJakubowski(chm, delta, name, header, path_out_Filter)

                    if EBDpeaklocalmax:
                        from EBD_Func_Laura import EBDpeaklocalmax
                        EBDpeaklocalmax(chm, image, versF, delta, raster_proj, x_values, y_values, minh, name, path_out_EBD)

                    if EBDlocalmaxima:
                        from EBD_Func_Laura import EBDlocalmaxima
                        EBDlocalmaxima(chm, image, versF, delta, raster_proj, x_values, y_values, name, path_out_EBD)

                    if EBDlinearind:
                        from EBD_Func_Laura import EBDlinearind
                        EBDlinearind(chm, image, versF, delta, raster_proj, x_values, y_values, name, path_out_EBD)

                if FilterEysn:
                    from EBD_Func_Laura import FilterEysn
                    image, versF = FilterEysn(chm, delta, name, header, path_out_Filter)

                    if EBDpeaklocalmax:
                        from EBD_Func_Laura import EBDpeaklocalmax
                        EBDpeaklocalmax(chm, image, versF, delta, raster_proj, x_values, y_values, minh, name, path_out_EBD)

                    if EBDlocalmaxima:
                        from EBD_Func_Laura import EBDlocalmaxima
                        EBDlocalmaxima(chm, image, versF, delta, raster_proj, x_values, y_values, name, path_out_EBD)

                    if EBDlinearind:
                        from EBD_Func_Laura import EBDlinearind
                        EBDlinearind(chm, image, versF, delta, raster_proj, x_values, y_values, name, path_out_EBD)

                if FilterOpening:
                    from EBD_Func_Laura import FilterOpening
                    image, versF = FilterOpening(chm, delta, name, header, path_out_Filter)

                    if EBDpeaklocalmax:
                        from EBD_Func_Laura import EBDpeaklocalmax
                        EBDpeaklocalmax(chm, image, versF, delta, raster_proj, x_values, y_values, minh, name, path_out_EBD)

                    if EBDlocalmaxima:
                        from EBD_Func_Laura import EBDlocalmaxima
                        EBDlocalmaxima(chm, image, versF, delta, raster_proj, x_values, y_values, name, path_out_EBD)

                    if EBDlinearind:
                        from EBD_Func_Laura import EBDlinearind
                        EBDlinearind(chm, image, versF, delta, raster_proj, x_values, y_values, name, path_out_EBD)

                if FilterClosing:
                    from EBD_Func_Laura import FilterClosing
                    image, versF = FilterClosing(chm, delta, name, header, path_out_Filter)

                    if EBDpeaklocalmax:
                        from EBD_Func_Laura import EBDpeaklocalmax
                        EBDpeaklocalmax(chm, image, versF, delta, raster_proj, x_values, y_values, minh, name, path_out_EBD)

                    if EBDlocalmaxima:
                        from EBD_Func_Laura import EBDlocalmaxima
                        EBDlocalmaxima(chm, image, versF, delta, raster_proj, x_values, y_values, name, path_out_EBD)

                    if EBDlinearind:
                        from EBD_Func_Laura import EBDlinearind
                        EBDlinearind(chm, image, versF, delta, raster_proj, x_values, y_values, name, path_out_EBD)

                if FilterKaartinenFGILOCM:
                    from EBD_Func_Laura import FilterKaartinenFGILOCM
                    image, versF = FilterKaartinenFGILOCM(chm, name, header, path_out_Filter)

                    if EBDpeaklocalmax:
                        from EBD_Func_Laura import EBDpeaklocalmax
                        EBDpeaklocalmax(chm, image, versF, delta, raster_proj, x_values, y_values, minh, name, path_out_EBD)

                    if EBDlocalmaxima:
                        from EBD_Func_Laura import EBDlocalmaxima
                        EBDlocalmaxima(chm, image, versF, delta, raster_proj, x_values, y_values, name, path_out_EBD)

                    if EBDlinearind:
                        from EBD_Func_Laura import EBDlinearind
                        EBDlinearind(chm, image, versF, delta, raster_proj, x_values, y_values, name, path_out_EBD)

                if FilterKaartinenMetla:
                    from EBD_Func_Laura import FilterKaartinenMetla
                    image, versF = FilterKaartinenMetla(chm, delta, name, header, path_out_Filter)

                    if EBDpeaklocalmax:
                        from EBD_Func_Laura import EBDpeaklocalmax
                        EBDpeaklocalmax(chm, image, versF, delta, raster_proj, x_values, y_values, minh, name, path_out_EBD)

                    if EBDlocalmaxima:
                        from EBD_Func_Laura import EBDlocalmaxima
                        EBDlocalmaxima(chm, image, versF, delta, raster_proj, x_values, y_values, name, path_out_EBD)

                    if EBDlinearind:
                        from EBD_Func_Laura import EBDlinearind
                        EBDlinearind(chm, image, versF, delta, raster_proj, x_values, y_values, name, path_out_EBD)



# writing statistics of all plots in table
statistik = pd.DataFrame()
statistik_tree = pd.DataFrame()
lst_dict = []
tree_dict = []


for root, dirs, files in os.walk(path_in_BDet):
    for fname in files:
        if not fname.startswith('.') and os.path.isfile(os.path.join(root, fname)):
            (base, ext) = os.path.splitext(fname)  # split base and extension
            if ext.lower() in '.csv':  # check the extension

                # defining version
                split = re.split('_', base)  #  split filename at symbol _
                plotname = split[3]
                datengrundlage = split[2]
                LM_Methode = split[0]
                F_Methode = split[1]

                vers_buffer = 'Buffer_' + base
                vers_ver = 'Ver_' + base

                loadname_BRef = path_in_BRef + '/' + 'BRef_EPSG2056_' + plotname + '.csv'
                loadname_BDet = path_in_BDet + '/' + fname
                loadname_Poly = path_in_Poly + '//' + 'Poly_' + plotname + '.shp'

                # for testing
                #plotname = '21316001'
                #loadname_BRef = path_in_BRef+'//'+'BRef_EPSG2056_' + plotname + '.csv'
                #loadname_BDet = path_in_BDet+'//'+'LM1_F3_Lidar_' + plotname + '.csv'
                #loadname_Poly = path_in_Poly + '//' + 'Poly_' + plotname + '.shp'
                #vers_buffer = 'Buffer_BHD_LM1_F3_Lidar_' + plotname + '.csv'
                #vers_ver = 'Ver_BHD_LM1_F3_Lidar_' + plotname + '.csv'


                from pandas import read_csv

                # input detected single trees (structure: X_Coord [LV95], Y_Coord [LV95], Baumhoehe [m], BHD_mm [mm])
                BRef_Data = read_csv(loadname_BRef, ',')
                BDet_Data = read_csv(loadname_BDet, ',')


                # input reference trees: changing height from [dm] to [m]
                BRef_Data['htot'] = BRef_Data['htot'] / 10

                # choosing year of field record
                if plotname == '34001002': # for plot 34001002 the record from 2004 was nearer to the aerial survey in the year 2003 than the newest record
                    aj = 2004
                else:
                    aj = BRef_Data['aj'].max()


                # defining treshold value for tree height
                # for each plot was defined, from which tree height there are height information for every tree available =
                # tree height, from which with high chance all the tree heights are measured

                Bed_HDiff = 5 # assignment condition for height difference between reference and detected tree
                cut = 0

                if plotname == '1054000':
                     cut = 22
                     cut_Anz = cut - Bed_HDiff
                elif plotname == '1058000':
                     cut = 24
                     cut_Anz = cut - Bed_HDiff
                elif plotname == '1059000':
                     cut = 21
                     cut_Anz = cut - Bed_HDiff
                elif plotname == '1061000':
                     cut = 23
                     cut_Anz = cut - Bed_HDiff
                elif plotname == '21310009':
                     cut = 23
                     cut_Anz = cut - Bed_HDiff
                elif plotname == '21315001':
                     cut = 25
                     BRef_Data = BRef_Data[BRef_Data.bnr != 181]
                     cut_Anz = cut - Bed_HDiff
                elif plotname == '21316001':
                     cut = 20
                     cut_Anz = cut - Bed_HDiff
                elif plotname == '21317001':
                     cut = 22
                     BRef_Data = BRef_Data[BRef_Data.bnr != 31]  # bnr = 10 ist sowieso zu klein (htot = 16m)
                     cut_Anz = cut - Bed_HDiff
                elif plotname == '21318001':
                     cut = 29
                     BRef_Data = BRef_Data[BRef_Data.bnr != 30]
                     cut_Anz = cut - Bed_HDiff
                elif plotname == '34001002':
                     cut = 23
                     BRef_Data = BRef_Data[BRef_Data.bnr != 621]
                     BRef_Data = BRef_Data[BRef_Data.bnr != 629]
                     BRef_Data = BRef_Data[BRef_Data.bnr != 640]
                     cut_Anz = cut - Bed_HDiff

                print('Grenzwert Baumhöhe:')
                print(cut)


                # Buffer around tree top
                from shapely.geometry import Point

                # rename columns
                BRef_Data = BRef_Data.rename(columns={'xcoord': 'x', 'ycoord': 'y'})
                BDet_Data = BDet_Data.rename(columns={'X_Coord': 'x', 'Y_Coord': 'y'})

                # change BRef and BDet to GeoDataFrame
                crs = {'init': 'epsg:2056'}

                BRef_Data['geometry'] = [Point(xy) for xy in zip(BRef_Data.x, BRef_Data.y)]
                BRef_gdf = gpd.GeoDataFrame(BRef_Data, crs=crs, geometry=BRef_Data['geometry'])

                BDet_Data['geometry'] = [Point(xy) for xy in zip(BDet_Data.x, BDet_Data.y)]
                BDet_gdf = gpd.GeoDataFrame(BDet_Data, crs=crs, geometry=BDet_Data['geometry'])

                # crop BDet to plot perimeter
                if Poly == True:
                    from EBD_Func_Laura import CutToPoly
                    BDet_gdf = CutToPoly(loadname_Poly, crs, BDet_gdf)


                # only trees with measured tree height and above the defined treshold for tree height
                BRef = BRef_gdf.loc[(BRef_gdf['htot'] != float('nan')) & (BRef_gdf['htot'] >= cut) & (BRef_gdf['aj'] == aj)].copy()
                BDet = BDet_gdf.loc[(BDet_gdf['Baumhoehe'] != float('nan')) & (BDet_gdf['Baumhoehe'] >= cut)].copy()
                BDet_Anz = BDet_gdf.loc[(BDet_gdf['Baumhoehe'] != float('nan')) & (BDet_gdf['Baumhoehe'] >= cut_Anz)].copy()


                if BDet.empty == False:

                    # defining buffer
                    BRef['geometry'] = BRef.buffer(3)  # radius = 5m
                    #BRef['geometry'] = BRef.buffer(5)  # radius = 3m

                    # cut Buffer-Polygon and BDet
                    #BDet_gdf = BDet_gdf.rename(columns={'index_BDet': 'ind_Det'})
                    BDet['ind_Det'] = BDet.index
                    join = gpd.sjoin(BDet, BRef, how="inner", op="within", lsuffix='BDet', rsuffix='BRef')

                    # calculate distance between BDet and BRef in same polygon
                    join = join.assign(dist="")
                    xy_BDet = join.filter(items=['x_BDet', 'y_BDet'])
                    xy_BRef = join.filter(items=['x_BRef', 'y_BRef'])

                    from scipy.spatial.distance import cdist

                    dist_matrix = cdist(xy_BDet, xy_BRef, 'euclidean')
                    join['dist'] = dist_matrix.diagonal()

                    # calculate tree height difference (htot - BRef, h - BDet)
                    join = join.assign(Diff_h=join['htot'] - join['Baumhoehe'])
                    join_diffh = join.copy()

                    # deleting detected trees outside of treshold
                    # treshold height difference Diff_h < 5m (Kaartinen, 2012)
                    join_kritH = join_diffh.query('Diff_h > -5 & Diff_h < 5')

                    # assign double assigned detected trees to the nearest reference tree
                    kritD_Det = join_kritH.groupby(['ind_Det'])['dist'].transform(min) == join_kritH['dist']
                    join_kritD_Det = join_kritH[kritD_Det]

                    # nearest tree to reference tree
                    join_kritD_Det["bnr"].astype('float64')
                    kritD_Ref = join_kritD_Det.groupby(['bnr'])['dist'].transform(min) == join_kritD_Det['dist']
                    join_kritD_Ref = join_kritD_Det[kritD_Ref]

                    # checking if no duplicate
                    duplicate_Ref = join_kritD_Ref['bnr'].duplicated()
                    duplicate_Det = join_kritD_Ref['ind_Det'].duplicated()
                    if any(duplicate_Ref) == True:
                        import sys

                        sys.exit("aa! errors!")
                    if any(duplicate_Det) == True:
                        import sys

                        sys.exit("aa! errors!")

                    # calculate height statistics
                    Mean_H_Ref = join_kritD_Ref['htot'].mean()
                    Mean_H_Det = join_kritD_Ref['Baumhoehe'].mean()
                    Mean_H_Diff = join_kritD_Ref['Diff_h'].mean()
                    StDev_H_Diff = join_kritD_Ref['Diff_h'].std()

                    # calculate BHD difference (bhd_BRef - BRef, bhd_mm - BDet)
                    join_kritD_Ref = join_kritD_Ref.assign(bhd_BRef=(join_kritD_Ref['d1'] + join_kritD_Ref['d2']) / 2)
                    join_kritD_Ref = join_kritD_Ref.assign(Diff_bhd=join_kritD_Ref['bhd_BRef'] - join_kritD_Ref['BHD_mm'])
                    join_kritD_Ref = join_kritD_Ref.assign(datengrundlage=datengrundlage)
                    join_kritD_Ref = join_kritD_Ref.assign(F_Methode=F_Methode)
                    join_kritD_Ref = join_kritD_Ref.assign(LM_Methode=LM_Methode)
                    join_diff_bhd = join_kritD_Ref.copy()

                    # calculate BHD statistics
                    Mean_BHD_Ref = join_diff_bhd['bhd_BRef'].mean()
                    Mean_BHD_Det = join_diff_bhd['BHD_mm'].mean()
                    Mean_BHD_Diff = join_diff_bhd['Diff_bhd'].mean()
                    StDev_BHD_Diff = join_diff_bhd['Diff_bhd'].std()
                    Mean_dist = join_diff_bhd['dist'].mean()

                    # calculate statistics number of trees
                    Anz_BRef = len(BRef.index)  # number of reference trees above treshold
                    Anz_BRef = float(Anz_BRef)
                    Anz_Zugeo_kritDH = len(join_kritD_Ref.index)  # number of correctly assigned trees (assigning condition distance <3m & height difference <5m)
                    Anz_Zugeo_kritDH = float(Anz_Zugeo_kritDH)
                    if len(BDet_Anz.index) < Anz_Zugeo_kritDH:
                        Anz_BDet = Anz_Zugeo_kritDH
                    else:
                        Anz_BDet = len(BDet_Anz.index)  # number of detected trees above tree height treshold
                    print('Anzahl Referenzbäume = ')
                    print(Anz_BRef)
                    print('Anzahl detektierte Bäume = ')
                    print(Anz_BDet)
                    print('Anzahl korrekt detektierte Bäume = ')
                    print(Anz_Zugeo_kritDH)


                    ## Definition statistics (Kaartinen 2012)
                    # Extraction rate = number of detected trees from all trees (reference trees > 10m)
                    # Matching rate = number of correctly assigned trees from all trees (reference trees >10m, dist <3m & hdiff <5m)
                    # Omission error = reference trees, which couldn't be assigned to a detected tree = not detected reference tree
                    # Comission error = detected tree, which couldn't be assigned to a reference tree
                    # false negative = not detected tree, which should have been detected
                    # false positive = detected tree, which was assign wrong
                    # true negative = not detected tree
                    # true positive = detected tree, which was assigned correctly

                    # http://coldregionsresearch.tpub.com/rsmnl/rsmnl0116.htm
                    # Producer's Accuracy (measure of omission error), User's Accuracy (measure of comission error)

                    # Statistics (Eysn 2015)
                    Extr_rate = Anz_BDet / Anz_BRef * 100  # percent
                    Match_rate = Anz_Zugeo_kritDH / Anz_BRef * 100  # percent
                    Om_error = (Anz_BRef - Anz_Zugeo_kritDH) / Anz_BRef * 100  # percent = false negative
                    if Anz_BDet == 0:
                        Com_error = 0
                    else:
                        Com_error = (Anz_BDet - Anz_Zugeo_kritDH) / Anz_BDet * 100  # percent = false positive

                    # calculate statistics
                    lst_dict.append({'Plotname': plotname, 'LM_Methode': LM_Methode, 'F_Methode': F_Methode,
                                     'Datengrundlage': datengrundlage, 'Mean_BHD_Ref': Mean_BHD_Ref,
                                     'Mean_BHD_Det': Mean_BHD_Det, 'Mean_BHD_Diff': Mean_BHD_Diff,
                                     'StDev_BHD_Diff': StDev_BHD_Diff, 'Mean_H_Ref': Mean_H_Ref, 'Mean_H_Det': Mean_H_Det,
                                     'Mean_H_Diff': Mean_H_Diff, 'StDev_H_Diff': StDev_H_Diff, 'Anz_BRef': Anz_BRef,
                                     'Anz_BDet': Anz_BDet, 'Extr_rate': Extr_rate,
                                     'Anz_Zugeo_kritDH': Anz_Zugeo_kritDH, 'Match_rate': Match_rate, 'Om_error': Om_error,
                                     'Com_error': Com_error, 'Mean_dist': Mean_dist})

                    tree_dict.append(join_diff_bhd)

                    if join_diff_bhd.empty == False:

                        # write SHP and CSV with results
                        join_diff_bhd.to_file(path_out_ver + '//' + vers_ver + '.shp')
                        join_diff_bhd.to_csv(path_out_ver + '//' + vers_ver + '.csv', sep=",")
                        print(vers_ver + '_done')
                    else:
                        print('join_diff_bhd = empty (keine richtig detektierten Bäume)')
                else:
                    print('BDet_gdf = empty (Bäume alle <20m)')

statistik = statistik.append(lst_dict)
statistik.to_csv(path_out_stat + '//statistik.csv', sep=",")

statistik_tree = statistik_tree.append(tree_dict)
statistik_tree.to_csv(path_out_stat + '//statistik_tree.csv', sep=",")

statistik_ZF = statistik.describe(include="all")
statistik_ZF.to_csv(path_out_stat + '//statistik_ZF.csv', sep=",")

print('statistik_done')



