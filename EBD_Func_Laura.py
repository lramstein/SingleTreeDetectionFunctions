
# -*- coding: utf-8 -*-
"""
Created on 5th of february 2020

@author: RamsteiL
Projekt "Neue Grundlagen für eine effiziente Seillinienplanung"


"""

### 0) Funktionen

def ReadHeaderAndLoadRaster(loadname, OptionReadHeader, OptionLoadRaster):
    #author: Leo Bont, WSL
    
    from osgeo import gdal
    import numpy as np

    #OptionReadHeader = True
    #OptionLoadRaster = True

    raster = []
    header = []
    # Lesen des Headers
    # Laden der Rasterdaten:
    loadname.lower()
    if loadname.endswith(".tif"):
        if OptionReadHeader:
            [header, nn, raster_proj] = read_header_of_tiff_file(loadname)
        if OptionLoadRaster:
            gdata = gdal.Open(loadname)
            raster = gdata.ReadAsArray().astype(np.float)
            gdata = None

    return header, raster, raster_proj


def read_header_of_tiff_file(path_tiff_file):
    #author: Leo Bont, WSL
    
    from osgeo import gdal, osr
    import numpy as np

    names = ['ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize', 'NODATA_value']
    formats = ['int', 'int', 'float', 'float', 'float', 'float']
    # Definition des Datentypes:
    header_info = np.zeros(1, dtype={'names': names, 'formats': formats})

    gdata = gdal.Open(path_tiff_file)
    sr_proj = gdata.GetProjection()
    raster_proj = osr.SpatialReference()
    raster_proj.ImportFromWkt(sr_proj)

    if gdata is None:
        print('WARNING: Corrupt File: ', path_tiff_file)
        header_info_vector = np.zeros(6)
    else:

        gt = gdata.GetGeoTransform()
        # data = gdata.ReadAsArray().astype(np.float)

        width = gdata.RasterXSize
        height = gdata.RasterYSize
        gt = gdata.GetGeoTransform()
        minx = gt[0]
        miny = gt[3] + width * gt[4] + height * gt[5]
        maxx = gt[0] + width * gt[1] + height * gt[2]
        maxy = gt[3]

        #        y1 = gt[3] + width*gt[4] + height*gt[5]
        #        maxx = gt[0] + width*gt[1] + height*gt[2]
        #        y2 = gt[3]
        #        miny = min(y1,y2)
        #        maxy = max(y1,y2)

        xllcorner = minx
        yllcorner = miny
        cellsize = gt[1]
        ncols = gdata.RasterXSize
        nrows = gdata.RasterYSize
        band = gdata.GetRasterBand(1)
        NODATA_value = band.GetNoDataValue()

        header_info['ncols'] = int(ncols)
        header_info['nrows'] = int(nrows)
        header_info['xllcorner'] = xllcorner
        header_info['yllcorner'] = yllcorner
        header_info['cellsize'] = cellsize
        header_info['NODATA_value'] = NODATA_value
        header_info_vector = np.array([int(ncols), int(nrows), xllcorner, yllcorner, cellsize, NODATA_value])

    return header_info, header_info_vector, raster_proj


def GivePropertiesOfHeader(header):
    #author: Leo Bont, WSL
    
    cellsize = header['cellsize']
    xmin = header['xllcorner']
    ymin = header['yllcorner']
    xmax = xmin + (header['ncols']) * cellsize
    ymax = ymin + (header['nrows']) * cellsize

    return (cellsize, xmin, xmax, ymin, ymax)


def writeMap(raster, header, path_basic_out, OutputNameOfMap):
    #author: Leo Bont, WSL
    
    import numpy as np

    delta = header['cellsize']
    xmin = header['xllcorner']
    ymin = header['yllcorner']
    xmax = xmin + (header['ncols']) * delta
    ymax = ymin + (header['nrows']) * delta

    anz_x = (xmax - xmin) / delta
    anz_y = (ymax - ymin) / delta

    xres = delta
    # yres = delta
    yres = delta * -1

    # geotransform = (xmin, xres, 0, ymin, 0, yres)
    geotransform = (xmin, xres, 0, ymax, 0, yres)
    print('yes')

    # print('Z 4509',geotransform)

    import gdal
    # create the 3-band raster file
    number_of_bands = 1
    dst_ds = gdal.GetDriverByName('GTiff').Create(path_basic_out + OutputNameOfMap, int(np.round(anz_x)),
                                                  int(np.round(anz_y)), number_of_bands, gdal.GDT_Float32)

    dst_ds.SetGeoTransform(geotransform)  # specify coords
    dst_ds.GetRasterBand(1).WriteArray(raster)  # write r-band to the raster
    # dst_ds.GetRasterBand(1).WriteArray(v_pixels)   # write r-band to the raster
    # dst_ds.GetRasterBand(2).WriteArray(g_pixels)   # write g-band to the raster
    # dst_ds.GetRasterBand(3).WriteArray(b_pixels)   # write b-band to the raster

    dst_ds.FlushCache()  # write to disk
    dst_ds = None


def CreateShapefile(outSHPfn, raster_proj, X, Y, local_maxi_heights1, bhd):
    #author: Leo Bont, WSL
    
    import os
    import ogr

    # Create output shapefile
    shpDriver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outSHPfn):
        shpDriver.DeleteDataSource(outSHPfn)
    outDataSource = shpDriver.CreateDataSource(outSHPfn)
    outLayer = outDataSource.CreateLayer(outSHPfn, geom_type=ogr.wkbPoint, srs=raster_proj)

    # create a field
    outLayer.CreateField(ogr.FieldDefn('X_coord', ogr.OFTReal))
    outLayer.CreateField(ogr.FieldDefn('Y_coord', ogr.OFTReal))
    outLayer.CreateField(ogr.FieldDefn('Baumhoehe', ogr.OFTReal))
    outLayer.CreateField(ogr.FieldDefn('bhd_mm', ogr.OFTReal))

    # Convert array to point coordinates
    count = 0
    for count in range(X.shape[0]-1):
        Xcoord = X[count]
        Ycoord = Y[count]
        x_float = float(Xcoord)
        y_float = float(Ycoord)
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(x_float, y_float)
        featureIndex = count + 1
        outFeature = ogr.Feature(outLayer.GetLayerDefn())
        outFeature.SetGeometry(point)
        outFeature.SetFID(featureIndex)
        outFeature.SetField('X_coord', x_float)
        outFeature.SetField('Y_coord', y_float)
        outFeature.SetField('Baumhoehe', local_maxi_heights1[count])
        outFeature.SetField('bhd_mm', bhd[count])
        outLayer.CreateFeature(outFeature)
        count += 1

    outFeature = None
    print('Shapefile done')


def CreateBuffer(inputfn, outputBufferfn, bufferDist, raster_proj):
    inputds = ogr.Open(inputfn)
    inputlyr = inputds.GetLayer()

    shpdriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outputBufferfn):
        shpdriver.DeleteDataSource(outputBufferfn)
    outputBufferds = shpdriver.CreateDataSource(outputBufferfn)
    bufferlyr = outputBufferds.CreateLayer(outputBufferfn, geom_type=ogr.wkbPolygon, srs=raster_proj)
    featureDefn = bufferlyr.GetLayerDefn()

    for feature in inputlyr:
        ingeom = feature.GetGeometryRef()
        geomBuffer = ingeom.Buffer(bufferDist)

        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(geomBuffer)
        bufferlyr.CreateFeature(outFeature)
        outFeature = None


def CreateCSV(outCSVfn, X, Y, local_maxi_heights1, bhd):
    f = open(outCSVfn, "w")
    f.write("{},{},{},{}\n".format("X_Coord", "Y_Coord", "Baumhoehe", "BHD_mm"))
    for a in range(X.size):
        f.write("{},{},{},{}\n".format(X[a], Y[a], local_maxi_heights1[a], bhd[a]))
    f.close()
    print('CSV done')


def CutToPoly(loadname_Poly, crs, BDet_gdf):
    import geopandas as gpd
    Poly_Data = gpd.read_file(loadname_Poly)
    Poly_Data.crs = crs

    gdf = gpd.sjoin(BDet_gdf, Poly_Data, how="inner", op="within",
                    lsuffix='BDet', rsuffix='Poly')
    gdf['ind_BDet'] = gdf.index
    return gdf

def sub2ind(array_shape, rows, cols):
    return rows*array_shape[1] + cols


def ind2sub(array_shape, ind):
    rows = (ind.astype('int') / array_shape[1])
    cols = (ind.astype('int') % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
    return (rows, cols)


def FilterNone(chm, name, header, path_basic_out):
    image = chm[:]
    image[image <= 2] = 0  # Werte unter 2m Höhe als Boden definieren

    versF = 'F0_'

    ## Gefiltertes Bild als .tif abspeichern
    OutputNameOfMap_Image = versF + name + '.tif'
    from Forest_Inventory_Func import writeMap

    writeMap(image, header, path_basic_out, OutputNameOfMap_Image)
    print('Filter_' + versF + name + '_done')

    return image, versF


def FilterMenk(chm, delta, name, header, path_basic_out):
    image = chm[:]

    ## Gaussfilter
    from scipy.ndimage.filters import gaussian_filter
    if delta >= 1:  # Erkentnisse aus Menk 2017
        sigma = 0
    elif delta >= 0.5:
        sigma = 1
    else:
        sigma = 2
    image = gaussian_filter(image, sigma=sigma)

    versF = 'F1_'

    ## Gefiltertes Bild als .tif abspeichern
    OutputNameOfMap_Image = versF + name + '.tif'
    from Forest_Inventory_Func import writeMap
    writeMap(image, header, path_basic_out, OutputNameOfMap_Image)
    print('Filter_' + versF + name + '_done')

    return image, versF


def FilterJakubowski(chm, delta, name, header, path_basic_out):
    import numpy as np
    image = chm[:]

    from skimage.morphology import square, closing, rectangle
    radius2 = int(np.round(2 / delta))
    selem2 = square(radius2)  # Define structuring element
    chm_smoothed = closing(image, selem2)  # Closing (dilation followed by an erosion)

    ## Erstellen einer Filter-Maske
    mask_diff = chm_smoothed - image
    mask_diff_max = mask_diff.max()
    mask_norm = mask_diff / mask_diff_max  # Maske normalisieren
    #                mask_boolean = mask_norm #Maske kopieren

    a = 0.1  # zwischen 0 und 1
    mask_tf = mask_norm >= a
    mask_boolean = mask_tf.astype(np.int)
    #                mask_boolean[mask_boolean <= a] = 0 #Maske in True/False umwandeln
    #                mask_boolean[mask_boolean > a] = 1 #Maske in True/False umwandeln

    from scipy.ndimage import convolve
    kernel = rectangle(3, 3)
    mask_convolve = convolve(mask_boolean, kernel)  # https://en.wikipedia.org/wiki/Kernel_(image_processing)
    b = 10  # zwischen 0 und 10
    mask_smooth = mask_convolve + mask_norm * b

    mask_smooth_max = mask_smooth.max()
    mask = mask_smooth / mask_smooth_max

    c = 1.5  # zwischen 1 und 2
    chm_corrected = image * (-1 * mask + 1) + chm_smoothed * mask * c

    chm_corrected[chm_corrected <= 2] = 0  # Werte unter 2m Höhe als Boden definieren

    versF = 'F2_'

    ## Gefiltertes Bild als .tif abspeichern
    OutputNameOfMap_Image = versF + name + '.tif'
    from Forest_Inventory_Func import writeMap
    writeMap(chm_corrected, header, path_basic_out, OutputNameOfMap_Image)
    print('Filter_' + versF + name + '_done')

    return image, versF


def FilterEysn(chm, delta, name, header, path_basic_out):
    import numpy as np
    image = chm[:]

    ## Non-linear filtering: closing filter
    from skimage.morphology import disk, closing
    radius3 = int(np.round(2 / delta))
    selem3 = disk(radius3)
    chm_closed = closing(image, selem3)  # Closing (dilation followed by an erosion)

    ## Lowpass filtering: gaussian filter
    from scipy.ndimage import gaussian_filter
    sigma3 = int(np.round(0.3 / delta))
    image = gaussian_filter(chm_closed, sigma=sigma3)

    image[image <= 2] = 0  # Werte unter 2m Höhe als Boden definieren

    versF = 'F3_'

    OutputNameOfMap_Image = versF + name + '.tif'
    from Forest_Inventory_Func import writeMap
    writeMap(image, header, path_basic_out, OutputNameOfMap_Image)
    print('Filter_' + versF + name + '_done')

    return image, versF


def FilterOpening(chm, delta, name, header, path_basic_out):
    import numpy as np
    image = chm[:]

    from skimage.morphology import disk, opening
    radius4 = int(np.round(2 / delta))
    selem4 = disk(radius4)
    image_opened = opening(image, selem4)  # Closing (dilation followed by an erosion)

    image_opened[image_opened <= 2] = 0  # Werte unter 2m Höhe als Boden definieren

    versF = 'F4_'

    OutputNameOfMap_Image = versF + name + '.tif'
    from Forest_Inventory_Func import writeMap
    writeMap(image_opened, header, path_basic_out, OutputNameOfMap_Image)
    print('Filter_' + versF + name + '_done')

    return image, versF


def FilterClosing(chm, delta, name, header, path_basic_out):
    import numpy as np
    image = chm[:]

    from skimage.morphology import disk, closing
    radius5 = int(np.round(2 / delta))
    selem5 = disk(radius5)
    image_closed = closing(image, selem5)  # Closing (dilation followed by an erosion)

    image_closed[image_closed <= 2] = 0  # Werte unter 2m Höhe als Boden definieren

    versF = 'F5_'

    OutputNameOfMap_Image = versF + name + '.tif'
    from Forest_Inventory_Func import writeMap
    writeMap(image_closed, header, path_basic_out, OutputNameOfMap_Image)
    print('Filter_' + versF + name + '_done')

    return image, versF


def FilterKaartinenFGILOCM(chm, name, header, path_basic_out):
    import numpy as np
    image = chm[:]

    ## Low-pass filter
    import cv2
    kernel6 = np.array([[1, 3, 1], [3, 12, 3], [1, 3, 1]]).astype(np.float32) / 28
    image = cv2.filter2D(image, -1, kernel6)

    image[image <= 2] = 0  # Werte unter 2m Höhe als Boden definieren

    versF = 'F6_'

    OutputNameOfMap_Image = versF + name + '.tif'
    from Forest_Inventory_Func import writeMap
    writeMap(image, header, path_basic_out, OutputNameOfMap_Image)
    print('Filter_' + versF + name + '_done')

    return image, versF


def FilterKaartinenMetla(chm, delta, name, header, path_basic_out):
    import numpy as np
    from numpy import matlib
    image = chm[:]

    ## Definition der Zellgrösse und des Suchradius:
    cellsize = float(delta)
    neighbour_radius_cell = 1.5
    #            radius = 3
    #            neighbour_radius_cell = float(radius) / cellsize

    ## Erster Schritt: Definition der Nachbarschaft
    sh = image.shape  # n rows, m cols = shape (n,m)
    x1 = np.arange(sh[0]) * cellsize
    y1 = np.arange(sh[1]) * cellsize

    mapY = np.matlib.repmat(x1, sh[1], 1).T
    mapX = np.matlib.repmat(y1, sh[0], 1)

    from EBD_Func_Laura import sub2ind, ind2sub
    center_index = np.round(np.array(sh) / 2.).astype('int')
    cent_lin_ind = sub2ind(sh, center_index[0], center_index[1])  ## stimmt noch nicht ganz!!!!

    center = center_index * cellsize

    dist_to_center = np.sqrt((mapX - center[1]) ** 2 + (mapY - center[0]) ** 2)

    neigh_within_ind = np.where(dist_to_center <= neighbour_radius_cell)
    neigh_within_lin_ind = sub2ind(sh, neigh_within_ind[0], neigh_within_ind[
        1])  # neigh_within_lin_ind[0] = rows/y, neigh_within_lin_ind[1] = cols/x

    rel_diff_lin_ind = neigh_within_lin_ind - cent_lin_ind
    rel_diff_ind = [neigh_within_ind[0] - center_index[0], neigh_within_ind[1] - center_index[1]]

    ## nur noch Punkte der Nachbarschaft
    to_del = np.where(rel_diff_lin_ind == 0)[0][0]
    rel_diff_lin_ind_neigh = np.delete(rel_diff_lin_ind, to_del)
    rel_diff_ind_neigh = [np.delete(rel_diff_ind[0], to_del), np.delete(rel_diff_ind[1], to_del)]

    ## Zweiter Schritt: Moving window
    prozent_constraint = 6. / 8.

    for i_l in range(sh[0] * sh[1]):

        i_y, i_x = ind2sub(sh, np.array(i_l))  # x,y von allen Matrix-Werten

        ## neighbours: check validity (nur Zellen innerhalb von image.shape)
        i_n_y = i_y + rel_diff_ind_neigh[0]
        i_n_x = i_x + rel_diff_ind_neigh[1]
        n_valid = np.logical_and(np.logical_and(i_n_y >= 0, i_n_y < sh[0]), np.logical_and(i_n_x >= 0, i_n_x < sh[1]))

        pixel_value = image.ravel()[i_l]
        neighbours_value = image.ravel()[i_l + rel_diff_lin_ind_neigh[n_valid]]

        ## Constraint:
        number_of_differing_pixels_required = np.round(prozent_constraint * len(neighbours_value))

        if np.sum(neighbours_value - pixel_value >= 5) >= number_of_differing_pixels_required:
            # print('Constraint active')
            ## median of the more-than-five-meters-larger neighbor pixel values
            image.ravel()[i_l] = np.median(neighbours_value[neighbours_value - pixel_value >= 5])

    image[image <= 2] = 0  # Werte unter 2m Höhe als Boden definieren

    versF = 'F7_'

    OutputNameOfMap_Image = versF + name + '.tif'
    from Forest_Inventory_Func import writeMap
    writeMap(image, header, path_basic_out, OutputNameOfMap_Image)
    print('Filter_' + versF + name + '_done')

    return image, versF


def EBDpeaklocalmax(chm, image, versF, delta, raster_proj, x_values, y_values, minh, name, path_out):

    import numpy as np
    ## Werte unte 2m Höhe als Boden definieren
    image[image <= 2] = 0

    ## Version
    vers = 'LM1_' + versF + name

    ## Lokale Maxima berechnen
    from skimage.feature import peak_local_max
    from skimage.morphology import disk

    radius = int(np.round(3 / delta))
    selem = disk(radius, dtype=bool)
    local_maxi1 = peak_local_max(image,
                                 threshold_abs=minh,
                                 footprint=selem)
    # Output mit Koordinaten der lokalen Maxima (y = Zeile, x = Spalte)
    print('loc_max_done')

    ## Output mit Höhenangaben der lokalen Maxima (Höhe [m])
    local_maxi_heights1 = chm[local_maxi1[:, 0], local_maxi1[:, 1]]

    ## X und Y Koordinaten (LV95) der lokalen Maxima berechnen
    Y = y_values[local_maxi1[:, 0]]  # Achtung: y-Wert = erste Spalte
    X = x_values[local_maxi1[:, 1]]  # Achtung: x-Wert = zweite Spalte

    ## Allometrische Formel (BHD) - für Nadelholz
    # Negative Werte melden Runtime error und werden als nan abgespeichert
    # mse für Nadelholz-Modell: 0.04972586
    bhd = np.exp(0.8796541115 * np.log(local_maxi_heights1) + 0.0002205558 *
                 np.power(local_maxi_heights1, 2) + 2.9810469865 + 0.04960369)  # Resultat in mm

    ## Shapefile und CSV als Output
    if local_maxi_heights1.size != 0:
        from EBD_Func_Laura import CreateShapefile
        from EBD_Func_Laura import CreateCSV

        outSHPfn = path_out + '//' + vers + '.shp'
        outCSVfn = path_out + '//' + vers + '.csv'
        CreateShapefile(outSHPfn, raster_proj, X, Y, local_maxi_heights1, bhd)
        CreateCSV(outCSVfn, X, Y, local_maxi_heights1, bhd)
        print(vers + " done")
    else:
        print('DataFrame = empty')



def EBDlocalmaxima(chm, image, versF, delta, raster_proj, x_values, y_values, name, path_out):

    ## Lokale Maxima berechnen
    import numpy as np
    from skimage.morphology import local_maxima

    ## Version
    vers = 'LM2_' + versF + name

    # In einer alten Version habe ich bei local_maxima noch die connectivity definiert (Kronengrösse).
    # Aber die Resultate waren mit und ohne connectivity Definition die gleichen. Zumindest in einem meiner Tests.
    #radius = int(np.round(3 / delta))
    #connectivity = 2 * radius

    local_maxi2 = local_maxima(image, allow_borders=False)  # True lokale Maxima / False Raster

    print('loc_max_done')
    local_maxi_hmatrix2 = np.multiply(image, local_maxi2)  # Berechnen der Höhen der lokalen Maxima (Raster mit Höhe [m] für lokale Maxima)

    ## X und Y Koordinaten (LV95) der lokalen Maxima berechnen
    hcoord2 = np.asarray(np.where(local_maxi_hmatrix2 != 0))
    local_maxi_hcoord2 = np.transpose(hcoord2)

    Y = y_values[local_maxi_hcoord2[:, 0]]
    X = x_values[local_maxi_hcoord2[:, 1]]

    ## Output mit Höhenangaben der lokalen Maxima (Höhe [m])
    local_maxi_heights2 = chm[local_maxi_hcoord2[:, 0], local_maxi_hcoord2[:, 1]]


    ## Allometrische Formel (BHD) - für Nadelholz
    # Negative Werte melden Runtime error und werden als nan abgespeichert
    # mse für Nadelholz-Modell: 0.04972586
    bhd = np.exp(0.8796541115 * np.log(local_maxi_heights2) + 0.0002205558 *
             np.power(local_maxi_heights2, 2) + 2.9810469865 + 0.04960369)  # Resultat in mm


    ## Shapefile und CSV als Output
    if local_maxi_heights2.size != 0:
        from EBD_Func_Laura import CreateShapefile
        from EBD_Func_Laura import CreateCSV

        outSHPfn = path_out + '//' + vers + '.shp'
        outCSVfn = path_out + '//' + vers + '.csv'
        CreateShapefile(outSHPfn, raster_proj, X, Y, local_maxi_heights2, bhd)
        CreateCSV(outCSVfn, X, Y, local_maxi_heights2, bhd)
        print(vers + " done")
    else:
        print('DataFrame = empty')


def EBDlinearind(chm, image, versF, delta, raster_proj, x_values, y_values, name, path_out):

    import numpy as np
    import numpy.matlib

    ## Version
    vers = 'LM3_' + versF + name

    ## Definition der Zellgrösse und des Suchradius:
    cellsize = float(delta)
    radius = int(np.round(3 / delta))
    neighbour_radius_cell = float(radius)

    ## Erster Schritt: Definition der Nachbarschaft
    sh = image.shape  # n rows, m cols = shape (n,m)
    x1 = np.arange(sh[0]) * cellsize
    y1 = np.arange(sh[1]) * cellsize

    mapY = np.matlib.repmat(x1, sh[1], 1).T
    mapX = np.matlib.repmat(y1, sh[0], 1)

    from EBD_Func_Laura import sub2ind, ind2sub

    center_index = np.round(np.array(sh) / 2.).astype('int')
    cent_lin_ind = sub2ind(sh, center_index[0], center_index[1])  ## stimmt noch nicht ganz!!!!

    center = center_index * cellsize

    dist_to_center = np.sqrt((mapX - center[1]) ** 2 + (mapY - center[0]) ** 2)

    neigh_within_ind = np.where(dist_to_center <= radius)
    neigh_within_lin_ind = sub2ind(sh, neigh_within_ind[0], neigh_within_ind[1])  # neigh_within_lin_ind[0] = rows/y, neigh_within_lin_ind[1] = cols/x

    rel_diff_lin_ind = neigh_within_lin_ind - cent_lin_ind
    rel_diff_ind = [neigh_within_ind[0] - center_index[0], neigh_within_ind[1] - center_index[1]]

    ## nur noch Punkte der Nachbarschaft
    to_del = np.where(rel_diff_lin_ind == 0)[0][0]
    rel_diff_lin_ind_neigh = np.delete(rel_diff_lin_ind, to_del)
    rel_diff_ind_neigh = [np.delete(rel_diff_ind[0], to_del), np.delete(rel_diff_ind[1], to_del)]

    ## Zweiter Schritt: Moving window
    local_max_res = np.zeros(sh)

    for i_l in range(sh[0] * sh[1]):
        i_y, i_x = ind2sub(sh, np.array(i_l))  # x,y von allen Matrix-Werten

        ## neighbours: check validity (nur Zellen innerhalb von image.shape)
        i_n_y = i_y + rel_diff_ind_neigh[0]
        i_n_x = i_x + rel_diff_ind_neigh[1]

        n_valid = np.logical_and(np.logical_and(i_n_y >= 0, i_n_y < sh[0]), np.logical_and(i_n_x >= 0, i_n_x < sh[1]))

        pixel_value = image.ravel()[i_l]
        neighbours_value = image.ravel()[i_l + rel_diff_lin_ind_neigh[n_valid]]

        ## Local maximum bestimmen:
        # evtl. nur > np.max
        if pixel_value >= np.max(neighbours_value):
            local_max_res.ravel()[i_l] = 1

    print('loc_max_done')

    ## Höhe [m] der lokalen Maxima berechnen
    local_maxi_hmatrix3 = np.multiply(chm, local_max_res)  # Raster mit Höhe [m] für lokale Maxima

    ## X und Y Koordinaten (LV95) der lokalen Maxima berechnen
    hcoord3 = np.asarray(np.where(local_maxi_hmatrix3 != 0))
    local_maxi_hcoord3 = np.transpose(hcoord3)

    Y = y_values[local_maxi_hcoord3[:, 0]]
    X = x_values[local_maxi_hcoord3[:, 1]]
    local_maxi_heights3 = chm[local_maxi_hcoord3[:, 0], local_maxi_hcoord3[:, 1]]

    ## Allometrische Formel (BHD) - für Nadelholz
    # Negative Werte melden Runtime error und werden als nan abgespeichert
    # mse für Nadelholz-Modell: 0.04972586
    bhd = np.exp(0.8796541115 * np.log(local_maxi_heights3) + 0.0002205558 *
             np.power(local_maxi_heights3, 2) + 2.9810469865 + 0.04960369)  # Resultat in mm


    ## Shapefile und CSV als Output
    if local_maxi_heights3.size != 0:
        from EBD_Func_Laura import CreateShapefile
        from EBD_Func_Laura import CreateCSV

        outSHPfn = path_out + '//' + vers + '.shp'
        outCSVfn = path_out + '//' + vers + '.csv'
        CreateShapefile(outSHPfn, raster_proj, X, Y, local_maxi_heights3, bhd)
        CreateCSV(outCSVfn, X, Y, local_maxi_heights3, bhd)
        print(vers + " done")
    else:
        print('DataFrame = empty')

