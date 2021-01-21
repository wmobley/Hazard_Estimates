import os
import pandas as pd
import geopandas as gpd
import gc
import numpy as np
import rasterio
import rasterio.mask
import shutil
import os
from Hazard_Estimates.Model import *

from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy import ndimage
import sys
import gdal
import richdem as rd
huc12 = sys.argv[1]
shapefile = gpd.read_file(r"./Shapefiles/huc12.shp")

shapefile= shapefile.loc[(shapefile.HUC12.str.startswith(huc12))] #running one of these currently
shapefile.reset_index(inplace=True)



roughness={21: 0.0404,

                22: 0.0678,
                23: 0.0678,
                24: 0.0404,
                31: 0.0113,
                41: 0.36,
                42: 0.32,
                43: 0.40,
                52: 0.40,
                71: 0.368,
                81: 0.325,
                90: 0.086,
                95: 0.1825}


def rescale(root, name):
    #This Function rescales the given rasters based on the highest resolution image used. Currently that is the hand.tif at 10m. 
    with rasterio.open(os.path.join(root, 'hand.tif')) as mask:
        # with rasterio.open(os.path.join(root, 'demfill.tif')) as mask2:
        shape = [{'type': 'Polygon', 'coordinates': [[(mask.bounds.left, mask.bounds.top),
                                                      (mask.bounds.left, mask.bounds.bottom),
                                                      (mask.bounds.right, mask.bounds.bottom),
                                                      (mask.bounds.right, mask.bounds.top)]]}]

        with rasterio.open(os.path.join(root, "temp.tif")) as src: #loads a tempory tif previously clipped. 
            transform, width, height = calculate_default_transform( #note temp.tif is developed in the clip to watershed function below. 
                src.crs, mask.crs, mask.width, mask.height, *mask.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': mask.crs,
                'transform': transform,
                'width': width,
                'height': height,

            })
            with rasterio.open(os.path.join(root, name), 'w', **kwargs) as dst: #save the temporary file now rescaled to 10-meters
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=mask.crs,
                        resampling=Resampling.bilinear)
    os.remove(os.path.join(root, "temp.tif"))


def safely_reduce_dtype(ser):  # pandas.Series or numpy.array
    #reduces the datatype to the lowest possible to reduce storage. 
    orig_dtype = "".join([x for x in ser.dtype.name if x.isalpha()])  # float/int
    mx = 1

    new_itemsize = np.min_scalar_type(ser).itemsize
    if mx < new_itemsize:
        mx = new_itemsize
    new_dtype = orig_dtype + str(mx * 8)
    return new_dtype


def clip_to_boundary(in_directory, out_directory, boundary_geom,
                     in_raster, out_raster):
    #Clips the in raster based on the boundary geometry. Usually a Hydrologic Unit.
    if os.path.exists(os.path.join(out_directory, out_raster)):return None
    with rasterio.open(os.path.join(in_directory, in_raster)) as src:

        out_image, out_transform = rasterio.mask.mask(src, boundary_geom, crop=True)
        out_meta = src.meta
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform,
                         "dtype":out_image.dtype}
                        )
        if out_raster == "hand.tif": #if the raster is a hand raster then it is the highest resolution so write the image otherwise rescale the image. 
            with rasterio.open(os.path.join(out_directory, out_raster), 'w', **out_meta) as dest:
                dest.write(out_image)
        else:
            with rasterio.open(os.path.join(out_directory, "temp.tif"), 'w', **out_meta) as dest:
                dest.write(out_image)
            rescale(out_directory, out_raster)


def clip_roughness(directory, boundary, year):
    #calcuate roughness coefficient based on landuse
    clip_to_boundary("RawFiles/Landcover", directory, boundary, #clip the landcover to the boundary
                     f"NLCD_{year}_Land_Cover_L48_20190424.img",
                     f"Landcover{year}.tif")
    with rasterio.open(os.path.join(directory, f"Landcover{year}.tif")) as src: #load landcover and use a lookup table to estimate roughness. 
        image = src.read(1)
        u, inv = np.unique(image, return_inverse=True)
        img = np.array([roughness.get(x, 0) for x in u])[inv].reshape(image.shape)

        out_meta = src.meta

        out_meta.update({"driver": "GTiff",
                         "height": img.shape[0],
                         "width": img.shape[1],
                         "dtype": img.dtype,
                         })
        with rasterio.open(os.path.join(directory, f"roughness{year}.tif"), 'w', **out_meta) as dst:
            dst.write(img, 1)
    os.remove(os.path.join(directory, f"Landcover{year}.tif"))

def clip_twi(directory):
    #Calculate TWI based on slope and flow accumulation. 
    TWI_Save = "{}/TWI.tif".format(directory)

    with rasterio.open(os.path.join(directory, f"FlowAccumulation.tif")) as src_acc:
        flow_accumulation = src_acc.read(1)
        with rasterio.open(os.path.join(directory, f"slope.tif")) as src_slope:
            slope = src_slope.read(1)
            img = np.log(((flow_accumulation * 900) + 1) / (np.tan((slope + 0.000001) / (180 / np.pi)))) 
            out_meta = src_acc.meta

            out_meta.update({"driver": "GTiff",
                             "height": img.shape[0],
                             "width": img.shape[1],
                             "dtype": img.dtype,
                             })
            with rasterio.open(TWI_Save, 'w', **out_meta) as dst:
                dst.write(img, 1)


def weighted_accum( in_dir, weight, out_directory, out_raster, boundary_geom):
    #uses the pysheds library to calculate flow accumulation, however its weighted with by a given array. 
    with rasterio.open(os.path.join(out_directory, weight))as src:
        weights = src.read(1)
        weights = np.where(weights == -9999, 0, 1)
        norm = np.linalg.norm(weights)
        weights = weights / norm
        #
        # g.accumulation(data='dir', weights=weights, out_name='weights_accum')
        # grid.to_raster('weights_accum', os.path.join(out_directory, out_raster),dtype=np.int32)
    
        accum = rd.FlowAccumulation(dem, method='D8', weights=weights.astype('float64') )

        rd.SaveGDAL(os.path.join(out_directory, "temp.tif"), accum)
        clip_to_boundary(out_directory, out_dir, boundary_geom, f"temp.tif",
                         out_raster)
        # os.remove(os.path.join(out_dir, "temp.tif"))
#Loop through each polygon in shapefile.
for index in shapefile.index:

    try:

        geom = [shapefile.iloc[index].geometry]
        huc12 = shapefile.iloc[index].HUC12
    except:
        continue
    print(huc12)

    dst_crs = 'EPSG:5070'

    out_dir = f"Spatial_Index/huc{huc12[:-4]}/huc{huc12}"

    if os.path.exists(os.path.join(out_dir, f'Distance2Streams.tif')): continue #This checks to see if I've iterated over the data. If I have don't do it again.
    if os.path.exists(f"Spatial_Index/huc{huc12[:-4]}") == False: os.makedirs(f"Spatial_Index/huc{huc12[:-4]}")
    if os.path.exists(out_dir) == False: os.makedirs(out_dir)                                                   # check to see if this directory exists if not make it.
    if os.path.exists(os.path.join(f"RawFiles/Hand/{huc12[:-6]}", f"hand_proj.tif")) == False:
        print(os.path.join(f"RawFiles/Hand/{huc12[:-6]}", f"{str(huc12)[:-6]}hand_proj.tif"))
        in_hand = os.path.join(f"RawFiles/Hand/{huc12[:-6]}",    f"hand.tif")
        out_hand = os.path.join(f"RawFiles/Hand/{huc12[:-6]}",    f"hand_proj.tif")
        with rasterio.open(in_hand) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            with rasterio.open(out_hand, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear)


    clip_to_boundary(
        f"RawFiles/Hand/{huc12[:-6]}",
        out_dir, geom, f"hand_proj.tif", "hand.tif") #Clip Hand

    clip_to_boundary("RawFiles/Topography", out_dir, geom, f"elevation.tif",
                     f"dem.tif")

    # clip_to_boundary("RawFiles/Topography", out_dir, geom, f"texas_slope.tif",
    #                  f"slope.tif")
    gc.collect() #clean up ram
    in_elevation = os.path.join(out_dir,f"dem.tif")
    dem = rd.LoadGDAL(in_elevation)
    rd.FillDepressions(dem, epsilon=True, in_place=True)
    slope = rd.TerrainAttribute(dem, attrib='slope_riserun')
    rd.SaveGDAL(os.path.join(out_dir, 'slope.tif'), slope)
    accum_d8 = rd.FlowAccumulation(dem, method='D8')
    rd.SaveGDAL(os.path.join(out_dir, 'FlowAccumulation.tif'),accum_d8)

    #Once slope and flow acculation are clipped then TWI can be calculated.
    clip_twi(out_dir)
    #This clips rainfall intensities for specific storms. Not necessary for the first analysis but needed later down the line.
    # for hr in [1, 2, 3, 4, 8, 12, 24, ]:
    #     for storm in [
    #         'taxday',
    #         'harvey']:
    #         clip_to_boundary(r"F:\test\{}\intensity\projected".format(storm), out_dir, geom,
    #                          f"{storm}{hr}hr.tif",
    #                          f"{storm}{hr}hr.tif")

    #Clip KSAT and then generate the accumulated KSAT.
    clip_to_boundary(r"RawFiles", out_dir, geom, "Ksat.tif",
                     f"ksat.tif")
    weighted_accum( out_dir,
                   "ksat.tif",
                   out_dir, 'AverageKSAT.tif', geom)

    #Thesea are the dynamic rasters. Using Imperviousness and Landcover.
    #It iterates over each year, and then clips a given raster. Impervious 2016 was named different so it stands along
    for i in [
        2001, 2004, 2006,
        2008, 2011, 2013,
        2016]:
        if i == 2016:
            clip_to_boundary("RawFiles/Impervious", out_dir, geom, f"impervious2016.tif",
                             f"impervious{i}.tif")
        elif i in [2001, 2006, 2011]:
            clip_to_boundary(r"RawFiles/Impervious/nlcd_{}_impervious_2011_edition_2014_10_10".format(i),
                             out_dir, geom,
                             f"nlcd_{i}_impervious_2011_edition_2014_10_10.img",
                             f"impervious{i}.tif")
        clip_roughness(out_dir, geom, i)

        weighted_accum( out_dir,
                       f"roughness{i}.tif",
                       out_dir, f'AverageRoughness{i}.tif', geom)

    #These probabilistic precipitations. 12hr and 60 minutes. We use three probabilities 25, 100, and 500 year.
    for year in [25, 100, 500]:
        for hour in ["12ha", "60ma"]:
            clip_to_boundary("RawFiles/Precip", out_dir, geom, f"Precip{year}yr_{hour}.tif",
                                         f"Precip{year}_{hour}.tif")
    #This for loop calculates euclidean distance.
    for water in ['Lakes', "Coast", "Streams"]:
        clip_to_boundary("RawFiles", out_dir, geom, f"{water}.tif",
                         f"{water}.tif")

        with rasterio.open(os.path.join(out_dir, f"{water}.tif"))as src:
            img = src.read(1)
        meta = src.meta.copy()
        #Inverse the image for the euclidean analysis.
        img = np.where(img == 1, 0, 1)

        #This is a check to ensure that coastlines exist in image and then calculates distance.
        if 0 in np.unique(img):
            img = ndimage.morphology.distance_transform_edt(img)
        meta.update({"dtype": "int16"})
        out_img = img.astype("int16")

        with rasterio.open(os.path.join(out_dir, f"D{water}.tif"), 'w+', **meta) as out:
            out.write_band(1, out_img)
        clip_to_boundary(out_dir, out_dir, geom, f"D{water}.tif",
                         f"Distance2{water}.tif")
        os.remove(os.path.join(out_dir, f"D{water}.tif"))
        os.remove(os.path.join(out_dir,  f"{water}.tif"))

def make_VRT(file, directory):
    sub_directories = [os.path.join(directory, name, file) for name in os.listdir(directory) if
                       os.path.isdir(os.path.join(directory, name))]
    gdal.BuildVRTOptions(VRTNodata="nan")
    gdal.BuildVRT(os.path.join(directory, f"{file[:-4]}.vrt"), sub_directories)


for huc in shapefile.HUC12.apply(lambda row: row[:8]):
    dir = f"Spatial_Index/huc{huc}"
    for year in [
        2001, 2004, 2006,
        2008, 2011, 2013,
        2016]:
        if year in [2001, 2006, 2011, 2016]:
            make_VRT(f'AverageRoughness{year}.tif', dir)
        make_VRT(f"roughness{year}.tif", dir)
    for year in [25, 100, 500]:
        for hour in ["12ha", "60ma"]:
              make_VRT(f"Precip{year}_{hour}.tif", dir)
    for water in ['Lakes', "Coast", "Streams"]:
         make_VRT(f"Distance2{water}.tif", dir)
    for file in [
        f"FlowAccumulation.tif",
        f"slope.tif",
        f"dem.tif",
        'AverageKSAT.tif',
        'hand.tif']:
        make_VRT(file, dir)

