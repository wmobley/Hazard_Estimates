from  Hazard_Estimates import Raster_Sets
import os
import timeit
from multiprocessing import Pool
from multiprocessing import cpu_count
import gdal
import time
import rasterio
rasters =[]
for root, dirs, files in os.walk(r"C:\Users\wmobley\Downloads\huc12010001\huc12010001", topdown=True):
    for filename in files:
        rasters.append(os.path.join(root,filename.split(".")[0]))
    break

#
# def make_VRT(file, directory):
#
#     sub_directories = [os.path.join(directory, name, file) for name in os.listdir(directory) if
#                        os.path.isdir(os.path.join(directory, name))]
#     print(sub_directories)
#     gdal.BuildVRTOptions(VRTNodata="nan")
#     gdal.BuildVRT(os.path.join(directory, f"{file[:-4]}.vrt"), sub_directories)
#
#
#
# dir = r"C:\Users\wmobley\Downloads\huc12010001\huc12010001"
# for year in [
#     2001, 2004, 2006,
#     2008, 2011, 2013,
#     2016]:
#
#     make_VRT(f"roughness{year}.tif", dir)
#     if year in [2001, 2011, 2006, 2016]:
#         make_VRT(f'impervious{year}.tif', dir)
#         make_VRT(f'AverageRoughness{year}.tif', dir)
# for year in [25, 100, 500]:
#     for hour in ["12ha", "60ma"]:
#         make_VRT(f"Precip{year}_{hour}.tif", dir)
# for water in ['Lakes', "Coast", "Streams"]:
#     make_VRT(f"Distance2{water}.tif", dir)
# for file in [
#     f"FlowAccumulation.tif",
#     f"slope.tif",
#     f"dem.tif",
#     'AverageKSAT.tif',
#     'hand.tif']:
#     make_VRT(file, dir)



def make_rasters(r, p):
    return Raster_Sets.raster_sets(files=r,extension='.vrt', pool=p)





#
# def handle_tiff(some_file):
#     with rasterio.open(f"{some_file}.tif") as src:
#         data_array = src.read(1)
#     return data_array

# def parallel_rasters(r, p):
#
#     all_data = p.map(handle_tiff, r)
#
if __name__=="__main__":
    pool = Pool(cpu_count()- 1)
    print(rasters)
    start = time.time()
    (make_rasters(rasters, pool))
    print(time.time()-start)
    start = time.time()
    print(make_rasters(rasters, None).__dict__)
    print(time.time() - start)
    # print(timeit.timeit('make_rasters(rasters, None)', 'from __main__ import make_rasters, rasters, pool', number=5))
    # print(timeit.timeit('parallel_rasters(rasters,pool)', 'from __main__ import parallel_rasters, rasters,pool', number=5))