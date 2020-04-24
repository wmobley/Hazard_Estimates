# from __future__ import print_function
#
# import csv
# import os
#
# import imageio as im
# import numpy as np
# import rasterio
#
# import raster_files as rf
#
# datasets = ["demFill", "Slope", "Soils", "impervious", "Distance2Coast", "distance2Stream", "CRS",
#             "Landcover"]
#
#
# def load(args):
#     dataAddress = args
#     """
#     Load Rasters
#     """
#
#     # if dataAddress.endswith("huc"): return
#     try:
#         raster = rf.ascii_raster()
#         raster.make_dataset( dataAddress)  # Load Raster make it 1-D
#     except Exception as e:
#         # print (e)
#         return False
#     return raster
#
#
# def get_raster_value(df_row, raster):
#     row, col = raster.src.index(df_row.X, df_row.Y)
#     try:
#         return raster.asciiFile[row, col]
#     except:
#         return -9999
#
#
#
#
# def make_dataset_1D(d, dataAddress="DataSets"):
#     rasters = []
#
#     ascii_raster_array = rf.ascii_raster()
#     ascii_raster_array.fileName = (d.split(".")[0])
#     ascii_raster_array.fileName = ascii_raster_array.fileName.split("/")[
#         len(ascii_raster_array.fileName.split("/")) - 1]
#     # print(ascii_raster_array.fileName)
#     ascii_file = os.path.join(dataAddress, "{}.tfw".format(d))
#     src = rasterio.open(os.path.join(dataAddress, f"{d}.tif"))
#     with open(ascii_file, 'r') as csvfile:
#         reader = csv.reader(csvfile, delimiter=",")
#         ascii_raster_array.cellsize = (float(next(reader)[0].split(" ")[-1]))
#         next(reader)
#         next(reader)
#         next(reader)
#         ascii_raster_array.xllcorner = (float(next(reader)[0].split(" ")[-1]))
#         ascii_raster_array.yllcorner = (float(next(reader)[0].split(" ")[-1]))
#
#     ascii_raster_array.src = src
#     ascii_raster_array.asciiFile = src.read(1, out_shape=(1, int(src.height), int(src.width)))
#
#     ascii_raster_array.nrows = ascii_raster_array.asciiFile.shape[0]
#     ascii_raster_array.ncols = ascii_raster_array.asciiFile.shape[1]
#     ascii_raster_array.asciiFile = ascii_raster_array.asciiFile.flatten('C')
#     ascii_raster_array.year = -9999
#     for y in [2001, 2004, 2006, 2008, 2011, 2013,
#               2016]:
#         if ascii_raster_array.fileName.endswith(str(y)):
#             ascii_raster_array.fileName = ascii_raster_array.fileName.split(str(y))[0]
#             ascii_raster_array.year = y
#
#     return ascii_raster_array
#
#
# def rescale(d, dataAddress="DataSets", scalar=1):
#     ascii_raster_array = d
#     ascii_raster_array.fileName = (d.fileName.split(".")[0])
#     ascii_raster_array.fileName = ascii_raster_array.fileName.split("/")[
#         len(ascii_raster_array.fileName.split("/")) - 1]
#     # print(ascii_raster_array.fileName)
#     ascii_file = os.path.join(dataAddress, "{}.tfw".format(d))
#     # with open(ascii_file, 'r') as csvfile:
#     #     reader = csv.reader(csvfile, delimiter=",")
#     #     ascii_raster_array.cellsize = (float(next(reader)[0].split(" ")[-1]))
#     #     next(reader)
#     #     next(reader)
#     #     next(reader)
#     #     ascii_raster_array.xllcorner = (float(next(reader)[0].split(" ")[-1]))
#     #     ascii_raster_array.yllcorner = (float(next(reader)[0].split(" ")[-1]))
#
#     # ascii_raster_array.asciiFile = im.imread(os.path.join(dataAddress, "{}.tif".format(d)))
#     file = np.reshape(ascii_raster_array.asciiFile, (ascii_raster_array.nrows, -1))
#     ascii_raster_array.asciiFile = np.asarray(
#         [[c for c in r for i in range(0, int(scalar))] for r in file for ii in
#          range(0, int(scalar))])
#     # [[value for value in enumerate(row) for scalar_y in range(0, int(scalar))] for row in
#     #  ascii_raster_array.asciiFile for scalar_x in range(0, int(scalar))])
#
#     ascii_raster_array.nrows = ascii_raster_array.asciiFile.shape[0]
#     ascii_raster_array.ncols = ascii_raster_array.asciiFile.shape[1]
#
#     ascii_raster_array.asciiFile = ascii_raster_array.asciiFile.flatten('C')
#     ascii_raster_array.year = -9999
#     for y in [2001, 2006, 2011]:
#         if ascii_raster_array.fileName.endswith(str(y)):
#             ascii_raster_array.fileName = ascii_raster_array.fileName.split(str(y))[0]
#             ascii_raster_array.year = y
#     ascii_raster_array.cellsize = ascii_raster_array.cellsize / scalar
#     return ascii_raster_array
#
#
# def make_dataset( dataAddress):
#     rasters = []
#
#
#     ascii_raster_array.fileName = (dataAddress.split("/")[-1])
#
#
#     # ascii_raster_array.fileName = ascii_raster_array.fileName.split("/")[
#     #     len(ascii_raster_array.fileName.split("/")) - 1]
#     # print(ascii_raster_array.fileName)
#     # ascii_file = os.path.join(dataAddress, "{}.tfw".format(dataAddress))
#     src = rasterio.open( f"{dataAddress}.tif")
#     # with open(ascii_file, 'r') as csvfile:
#     #     reader = csv.reader(csvfile, delimiter=",")
#     #     ascii_raster_array.cellsize = (float(next(reader)[0].split(" ")[-1]))
#     #     next(reader)
#     #     next(reader)
#     #     next(reader)
#     #     ascii_raster_array.xllcorner = (float(next(reader)[0].split(" ")[-1]))
#     #     ascii_raster_array.yllcorner = (float(next(reader)[0].split(" ")[-1]))
#
#     ascii_raster_array.src = src
#
#     ascii_raster_array.asciiFile = src.read(1, out_shape=(1, int(src.height), int(src.width)))
#     ascii_raster_array.nrows = ascii_raster_array.asciiFile.shape[0]
#     ascii_raster_array.ncols = ascii_raster_array.asciiFile.shape[1]
#     ascii_raster_array.year = -9999
#     for y in [2001, 2006, 2011, 2016]:
#         if ascii_raster_array.fileName.endswith(str(y)):
#             ascii_raster_array.fileName = ascii_raster_array.fileName.split(str(y))[0]
#             ascii_raster_array.year = y
#     return ascii_raster_array
#
#
# def add_Pseudo_Random_Points(locations, rasters, n, random_seed):
#     np.random.seed(random_seed)
#     for i in range(n):
#         point = ["Dry",
#                  str(np.random.randint(low=rasters[0].xllcorner,
#                                        high=rasters[0].xllcorner + (rasters[0].cellsize * rasters[0].ncols), size=1)[
#                          0]),
#                  str(np.random.randint(high=rasters[0].yllcorner,
#                                        low=rasters[0].yllcorner - (rasters[0].cellsize * rasters[0].nrows), size=1)[
#                          0]), np.random.randint(1998, 2014)]
#         locations.append(point)
#     return locations
#
#
# def get_values_1D(location, rasters):
#     data = []
#     test = []
#
#     for i, x, y, year in location:
#         ind2 = []
#         try:
#             if (x.upper() == 'X'):
#                 continue
#         except AttributeError:
#             pass
#         independents = {}
#         for r in (rasters):
#             if r.fileName == "landcover" and r.year == min([2001, 2004, 2006, 2008, 2011, 2013, 2016],
#                                                            key=lambda sub_year: abs(sub_year - year)):
#                 try:
#                     temp = r.fileName.split("\\")
#                     r.fileName = temp[-1]
#                     # print(r.fileName)
#                 except:
#
#                     r = r
#
#             if r.year == -9999 or r.year == min([2001, 2006, 2011, 2016], key=lambda sub_year: abs(sub_year - year)):
#                 try:
#                     temp = r.fileName.split("\\")
#                     r.fileName = temp[-1]
#                     r.fileName
#                 except:
#
#                     r = r
#                 top = (r.yllcorner + (r.cellsize * r.nrows))
#
#                 i_x = int((float(x) - r.xllcorner) / r.cellsize)
#                 i_y = int((r.yllcorner - float(y)) / r.cellsize)
#                 # print(y, top)
#                 if (i_x > 0 and i_y > 0 and i_x < r.ncols and i_y < r.nrows):
#                     if (r.asciiFile[((i_y) * r.ncols) + i_x] == np.nan):
#                         independents[r.fileName] = None
#                         continue
#                     independents[r.fileName] = r.asciiFile[((i_y) * r.ncols) + i_x]
#                     independents['Inundated'] = i
#                     independents['year'] = year
#
#             else:
#                 continue
#
#         try:
#
#             data.append(independents)
#         except NameError:
#             continue
#     return data
#
#
# def get_values(location, rasters):
#     data = []
#     test = []
#
#     for i, x, y, year in location:
#         ind2 = []
#         if (x.upper() == 'X'):
#             continue
#         independents = {}
#         for r in (rasters):
#
#             if r.year == -9999 or r.year == min([2001, 2006, 2011], key=lambda sub_year: abs(sub_year - year)):
#                 top = (r.yllcorner + (r.cellsize * r.nrows))
#
#                 i_x = int((float(x) - r.xllcorner) / r.cellsize)
#                 i_y = int((r.yllcorner - float(y)) / r.cellsize)
#                 # print(y, top)
#                 if (i_x > 0 and i_y > 0 and i_x < r.ncols and i_y < r.nrows):
#                     if (r.asciiFile[i_x - 1].iloc[i_y - 1] == np.nan):
#                         print("Nan")
#                         continue
#                     independents[r.fileName] = r.asciiFile[i_x - 1].iloc[i_y - 1]
#                     test_dict = (independents, i)
#             else:
#                 continue
#         try:
#             test.append(test_dict)
#             data.append(independents)
#         except NameError:
#             continue
#     return test, data
#
#
# def trim_raster(raster, raster_standard):
#     One_D = False
#     if len(raster.asciiFile.shape) == 1:
#         One_D = True
#         raster.asciiFile = np.reshape(raster.asciiFile, (raster.nrows, -1))
#
#     yll = raster.yllcorner - raster_standard.yllcorner
#     xll = raster.xllcorner - raster_standard.xllcorner
#     x_change = raster.ncols - raster_standard.ncols
#     y_change = raster.nrows - raster_standard.nrows
#
#     if xll < 0:
#         raster.asciiFile = raster.asciiFile[0:raster.nrows, x_change: raster.ncols]
#     else:
#         raster.asciiFile = raster.asciiFile[0:raster.nrows, 0: raster_standard.ncols]
#     if yll > 0:
#         raster.asciiFile = raster.asciiFile[y_change: raster.nrows]
#     else:
#         raster.asciiFile = raster.asciiFile[0: raster_standard.nrows]
#     if One_D:
#         raster.asciiFile = raster.asciiFile.flatten('C')
#     print(len(raster.asciiFile), len(raster_standard.asciiFile), len(raster.asciiFile) - len(raster_standard.asciiFile))
#     raster.ncols = raster_standard.ncols
#     raster.nrows = raster_standard.nrows
#     raster.xllcorner = raster_standard.xllcorner
#     raster.yllcorner = raster_standard.yllcorner
#     return raster
#     # check if its 1D or 2D
#     # convert to 2d
#     # find which side to trim
#     # trim
#
#
# def trim_rows(raster, raster_standard):
#     One_D = False
#     if len(raster.asciiFile.shape) == 1:
#         One_D = True
#         raster.asciiFile = np.reshape(raster.asciiFile, (raster.nrows, -1))
#
#     yll = raster.yllcorner - raster_standard.yllcorner
#
#     y_change = raster.nrows - raster_standard.nrows
#     # print(y_change)
#     if yll > 0:
#         raster.asciiFile = raster.asciiFile[y_change: raster.nrows]
#     else:
#         raster.asciiFile = raster.asciiFile[0: raster.nrows - y_change]
#
#     if One_D:
#         raster.asciiFile = raster.asciiFile.flatten('C')
#     # print(len(raster.asciiFile), len(raster_standard.asciiFile), len(raster.asciiFile) - len(raster_standard.asciiFile))
#
#     raster.nrows = raster_standard.nrows
#
#     raster.yllcorner = raster_standard.yllcorner
#     return raster
#
#
# def trim_cols(raster, raster_standard):
#     One_D = False
#     if len(raster.asciiFile.shape) == 1:
#         One_D = True
#         raster.asciiFile = np.reshape(raster.asciiFile, (raster.nrows, -1))
#
#     xll = raster.xllcorner - raster_standard.xllcorner
#     x_change = raster.ncols - raster_standard.ncols
#
#     # print(x_change)
#     if xll < 0:
#         raster.asciiFile = raster.asciiFile[0:raster.nrows, x_change: raster.ncols]
#     else:
#         raster.asciiFile = raster.asciiFile[0:raster.nrows, 0: raster.ncols - x_change]
#
#     if One_D:
#         raster.asciiFile = raster.asciiFile.flatten('C')
#     # print(len(raster.asciiFile), len(raster_standard.asciiFile), len(raster.asciiFile) - len(raster_standard.asciiFile))
#     raster.ncols = raster_standard.ncols
#
#     raster.xllcorner = raster_standard.xllcorner
#
#     return raster
#
#
# if __name__ == '__main__':
#     elevation = make_dataset_1D(datasets[0])
#     dist2coast = make_dataset_1D(datasets[5])
#     print(elevation.asciiFile.shape)
#     elevation = trim_raster(elevation, dist2coast)
#     print(elevation.asciiFile.shape)
#     print(dist2coast.asciiFile.shape)
#
