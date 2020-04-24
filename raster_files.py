import numpy as np
import rasterio
import numba
import os

# Raster Ascii Datastructure
# Provides a structure to load an ESRI ascii file.
# variables
# Ncols = number of cols in the file
# Nrows = number of rows in the file
# xllcorner = x value of lower left corner location this and yllcorner is (0,0) in the array
# yllcorner = y value of lower left corner location this and yllcorner is (0,0) in the array
# cellsize = spatial size of each cell in the array
# nodata = value for nodata in raster
# asciiFile = 2d array for raster data
# fileName = Name of File

class ascii_raster:
    def _init_(self, year=-9999):

        self.year = year
    
    # def ncols(self, ncols=()):
    #     self.ncols = ncols
    #
    #
    # def nrows(self, nrows=()):
    #     self.nrows = nrows
    #
    #
    # def xllcorner(self, xllcorner):
    #     self.xllcorner = xllcorner
    #
    #
    # def yllcorner(self, yllcorner):
    #     self.yllcorner = yllcorner
    #
    #
    # def cellszie(self, cellsize):
    #     self.cellsize = cellsize
    #
    #
    # def nodata(self, nodata):
    #     self.nodata = nodata
    #
    #
    # def asciiFile(self, asciiFile):
    #     self.asciiFile = asciiFile
    #
    #
    # def year(self, year):
    #     self.year = year
    #
    # def src(self, src):
    #     self.src = src
    # @classmethod
    # def load(self, args):
    #     '''
    #
    #     :param args:
    #     :return:
    #     '''
    #
    #     dataAddress = args
    #     """
    #     Load Rasters
    #     """
    #
    #     # if dataAddress.endswith("huc"): return
    #     try:
    #         raster = self.make_dataset(dataAddress)  # Load Raster make it 1-D
    #     except Exception as e:
    #         print (e)
    #         return False
    #     return raster

    def load(self, dataAddress):
        try:
            self.fileName = (dataAddress.split("/")[-1])

            self.src = rasterio.open(f"{dataAddress}.tif")
            self.asciiFile = self.src.read(1, out_shape=(1, int(self.src.height), int(self.src.width)))
            self.nrows = self.asciiFile.shape[0]
            self.ncols = self.asciiFile.shape[1]
            self.year = -9999
            for y in [2001, 2006, 2011, 2016]:
                if self.fileName.endswith(str(y)):
                    self.fileName = self.fileName.split(str(y))[0]
                    self.year = y
            return self

        except Exception as e:
            print (e)
            return False


    def make_dataset_1D(self):

        self.asciiFile = self.asciiFile.flatten('C')
        return self

    def return_dataset_2d(self, nrows):
        self.asciiFile = np.reshape(self.asciiFile, (nrows, -1))
        return self
    def get_raster_value(self, df_row):
        row, col = self.src.index(df_row.X, df_row.Y)
        try:
            return self.asciiFile[row, col]
        except:
            return -9999

    def save_image(self, asciiFile,location, file):


        profile = {
            "driver": 'GTiff',
            "count": 1,
            "height": asciiFile.shape[0],
            "width": asciiFile.shape[1],
            'dtype': asciiFile.dtype,
            'transform': self.src.transform,
            'crs': self.src.crs,
            'nodata': -9999
        }
        with rasterio.open(
               f"{location}.tif",
                'w',
                **profile

        ) as dst:
            dst.write(asciiFile, 1)

        with open(f"{location}.tfw", "w") as tfw:

            for i, line in enumerate(file):
                if i == 0:
                    tfw.write("10.0\n")
                elif i == 3:
                    tfw.write("-10.0\n")
                else:
                    tfw.write(line)

        if os.path.exists(f"{location}.tif.ovr"):
            os.remove(f"{location}.tif.ovr")