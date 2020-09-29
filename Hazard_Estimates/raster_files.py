import numpy as np
import rasterio
import os

from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask

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

    ''' Raster Datastructure Built on Raster IO'''
    def _init_(self, year=-9999):

        self.year = year


    def load(self, dataAddress):
        '''
        Loads raster from the dataAdress. Specifically added files are
        :param dataAddress: Location of the raster
        :return:
        '''
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
        '''
        convert from 2-d numpy array to 1-d.
        :return:
        '''
        self.asciiFile = self.asciiFile.flatten('C')
        return self

    def return_dataset_2d(self, nrows):
        '''
        convert from 1-d numpy array to 2-d.
        :param nrows: Rows of initial dataset.
        :return:
        '''

        self.asciiFile = np.reshape(self.asciiFile, (nrows, -1))
        return self

    def get_raster_value(self, df_row):
        ''' use Raster IO index to look up location. '''
        row, col = self.src.index(df_row.X, df_row.Y)
        try:
            return self.asciiFile[row, col]
        except:
            return -9999

    def save_image(self, asciiFile,location, file):
        '''
        Save file image
        :param asciiFile: 2d array
        :param location: Location to save the file
        :param file: TFW file name to convert.
        :return:
        '''

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

        # with open(f"{location}.tfw", "w") as tfw:
        #
        #     for i, line in enumerate(file):
        #         if i == 0:
        #             tfw.write("10.0\n")
        #         elif i == 3:
        #             tfw.write("-10.0\n")
        #         else:
        #             tfw.write(line)

        if os.path.exists(f"{location}.tif.ovr"):
            os.remove(f"{location}.tif.ovr")



def rescale(root, name, template="hand.tif"):
    '''
    Rescale a raster based on the
    :param root: root directory
    :param name: name of raster to be rescaled
    :param template: Name of file used for rescale template. Defaults to Hand in the output folder.
    :return: none
    '''
    with rasterio.open(os.path.join(root, template)) as mask_file:
        # with rasterio.open(os.path.join(root, 'demfill.tif')) as mask2:
        shape = [{'type': 'Polygon', 'coordinates': [[(mask_file.bounds.left, mask_file.bounds.top),
                                                      (mask_file.bounds.left, mask_file.bounds.bottom),
                                                      (mask_file.bounds.right, mask_file.bounds.bottom),
                                                      (mask_file.bounds.right, mask_file.bounds.top)]]}]

        with rasterio.open(os.path.join(root, "temp.tif")) as src:
            # resample_raster(src, upscale_factor, huc, name)
            transform, width, height = calculate_default_transform(
                src.crs, mask_file.crs, mask_file.width, mask_file.height, *mask_file.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': mask_file.crs,
                'transform': transform,
                'width': width,
                'height': height,

            })
            with rasterio.open(os.path.join(root, name), 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=mask_file.crs,
                        resampling=Resampling.bilinear)
    return None

def safely_reduce_dtype(ser):  # pandas.Series or numpy.array
    """
    reduces dtype to lowest memory size
    :param ser: Pandas series or array
    :return: dtype
    """
    orig_dtype = "".join([x for x in ser.dtype.name if x.isalpha()])  # float/int
    mx = 1

    new_itemsize = np.min_scalar_type(ser).itemsize
    if mx < new_itemsize:
        mx = new_itemsize
    new_dtype = orig_dtype + str(mx * 8)
    return new_dtype


def clip_to_boundary(in_directory, out_directory, boundary_geom,
                     in_raster, out_raster, template = "hand.tif"):
    '''

    :param in_directory: root directory for original raster
    :param out_directory: directory for new raster
    :param boundary_geom: geomerty to clip
    :param in_raster: initial raster
    :param out_raster: File name to save new raster
    :param template: Template file name loctated in the output directory.
    :return:
    '''
    with rasterio.open(os.path.join(in_directory, in_raster)) as src:

        out_image, out_transform = mask(src, boundary_geom, crop=True)
        out_meta = src.meta
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform,
                         "dtype": safely_reduce_dtype(out_image)}
                        )
        if out_raster == template:
            with rasterio.open(os.path.join(out_directory, out_raster), 'w', **out_meta) as dest:
                dest.write(out_image)
        elif template is not None:
            with rasterio.open(os.path.join(out_directory, "temp.tif"), 'w', **out_meta) as dest:
                dest.write(out_image)
            rescale(out_directory, out_raster)