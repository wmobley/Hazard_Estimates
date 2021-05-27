from Hazard_Estimates import raster_files as rf
import pandas as pd
import numpy as np
import gc
from multiprocessing import Pool
from multiprocessing import cpu_count
import psutil
class raster_sets:
    def __init__(self, files, storm="", year_range=[], extension=".tif", pool=None):
        '''
        Initializes a dataset of rasters, loads them and prunes extraneous data.
        :param files:List of files to load
        :param storm: Storm event of interest.
        '''

        self.storm = storm
        self.year_range = year_range
        self.extension = extension

        self.rasters = []
        if pool==None:
            self.rasters = [self.load_rasters(file) for file in files]
        else:

            self.rasters.append(pool.map(self.load_rasters, files))
            self.rasters=self.rasters[0]
        self.prune_falses()
    def annualization(self, probability, last_year, first_year):
       '''
       Annualizes the probability given a probaility and the first and last year.
       :param probability: Float
       :param last_year: int latest year within the sample
       :param first_year: int first year within the sample
       '''
       return  (1 - np.power((1 - probability), (1 / (last_year - first_year + 1))))

    def prune_falses(self):
        '''
        removes any files that could not be loaded in the list.
        :return:
        '''
        self.rasters = [self.prefix(r) for r in self.rasters if r != False]
        return self.rasters

    def load_rasters(self, file):
        '''
        generates a raster data structure, return a loaded raster.
        :param file: Location of the file
        :return:
        '''
        return rf.ascii_raster(extension = self.extension, dataAddress=file, years = self.year_range)
     


    def prefix(self, x):
        '''
        Removes prefix from columns based on rain event from rasters.
        :param x: Row and column location
        :return:
        '''
        if x.fileName.startswith(self.storm):
            x.fileName = x.fileName[len(self.storm):]

        return x
    def Convert_Dimensions(self):
        '''
        Convert from 1d to 2d or back
        :return:
        '''
        for raster in self.rasters:
            if len(raster.asciiFile.shape)== 2:
                raster.make_dataset_1D()
            else: raster.return_dataset_2d(raster.nrows)



    def make_dataframe(self):
        ''' loads rasters as Dataframe'''
        dataset = pd.DataFrame()
        for r in self.rasters:
            if r.fileName not in [0, '0']:
                gc.collect()
                dataset[r.fileName] = r.asciiFile
        return dataset

    def generate_probability_raster(self, model_structure, location, ignore_column,
                                    nodata, file,  annualize = False):
        '''

        :param model: SKlearn Model
        :param location: Save Location
        :param ignore_column: Column to base ignore values on.
        :param nodata: Ignore values
        :param file: TFW file name.
        :return:
        '''
        df = self.make_dataframe()
        df = df[model_structure.XColumns]
        predictions =  rf.ascii_raster()
        gc.collect()
        values = psutil.virtual_memory()
        chunks = (int(df.memory_usage(deep=True).sum()/values.available)+2)
        print(chunks)

        split_data = np.array_split(df, chunks)


        predictions.asciiFile = np.array([])
        for data in split_data:

            predictions.asciiFile = np.concatenate( (predictions.asciiFile,
                                                     np.where(data[ignore_column] != nodata,
                                                         model_structure.model.predict_proba(data.values)[:, 1], nodata))
                                                    , axis=None)
            gc.collect()
        if annualize:
            print(annualize)
            predictions.asciiFile = np.where(predictions.asciiFile != nodata, annualize(predictions.asciiFile, max=model_structure.max,min= model_structure.min) ), nodata)
        else:

            predictions.asciiFile = np.where(predictions.asciiFile != nodata,  predictions.asciiFile, nodata)

        #
        gc.collect()


        predictions.return_dataset_2d(self.rasters[0].nrows)
        gc.collect()
        self.rasters[0].save_image(predictions.asciiFile.astype("float32"),location, file, nodata)
        return predictions
