from Hazard_Estimates import raster_files as rf
import pandas as pd
import numpy as np
import gc
import psutil
p = psutil.Process()

class raster_sets:
    def __init__(self, files, storm=""):
        '''
        Initializes a dataset of rasters, loads them and prunes extraneous data.
        :param files:List of files to load
        :param storm: Storm event of interest.
        '''

        self.storm = storm
        self.rasters = [self.load_rasters(file) for file in files]
        self.prune_falses()

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
        raster = rf.ascii_raster()

        return raster.load(file)

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

    def generate_probability_raster(self, model_structure, location, ignore_column, nodata, file):
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
                                                         model_structure.model.predict_proba(data.values)[:, 1], -9999))
                                                    , axis=None)


        #
        gc.collect()


        predictions.return_dataset_2d(self.rasters[0].nrows)
        gc.collect()
        self.rasters[0].save_image(predictions.asciiFile,location, file)
        return predictions
