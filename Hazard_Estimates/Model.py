import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from Hazard_Estimates.metrics import *
from Hazard_Estimates.XY_Dataset import *
from joblib import dump, load
import psutil
p = psutil.Process()
pd.set_option('display.float_format', lambda x: '%.3f' % x)

from Hazard_Estimates import Raster_Sets as sets


reg_model = RandomForestRegressor(n_estimators=500, criterion='mse', max_depth=50,
                                                  min_samples_split=3, min_samples_leaf=2, random_state=42,
                                                  n_jobs=-1)
cat_model = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=90,
                                            min_samples_split=3, min_samples_leaf=2, n_jobs=-1)

def default_rescale_y(y, y_=""):
    '''place holder funciton. THis should be set up within the exposure model notebook.'''
    return y


class model_framework:
    def __init__(self, model_name, spatialIndex, ycolumn, XColumns, file_location, storm="", split_model=None,
                 create_Y=None, rescale_y=default_rescale_y):

        """ Data structure for housing sklearn model. designed for predicitons on a Raster Dataset.
        The Structure uses a huc spatial index
        :param model_name:   String variable to help organize models during tests.
        :param spatialIndex: Variable in dataframe that is used for spatial index.
                            This variable needs to be spelled the same way as in the folder system.
        :param ycolumn:     Pandas column used for either regression or category dependent variable.
        :param XColumns:    Columns that will be in the final model.
        :param file_location: string to locate raster locations. This is combined with the spatial index.
        :param storm:       Optional The storm name to use when predicting an event.
        :param split_model: Optional Dataframe column used for spliting models by a category.
        :param create_Y:    Optional Function for regressions. Allows for normalization of the y variable.
        :param rescale_y:   Optional Function for regressions. inverse of create_y

        train: Set up data structure for Train independent  and dependent variables data
        test:  Set up data structure Test independent dependent Varaibles must be the same as the train dataset.
        """
        self.FileLocation = file_location
        self.ModelName = model_name
        self.Spatial_Index = spatialIndex
        self.YColumn = ycolumn
        self.XColumns = XColumns
        self.Dynamic_rasters = None
        self.metrics = None
        self.storm = storm
        self.test = X_Y()
        self.train = X_Y()
        self.dummies = ""
        self.split_model = split_model
        self.create_Y = create_Y
        self.rescale_y = rescale_y
        self.min = 1976
        self.max = 2017
    def min_max_years(self, year_column):
        self.min = self.train.X_[year_column].min()
        self.max = self.train.X_[year_column].max()
    def save_model(self, location):
        dump(self.model, location)

    def load_model(self, location):
        self.model = load(location)

    def Add_Model(self, model):
        '''

        :param model: adds the model to  data structure

        '''
        self.model = model

    def get_model(self, model_name, regression_model = reg_model):
        '''

        :param model_name: used for dictionary key
        :return: returns dictionary entry with key and random forest model.
        '''
        return {model_name: regression_model}

    def add_metrics(self, classification):
        '''

        :param classification: Boolean that directs metrics

        '''
        if self.metrics ==None:
            if classification:
                self.metrics = classification_metrics()
            else:
                self.metrics = regression_metrics()

    def set_up_categorical(self, categorical_model = cat_model):
        '''
        Function used to add categorical model to the datastructure. Currently hard coded as a Random forest Datastructure.
        Add's metrics, and accuracies.
        :return:
        '''
        self.equalize_train_test_columns()
        self.model = categorical_model
        self.train.Y_['inundated'] = self.train.Y_.apply(lambda row: self.label_y(row['inundated']), axis=1)

        self.model.fit(self.train.X_[self.XColumns], self.train.Y_['inundated'] )

        self.add_metrics(classification=True)
        self.metrics.add_accuracies(self)

    def label_y(self, y):
        return {1:"Flooded", 0:"Dry", 'Flooded':"Flooded", 'Dry':"Dry"}[y]


    def fit_model(self, subset, key):
        '''

         :param subset: Boolean or an array used fot reduce DataFrame.
        :param key:     Dictionary Key to reduce dataset and fit the model.
        :return:
        '''
        self.equalize_train_test_columns()
        if  subset.all()==None:
            X_ = self.train.X_
        else:
            X_ = subset
        X_ = X_.loc[X_[self.split_model] == key]

        #
        X_["y_col"] = self.create_Y(X_, self.YColumn, self.split_model)

        # Y_ = pd.Series(Y_)
        # Y_ = Y_.replace([np.inf, -np.inf, np.nan], 0)

        try:

            self.model[key].fit(X_[self.XColumns], X_["y_col"])
        except:
            print(X_["y_col"])
            print(X_["y_col"].max())
            print(np.where(np.isnan(X_["y_col"]))
                  )

    def set_up_continious(self, subset=None):
        '''
        Generates a dictionary of models.
        If Split the dictionary is based on the split_model variable,
        else: Dictionary key is "ALL"
        run fit_model funciton
        :param subset: Boolean or an array used fot reduce DataFrame.

        :return:
        '''

        # self.train.Y_ =  self.create_Y(self.train.X_, self.YColumn,self.split_model)
        if self.split_model != None:
            self.Add_Model({})
            for category, subset in self.train.X_.groupby(self.split_model):
                self.model.update(self.get_model(category))
                self.fit_model(subset, category)

        else:
            self.Add_Model(self.get_model("All"))
            self.fit_model(subset, 'All')
        self.add_metrics(False)


    def predict(self, data):
        Y_ = pd.DataFrame( index=data.X_.index.copy())

        Y_[ 'actual'] =  self.create_Y(data.X_, self.YColumn, self.split_model)
        Y_['actual'] =   self.rescale_y(  Y_['actual'], data.X_, self.split_model)
        Y_['predict'] = 0

        for category in data.X_[self.split_model].unique():

            X_loc_ = data.X_[self.split_model] == category
            if len(data.X_.loc[X_loc_]) > 0:
                try:
                    Y_.loc[X_loc_, 'predicted'] = self.model[category].predict(data.X_[self.XColumns].loc[X_loc_])
                except:
                    print("broken")
                    Y_.loc[X_loc_, 'predicted'] = -9999
        Y_['predict']= self.rescale_y(  Y_['predicted'], data.X_, self.split_model)
        return Y_

    def locate_and_load(self, spatial_index, dimension="2D"):
        ''' Locate and load rasters
        Create list of drivers and add precipitation variable if they exist.
        Load rasters then remove

        :param spatial_index: single index location.
        :return: list of loaded rasters. Numpy array's and locations.
        '''

        fileLocation = r"{}/{}".format(self.FileLocation, spatial_index)

        files = [f'{fileLocation}/{column}'

                 for column in self.XColumns]

        if self.Dynamic_rasters != None:
            files.extend([f"{fileLocation}/{key['filename']}{key['time']}"
                          for key in self.Dynamic_rasters])

        raster_sets = sets.raster_sets( files, self.storm)

        return raster_sets



    def iterate_rasters(self, dataset, hucNumber, ):
        '''
        Dataframe function to iterate through rasters list and add to Dataframe.
        :param dataset:  Subset of dataframe X_
        :param hucNumber: Spatial index
        :return: Subset of dataframe with all raster variables added.
        '''

        raster_sets = self.locate_and_load(hucNumber)

        for r in raster_sets.rasters:
            dataset[r.fileName] = dataset.apply(lambda row: r.get_raster_value(row), axis=1)

        if self.storm != "":
            ### makes sure that all precipitation variables have the same columns. THis will be important for future modelling.
            for hr in [1, 2, 3, 4, 8, 12, 24, 48, 96, 168]:
                column = f"{self.storm}{hr}hr"
                if column in dataset.columns:
                    dataset[f"{hr}hr"] = dataset[column]
                    dataset.drop(column, axis=1, inplace=True)

        return dataset


    def equalize_train_test_columns(self):
        '''
        Makes sure that all test and training columns are the same if not add 0's. This will keep the model from breaking.
        :return:
        '''
        for column in self.train.X_.columns:
            if column not in self.test.X_.columns:
                self.test.X_[column] = 0

            if column not in self.train.X_.columns:
                self.train.X_[column] = 0
        for column in self.test.X_.columns:
            if column not in self.test.X_.columns:
                self.test.X_[column] = 0

            if column not in self.train.X_.columns:
                self.train.X_[column] = 0

        self.train.X_ = self.train.X_.loc[:, ~self.train.X_.columns.duplicated()]

        self.test.X_=  self.test.X_.loc[:, ~self.test.X_.columns.duplicated()]
