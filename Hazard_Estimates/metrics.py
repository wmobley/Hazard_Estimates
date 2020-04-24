from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
from copy import *
import pandas as pd
import numpy as np

class regression_metrics:
    def __init__(self):

        self.r2 = []
        self.median_percent = []
        self.mean_percent = []
        self.mae = []
        self.mse = []

        self.mbe = []
        self.oob = []

    def variable_removed(self, variable_removed):
        self.variable_removed = variable_removed

    def add_accuracies(self, data_structure, subset=False, loc=[], test=False):
        if data_structure.split_model == None:
            ''''''
            if test == True:
                data = data_structure.test
            else:
                data = data_structure.train
            if not subset:
                X_ = data.X_

                Y_ = data_structure.create_Y(X_, data_structure.YColumn, "All")
            else:

                X_ = data.X_.loc[loc]
                Y_ = data_structure.create_Y(X_, data_structure.YColumn, 'All')
            y_predict = (data_structure.model['All'].predict(X_[data_structure.XColumns]))
            self.oob.append(data_structure.model['All'].feature_importances_)
        else:
            print("multi- model")
            if test == True:
                data = deepcopy(data_structure.test)
            else:
                data = deepcopy(data_structure.train)
            Y_ = pd.DataFrame(index = data.X_.index)
            Y_['actual'] = 0
            Y_['predict'] = 0
            for category in data.X_[data_structure.split_model].unique():
                X_loc_ = data.X_[data_structure.split_model] == category
                Y_.loc[X_loc_, 'actual'] = data_structure.rescale_y(
                    data_structure.create_Y(data.X_, data_structure.YColumn, category),
                    category)

                if len(data.X_.loc[X_loc_]) > 0:
                    Y_.loc[X_loc_,'predict'] =  data_structure.rescale_y( data_structure.model[category].predict(data.X_[data_structure.XColumns].loc[X_loc_]),
                                               category)





        Y_.replace(np.nan, 0)


        Y_['diff'] =   Y_['predict'] - Y_['actual']
        Y_['percent_error'] = ((Y_['diff'] / Y_['actual']) * 100)
        Y_['percent_error'] = Y_['percent_error'].fillna(0)

        Y_ = Y_.replace([np.inf, -np.inf, np.nan], 0)

        self.r2.append(r2_score(Y_['actual'], Y_['predict'] ))
        self.median_percent.append(np.median(np.abs(Y_['percent_error'])))
        self.mean_percent.append(np.sum(Y_['percent_error'] / len(Y_)))
        self.mae.append(median_absolute_error(Y_['actual'], Y_['predict'] ))
        self.mse.append(mean_squared_error(Y_['actual'], Y_['predict'] ))
        self.mbe.append(np.sum(Y_['predict']  - Y_['actual']) / len(Y_))


class classification_metrics:
    def __init__(self):
        self.variable_removed = []
        self.auc = []
        self.f1 = []
        self.recall = []
        self.precision = []
        self.accuracy = []

    def variable_removed(self, variable_removed):
        self.variable_removed.append(variable_removed)


    def add_accuracies(self, data_structure):
        data_structure.test.Y_['pred'] = data_structure.model.predict(data_structure.test.X_[data_structure.XColumns])
        prob = data_structure.model.predict_proba(data_structure.test.X_[data_structure.XColumns])
        data_structure.test.Y_['prob'] = split_probabilities(prob)
        data_structure.test.Y_['inundated']= data_structure.test.Y_.apply(lambda row: data_structure.label_y(row['inundated']), axis=1)
        self.accuracy.append(
            round(accuracy_score(data_structure.test.Y_['inundated'], data_structure.test.Y_['pred']), 3))
        self.auc.append(roc_auc_score(data_structure.test.Y_['inundated'], data_structure.test.Y_["prob"]))
        self.f1.append(f1_score(data_structure.test.Y_['inundated'], data_structure.test.Y_['pred'], pos_label='Flooded'))
        self.precision.append(precision_score(data_structure.test.Y_['inundated'], data_structure.test.Y_['pred'],pos_label='Flooded'))
        self.recall.append(recall_score(data_structure.test.Y_['inundated'], data_structure.test.Y_['pred'], pos_label='Flooded'))
        '''Report Accuracy, Will not work for Continuous Variable'''
        print(f"Random Forest tests for \n{data_structure.ModelName} Confusion Matrix\n ")
        print(pd.crosstab(data_structure.test.Y_['inundated'], data_structure.test.Y_['pred'], rownames=["Actual"],
                          colnames=["Predicted"]))

        print(f"{data_structure.ModelName} accuracy: {self.accuracy[-1]}")

        print("AUC= {}".format(self.auc[-1]))

def split_probabilities( prob):
    '''Probabilities are provided as a tuple array. We only need the second probability.
    This function returns that secion probabiilty as a list'''
    flood_score = []

    for s, s_2 in prob:
        flood_score.append(s_2)
    return flood_score
