# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 21:26:50 2018

@author: mmann
"""

import os 
import pandas as pd
from tsraster.calculate import calculateFeatures2
from tsraster.calculate import checkRelevance2
 
import tsraster.prep  as tr
import tsraster.model  as md


#%% Extract features 

path = r"F://5year_Copy/"

parameters = {
    "mean": None,
    "maximum": None,
    "median":None,
    "minimum":None,
    "sum_values":None,
    "agg_linear_trend": [{"attr": 'slope', "chunk_len": 6, "f_agg": "min"},
                          {"attr": 'slope', "chunk_len": 6, "f_agg": "max"}],
    "last_location_of_maximum":None,
    "last_location_of_minimum":None,
    "longest_strike_above_mean":None,
    "longest_strike_below_mean":None,
    "count_above_mean":None,
    "count_below_mean":None,
    #"mean_abs_change":None,
    "mean_change":None,
    "number_cwt_peaks":[{"n": 6},{"n": 12}],
    "quantile":[{"q": 0.15},{"q": 0.05},{"q": 0.85},{"q": 0.95}],
    "ratio_beyond_r_sigma":[{"r": 2},{"r": 3}], #Ratio of values that are more than r*std(x) (so r sigma) away from the mean of x.
    "skewness":None }


#%%
extracted_features = calculateFeatures2(path, parameters, reset_df=False, tiff_output=True)

extracted_features.head()



#%% # Mask all values that are outside of the buffered state boundary

raster_mask = u"F:/Boundary/StatePoly_buf.tif"
extracted_features_mask  = tr.mask_df(raster_mask,
                                   original_df=extracted_features)


#%% read target data & mask out non-CA values

target_data = tr.targetData("F:/5year/Fires/")

target_data_mask  = tr.mask_df(raster_mask, 
                               original_df = target_data)


#%% join and test train split yX data

obj = [target_data_mask, extracted_features_mask]

#from sklearn.preprocessing import StandardScaler as scaler
X_train, X_test, y_train, y_test = md.get_data(obj, 
                                               stratify=True,
                                               test_size=0.9,
                                               scale=False)


#%%
 

#%% Find relevant variables and combine Y and X data

relevance_test, X_train_relevant = \
                        checkRelevance2(X_train,y_train,fdr_level=0.01) 
                        
print(relevance_test)


#%%

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix, cohen_kappa_score

def RandomForestClass(X_train, y_train, X_test, y_test):
    RF = RandomForestClassifier(n_estimators=100,
                               max_depth=10,
                               min_samples_leaf=5,
                               min_samples_split=5,
                               random_state=42,
                               oob_score = True)

    model = RF.fit(X_train, y_train)
    predict_test = model.predict(X=X_test)
    
    print("Train Accuracy :: ", accuracy_score(y_train, model.predict(X_train)))
    print("Test Accuracy  :: ", accuracy_score(y_test, predict_test))
    print("Kappa  :: ", cohen_kappa_score(y_test, predict_test))
    
    print("Test Confusion matrix ")
    print(confusion_matrix(y_test, predict_test))
    
    return RF  

RF =  RandomForestClass(X_train[X_train_relevant.columns], y_train, 
                        X_test[X_train_relevant.columns], 
                        y_test)

