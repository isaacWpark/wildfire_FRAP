# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 20:34:08 2018

@author: MMann
"""

import os 
import pandas as pd
from tsraster.calculate import checkRelevance
from tsraster.calculate import checkRelevance2

from tsraster.prep import sRead as tr
from tsraster.model import model as md


#%% append all features to one dataframe

path = r'F:/5year/'

all_files = [os.path.join(root, name)
             for root, dirs, files in os.walk(path)
             for name in files
             if name.endswith(( "features.csv"))]

df_from_each_file = (pd.read_csv(f) for f in all_files)
concatenated_df   = pd.concat(df_from_each_file,axis=1, ignore_index=False)
concatenated_df.columns

del df_from_each_file


#%% read target data

target_variable = "F:/5year/Fires/"

target_data = tr.targetData(target_variable)
 

#%% join and test train split yX data

obj = [target_data,concatenated_df]

#from sklearn.preprocessing import StandardScaler as scaler
X_train, X_test, y_train, y_test = md.get_data(obj, scale=False)

# allow for garbage collection of large objects
del concatenated_df

#%% Find relevant variables and combine Y and X data

relevant_vars, X_train_relevant = checkRelevance2(X_train,y_train,fdr_level=0.01) #
#print(relevant_vars)
#X_train_relevant.head()

#%% 

RF, MSE, R_Squared =  md.RandomForestReg(X_test, y_test)



#%%

y_pred=RF1.predict(X_test)

from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))