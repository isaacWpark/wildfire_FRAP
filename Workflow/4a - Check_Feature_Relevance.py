# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 20:34:08 2018

@author: MMann
"""

import pandas as pd
from tsraster.prep import combine_extracted_features, combine_target_rasters
from tsraster.calculate import checkRelevance2

import tsraster.prep  as tr
import tsraster.model  as md


#%% append all features to one dataframe

path = r'G:\Climate_feature_subset_train'

concatenated_attribute_df = combine_extracted_features(path,write_out=False)


#%%  collect multitple years of Y (target) data

path = r"G:\Fire_target_train"
target_file_prefix = 'fire_'

concatenated_target_df = combine_target_rasters(path,target_file_prefix,write_out=False)



#%% mask both the attribute data and targets 

raster_mask =u"F:/Boundary/StatePoly_buf.tif"
original_df = [concatenated_attribute_df, concatenated_target_df]

mask_attributes_df, mask_target_df = tr.mask_df(raster_mask, original_df)



#%% switch panel data from wide to long format
import re 
target = mask_target_df
features = mask_attributes_df
sep='-'

target_ln, features_ln = wide_to_long_target_features(target,features,sep='-')



#%% add lagged variables 




#%% join and test train split yX data

obj = [target_ln,features_ln]

#from sklearn.preprocessing import StandardScaler as scaler
X_train, X_test, y_train, y_test = md.get_data(obj, 
                                               stratify=True,
                                               test_size=0.9,
                                               scale=False)

# allow for garbage collection of large objects
#del concatenated_df, concatenated_df_mask, target_data_mask

#%% Find relevant variables and combine Y and X data

relevant_vars, X_train_relevant = checkRelevance2(X_train,y_train,fdr_level=0.01) #
print(relevant_vars)

X_test_relevant = X_test[X_train_relevant.columns]
 

#%% Run random forest on relevant features and get out of sample accuracy measures

RF, predict_test, MSE, R_Squared =  md.RandomForestReg(X_train_relevant, y_train, X_test_relevant, y_test)
print(RF)
print(MSE)
print(R_Squared)


#%%
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix, cohen_kappa_score
from sklearn.ensemble import GradientBoostingClassifier

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
    print("Test Confusion matrix ", confusion_matrix(y_test, predict_test))
    
    return RF  

RF =  RandomForestClass(X_train_relevant, y_train, X_test_relevant, y_test)


#%%

clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1,
                                 max_depth=3, random_state=0).fit(X_train[X_train_relevant.columns], y_train)

#%%
predict_test = clf.predict(X=X_test[X_train_relevant.columns])

test_acc = accuracy_score(y_test, predict_test)
kappa = cohen_kappa_score(y_test, predict_test)
confusion = confusion_matrix(y_test, predict_test)

print(kappa)
print(confusion)


#%% Look at feature importance

feature_importances = pd.DataFrame(RF.feature_importances_,
                                   index = X_test_relevant.columns,
                                    columns=['importance']).sort_values('importance',                                                                 ascending=False)
 
feature_importances

#%% unmask, predict

concatenated_df_relevant = concatenated_df[X_train_relevant.columns]
concatenated_df_predict = RF.predict(X=concatenated_df_relevant)


#%% rasterize and plot prediction
import rasterio
import matplotlib.pyplot as plt

raster_ex = "F:/5year/aet/aet-201201.tif"
ex_row, ex_cols =  rasterio.open("F:/5year/aet/aet-201201.tif").shape

f2Array = concatenated_df_predict.reshape(ex_row, ex_cols)
print(f2Array.shape)
# Plot the grid

plt.imshow(f2Array)
plt.gray()
plt.show()

#%%
path = "F:/5year/Fires/"
image_name = tr.image_names(path)
rasters = tr.image_to_array(path)[:,:,0]

plt.imshow(rasters)
plt.gray()
plt.show()


#%%
# first, get the original dimension/shape of image 
og_rasters = tr.image2array(path)
rows, cols, nums = og_rasters.shape


# convert df to matrix array
matrix_features = ts_features.values
num_of_layers = matrix_features.shape[1]


f2Array = matrix_features.reshape(rows, cols, num_of_layers)
print(f2Array.shape)



