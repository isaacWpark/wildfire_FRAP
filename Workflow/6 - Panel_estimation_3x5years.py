# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 20:34:08 2018

@author: MMann
"""

from tsraster.prep import combine_extracted_features, combine_target_rasters, \
    wide_to_long_target_features,unmask_df,panel_lag_1, image_to_series_simple
from tsraster.calculate import checkRelevance2
from numpy import NaN

import tsraster.prep  as tr
import tsraster.model  as md
import rasterio as rio
import matplotlib.pyplot as plt
import pandas as pd
import os
from tsraster.prep import *

#%% append all features to one dataframe

path = r'C:\Users\mmann\Documents\wildfire_FRAP\Data\Examples\Panel_Example\Climate'

concatenated_attribute_df = combine_extracted_features(path,write_out=False)
 


#%% append other shapefile data 

# rasterize the state polygon to match inputs 

#jepson = tr.poly_to_series(poly = 'F:/Boundary/Jepson.shp',
#               raster_ex = r'F:/5year/aet/aet-201201.tif', 
#               field_name = 'JEPSON_ID', 
#               nodata=-9999, 
#               plot_output=True)
#
#concatenated_attribute_df = pd.concat([concatenated_attribute_df,jepson],
#                                axis=1, 
#                                ignore_index=False)


#%%  collect multitple years of Y (target) data

path = r"C:\Users\mmann\Documents\wildfire_FRAP\Data\Examples\Panel_Example\Fire"
target_file_prefix = 'fire_'

concatenated_target_df = combine_target_rasters(path,
                                                target_file_prefix,
                                                write_out=False)
 

#%% mask both the attribute data and targets 

raster_mask =r"C:\Users\mmann\Documents\wildfire_FRAP\Data/Examples/buffer/StatePoly_buf.tif"
original_df = [concatenated_attribute_df, concatenated_target_df]

mask_attributes_df, mask_target_df = tr.mask_df(raster_mask, 
                                                original_df,  
                                                missing_value=-9999,
                                                reset_index = False)

#%% switch panel data from wide to long format

# stub name for jepson issue !!!!!!  

target_ln, features_ln = wide_to_long_target_features(target = mask_target_df,
                                                      features = mask_attributes_df,
                                                      sep='-')

#%%
import io
data = io.StringIO('''Fruit,Color,Count,Price
Apple,Red,3,$1.29
Apple,Green,9,$0.99
Pear,Red,25,$2.59
Pear,Green,26,$2.79
Lime,Green,99,$0.39
''')
df_unindexed = pd.read_csv(data)
df = df_unindexed.set_index(['Fruit', 'Color'])
df

join2 = io.StringIO('''Fruit,Count2
Apple,3
Pear,25
Lime,99
''')
join2 = pd.read_csv(join2)
join2 = join2.set_index(['Fruit'])
join2
#df.join(join2, on='Fruit')
df['count2'] = df.index.get_level_values('Fruit').map(join2['Count2'].get)


#%% append other shapefile data 

# rasterize the state polygon to match inputs 

jepson =  tr.poly_to_series(poly = 'F:/Boundary/Jepson.shp',
               raster_ex = r'F:/5year/aet/aet-201201.tif', 
               field_name = 'JEPSON_ID', 
               nodata=-9999, 
               plot_output=True)
jepson.head()

features_ln  = features_ln.join(jepson, on=['index'])


#%% add lagged variables 
 
lag_vars = ['ppt__median',
 'cwd__mean',
 'cwd__mean']    


features_ln = panel_lag_1(features_ln, 
                          col_names=lag_vars, 
                          group_by_index ='pixel_id')

#%%  add other polygon data 

df.join(join2,on = 'Fruit')


#%% join and test train split yX data with pixels as indep groupings 


obj = [target_ln,features_ln]

X_train, X_test, y_train, y_test = md.get_data(obj,
                                            stratify=True,
                                            test_size=0.9,
                                            scale=False,
                                            groups =  features_ln.index.get_level_values('pixel_id') )


#%% Find relevant variables and combine Y and X data

relevant_vars, X_train_relevant = checkRelevance2(x=X_train,
                                                  y=y_train,
                                                  ml_task='auto',
                                                  fdr_level=0.001) #
print(relevant_vars)

X_test_relevant = X_test[X_train_relevant.columns]

#%%

#%%
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix, cohen_kappa_score
from sklearn.ensemble import GradientBoostingClassifier

#%%

clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1,
                                 max_depth=3, random_state=0).fit(X_train[X_train_relevant.columns], y_train)

pred_prob = clf.predict_proba(X_train[X_train_relevant.columns])



#%%
predict_test = clf.predict(X=X_test[X_train_relevant.columns])

test_acc = accuracy_score(y_test, predict_test)
kappa = cohen_kappa_score(y_test, predict_test)
confusion = confusion_matrix(y_test, predict_test)

print('Testing accuracy:',test_acc)
print('Testing Kappa: ',kappa)
print('Testing Conf: 'confusion)


#%% Look at feature importance

feature_importances = pd.DataFrame(clf.feature_importances_,
                                   index = X_test_relevant.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)
 
feature_importances

#%% unmask, predict

concatenated_df_predict = md.model_predict(model = clf,
                                        new_X = features_ln[X_train_relevant.columns])

concatenated_df_prob =  md.model_predict_prob(model = clf,
                                        new_X = features_ln[X_train_relevant.columns])


 
#%% isolate one 5 year period for mapping 


concatenated_df_prob_1996 = concatenated_df_prob.query('time == "19962000" ')
concatenated_df_prob_1996.index = concatenated_df_prob_1996.index.get_level_values(0) 

concatenated_df_prob_1996.describe()


#%% unmask values 

unmask_concatenated_df_prob_1996 = tr.unmask_df(concatenated_attribute_df, 
                                                concatenated_df_prob_1996)

 
unmask_concatenated_df_prob_1996 = unmask_df(original_df = concatenated_attribute_df,
                                             mask_df_output = concatenated_df_prob_1996) 

# keep only class prob of fire event 
unmask_concatenated_df_prob_1996 = unmask_concatenated_df_prob_1996.iloc[:,1]


unmask_concatenated_df_prob_1996[unmask_concatenated_df_prob_1996< 0]= NaN 

unmask_concatenated_df_prob_1996.describe()


#%% rasterize and plot prediction

raster_ex = "F:/5year/aet/aet-201201.tif"
ex_row, ex_cols =  rasterio.open(raster_ex).shape

f2Array = unmask_concatenated_df_prob_1996.reshape(ex_row, ex_cols)
 # Plot the grid

plt.imshow(f2Array)
plt.set_cmap("Reds")
plt.colorbar( )
plt.show()

#%%

path = r"G:\Fire_target_train\fire_1996_2000.tif"
image_name = tr.image_names(path)
rasters = tr.image_to_array(path)[:,:,0]

plt.imshow(rasters)
plt.set_cmap("Reds")
plt.show()



 # fit cv will groups 
rs=sklearn.model_selection.RandomizedSearchCV(forest,parameters,scoring='roc_auc',cv=gkf,n_iter=10)

rs.fit(X,y,groups=groups)

 
