# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 21:26:50 2018

@author: mmann
"""

from tsraster.calculate import extract_features,calculateFeatures, MultiprocessingDistributor,checkRelevance
from tsraster.prep import mask_df, image_to_series, read_my_df, if_series_to_df, path_to_var, \
set_df_mindex,image_to_series_simple, unmask_from_mask, set_df_index, reset_df_index
import tsraster.prep  as tr
import tsraster.calculate  as ca

import tsraster.model  as md
import pandas as pd
import os

#%% Extract features 



parameters = {
    "mean": None,
    "maximum": None
    #"median":None,
    #"minimum":None,
    #"sum_values":None,
    #"agg_linear_trend": [{"attr": 'slope', "chunk_len": 6, "f_agg": "min"},
    #                      {"attr": 'slope', "chunk_len": 6, "f_agg": "max"}],
    #"last_location_of_maximum":None,
    #"last_location_of_minimum":None,
    #"longest_strike_above_mean":None,
    #"longest_strike_below_mean":None,
    #"count_above_mean":None,
    #"count_below_mean":None,
    #"mean_change":None,
    #"number_cwt_peaks":[{"n": 6},{"n": 12}],
    #"quantile":[{"q": 0.15},{"q": 0.05},{"q": 0.85},{"q": 0.95}],
    #"ratio_beyond_r_sigma":[{"r": 2},{"r": 3}], #Ratio of values that are more than r*std(x) (so r sigma) away from the mean of x.
    #"skewness":None 
    }
#%%


#%%
extracted_features = ca.calculateFeatures(path = r"F://3month/", 
                                          parameters = {
                                                "mean": None,
                                                "maximum": None}, 
                                          reset_df=False,
                                          raster_mask =  r"F:/Boundary/StatePoly_buf.tif"  ,
                                          tiff_output=True,
                                          workers = 1)

#%%
target_data = tr.image_to_series_simple("F:/5year/Fires/")
raster_mask = u"F:/Boundary/StatePoly_buf.tif"

original_df = [ target_data, extracted_features]
 
target_data_mask, extracted_features_mask  = tr.mask_df(raster_mask,
                                   original_df=original_df,
                                   reset_index = False)
print(target_data_mask.head())
extracted_features_mask.head()



#%% join and test train split yX data

obj = [target_data_mask,extracted_features_mask]
 
#from sklearn.preprocessing import StandardScaler as scaler
X_train, X_test, y_train, y_test = md.get_data(obj, 
                                               stratify=True,
                                               test_size=0.9,
                                               scale=False)

 
#%% Find relevant variables and combine Y and X data

relevance_test, X_train_relevant = \
                        checkRelevance2(X_train,y_train,fdr_level=0.01) 
                        
print(relevance_test)


#%%


RF =  RandomForestClass(X_train[X_train_relevant.columns], y_train, 
                        X_test[X_train_relevant.columns], 
                        y_test)

#%% 

GBoost, MSE, R_Squared = GradientBoosting(X_train[X_train_relevant.columns], 
                                          y_train,
                                           X_test[X_train_relevant.columns], 
                                           y_test)



#%%

from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


#%%

clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1,
                                 max_depth=3, random_state=0).fit(X_train[X_train_relevant.columns], y_train)

#%%
names = X_train_relevant.columns
features = [0,1 , (1, 2)]
fig, axs = plot_partial_dependence(clf,    
                                   X_train[X_train_relevant.columns], 
                                   features, 
                                   feature_names=names) 


#%%

target_feature = (0, 1)
pdp, axes = partial_dependence(clf, target_feature,
                               X=X_train[X_train_relevant.columns], grid_resolution=50)
XX, YY = np.meshgrid(axes[0], axes[1])
Z = pdp[0].reshape(list(map(np.size, axes))).T
ax = Axes3D(fig)
surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1,
                       cmap=plt.cm.BuPu, edgecolor='k')
ax.set_xlabel(names[target_feature[0]])
ax.set_ylabel(names[target_feature[1]])
ax.set_zlabel('Partial dependence')
#  pretty init view
ax.view_init(elev=22, azim=122)
plt.colorbar(surf)
plt.suptitle('Partial dependence of house value on median\n'
             'age and average occupancy')
plt.subplots_adjust(top=0.9)

plt.show()

#%%
predict_test = clf.predict(X=X_test[X_train_relevant.columns])

test_acc = accuracy_score(y_test, predict_test)
kappa = cohen_kappa_score(y_test, predict_test)
confusion = confusion_matrix(y_test, predict_test)

print(kappa)
print(confusion)