# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 21:26:50 2018

@author: mmann
"""

from tsraster.calculate import extract_features,calculateFeatures, MultiprocessingDistributor,checkRelevance
from tsraster.prep import mask_df
 
import tsraster.prep  as tr
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

path = r"F://3month/"

mask =  r"F:/Boundary/StatePoly_buf.tif"

missing_value =-9999

reset_df=False
tiff_output=False
#%%
extracted_features = calculateFeatures(path, 
                                        parameters, 
                                        #mask=mask, 
                                        reset_df=True, 
                                        tiff_output=True#, 
                                        #missing_value =-9999
                                        )

extracted_features.head()

#%%
from tsfresh.utilities.distribution import LocalDaskDistributor

def read_my_df(path):
    my_df = pd.read_csv(os.path.join(path,'my_df.csv'))
    my_df.set_index(['pixel_id', 'time'], inplace=True)
    # add columns needed for tsfresh
    my_df['pixel_id'] = my_df.index.get_level_values('pixel_id') 
    my_df['time'] = my_df.index.get_level_values('time') 
    return(my_df)

reset_df=False
tiff_output=False
raster_mask =  u"F:/Boundary/StatePoly_buf.tif"
workers = 1

def calculateFeatures(path, parameters, reset_df,raster_mask=None ,tiff_output=True, workers = None):
    '''
    Calculates features or the statistical characteristics of time-series raster data.
    It can also save features as a csv file (dataframe) and/or tiff file.
    
    :param path: directory path to the raster files
    :param parameters: a dictionary of features to be extracted
    :param reset_df: boolean option for existing raster inputs as dataframe
    :param raster_mask: path to binary raster mask
    :param tiff_output: boolean option for exporting tiff file
    :return: extracted features as a dataframe and tiff file
    '''
    
    if reset_df == False:
        #if reset_df =F read in csv file holding saved version of my_df
        my_df = read_my_df(path)
            
    else:
        #if reset_df =T calculate ts_series and save csv
        my_df = image_to_series(path)
        print('df: '+os.path.join(path,'my_df.csv'))
        my_df.to_csv(os.path.join(path,'my_df.csv'), chunksize=10000, index=False)
    
    
    # mask 
    if raster_mask is not None:
        my_df = mask_df(raster_mask = raster_mask, 
                        original_df = my_df)
    
    
    if workers is not None:
        #Distributor = MultiprocessingDistributor(n_workers=workers,
        #                                         disable_progressbar=False,
        #                                         progressbar_title="Feature Extraction")
        Distributor = LocalDaskDistributor(n_workers=workers)
    else:
        Distributor = None
    
    extracted_features = extract_features(my_df, 
                                          default_fc_parameters = parameters,
                                          column_sort = "time",
                                          column_value = "value",
                                          column_id = "pixel_id",
                                          chunksize = 1000,
                                          distributor=Distributor
                                          )
    
    
    # change index name to match 
    extracted_features.index.rename('pixel_id',inplace=True)
    
    return(extracted_features)
    
calculateFeatures(path, parameters, reset_df,raster_mask=None ,tiff_output=True, 
                  workers = 1)

#%%

# deal with output location 
out_path = Path(path).parent.joinpath(Path(path).stem+"_features")
out_path.mkdir(parents=True, exist_ok=True)

# get file prefix
if os.path.isdir(path):
    prefix = os.path.basename(glob.glob("{}/**/*.tif".format(path), recursive=True)[0])[0:4]
    
# write out features to csv file
print("features:"+os.path.join(out_path,'extracted_features.csv'))
extracted_features.columns = [prefix + str(col) for col in extracted_features.columns]
extracted_features.to_csv(os.path.join(out_path,'extracted_features.csv'), chunksize=10000)

# write data frame
kr = pd.DataFrame(list(extracted_features.columns))
kr.index += 1
kr.index.names = ['band']
kr.columns = ['feature_name']
kr.to_csv(os.path.join(out_path,"features_names.csv"))

# write out features to tiff file
if tiff_output == False:
    return extracted_features
else:
    # get image dimension from raw data
    rows, cols, num = image_to_array(path).shape
    # get the total number of features extracted
    matrix_features = extracted_features.values
    num_of_layers = matrix_features.shape[1]
    
    #reshape the dimension of features extracted
    f2Array = matrix_features.reshape(rows, cols, num_of_layers)
    output_file = 'extracted_features.tiff'  
    
    #Get Meta Data from raw data
    raw_data = read_images(path)
    GeoTransform = raw_data[0].GetGeoTransform()
    driver = gdal.GetDriverByName('GTiff')
    
    noData = -9999
    
    Projection = raw_data[0].GetProjectionRef()
    DataType = gdal.GDT_Float32
    
    #export tiff
    CreateTiff(output_file, f2Array, driver, noData, GeoTransform, Projection, DataType, path=out_path)
    return extracted_features



#%% # Mask all values that are outside of the buffered state boundary

raster_mask = u"F:/Boundary/StatePoly_buf.tif"
extracted_features_mask  = tr.mask_df(raster_mask,
                                   original_df=extracted_features)


#%% read target data & mask out non-CA values

target_data = tr.targetData("F:/5year/Fires/")

target_data_mask  = tr.mask_df(raster_mask, 
                               original_df = target_data)


#%%
missing_mask =(extracted_features_mask['tmx__number_cwt_peaks__n_12'] == -9999)  

target_data_mask = target_data_mask[~missing_mask]

extracted_features_mask = extracted_features_mask[~missing_mask]

#%% join and test train split yX data

obj = [target_data_mask, extracted_features_mask]

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