# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:52:17 2018

@author: MMann
"""

from tsraster.calculate import calculateFeatures
 
fc_parameters = {
	    "mean": None,
	    "maximum": None,
	    "median":None
#	    "minimum":None,
#        "sum_values":None,
#	    "agg_linear_trend": [{"attr": 'slope', "chunk_len": 6, "f_agg": "min"},
#                              {"attr": 'slope', "chunk_len": 6, "f_agg": "max"}],
#	    "last_location_of_maximum":None,
#	    "last_location_of_maximum":None,
#	    "last_location_of_minimum":None,
#	    "longest_strike_above_mean":None,
#	    "longest_strike_below_mean":None,
#	    "count_above_mean":None,
#	    "count_below_mean":None,
#	    #"mean_abs_change":None,
#	    "mean_change":None,
#	    "number_cwt_peaks":[{"n": 6},{"n": 12}],
#	    "quantile":[{"q": 0.15},{"q": 0.05},{"q": 0.85},{"q": 0.95}],
#	    "ratio_beyond_r_sigma":[{"r": 2},{"r": 3}], #Ratio of values that are more than r*std(x) (so r sigma) away from the mean of x.
#	    "skewness":None,
	}

#%%
from pathlib import Path 
p = Path('F:/5year')
folders = [str(x) for x in p.iterdir() if x.is_dir() and '_features' not in str(x) and 'Fire' not in str(x) ]

for folder in folders:
    if not os.listdir(folder):
        print("Directory is empty")
    else:    
        print("Directory is not empty")
        print(folder)
        ts_features = calculateFeatures(folder,parameters=fc_parameters,reset_df=False,tiff_output=True)
        print(ts_features.describe())
     
    
    
    #%%
    
from pathlib import Path 
import os
p = Path('./Documents/wildfire_FRAP/Data/Actual/Climate/')
folders = [str(x) for x in p.glob('*/*') if x.is_dir() and '_features' not in str(x) and 'Fire' not in str(x) ]
folders = folders[3:]
folders

for folder in folders:
    if not os.listdir(folder):
        print("Directory is empty")
    else:    
        print("Directory is not empty")
        print(folder)
        ts_features = calculateFeatures(folder,parameters=fc_parameters,reset_df=True,tiff_output=True)
        print(ts_features.describe())
        del ts_features
 