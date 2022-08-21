# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 10:12:40 2022

@author: Ann
"""

#https://elitedatascience.com/python-data-wrangling-tutorial

#sql: case  when then
python: np.where

# condition statement
np.select
df.case_when 
df.shift(7) #(shift previous 7 rows, exluding index)

[df1]+[df2] = [df1, df2]

from functools import reduce
reduce(lambda x,y: x+y, [1,2,3,4,5]) #calculates ((((1+2)+3)+4)+5)

reduce(lambda x,y: pd.merge(x,y,on=['Date', 'Code']),
             feature_dfs)

#groupby and take the .first() observation (excluding Nan)
abt.groupby(['Code', 'month']).first()

# =============================================================================
# extract
# =============================================================================

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 




df =pd.read_csv (r"D:\Users\Ann\Downloads\BNC2_sample.csv",
                  names=['Code', 'Date', 'Open', 'High', 'Low', 
                        'Close', 'Volume', 'VWAP', 'TWAP'])

df.Code.unique()


gwa_codes = [code for code in df.Code.unique() if 'GWA_' in code ]
df = df[df.Code.isin(gwa_codes)]

pivoted_df = df.pivot(index='Date', columns='Code', values='VWAP')

# Calculate returns over 7 days prior == (current - previous)/previous
delta_7 = pivoted_df / pivoted_df.shift(7) - 1.0


# Calculate returns over each window and store them in dictionary
delta_dict = {}
for offset in [7, 14, 21, 28]:
    delta_dict['delta_{}'.format(offset)] = pivoted_df / \
        pivoted_df.shift(offset) - 1.0



# Melt delta_7 returns
melted_7 = delta_7.reset_index().melt(id_vars=['Date'], value_name='delta_7')
 

# Melt all the delta dataframes and store in list
melted_dfs = []
for key, delta_df in delta_dict.items():
    melted_dfs.append( delta_df.reset_index().melt(id_vars=['Date'], value_name=key) )


# Calculate 7-day returns after the date
return_df = pivoted_df.shift(-7) / pivoted_df - 1.0
 
# Melt the return dataset and append to list
melted_dfs.append( return_df.reset_index().melt(id_vars=['Date'], value_name='return_7') )


from functools import reduce

# Grab features from original dataset
base_df = df[['Date', 'Code', 'Volume', 'VWAP']]
 
# Create a list with all the feature dataframes
feature_dfs = [base_df] + melted_dfs

# Reduce-merge features into analytical base table
abt = reduce(lambda left,right: pd.merge(left,right,on=['Date', 'Code']), feature_dfs)
 
# Create 'month' feature
abt['month'] = abt.Date.apply(lambda x: x[:7])

 
# Group by 'Code' and 'month' and keep first date
gb_df = abt.groupby(['Code', 'month']).first().reset_index()
 


















