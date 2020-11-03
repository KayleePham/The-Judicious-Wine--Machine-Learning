#!/usr/bin/env python
# coding: utf-8

#Load libaries
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np 
get_ipython().run_line_magic('matplotlib', 'inline')

#Load datasets
data_red = pd.read_csv('dataset_WineQuality-Red.csv')
data_red

data_red.describe()

data_red.info()

data_red.skew(axis = 0) 

data_red.corr() #return correlation between columns

data_red.cov() #return co-variance betweent columns

