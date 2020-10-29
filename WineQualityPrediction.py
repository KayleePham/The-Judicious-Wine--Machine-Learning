#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Load libaries
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC  
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np 
get_ipython().run_line_magic('matplotlib', 'inline')

#Load datasets
data_white = pd.read_csv('dataset_WineQuality-White.csv')
data_red = pd.read_csv('dataset_WineQuality-Red.csv')
#data_white
data_red


# In[2]:


data_red.describe()


# In[3]:


data_red.info()


# In[4]:


data_red.skew(axis = 0) 


# In[5]:


data_red.corr() #return correlation between columns


# In[6]:


data_red.cov() #return co-variance betweent columns

