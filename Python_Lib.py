#!/usr/bin/env python
# coding: utf-8

# In[1]:

# load libraries
import numpy as np
import pandas as pd

# for visualization 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Set option to display max 100 columns & 100 rows
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

# for train -test split
import sklearn
from sklearn.model_selection import train_test_split

# for feature scaling (MinMax Scaler)
from sklearn.preprocessing import MinMaxScaler

# For Standardisation & Scaling
from sklearn.preprocessing import StandardScaler

# For building logistic regression model
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

# Import RFE used in feature elimination
from sklearn.feature_selection import RFE

# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# model evaluation
from sklearn.metrics import r2_score

# compute accuracy scores & Confusion matrix
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve


# In[ ]:




