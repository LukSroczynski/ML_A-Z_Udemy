# Data Preprocessing

# Importing the libraries
import numpy as np # contains math tools
import matplotlib.pyplot as plt
import pandas as pd # for import data-sets and to manage them

# Importing the data-set
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values # Take all columns and removes last one
Y = dataset.iloc[:, 3].values # Take all columns returns fourth one

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:, 1:3])

X[:, 1:3] = imputer.transform(X[:, 1:3]) # Fill data

