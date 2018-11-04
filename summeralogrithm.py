import os 
import json
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from googletrans import Translator
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import make_pipeline

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("summermldata.csv")

print(len(data.columns))

print("Data info")
print(data.info)

headers=[len(data.columns)]
headers=data.columns

print("length of the header")
print("length of the headers "+str(len(headers)))
translator = Translator()
#for _ in data.columns:
    #print("the number of null values in : {} == {}".format(_, data[_].isnull().sum()))

#for line in data.values:
    #print("line >>>"+str(translator.translate(line)))
    #print("line >>>"+data.head())

#lb = LabelBinarizer()


#lb_results = lb.fit_transform(data['dayoftheweek'])
#print(" results >>"+str(lb_results))

from sklearn.preprocessing import LabelEncoder

dataLabelEncoder = LabelEncoder()
def encodeData(data):
    for col in range(1, 4):
        values = data[data.columns[col]]
        #print("values in for loop")
        #print(values)
        dataLabelEncoder.fit(values)
        values = dataLabelEncoder.transform(values)
        data[data.columns[col]] = pd.Series(values);
    return data

# daysoftheweek = data['dayoftheweek']
# dataLabelEncoder.fit(daysoftheweek)
# daysoftheweek = dataLabelEncoder.transform(daysoftheweek)
# data['dayoftheweek'] = pd.Series(daysoftheweek)

print("calling encode data method")
data = encodeData(data)
print("after encode the data>>> ")
print(data)






