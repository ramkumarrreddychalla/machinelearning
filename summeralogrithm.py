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

data = pd.read_csv(str(os.curdir)+"/summermldata.csv")

print(len(data.columns))

print("Data info")
print(data.info)

headers=[len(data.columns)]
headers=data.columns

print('headers >>>>')
print(headers)


print("length of the header")
print("length of the headers "+str(len(headers)))
translator = Translator()
#for _ in data.columns:
    #print("the number of null values in : {} == {}".format(_, data[_].isnull().sum()))

#for line in data.values:
    #print("line >>>"+str(translator.translate(line)))
    #print("line >>>"+data.head())

#lb = LabelBinarizer()


#lb_results = lb.fit_transform(data['scheduledoncalendar'])
#print(" results >>"+str(lb_results))

from sklearn.preprocessing import LabelEncoder

print("length is >>> "+str(len(data.columns)-2))

dataLabelEncoder = LabelEncoder()
def encodeData(data):
    for col in range(1, (len(data.columns)-2)):
        values = data[data.columns[col]]
        #print("values in for loop")
        #print(values)
        dataLabelEncoder.fit(values)
        values = dataLabelEncoder.transform(values)
        data[data.columns[col]] = pd.Series(values)
    return data

# daysoftheweek = data['dayoftheweek']
# dataLabelEncoder.fit(daysoftheweek)
# daysoftheweek = dataLabelEncoder.transform(daysoftheweek)
# data['dayoftheweek'] = pd.Series(daysoftheweek)

print("calling encode data method")
data = encodeData(data)
print("after encode the data>>> ")
print(data)

print("printing data.shape below")
print(data.shape)
print(data.shape[0])
print(data.shape[1])

#data['split'] = np.random.randn(data.shape[1], 1)

print(data['topofmindfocus'])

# print(data[data.shape[1] - 1])

from sklearn.preprocessing import LabelBinarizer

# lb = LabelBinarizer()
# lb_results = lb.fit_transform(data['acceptedoncalendar'])
# print('lb results >>>')
# print(lb_results)
# print(lb.classes_);

# lb_results_df = pd.DataFrame(lb_results, columns=lb.classes_)
# print(lb_results_df.head())

from sklearn.model_selection import train_test_split

features = [headers[i] for i in range(1, 6)]


print('features')
print(features)

from sklearn.feature_extraction.text    import CountVectorizer
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate, BaseCrossValidator
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression


X = data.iloc[:, :2].values
Y = data.iloc[:, 6].values
# Encode Categorical Data

print(X)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Feature Scaling

# Split the data between the Training Data and Test Data

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2
                                                    ,random_state = 0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


print(X_train)
print(X_test)
print(Y_train)
print(Y_test)



