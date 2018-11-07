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

#data = data.sort_values([data.columns[0]], ascending=False)

print(" after sorted ")
print(data)


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
    for col in range(0, (len(data.columns)-2)):
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
print(data.dtypes)
print 


#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import KFold   #For K-fold cross validation
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print ("Accuracy : %s" % "{0:.3%}".format(accuracy))

  #Perform k-fold cross-validation with 5 folds
  kf = KFold(n_splits=3)
  kf.get_n_splits(data)

  print('kf')
  print(kf)


  error = []
  for train, test in kf.split(data):
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    print('model scores')
    print('test')
    print(test)
    print('data[predictors].iloc[test,:]')
    print(data[predictors].iloc[test,:])
    print('data[outcome].iloc[test]')
    print(data[outcome].iloc[test])
    print(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
    
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
  print ("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
  #print ("Cross-Validation Score : %s" % "{0:.3%}".format(error))
  #print (data[outcome])

  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome]) 
  return data[outcome]
 

outcome_var = 'topofmindfocus'
#model = LogisticRegression()
model = DecisionTreeClassifier()
#predictor_var = ['Credit_History']
predictor_var = ['dayoftheweek', 'timeoftheday', 'location', 'scheduledoncalendar',
       'acceptedoncalendar', 'timescheduled']
       #data.columns[0:6]
#print('predictor var')
#print(predictor_var)
output = classification_model(model, data[0:3], predictor_var, outcome_var)
print("output is "+output)




# impute_grps = data.pivot_table(values=["topofmindfocus"], 
#                         index=["dayoftheweek","timeoftheday","location",	
#                                 "scheduledoncalendar",
#                             	"acceptedoncalendar", "timescheduled"],
#                  aggfunc=np.mean)

# print(impute_grps)



# print("printing data.shape below")
# print(data.shape)
# print(data.shape[0])
# print(data.shape[1])


# X = data.iloc[:,[2,4]].values
# Y = data.iloc[:,[5,6]].values

# print(X)
# print(Y)

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn import svm

# def larger_model():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(20, input_dim=5, init='normal', activation='relu'))
# 	model.add(Dense(2, init='normal'))
# 	# Compile model
# 	model.compile(loss='mean_squared_error', optimizer='adam')
# 	return model
# seed=7
# np.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# #nb_epoch=50, batch_size=5
# estimators.append(('mlp', KerasRegressor(build_fn=larger_model, nb_epoch=50, batch_size=20, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=10, random_state=seed)


# scalar = StandardScaler()
# clf = svm.LinearSVC()

# pipeline = Pipeline([('transformer', scalar), ('estimator', clf)])
# cv = KFold(n_splits=4)

# #scores = cross_val_score(pipeline, X, Y, cv = cv)

# #results = cross_val_score(pipeline, X, Y, cv=kfold)
# #print("Larger: %.2f (%.2f) MSE" % (scores.mean(), scores.std()))