##!pip install Keras

import matplotlib
matplotlib.use('Agg')
import math
import itertools

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sys
import csv
import numpy as ny
import pandas
import matplotlib.pyplot as plt

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed

set_random_seed(2)

Filename = 'crimepattern.csv'
inputsize = 1
TrainOutputFile='crimepattern_Pred.csv'
PlotFile='crimepattern.png'
TestInputFile='hypothetical_sentiment.csv'
TestOutputFile='hypothetical_output.txt'
trials=1000
verboseVal=1

X = ny.empty([0,inputsize])
Y = ny.empty([0,1])

with open(Filename) as csvfile:
    readCSV=csv.reader(csvfile,delimiter=',')
    for row in readCSV:
        X = ny.vstack([X,[row[:inputsize]]])
        Y = ny.vstack([Y,row[inputsize]])

#
print X
#
print Y

sc_X=StandardScaler()
X_train = sc_X.fit_transform(X)

Y=ny.reshape(Y,(-1,1))
sc_Y=StandardScaler()
Y_train = sc_Y.fit_transform(Y)


N = len(Y_train)
print N

def brain():
    #Create the brain
    br_model=Sequential()
    br_model.add(Dense(40, input_dim=inputsize, kernel_initializer='normal',activation='relu'))
    br_model.add(Dense(30, kernel_initializer='normal',activation='relu'))


    br_model.add(Dense(20, kernel_initializer='normal',activation='relu'))

    br_model.add(Dense(20, kernel_initializer='normal',activation='relu'))
  
    br_model.add(Dense(15, kernel_initializer='normal',activation='relu'))

    br_model.add(Dense(1,kernel_initializer='normal'))
    
    #Compile the brain
    br_model.compile(loss='mean_squared_error',optimizer='adam')
#    br_model.compile(loss='logcosh',optimizer='adam')
    return br_model

def predict(X,sc_X,sc_Y,estimator):
    prediction = estimator.predict(sc_X.transform(X))
    return sc_Y.inverse_transform(prediction)

estimator = KerasRegressor(build_fn=brain, epochs=trials, batch_size=5,verbose=verboseVal)
#
print "Done"

# seed = 21
# ny.random.seed(seed)
# kfold = KFold(n_splits=N, random_state=seed)
# results = cross_val_score(estimator, X_train, Y_train, cv = kfold)
estimator.fit(X_train,Y_train)
prediction = estimator.predict(X_train)

# print Y_train
# print prediction

# print Y

# pred_final= sc_Y.inverse_transform(prediction)
pred_final = predict(X,sc_X,sc_Y,estimator)
# print pred_final
X_trainOut=ny.empty([0,3])
Base = ny.empty([0,1])
errorVal=0
X_trainOut=ny.vstack([X_trainOut,['SNo','Actual','Predicted']])
for i in range(0, len(Y)):
   row_new = [i,Y[i][0], pred_final[i]]
   X_trainOut=ny.vstack([X_trainOut,row_new])
   errorVal=errorVal+pow(float(Y[i][0])-float(pred_final[i]),2)
   Base=ny.vstack([Base,i])

errorVal=errorVal/len(Y)
   
print 'Average Deviation:'
print math.sqrt(errorVal)
with open(TrainOutputFile,'wb') as csvWriteFile:
    writeCSV=csv.writer(csvWriteFile,delimiter=",")
    writeCSV.writerows(X_trainOut)

