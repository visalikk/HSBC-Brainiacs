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
