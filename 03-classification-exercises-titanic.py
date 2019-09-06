#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 18:00:05 2017

@author: cbilgili
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('datasets/exercises/03-titanic/train.csv')

feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp',
                'Parch', 'Fare', 'Embarked']

#'Survived'

from pandas.tools.plotting import scatter_matrix
scatter_matrix(dataset, figsize=(12,8))

#sns.pairplot(dataset,x_vars=feature_cols,y_vars="Survived",size=7,aspect=0.7,kind = 'reg')