#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 17:58:14 2017

@author: cbilgili

A-Z Project Exercise
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

HOUSING_PATH = "datasets/housing"


def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "kc_house_data.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()

#import matplotlib.pyplot as plt
#housing.hist(bins=50, figsize=(20,15))
#plt.show()

housing["age"] = 2015 - housing["yr_built"]
housing["age"] = 2015 - housing["yr_renovated"]
housing.loc[housing.age == 2015, 'age'] = 2015 - housing["yr_built"]

feature_cols = ['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'age']

best_feature_cols = ['grade', 'sqft_living']

#housing.plot(kind="scatter", x="long", y="lat", alpha=0.4, 
#             s=housing["population"]/100, label="population", figsize=(10,7),
#             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar = True
#             )

# Correlation
#from pandas.tools.plotting import scatter_matrix
#scatter_matrix(housing, figsize=(12,8))

# Seaborn Better Plotting
#sns.pairplot(housing,x_vars=feature_cols,y_vars="price",size=7,aspect=0.7,kind = 'reg')
X = housing[best_feature_cols]
y = housing["price"]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
        ('std_scaler', StandardScaler())
    ])

housing_prepared = pipeline.fit_transform(X_train)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("standard deviation:", scores.std())
    
from sklearn.model_selection import cross_val_score
lin_scores = cross_val_score(regressor, X_test, y_test,
                         scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(X_train, y_train)
y_pred_forest = forest_reg.predict(X_test)


# Grid Search for Fine tuning model
from sklearn.model_selection import GridSearchCV

param_grid = [
        {'n_estimators': [30, 50, 60], 'max_features': [8, 10, 12]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
    ]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(X_train, y_train)

grid_search.best_params_

feature_importances = grid_search.best_estimator_.feature_importances_
sorted(zip(feature_importances, feature_cols), reverse=True)

#sns.lmplot(x='sqft_living',y='price',data=X_test,fit_reg=True) 

# Visualising the Training set results
plt.scatter(X_train[["sqft_living"]], y_train, color = 'red')
plt.plot(X_train[["sqft_living"]], regressor.predict(X_train), color = 'blue')
plt.title('Grade vs Experience (Training set)')
plt.xlabel('1.')
plt.ylabel('Price')
plt.show()


# Visualising the Training set results
plt.scatter(y_test, y_pred, color = 'red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.title('Grade vs Experience (Training set)')
plt.xlabel('1.')
plt.ylabel('Price')
plt.show()


accuracy = regressor.score(X_test, y_test)
"Accuracy: {}%".format(int(round(accuracy * 100)))