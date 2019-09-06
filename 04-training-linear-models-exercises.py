#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 22:28:39 2017

@author: cbilgili
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('datasets/exercises/04-linear/ex1data1.txt', names = ['population', 'profit'])

# Split population and training data
X_df = pd.DataFrame(data.population)
y_df = pd.DataFrame(data.profit)

# Length, or number of observations in data
m = len(y_df)

plt.figure(figsize=(10,8))
plt.plot(X_df, y_df, 'kx')
plt.xlabel('Population of City')
plt.ylabel('Profit in 10k$')

## Try to draw random lines
plt.figure(figsize=(10,8))
plt.plot(X_df, y_df, 'k.')
plt.plot([5, 22], [6,6], '-')
plt.plot([5, 22], [0,20], '-')
plt.plot([5, 15], [-5,25], '-')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')

iterations = 1500
alpha = 0.01

## Add a columns of 1s as intercept to X. This becomes the 2nd column
X_df['intercept'] = 1
X = np.array(X_df)
y = np.array(y_df).flatten()
theta = np.array([0, 0])

def cost_function(X, y, theta):
    """
    cost_function(X, y, theta) computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y
    """
    ## number of training examples
    m = len(y) 
    
    ## Calculate the cost with the given parameters
    J = np.sum((X.dot(theta)-y)**2)/2/m
    
    return J

cost_function(X, y, theta)

## Try to draw cost function
best_fit_x = np.linspace(-15, 10, 50)
best_fit_y = [cost_function(X, y, [1, xx]) for xx in best_fit_x]
plt.figure(figsize=(10,6))
plt.plot(best_fit_x, best_fit_y, 'r-')
plt.xticks(np.arange(min(best_fit_x), max(best_fit_x)+1, 1))
plt.axis([-15,10,-5,55])


def gradient_descent(X, y, theta, alpha, iterations):
    """
    gradient_descent Performs gradient descent to learn theta
    theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
    taking num_iters gradient steps with learning rate alpha
    """
    cost_history = [0] * iterations
    
    for iteration in range(iterations):
        hypothesis = X.dot(theta)
        loss = hypothesis-y
        gradient = X.T.dot(loss)/m
        theta = theta - alpha*gradient
        cost = cost_function(X, y, theta)
        cost_history[iteration] = cost
        
        ## If you really want to merge everything in one line:
        # theta = theta - alpha * (X.T.dot(X.dot(theta)-y)/m)
        # cost = cost_function(X, y, theta)
        # cost_history[iteration] = cost

    return theta, cost_history

(t, c) = gradient_descent(X,y,theta,alpha, iterations)

## Prediction
print(np.array([3.5, 1]).dot(t))
print(np.array([7, 1]).dot(t))
# Profits are about $4,519 and $45,342 respectively.

## Plotting the best fit line
best_fit_x = np.linspace(0, 25, 20)
best_fit_y = [t[1] + t[0]*xx for xx in best_fit_x]
plt.figure(figsize=(10,6))
plt.plot(X_df.population, y_df, '.')
plt.plot(best_fit_x, best_fit_y, '-')
plt.axis([0,25,-5,25])
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.title('Profit vs. Population with Linear Regression Line')