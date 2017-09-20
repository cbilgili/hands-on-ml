#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 19:14:52 2017

@author: cbilgili
"""

from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt

a = np.array([[-2.5, 3.1, 7], [10, 11, 12]])
for func in (a.min, a.max, a.sum, a.prod, a.std, a.var):
    print(func.__name__, "=", func())