#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 17:37:46 2017

@author: cbilgili
"""

from __future__ import division, print_function, unicode_literals
import pandas as pd

s = pd.Series([2,-1,3,5])

s2 = pd.Series([68, 83, 112, 68], index=["alice", "bob", "charles", "darwin"])

weights = {"alice": 68, "bob": 83, "colin": 86, "darwin": 68}
s3 = pd.Series(weights)

s4 = pd.Series(weights, index = ["colin", "alice"])

dates = pd.date_range('2016/10/29 5:30pm', periods=12, freq='H')

import matplotlib.pyplot as plt
temperatures = [4.4,5.1,6.1,6.2,6.1,6.1,5.7,5.2,4.7,4.1,3.9,3.5]
s7 = pd.Series(temperatures, name="Temperature")
s7.plot()
plt.show()

temp_series = pd.Series(temperatures, dates)

temp_series.plot(kind="bar")

plt.grid(True)
plt.show()


people_dict = {
    "weight": pd.Series([68, 83, 112], index=["alice", "bob", "charles"]),
    "birthyear": pd.Series([1984, 1985, 1992], index=["bob", "alice", "charles"], name="year"),
    "children": pd.Series([0, 3], index=["charles", "bob"]),
    "hobby": pd.Series(["Biking", "Dancing"], index=["alice", "bob"]),
}
people = pd.DataFrame(people_dict)

import numpy as np

d5 = pd.DataFrame(
  {
    ("public", "birthyear"):
        {("Paris","alice"):1985, ("Paris","bob"): 1984, ("London","charles"): 1992},
    ("public", "hobby"):
        {("Paris","alice"):"Biking", ("Paris","bob"): "Dancing"},
    ("private", "weight"):
        {("Paris","alice"):68, ("Paris","bob"): 83, ("London","charles"): 112},
    ("private", "children"):
        {("Paris", "alice"):np.nan, ("Paris","bob"): 3, ("London","charles"): 0}
  }
)
    
# people[people["birthyear"] > 1990]
people["age"] = 2016 - people["birthyear"]
people["over 30"] = people["age"] > 30
birthyears = people.pop("birthyear")
people["pets"] = pd.Series({"bob": 0, "charles": 5, "eugene":1})
people.insert(1, "height", [172, 181, 185])
# people.query("age > 30 and pets == 0")
people.eval("weight / (height/100) ** 2 > 25")
people.eval("body_mass_index = weight / (height/100) ** 2")
people.sort_values(by="age", inplace=True)
people.plot(kind = "line", x = "body_mass_index", y = ["height", "weight"])
people.plot(kind = "scatter", x = "height", y = "weight", s=[40, 120, 200])

much_data = np.fromfunction(lambda x,y: (x+y*y)%17*11, (10000, 26))
large_df = pd.DataFrame(much_data, columns=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
large_df[large_df % 16 == 0] = np.nan
large_df.insert(3,"some_text", "Blabla")

city_pop = pd.DataFrame(
    [
        [808976, "San Francisco", "California"],
        [8363710, "New York", "New-York"],
        [413201, "Miami", "Florida"],
        [2242193, "Houston", "Texas"]
    ], index=[3,4,5,6], columns=["population", "city", "state"])
city_eco = city_pop.copy()
city_eco["eco_code"] = [17, 17, 34, 20]
city_eco["economy"] = city_eco["eco_code"].astype('category')
city_eco["economy"].cat.categories = ["Finance", "Energy", "Tourism"]