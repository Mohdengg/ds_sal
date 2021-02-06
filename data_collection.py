# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 08:36:31 2021

@author: Katpadi
"""


import glassdoor_scrapper as gs
import pandas as pd
path ="C:/Users/Katpadi/Documents/DataScience/Ken Jee/sal_proj/chromedriver"
slp_time= 15
df = gs.get_jobs("data scientist",40, False, path, slp_time)
