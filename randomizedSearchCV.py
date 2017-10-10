import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer


import pickle

import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


# Load saved X,y data
with open("Xy.pickle", 'rb') as picklefile: 
    X_train, X_test, y_train, y_test, X, y = pickle.load(picklefile)

    
h = ['h_x1', 'h_y1', 'h_z1','h_g1', 'h_g2', 'h_g3', 'h_m1', 'h_m2', 'h_m3']
c = ['c_x1', 'c_y1', 'c_z1','c_g1', 'c_g2', 'c_g3', 'c_m1', 'c_m2', 'c_m3']
a = ['a_x1', 'a_y1', 'a_z1','a_g1', 'a_g2', 'a_g3', 'a_m1', 'a_m2', 'a_m3']

acce= h+c+a# Features belonging to wrist sensor


# Create grid
ne = [10,15,20]
md = [5,10,15,30]
mf = [6,9]
lr = [0.1,0.2]

param_grid_gbc = dict(n_estimators=ne,max_depth=md,max_features = mf) # number of combinations = 48

#param_grid_tree = dict(n_estimators= ne, criterion=cr,max_depth=md,
#                 max_features = mf)# number of combinations = 96

#print('set up grid_tree')
#grid_tree = RandomizedSearchCV(RandomForestClassifier(n_jobs=-1), param_grid_tree, cv=5,
#                          scoring=['f1_macro','recall_macro','precision_macro'],refit=False,
#                          n_jobs=-1,random_state=4444,
#                          n_iter=96/3)


# Fit randomized CV grid
#print('grid_tree fitting')
#grid_tree.fit(X_train[c],y_train)
#print(grid_tree.cv_results_)
#print('grid_tree fitted')

#with open('tree_randomSearchCV.pickle', 'wb') as f: 
#    pickle.dump(grid_tree,f )


print('Start Process')
start = default_timer()
# Set up randomized Cross Validated grid search.
grid_gbc = GridSearchCV(GradientBoostingClassifier(), param_grid_gbc, cv=5,
                        scoring=['f1_macro','recall_macro','precision_macro'],
                        refit=False,n_jobs=-1,verbose=1)

# Fit randomized CV grid
grid_gbc.fit(X_train[h],y_train)

print('{:.2f}'.format(default_timer()-start))
print('grid_gbc fit')


with open('gbc_randomSearchCV.pickle', 'wb') as f:
    pickle.dump(grid_gbc,f )

    