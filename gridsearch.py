#importing libraries
import pandas as pd 
import numpy as np

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection 


if __name__ == "__main__":
  data = pd.read_csv("...../input/mobile_train.csv")
  X = df.drop("price_range", axis=1).values
  Y = df.price_range.values
  
  #defining classifier model
  clf = ensemble.RandomForestClassifier(n_jobs = -1)
  
  param_grid = { 
                "n_estimators" : [50, 100, 150, 200, 250],
                "max_depth" : [2,4,6,8,10]
                "criterion" : ["gini", "entropy"]
               }
               
  #starting the hyper parameter optimization              
  model = model_selection.GridsearchCV(
                                        estimator = clf,
                                        param_grid = param_grid,
                                        scoring = "accuracy",
                                        verbose = 10,
                                        n_jobs = 1
                                        cv = 5
                                       )
  #traing the model                                     
  model.fit(X,Y)
  print(model.best_score)
  print(model.best_estimator_.get_params())
                                      
