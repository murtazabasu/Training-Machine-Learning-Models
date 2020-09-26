from sklearn import tree
from sklearn import ensemble
from sklearn import linear_model
import xgboost as xgb


"""
This file contains the models to be dispatched
to the training script.   
"""

models = {
    "log_reg": linear_model.LogisticRegression(),
    "random_forest": ensemble.RandomForestClassifier(n_jobs=-1)
    "XGboost": xgb.XGBClassifier(n_jobs=-1,
                              max_depth=7,
                              n_estimators=200
                              )
}