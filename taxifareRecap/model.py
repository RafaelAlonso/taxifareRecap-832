from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import xgboost

def get_model(model_name='RandomForest', hyperparams={}):
  if model_name == 'RandomForest':
    model = RandomForestRegressor()
  elif model_name == 'XGBoost':
    model = xgboost()
  elif model_name == 'SVR':
    model = SVR()
  else:
    model = LinearRegression()
    
  model.set_params(**hyperparams)
    
  return model
