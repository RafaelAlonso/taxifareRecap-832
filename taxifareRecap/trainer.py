from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from taxifareRecap.data import clean_df, get_data, split_df
from taxifareRecap.model import get_model
from taxifareRecap.transformers import DistanceTransformer
from taxifareRecap.utils import compute_rmse

class Trainer():
  # guardar os dados de TREINO
  # montar o pipeline
  # treinar o pipeline
  # avaliar o pipeline
  def __init__(self, X_train, y_train):
    self.X_train = X_train
    self.y_train = y_train
    self.pipeline = None
    
  def build_pipeline(self, model_name='RandomForest', hyperparams={}):
    pipe_distance = make_pipeline(
                                  DistanceTransformer(),
                                  StandardScaler()
                                )

    cols = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]

    feateng_blocks = [
        ('distance', pipe_distance, cols),
    ]

    features_encoder = ColumnTransformer(feateng_blocks)

    model = get_model(model_name, hyperparams)


    self.pipeline = Pipeline(steps=[
                ('features', features_encoder),
                ('model', model)])
  
  def train(self):
    self.pipeline.fit(self.X_train, self.y_train)
    
  def evaluate(self, X_test, y_test):
    y_pred = self.pipeline.predict(X_test)
    rmse = compute_rmse(y_pred, y_test)
    return rmse

if __name__ == '__main__':
  # pegar os dados
  print('coletando dados')
  df = get_data()
  
  # limpar os dados
  print('limpando a bagunça')
  df = clean_df(df)
  
  # separar entre X_train, X_test, y_train, y_test
  print('separando entre treino e teste')
  X_train, X_test, y_train, y_test = split_df(df)
  
  # bonus secreto: instanciar meu trainer
  print('instanciando o nosso trainer')
  trainer = Trainer(X_train, y_train)
  
  # montar meu pipeline
  print('montando o pipeline')
  trainer.build_pipeline(model_name='LinearRegression', hyperparams={'alpha': 0.1, 'learning_rate': 0.01})
  
  # trainar meu pipeline
  print('treinando o pipeline')
  trainer.train()
  
  # avaliar meu pipeline
  print('avaliando o pipeline')
  rmse = trainer.evaluate(X_test, y_test)
  print(f'RMSE: {rmse}')
  
  # dar tchau
  print('guhbai')
  