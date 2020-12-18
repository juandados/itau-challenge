import pandas as pd
from pathlib import Path
import os

class Dataset():
  BASE_PATH = Path('/content/drive/My Drive/itau')
  BASE_DATA_PATH = BASE_PATH / 'data'
  
  def __init__(self):
    self.PRODUCTS = ["A-A", "B-B", "C-D", "D-E", "E-E"]

  def load_data(self):
    self.campains_train_df = pd.read_csv(self.BASE_DATA_PATH / 'Campanas_train.csv', index_col=['Unnamed: 0'])
    dtypes = {"id": "category", "Id_Producto": "category", "Tipo": "category", "Producto-Tipo": "category",
              "Fecha_Campaña": "datetime64[ns]", "Periodo": "int64", "Canal": "category", "Duracion_Campaña": "int64"}
    self.campains_train_df = self.campains_train_df.astype(dtypes)

    self.campains_test_df = pd.read_csv(self.BASE_DATA_PATH /'Campanas_test.csv', index_col=['Unnamed: 0'])
    dtypes = {"id": "category", "Id_Producto": "category", "Tipo": "category", "Producto-Tipo": "category",
              "Fecha_Campaña": "datetime64[ns]", "Periodo": "int64", "Canal": "category", "Duracion_Campaña": "int64"}
    self.campains_test_df = self.campains_test_df.astype(dtypes)

    self.comunications_train_df = pd.read_csv(self.BASE_DATA_PATH / 'Comunicaciones_train.csv', index_col=['Unnamed: 0'])
    dtypes = {"id": "category", "Id_Producto": "category", "Tipo": "category", "Producto-Tipo": "category", 
              "Tipo_comunicacion": "category", "Fecha": "datetime64[ns]", "Periodo": "int64", "Lectura": "category"}
    self.comunications_train_df = self.comunications_train_df.astype(dtypes)

    self.comunications_test_df = pd.read_csv(self.BASE_DATA_PATH / 'Comunicaciones_test.csv', index_col=['Unnamed: 0'])
    dtypes = {"id": "category", "Id_Producto": "category", "Tipo": "category", "Producto-Tipo": "category",
              "Tipo_comunicacion": "category", "Fecha": "datetime64[ns]", "Periodo": "int64", "Lectura": "category"}
    self.comunications_test_df = self.comunications_test_df.astype(dtypes)

    self.customers_df = pd.read_csv(self.BASE_DATA_PATH / 'Consumidores.csv', index_col=['Unnamed: 0'])
    dtypes = {'Edad': 'category', 'Sexo': 'category', 'Renta': 'category', 'Recibe_sueldo_en_cuenta': 'category',
          'Segmento_consumidor': 'category', 'Meses_antiguedad': 'category', 'Comuna':'category', 'Ciudad': 'category',
          'Estado_civil': 'category', 'Principalidad': 'category', 'Profesion': 'category', 'id': 'category'}
    self.customers_df = self.customers_df.astype(dtypes)

    self.transactions_train_df = pd.read_csv(self.BASE_DATA_PATH / 'Transaccion_train.csv',index_col=['Unnamed: 0'])
    dtypes = {'id':'category', 'Id_Producto':'category', 'Tipo':'category', 'Producto-Tipo':'category', 'Signo':'category', 'Monto':'float64', 
              'Fecha':'datetime64[ns]','Periodo':'int'}
    self.transactions_train_df = self.transactions_train_df.astype(dtypes)

    self.transactions_test_df = pd.read_csv(self.BASE_DATA_PATH / 'Transaccion_test.csv', index_col=['Unnamed: 0'])
    dtypes = {'id':'category', 'Id_Producto':'category', 'Tipo':'category', 'Producto-Tipo':'category', 'Signo':'category', 'Monto':'float64', 
              'Fecha':'datetime64[ns]','Periodo':'int'}
    self.transactions_test_df = self.transactions_test_df.astype(dtypes)
    # products_of_interest:
    interest_mask = self.transactions_train_df['Producto-Tipo'].apply(lambda x: x in self.PRODUCTS)
    self.products_df = (self.transactions_train_df.loc[interest_mask]
                   .astype({'Producto-Tipo':'object'})
                   .astype({'Producto-Tipo':'category'}))
    
  def get_transactions_counts(self):
    if not os.path.exists(self.BASE_DATA_PATH/'data_counts.pkl'):
      counts = self.products_df.groupby(['id','Producto-Tipo', 'Periodo']).agg({'Monto':'count'}).unstack(-1)
      counts.rename(columns={'Monto':'count'}, level=0, inplace=True)
      counts = counts
      counts.to_pickle(self.BASE_DATA_PATH/'data_counts.pkl')
    else:
      counts = pd.read_pickle(self.BASE_DATA_PATH/'data_counts.pkl')
    return counts