import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestRegressor


def get_estimator():
  K_imp = KNNImputer(missing_values=np.nan, n_neighbors=3, weights="distance")
  reg = RandomForestRegressor(n_estimators=10, max_depth=8)
  cols = ['P_MHD', 'DAUD', 'PDD', 'PAD', 'PADHD', 'DMSUD', 'PBD',
       'Current health expenditure',
       'Current health expenditure per capita',
       'Out-of-pocket expenditure',
       'Unemployment',
       'School enrollment primary',
       'School enrollment secondary',
       'School enrollment tertiary', 'ghs',
        'media integrity', 'military expenditure']

  prep = ColumnTransformer(transformers=[
          ('prep',  make_pipeline(K_imp, StandardScaler()), cols),
          ], remainder='drop')

  estimator = Pipeline(steps=[
      ('prep', prep),
      ('classifier', reg)])

  return estimator