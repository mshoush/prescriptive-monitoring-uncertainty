from sklearn.base import TransformerMixin
import pandas as pd
from time import time
import numpy as np
import os

class StaticTransformer(TransformerMixin):
    
    def __init__(self, results_dir, dataset_name,case_id_col, cat_cols, num_cols, model="catboost", fillna=True):
        self.case_id_col = case_id_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.fillna = fillna
        self.model = model
        self.dataset_name=dataset_name
        self.results_dir = results_dir
        
        self.columns = None
        
        self.fit_time = 0
        self.transform_time = 0
        #print(f"self.cat_cols: {self.cat_cols}")
        #print(f"self.num_cols: {self.num_cols}")
        #print(f"X: {X}")
    
    def fit(self, X, y=None):
        return self
    
    
    def transform(self, X, y=None):
        start = time()
        
        dt_first = X.groupby(self.case_id_col).first()
        
        # transform numeric cols
        dt_transformed = dt_first[self.num_cols]
        
        # transform cat cols
        if len(self.cat_cols) > 0 and self.model == "catboost":
            #dt_cat = pd.get_dummies(dt_first[self.cat_cols])
            dt_transformed = pd.concat([dt_transformed, dt_first[self.cat_cols]], axis=1).reset_index(drop=True)
            #print(f"np.where(dt_transformed.dtypes != float)[0]: {np.where(dt_transformed.dtypes != float)[0]}")
        else:
            if len(self.cat_cols) > 0:
                dt_cat = pd.get_dummies(dt_first[self.cat_cols])
                dt_transformed = pd.concat([dt_transformed, dt_cat], axis=1)
            else:
                print("No Cat Cols...")


        # fill NA with 0 if requested
        if self.fillna:
            dt_transformed = dt_transformed.fillna(0)
            
        # add missing columns if necessary
        if self.columns is not None:
            missing_cols = [col for col in self.columns if col not in dt_transformed.columns]
            for col in missing_cols:
                dt_transformed[col] = 0
            dt_transformed = dt_transformed[self.columns]
        else:
            self.columns = dt_transformed.columns
        
        self.transform_time = time() - start
        print("Save Static encoding")
        # dt_transformed.to_csv('dt_transformed_static_%s.csv'%self.dataset_name, index=False, sep=';')
        dt_transformed.to_csv(os.path.join(self.results_dir, 'dt_transformed_static_%s.csv'%self.dataset_name), index=False, sep=';')

        #print(dt_transformed)
        #print("==============")
        return dt_transformed
