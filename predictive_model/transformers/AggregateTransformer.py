from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np
from time import time
import sys
import os
class AggregateTransformer(TransformerMixin):
    
    def __init__(self, results_dir, dataset_name, case_id_col, cat_cols, num_cols, boolean=False, model="catboost", fillna=True):
        self.case_id_col = case_id_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.model = model
        self.dataset_name = dataset_name
        self.results_dir = results_dir
        
        self.boolean = True
        self.fillna = fillna
        
        self.columns = None
        
        self.fit_time = 0
        self.transform_time = 0
    
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        start = time()
        
        # transform numeric cols
        if len(self.num_cols) > 0:
            dt_numeric = X.groupby(self.case_id_col)[self.num_cols].agg(["mean", "max", "min", "sum", "std"])
            dt_numeric.columns = ['_'.join(col).strip() for col in dt_numeric.columns.values]
            
        # transform cat cols
        # transform cat cols
        if self.model!="catboost":
            dt_transformed = pd.get_dummies(X[self.cat_cols])
            dt_transformed[self.case_id_col] = X[self.case_id_col]
            del X
        else:
            #dt_transformed = pd.get_dummies(X[self.cat_cols])
            dt_transformed = X[self.cat_cols]
            dt_transformed[self.case_id_col] = X[self.case_id_col]
            del X
        if self.boolean:
            dt_transformed = dt_transformed.groupby(self.case_id_col).max()
        else:
            dt_transformed = dt_transformed.groupby(self.case_id_col).sum()

        # concatenate
        if len(self.num_cols) > 0:
            dt_transformed = pd.concat([dt_transformed, dt_numeric], axis=1)
            del dt_numeric
        
        # fill missing values with 0-s
        if self.fillna:
            dt_transformed = dt_transformed.fillna(0)
            
        # add missing columns if necessary
        if self.columns is None:
            self.columns = dt_transformed.columns
        else:
            missing_cols = [col for col in self.columns if col not in dt_transformed.columns]
            for col in missing_cols:
                dt_transformed[col] = 0
            dt_transformed = dt_transformed[self.columns]
        
        self.transform_time = time() - start
        print("Save Aggregate encoding")
        #print(dt_transformed)
        #dt_transformed.to_csv('dt_transformed_agg_%s.csv'%self.dataset_name, index=False, sep=';')
        dt_transformed.to_csv(os.path.join(self.results_dir, 'dt_transformed_agg_%s.csv'%self.dataset_name),  index=False, sep=';')
        #print(f"np.where(dt_transformed.dtypes != float)[0]: {np.where(dt_transformed.dtypes != float)[0]}")
        return dt_transformed
