import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, OrdinalEncoder

import pandas as pd
import numpy as np
from os import path

from config.common import BASE_DIR


class CustomScaler(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):

        num_index = []
        str_index = []
        
        if isinstance(X, np.ndarray):
            # 결측치 처리
            for i in range(len(X)):
                for j in range(len(X[i])):
                    if X[i][j] in {'null', 'Null', 'NULL', 'N', 'n', '', np.nan}:
                        X[i][j] = 0
                    elif X[i][j] in {'Y', 'y'}:
                        X[i][j] = 1
            # print(f"missing values : {X}")

            # 문자열 데이터, 숫자형 데이터 분류 ( numpy array )
            shape = X.shape
            for index in range(shape[1]):
                if np.char.isnumeric(X[0, index]):
                    num_index.append(index)
                else:
                    str_index.append(index)

        else:
            # 문자열 데이터, 숫자형 데이터 분류 ( dask array )
            shape = X.compute_chunk_sizes().shape
            for index in range(shape[1]):
                data_type = type(X[0, index].compute())
                if data_type is float or data_type is int:
                    num_index.append(index)
                elif data_type is str or data_type is object:
                    str_index.append(index)

        # 숫자형 데이터 scaler 처리
        for index in num_index:
            scaler = joblib.load(f'{BASE_DIR}/dask/output/scaler_{index}.pkl') if path.exists(
                f'{BASE_DIR}/dask/output/scaler_{index}.pkl') else RobustScaler()
            try:
                X[:, index] = scaler.fit_transform(pd.DataFrame(X[:, index]))
            except:
                continue
            finally:
                joblib.dump(scaler, f'{BASE_DIR}/dask/output/scaler_{index}.pkl')

        # 문자형 데이터 encoder 처리
        encoder = joblib.load(f'{BASE_DIR}/dask/output/encoder.pkl') if path.exists(
            f'{BASE_DIR}/dask/output/encoder.pkl') else OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=9999)
        # print(f"encoder category : {encoder.categories_}")
        try:
            X = encoder.transform(X)
        except:
            X = encoder.fit_transform(X)
        finally:
            joblib.dump(encoder, f'{BASE_DIR}/dask/output/encoder.pkl')
        return X
