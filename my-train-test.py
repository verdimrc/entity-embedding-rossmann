#!/usr/bin/env python3

from itertools import chain
import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Activation, Reshape, Concatenate
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.embeddings import Embedding
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, PowerTransformer, MinMaxScaler
from typing import List, Tuple, Dict

all_features = 'Store DayOfWeek Open Promo StateHoliday SchoolHoliday '\
               'Year Month Date State Holiday'.split()
features_ori = 'Store DayOfWeek Promo Year Month Day State'.split()
features_hol = 'Store DayOfWeek Promo Holiday Year Month Day State'.split()

seed = 7
np.random.seed(seed)

def df2list(df):
    return [df[f] for f in features]

def main():
    '''Experiment rossmann embedding approaches: (i) original features vs with
    squashed holiday, and (ii) log normal transformation on dependent variables
    vs box-cox transformation.

    NOTE: failed experiment keras + sklearn; code deleted from this file.
          See https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/

    REASON: validation curve per epoch not easy; must use one or potentially
            both ofpartial_fit + warm_startup.
    '''

    # Load & preprocess data
    df = load_data()
    df = df[(df.Open == 1) & (df.Sales > 0)]    # Same as original
    df = pd.concat((df, squash_holidays(df, 'Holiday')), axis=1, sort=False)
    encode_category_inplace(df, all_features)

    train_df, val_df = split(df, train_ratio=0.9, train_samples=200*1000)

    Y_train    : np.ndarray = train_df['Sales'].values
    Y_val      : np.ndarray = val_df['Sales'].values
    X_ori_train: np.ndarray = df2list(train_df, features_ori)
    X_ori_val  : np.ndarray = df2list(val_df, features_ori)
    X_hol_train: np.ndarray = df2list(train_df, features_hol)
    X_hol_val  : np.ndarray = df2list(val_df, features_hol)

    ori_set = (X_ori_train, Y_train, X_ori_val, Y_val)
    do_experiment_ori_v1(*ori_set)
    do_experiment_ori_v2(*ori_set)

    hol_set = (X_hol_train, Y_train, X_hol_val, Y_val)
    do_experiment_hol_v1(*hol_set)
    do_experiment_hol_v2(*hol_set)

def do_experiment_ori_v1(X_train, Y_train: np.ndarray, X_val, Y_val: np.ndarray):
    print_header('ori', 'log_normal', 'sigmoid')
    log_y_train = np.log(Y_train)
    scale = np.max(log_y_train)
    y_train = log_y_train / scale
    y_val = np.log(Y_val) / scale

    model = keras_model_ori()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=128)

def do_experiment_ori_v2(X_train, Y_train: np.ndarray, X_val, Y_val: np.ndarray):
    print_header('ori', 'box-cox', 'sigmoid')
    scaler_pln = Pipeline(steps=[
            ('box-cox', PowerTransformer(method='box-cox')),
            ('[0,1]', MinMaxScaler(feature_range=(0,1)))
        ])
    
    Y_train_2d = Y_train.reshape((-1,1))            # Pipeline needs 2d array (#samples, #features)
    y_train = scaler_pln.fit_transform(Y_train_2d)
    y_train = y_train.reshape((-1,))                # keras fit() needs 1d array

    y_val = scaler_pln.transform(Y_val.reshape((-1,1))).reshape((-1,))

    model = keras_model_ori()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=128)

def do_experiment_hol_v1(X_train, Y_train: np.ndarray, X_val, Y_val: np.ndarray):
    print_header('hol', 'log_normal', 'sigmoid')
    log_y_train = np.log(Y_train)
    scale = np.max(log_y_train)
    y_train = log_y_train / scale
    y_val = np.log(Y_val) / scale

    model = keras_model_hol()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=128)

def do_experiment_hol_v2(X_train, Y_train: np.ndarray, X_val, Y_val: np.ndarray):
    print_header('hol', 'box-cox', 'sigmoid')
    scaler_pln = Pipeline(steps=[
            ('box-cox', PowerTransformer(method='box-cox')),
            ('[0,1]', MinMaxScaler(feature_range=(0,1)))
        ])
    
    Y_train_2d = Y_train.reshape((-1,1))            # Pipeline needs 2d array (#samples, #features)
    y_train = scaler_pln.fit_transform(Y_train_2d)
    y_train = y_train.reshape((-1,))                # keras fit() needs 1d array

    y_val = scaler_pln.transform(Y_val.reshape((-1,1))).reshape((-1,))

    model = keras_model_hol()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=128)

def print_header(f, ys, oa):
    print('',
          '*'*30,
          f'features={f}, y_scaler={ys}, output_activation={oa}',
          '*'*30,
          sep='\n')

def load_data() -> pd.DataFrame:
    txn_df = pd.read_parquet('train.parquet.gzip')
    store_state_df = pd.read_csv('store_states.csv')

    # split date strings to year, month, and date components
    yyyy_mm_dd_df = txn_df.Date.str.split('-', expand=True)
    yyyy_mm_dd_df.columns = 'Year Month Day'.split()
    txn_df2 = pd.concat((txn_df, yyyy_mm_dd_df), axis=1, sort=False)

    # Add State feature to each training datum
    txn_df3 = txn_df2.merge(store_state_df, how='left', on='Store')
    return txn_df3

def squash_holidays(df, colname:str) -> pd.Series:
    '''Squash holiday features.'''
    state_holiday_map = {'0': 0, 'a': 1, 'b': 1, 'c': 1}
    binary_state_holiday: pd.Series = df.StateHoliday.replace(state_holiday_map)
    holiday: pd.Series = binary_state_holiday | df.SchoolHoliday
    return holiday.rename(colname)

LabelEncoder_T = Dict[str, LabelEncoder]
def encode_category_inplace(df, features=List[str]) -> Tuple[pd.DataFrame, LabelEncoder_T]:
    les = { f: LabelEncoder().fit(df[f]) for f in features }
    for f in features:
        le = les[f]
        df[f] = le.transform(df[f])
    return df, les

def split(df, train_ratio: float, train_samples: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_size = int(train_ratio * len(df))
    train_df, val_df = df[:train_size], df[train_size:]
    return sample(train_df, samples=200*1000), val_df

def sample(df, samples: int) -> pd.DataFrame:
    num_rows = df.shape[0]
    indices = np.random.randint(num_rows, size=samples)
    return df.iloc[indices]

def df2list(df, features: List[str]) -> List[np.ndarray]:
    return [df[f].values for f in features]

def keras_model_ori(output_act: str='sigmoid') -> keras.models.Model:
    input_store = Input(shape=(1,))    # 1d vectors
    output_store = Embedding(1115, 10, # 1,115 stores embedded to 10d vectors
                             name='store_embedding')(input_store)
    output_store = Reshape(target_shape=(10,))(output_store)
    
    input_dow = Input(shape=(1,))
    output_dow = Embedding(7, 6, name='dow_embedding')(input_dow)
    output_dow = Reshape(target_shape=(6,))(output_dow)
    
    input_promo = Input(shape=(1,))
    output_promo = Dense(1)(input_promo)
    
    # Interesting choice to assume year as categorical. Just remember that this
    # constraints our model to recognize only years seen in the training set. 
    input_year = Input(shape=(1,))
    output_year = Embedding(3, 2, name='year_embedding')(input_year)
    output_year = Reshape(target_shape=(2,))(output_year)
    
    input_month = Input(shape=(1,))
    output_month = Embedding(12, 6, name='month_embedding')(input_month)
    output_month = Reshape(target_shape=(6,))(output_month)
    
    input_day = Input(shape=(1,))
    output_day = Embedding(31, 10, name='day_embedding')(input_day)
    output_day = Reshape(target_shape=(10,))(output_day)
    
    input_state = Input(shape=(1,))
    output_state = Embedding(12, 6, name='state_embedding')(input_state)
    output_state = Reshape(target_shape=(6,))(output_state)

    input_model = [input_store, input_dow, input_promo,
                   input_year, input_month, input_day, input_state]
    output_embeddings = [output_store, output_dow, output_promo,
                         output_year, output_month, output_day, output_state]
    output_model = Concatenate()(output_embeddings)
    output_model = Dense(1000, kernel_initializer='uniform')(output_model)
    output_model = PReLU()(output_model)
    output_model = Dense(500, kernel_initializer='uniform')(output_model)
    output_model = PReLU()(output_model)
    output_model = Dense(1)(output_model)
    output_model = Activation(output_act)(output_model)

    model = keras.models.Model(inputs=input_model, outputs=output_model)

    loss_fn = 'mean_absolute_error'
    print('Loss function:', loss_fn)
    model.compile(loss=loss_fn, optimizer='adam')
    return model

def keras_model_hol(output_act: str='sigmoid') -> keras.models.Model:
    input_store = Input(shape=(1,))
    output_store = Embedding(1115, 10, name='store_embedding')(input_store)
    output_store = Reshape(target_shape=(10,))(output_store)
    
    input_dow = Input(shape=(1,))
    output_dow = Embedding(7, 6, name='dow_embedding')(input_dow)
    output_dow = Reshape(target_shape=(6,))(output_dow)
    
    input_promo = Input(shape=(1,))
    output_promo = Dense(1)(input_promo)
    
    input_holiday = Input(shape=(1,))
    output_holiday = Dense(1)(input_holiday)

    input_year = Input(shape=(1,))
    output_year = Embedding(3, 2, name='year_embedding')(input_year)
    output_year = Reshape(target_shape=(2,))(output_year)
    
    input_month = Input(shape=(1,))
    output_month = Embedding(12, 6, name='month_embedding')(input_month)
    output_month = Reshape(target_shape=(6,))(output_month)
    
    input_day = Input(shape=(1,))
    output_day = Embedding(31, 10, name='day_embedding')(input_day)
    output_day = Reshape(target_shape=(10,))(output_day)
    
    input_state = Input(shape=(1,))
    output_state = Embedding(12, 6, name='state_embedding')(input_state)
    output_state = Reshape(target_shape=(6,))(output_state)

    input_model = [input_store, input_dow, input_promo, input_holiday,
                   input_year, input_month, input_day, input_state]
    output_embeddings = [output_store, output_dow, output_promo, output_holiday,
                         output_year, output_month, output_day, output_state]
    output_model = Concatenate()(output_embeddings)
    output_model = Dense(1000, kernel_initializer='uniform')(output_model)
    output_model = PReLU()(output_model)
    output_model = Dense(500, kernel_initializer='uniform')(output_model)
    output_model = PReLU()(output_model)
    output_model = Dense(1)(output_model)
    output_model = Activation(output_act)(output_model)

    model = keras.models.Model(inputs=input_model, outputs=output_model)

    loss_fn = 'mean_absolute_error'
    print('Loss function:', loss_fn)
    model.compile(loss=loss_fn, optimizer='adam')
    return model

if __name__ == '__main__':
    main()
