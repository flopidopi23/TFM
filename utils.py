import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler

"""This script contains utility functions for data processing and evaluation metrics.
It includes functions to divide train from test, evaluate model performance using MAE and RMSE 
and plot the results. Thi last function is used to plot the training, test and forecasted data.
"""
def eval_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # remove nan or not valid values like infinity
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan, np.nan
    y_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    mae = mean_absolute_error(y_clean, y_pred_clean)
    rmse = root_mean_squared_error(y_clean, y_pred_clean)
    return mae, rmse

def train_data(df,location,date):
    df = df[df['location']==location]
    df = df.set_index('truth_date')
    train = df[df.index<=date]
    test = df[df.index>date]
    return train,test

def create_features(data):
    """
    Create additional features for the non sequential dataset.
    """
    data = data.copy()

    # Extract year, month, day, weekday, and week from 'truth_date'
    data['year'] = data.index.year
    data['month'] = data.index.month

    week = data['year_week'].str.split('-W').str[1]
    data['week'] = week.astype(int)
    for h in range(1,5):
        data[f'lag_value_{h}'] = data['value'].shift(h)
        data[f'lag_humidity_{h}'] = data['relative_humidity_2m'].shift(h)
        data[f'lag_temp_max_{h}'] = data['temperature_2m_max'].shift(h)
        data[f'lag_temp_min_{h}'] = data['temperature_2m_min'].shift(h)
    data = data.dropna()
    # Convert cyclical categorical variables to category type
    data['month_sin'] = np.sin(2 * np.pi * data['month']/12)
    data['month_cos'] = np.cos(2 * np.pi * data['month']/12)
    data['week_sin'] = np.sin(2 * np.pi * data['week']/52)
    data['week_cos'] = np.cos(2 * np.pi * data['week']/52)
    data = data.drop(columns=['month', 'week','year_week'])
    data['week_mas_1'] = data['value'].shift(-1)
    data['week_mas_2'] = data['value'].shift(-2)
    data['week_mas_3'] = data['value'].shift(-3)
    return data



def train_data_ml(df,location,date):
    df = df[df['location']==location]
    df = df.set_index('truth_date')
    df = create_features(df)
    train = df[df.index<=date]
    test = df[df.index>date]
    return train,test


def plot_train_test(train, test, model_name, location,folder):
    plt.figure(figsize=(15, 7))

    # Plot actual data
    plt.plot(train.index, train["value"], color='blue', label='Training Data')
    plt.plot(test.index, test["value"], color='orange', label='Actual Test Data')

    # Unified forecast column names
    forecast_horizons = {
        "prediction_1_weeks": ("1 Week Ahead", "green", 0),
        "prediction_2_weeks": ("2 Weeks Ahead", "red", 1),
        "prediction_3_weeks": ("3 Weeks Ahead", "purple", 2),
        "prediction_4_weeks": ("4 Weeks Ahead", "brown", 3),
    }

    for col, (label, color, shift_val) in forecast_horizons.items():
        if col in test.columns:
            # Shift predictions forward by their horizon
            plt.plot(test.index, test[col].shift(shift_val), linestyle='--', color=color, label=f'{model_name} {label}')

    plt.legend(loc='upper left')
    plt.title(f"{model_name} Forecasting – {location}")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    plt.savefig(f"plots_{folder}/plot_{location}_{model_name}.jpg", format='jpg', dpi=300)

def train_data_rnn(df,location,date):
    data = df.copy()
    data = data[data['location']==location]
    data = data.set_index('truth_date')
    data = create_features(data)
    train = data[data.index<=date]
    test = data[data.index>date]
    scal = MinMaxScaler()
    var = ['relative_humidity_2m', 'temperature_2m_max',
       'temperature_2m_min', 'lag_value_1', 'lag_humidity_1',
       'lag_temp_max_1', 'lag_temp_min_1', 'lag_value_2', 'lag_humidity_2',
       'lag_temp_max_2', 'lag_temp_min_2', 'lag_value_3', 'lag_humidity_3',
       'lag_temp_max_3', 'lag_temp_min_3', 'lag_value_4', 'lag_humidity_4',
       'lag_temp_max_4', 'lag_temp_min_4']
    scal_2 = MinMaxScaler()
    var2 = ['value', 'week_mas_1', 'week_mas_2', 'week_mas_3']
    train[var2] = scal_2.fit_transform(train[var2])
    test[var2] = scal_2.transform(test[var2])
    
    train[var] = scal.fit_transform(train[var])
    test[var] = scal.transform(test[var])
    return train,test,scal_2,scal

def generate_sequences(df, target_col='value', n_input=16,n_output=4):
    features = df.drop(columns=[target_col]).values
    target = df[target_col].values

    X, y = [], []
    for i in range(len(df) - n_input - n_output + 1):
        X.append(features[i:i+n_input])
        y.append(target[i+n_input:i+n_input+n_output])
    return np.array(X), np.array(y)
