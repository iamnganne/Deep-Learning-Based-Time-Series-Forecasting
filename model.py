import os
import numpy as np
import io
import pandas as pd
import random
import streamlit as st
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D,LSTM, Dense, Flatten, Dropout,SimpleRNN,MaxPooling1D,Embedding, Bidirectional,GRU,RepeatVector,TimeDistributed,GlobalAveragePooling1D, BatchNormalization,Input
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from keras.optimizers import Adam
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from skopt.space import Real, Integer
from skopt import BayesSearchCV
import plotly.graph_objs as go
import nbformat
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import ParameterGrid
from scikeras.wrappers import KerasClassifier, KerasRegressor
import pickle
from keras.regularizers import l2
import pandas as pd
import plotly.graph_objs as go
from tensorflow.keras.layers import (
    Dense, LSTM, Conv1D, MaxPooling1D, Flatten, 
    Dropout, BatchNormalization, GlobalAveragePooling1D
)
# Khai báo các biến ngoài class
data = None  # hoặc gán dữ liệu của bạn vào đây
window_size = None  # Gán giá trị của window_size (ví dụ: 7)
scaler = MinMaxScaler() 
def normalize_data(X_value, y_value):
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

        # Huấn luyện scaler và chuẩn hóa dữ liệu
    X_scale_dataset = X_scaler.fit_transform(X_value)
    y_scale_dataset = y_scaler.fit_transform(y_value)

    return X_scale_dataset, y_scale_dataset

def get_data(data,input_dim, output_dim,target_column=None):
        # Chuẩn bị dữ liệu
    if data is not None:
        data = data.copy()
            
            # Chuyển đổi cột thời gian
        date_column = data.columns[0]  # Giả định cột đầu tiên là thời gian
        data[date_column] = pd.to_datetime(data[date_column], format='%m/%d/%Y', errors='coerce')
        data = data.sort_values(by=[date_column], ascending=True)
        data.set_index(date_column, inplace=True)
        
            # Xử lý dữ liệu số
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            data[column] = pd.to_numeric(data[column].astype(str).str.replace(',', ''), errors='coerce')
            
            # Chọn cột mục tiêu
        if target_column is None:
            target_column = numeric_columns[0]  # Sử dụng cột số đầu tiên nếu không chỉ định
            
        X_value = data[[target_column]]
        y_value = data[[target_column]]
            
            # Xử lý missing values
        X_value = X_value.fillna(method='ffill').fillna(method='bfill')
        y_value = y_value.fillna(method='ffill').fillna(method='bfill')
            
            # Chuẩn hóa dữ liệu
        X_scale_dataset, y_scale_dataset = normalize_data(X_value, y_value)
            
            # Tạo chuỗi thời gian
        X, y = get_X_y(X_scale_dataset, y_scale_dataset, input_dim, output_dim)
        print(f"X shape: {X.shape}, y shape: {y.shape}") 
        return X,y  
def get_X_y(X_data, y_data, input_dim, output_dim):
    X = []
    y = []     
    length = len(X_data)

    for i in range(length - input_dim - output_dim + 1):  # Đảm bảo không vượt quá chiều dài
        X_value = X_data[i: i + input_dim][:, :]
        y_value = y_data[i + input_dim: i + (input_dim + output_dim)][:, 0] # Flatten để giảm chiều cho y
        if len(X_value) == input_dim and len(y_value) == output_dim:
            X.append(X_value)
            y.append(y_value)

    return np.array(X), np.array(y)

def split_data (X,y,test_ratio=0.2, val_ratio=0.2):
    # Chia dữ liệu thành 2 tập train_val và test 
    X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size = test_ratio, random_state = 42,shuffle=False)

    # Tiếp theo, chia tập train_val thành 2 tập train và validation set 
    X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size = val_ratio, random_state = 42,shuffle=False)

    return X_train, X_val, X_test, y_train, y_val, y_test
def get_model(model_name, input_dim, output_dim, learning_rate):
    if model_name == "LSTM":
        return LSTM_Model(input_dim=input_dim, output_dim=output_dim, learning_rate=learning_rate)
    elif model_name == "CNN":
        return CNN_Model(input_dim=input_dim, output_dim=output_dim, learning_rate=learning_rate)
    elif model_name == "LSTM-CNN":
        return LSTM_CNN_Model(input_dim=input_dim, output_dim=output_dim, learning_rate=learning_rate)
    else:
        raise ValueError(f"Không tìm thấy mô hình: {model_name}")
    
def train_and_evaluate(X_train, y_train, X_val, y_val, model, epochs, batch_size,learning_rate, return_history=False):
    # Early stopping để tránh overfitting
    early_stopping = EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
    
    # Checkpoint để lưu model tốt nhất
    checkpoint = ModelCheckpoint('best_model.keras',monitor='val_loss',save_best_only=True,mode='min')
    
    # Train model và lưu history
    history = model.fit(X_train, y_train, epochs=epochs,batch_size=batch_size,validation_data=(X_val, y_val),
                        verbose=2, shuffle=False, callbacks=[early_stopping, checkpoint])
    # Tải mô hình từ tệp .h5
    model = load_model('best_model.keras')

    return model, history

def rescale_data(test_predicted, actual_test):
    scaler = MinMaxScaler()
    scaler.fit(actual_test)
    # Chuyển đổi dữ liệu test_predicted và actual_test
    actual_test = scaler.inverse_transform(actual_test)
    #test_predicted = test_predicted.reshape(-1, 1)
    test_predicted = scaler.inverse_transform(test_predicted)
    return test_predicted, actual_test

def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_mape(y_true, y_pred):
    # Tránh chia cho 0 bằng cách thay thế các giá trị 0 trong y_true bằng một giá trị nhỏ (epsilon)
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return mape

def predict(model,X_test,y_test):
    predictions = model.predict(X_test)
    actual_test=y_test
    predictions, actual_test = rescale_data(predictions,actual_test)
    #tính toán các chỉ số đánh giá
    mae = mean_absolute_error(actual_test, predictions)
    mse = mean_squared_error(actual_test, predictions)
    rmse = calculate_rmse(actual_test, predictions)
    mape = calculate_mape(actual_test, predictions)
    #vẽ biểu đồ hiển thị
    plt.figure(figsize=(12, 6))
    if predictions.shape[1] == 1: 
        plt.plot(actual_test, label='Actual')
        plt.plot(predictions, label='Predicted')
    else: 
        for i in range(predictions.shape[1]):  # Vẽ mỗi cột output_dim
            plt.plot(actual_test[:, i], label=f'Actual {i+1}')
            plt.plot(predictions[:, i], label=f'Predicted {i+1}')
    plt.legend()
    plt.title('Actual vs Predicted Values')
    buf = io.BytesIO() 
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return predictions,actual_test, mae, mse, rmse, mape, buf
def LSTM_Model(input_dim=10, output_dim=1, units=32, learning_rate=0.0001, activation='tanh'):
    model = Sequential()
    model.add(Input(shape=(input_dim, 1)))
    model.add(LSTM(units=units, return_sequences=True, 
                  activation=activation,
                  kernel_regularizer=l2(0.01)))  # Thêm L2 regularization
    #model.add(Dropout(0.2))  # Thêm dropout
    model.add(LSTM(units=units, activation=activation,
                  kernel_regularizer=l2(0.01)))
    #model.add(Dropout(0.2))
    model.add(Dense(64, activation=activation,
                   kernel_regularizer=l2(0.01)))
    #model.add(Dropout(0.2))
    model.add(Dense(output_dim))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model
def CNN_Model(input_dim, output_dim, units=32, learning_rate=0.0001, activation='relu'): 
    model = Sequential() 
    model.add(Conv1D(filters=units, input_shape=(input_dim, 1), kernel_size=3, strides=1, padding='same', activation=activation)) 
    model.add(Conv1D(filters=units, kernel_size=3, strides=1, padding='same', activation=activation)) 
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same')) 
    model.add(Flatten()) 
    # model.add(GlobalAveragePooling1D())
    model.add(Dense(64, activation=activation)) 
    model.add(Dense(output_dim)) 
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse') 
    return model

def LSTM_CNN_Model(input_dim=10, output_dim=1, units=32, learning_rate=0.001, activation='relu'):
    model = Sequential()
    model.add(LSTM(units=128,return_sequences=True,input_shape=(input_dim, 1),activation=activation))
    model.add(LSTM(units=64,return_sequences=True,activation=activation))
    model.add(Conv1D(filters=32,kernel_size=2,padding='same',activation=activation))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=16,kernel_size=2,padding='same',activation=activation))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(32, activation=activation))
    model.add(Dropout(0.2))
    model.add(Dense(units=output_dim))
    model.compile(optimizer=Adam(learning_rate=learning_rate),loss='mse')
    return model

def Optimize_RandomsearchCV(build_fn,param_dist,X_val,y_val,input_dim,output_dim):
    model_regressor = KerasRegressor(build_fn=build_fn, units =32, learning_rate=0.0001, activation = 'relu',input_dim = input_dim, output_dim = output_dim)
    random_search = RandomizedSearchCV(model_regressor, param_distributions=param_dist,cv = 3, n_iter=10, n_jobs=-1,scoring='neg_mean_squared_error')
    random_search.fit(X_val,y_val)
    results = random_search.cv_results_
    best_params = random_search.best_params_
    return results, best_params
def Optimize_GridsearchCV(build_fn,param_grid,X_val,y_val):
    model_regressor = KerasRegressor(build_fn=build_fn,input_dim = 10 , output_dim=1, units =32, learning_rate=0.0001, activation = 'relu')
    grid_search = GridSearchCV(model_regressor, param_grid=param_grid,cv = 3, n_jobs=-1,scoring='neg_mean_squared_error')
    grid_search.fit(X_val,y_val)
    results = grid_search.cv_results_
    best_params = grid_search.best_params_
    return results, best_params
def Optimize_Bayesian(build_fn,param_space,X_val,y_val):
    model_regressor = KerasRegressor(build_fn=build_fn,input_dim = 10 , output_dim=1, units =32, learning_rate=0.0001, activation = 'relu')
    bayesian_search = BayesSearchCV(model_regressor, search_spaces=param_space,n_iter = 10,cv = 3, n_jobs=-1,scoring='neg_mean_squared_error')
    bayesian_search.fit(X_val,y_val)
    results = bayesian_search.cv_results_
    best_params = bayesian_search.best_params_
    return results, best_params

# Hàm Visualize với kiểm tra và in thông tin chi tiết
def Visualize(data_old, selected_column_name):
    if data_old is not None:
        # Tạo bản sao của dữ liệu
        data = data_old.copy()
        
        # Lấy tên cột ngày (giả sử cột ngày nằm ở đầu tiên)
        date_column = data.columns[0]

        # Chuyển đổi cột ngày thành datetime, nếu có lỗi thì sẽ bỏ qua và đưa giá trị NaT
        try:
            data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
        except Exception as e:
            print(f"Error converting date column: {e}")
            return None
        
        # Loại bỏ các dòng có giá trị ngày lỗi (NaT)
        data.dropna(subset=[date_column], inplace=True)

        # Kiểm tra nếu cột được chọn có trong dữ liệu hay không
        if selected_column_name not in data.columns:
            print(f"Column '{selected_column_name}' not found in data. Available columns: {data.columns.tolist()}")
            return None

        # Thiết lập cột ngày làm chỉ mục
        data.set_index(date_column, inplace=True)

        # In ra dữ liệu đầu tiên để kiểm tra
        print(f"Data after processing:\n{data.head()}")

        # Vẽ biểu đồ lineplot sử dụng seaborn và matplotlib
        plt.figure(figsize=(20,8))
        sns.lineplot(x=data.index, y=selected_column_name, data=data)
        plt.title(f'Lineplot of {selected_column_name} Vs Time')
        plt.xlabel('Date')
        plt.ylabel(selected_column_name)
        plt.grid(True)

        # Hiển thị biểu đồ
        st.pyplot(plt)

def save_optimize_results(results, best_params, model_name, optimize_method):
    """Lưu kết quả optimize"""
    optimize_data = {
        'results': results,
        'best_params': best_params,
        'timestamp': pd.Timestamp.now()
    }
    filename = f'optimize_{model_name}_{optimize_method}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(optimize_data, f)
    return filename

def load_optimize_results(model_name, optimize_method):
    """Tải kết quả optimize đã lưu"""
    filename = f'optimize_{model_name}_{optimize_method}.pkl'
    try:
        with open(filename, 'rb') as f:
            optimize_data = pickle.load(f)
        return optimize_data['results'], optimize_data['best_params']
    except FileNotFoundError:
        return None, None
