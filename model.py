import numpy as np
import io
import pandas as pd
import streamlit as st
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D,LSTM, Dense, Flatten, Dropout,AveragePooling1D,MaxPooling1D,RepeatVector,GlobalAveragePooling1D, BatchNormalization,Input
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.optimizers import Adam
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from skopt.space import Real, Integer
from skopt import BayesSearchCV
import plotly.graph_objs as go
import nbformat
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from scikeras.wrappers import KerasRegressor
import pickle
from keras.regularizers import l2
import pandas as pd
import plotly.graph_objs as go
import optuna
from tensorflow.keras.layers import (
    Dense, LSTM, Conv1D, MaxPooling1D, Flatten, 
    Dropout, BatchNormalization, GlobalAveragePooling1D
)

data = None 
window_size = None  
scaler = MinMaxScaler() 
def normalize_data(X_value, y_value):
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_scale_dataset = X_scaler.fit_transform(X_value)
    y_scale_dataset = y_scaler.fit_transform(y_value)
    return X_scale_dataset, y_scale_dataset

def get_data(data,input_dim, output_dim,target_column=None):
    if data is not None:
        data = data.copy()
            
        #change date/time
        date_column = data.columns[0] 
        data[date_column] = pd.to_datetime(data[date_column], format='%m/%d/%Y', errors='coerce')
        data = data.sort_values(by=[date_column], ascending=True)
        data.set_index(date_column, inplace=True)

        #Numerical data
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            data[column] = pd.to_numeric(data[column].astype(str).str.replace(',', ''), errors='coerce')
            
        #choose target column
        if target_column is None:
            target_column = numeric_columns[0] 
            
        X_value = data[[target_column]]
        y_value = data[[target_column]]
            
        # missing values
        X_value = X_value.fillna(method='ffill').fillna(method='bfill')
        y_value = y_value.fillna(method='ffill').fillna(method='bfill')
            
        # normailize data
        X_scale_dataset, y_scale_dataset = normalize_data(X_value, y_value)
            
        # create time series data
        X, y = get_X_y(X_scale_dataset, y_scale_dataset, input_dim, output_dim)
        print(f"X shape: {X.shape}, y shape: {y.shape}") 
        return X,y  
    
def get_X_y(X_data, y_data, input_dim, output_dim):
    X = []
    y = []     
    length = len(X_data)

    for i in range(length - input_dim - output_dim + 1): 
        X_value = X_data[i: i + input_dim][:, :]
        y_value = y_data[i + input_dim: i + (input_dim + output_dim)][:, 0] 
        if len(X_value) == input_dim and len(y_value) == output_dim:
            X.append(X_value)
            y.append(y_value)

    return np.array(X), np.array(y)

def split_data (X,y,test_ratio=0.2, val_ratio=0.2):

    X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size = test_ratio, random_state = 42,shuffle=False)

    X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size = val_ratio, random_state = 42,shuffle=False)

    return X_train, X_val, X_test, y_train, y_val, y_test

def train_and_evaluate(X_train, y_train, X_val, y_val,input_dim,output_dim, 
                      model_chose, epochs, batch_size, learning_rate, return_history=False):
    # Chọn mô hình dựa trên model_chose
    if model_chose == 'LSTM':
        model = LSTM_Model(input_dim=input_dim,output_dim = output_dim, learning_rate=learning_rate)
    elif model_chose == 'CNN':
        model = CNN_Model(input_dim=input_dim,output_dim = output_dim, learning_rate=learning_rate)
    elif model_chose == 'CNN-LSTM': 
        model = CNN_LSTM_Model(input_dim=input_dim,output_dim = output_dim, learning_rate=learning_rate)
    elif model_chose == 'LSTM-CNN':
        model = LSTM_CNN_Model(input_dim=input_dim,output_dim = output_dim, learning_rate=learning_rate)
    else:
        raise ValueError(f"Không tìm thấy mô hình: {model_chose}")

    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    )
    
    history = model.fit(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=(X_val, y_val), 
        verbose=2, 
        shuffle=False,
        callbacks=[early_stopping, checkpoint]
    )
    if return_history:
        return model, history
    return model

def train_sequential_additive(X_train, y_train, X_val, y_val,input_dim,output_dim,
                            first_model_name, second_model_name, 
                            epochs=50, batch_size=32, learning_rate=0.001,return_history=False):

    model1, history1 = train_and_evaluate(
        X_train, y_train,
        X_val, y_val,
        input_dim,
        output_dim,
        model_chose=first_model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        return_history=True
    )
    
    train_pred1 = model1.predict(X_train)
    val_pred1 = model1.predict(X_val)

    train_residuals = y_train - train_pred1
    val_residuals = y_val - val_pred1
    
    model2, history2 = train_and_evaluate(
        X_train, train_residuals,
        X_val, val_residuals,
        input_dim,
        output_dim,
        model_chose=second_model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate, 
        return_history=True
    )
    return model1, model2, history1, history2

def train_sequential_multi(X_train, y_train, X_val, y_val,input_dim,output_dim,
                            first_model_name, second_model_name, 
                            epochs=50, batch_size=32, learning_rate=0.001,return_history=False):

    model1, history1 = train_and_evaluate(
        X_train, y_train,
        X_val, y_val,
        input_dim,
        output_dim,
        model_chose=first_model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        return_history=True
    )
    
    train_pred1 = model1.predict(X_train)
    val_pred1 = model1.predict(X_val)
    
    train_residuals = y_train / train_pred1
    val_residuals = y_val / val_pred1
    
    model2, history2 = train_and_evaluate(
        X_train, train_residuals,
        X_val, val_residuals,
        input_dim,
        output_dim,
        model_chose=second_model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate, 
        return_history=True
    )
    return model1, model2, history1, history2

def rescale_data(test_predicted, actual_test):
    scaler = MinMaxScaler()
    scaler.fit(actual_test)
    actual_test = scaler.inverse_transform(actual_test)
    test_predicted = scaler.inverse_transform(test_predicted)
    return test_predicted, actual_test

def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_cvrmse(y_true,y_pred):
    mean_y = np.mean(y_true)
    cvrmse = (calculate_rmse(y_true,y_pred) / mean_y) * 100
    return cvrmse

def calculate_omega(y, y1, y2):
    numerator = np.sum((y2 - y1) * (y - y1))
    denominator = np.sum((y2 - y1) ** 2)
    omega = numerator / denominator if denominator != 0 else 0
    return omega

def calculate_square_deviation(y,y1,y2):
    d1 = np.sum((y-y1)**2)
    d2 = np.sum((y-y2)**2)
    d1 = max(d1,1e-10)
    d2 = max(d2,1e-10)
    omega = (1/d1)/(1/d1 + 1/d2)
    omega2 = (1/d2)/(1/d1 + 1/d2)
    return omega,omega2

def calculate_score (predictions,actual_test):
    mae = mean_absolute_error(actual_test, predictions)
    mse = mean_squared_error(actual_test, predictions)
    rmse = calculate_rmse(actual_test, predictions)
    cv_rmse = calculate_cvrmse(actual_test, predictions)
    return mae, mse, rmse, cv_rmse

def predict(model,X_test,y_test):
    predictions = model.predict(X_test)
    actual_test=y_test
    predictions, actual_test = rescale_data(predictions,actual_test)
    print(f"Warning: Predictions shape {predictions.shape} does not match y_test shape {y_test.shape}.")
    mae = mean_absolute_error(actual_test, predictions)
    mse = mean_squared_error(actual_test, predictions)
    rmse = calculate_rmse(actual_test, predictions)
    cv_rmse = calculate_cvrmse(actual_test, predictions)

        # Create visualization
    plt.figure(figsize=(12, 6))
    if predictions.shape[1] == 1:  # Nếu output_dim = 1
        plt.plot(actual_test, label='Actual')
        plt.plot(predictions, label='Predicted')
    else:  # Nếu output_dim > 1
        plt.plot(actual_test, label='Actual')
        for i in range(predictions.shape[1]):  # Vẽ mỗi cột output_dim
            plt.plot(predictions[:, i], label=f'Predicted {i+1}')
    plt.legend()
    plt.title('Actual vs Predicted Values')
        
        # Save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return predictions,actual_test, mae, mse, rmse, cv_rmse, buf

def combine_predictions_add(predictions_1, predictions_2, y_test):
    # Kết hợp dự đoán bằng cách cộng
    final_predictions = predictions_1 + predictions_2
    mae = mean_absolute_error(y_test, final_predictions)
    mse = mean_squared_error(y_test, final_predictions)
    rmse = calculate_rmse(y_test, final_predictions)
    cv_rmse = calculate_cvrmse(y_test, final_predictions)
    
    # Vẽ biểu đồ so sánh giá trị thực và dự đoán cuối cùng
    buf = plot_final_prediction(y_test, final_predictions, title="Biểu đồ Dự đoán - Mô hình Tuần Tự Cộng")
    
    return final_predictions,mae, mse, rmse, cv_rmse, buf

def combine_predictions_mul(predictions_1, predictions_2, y_test):
    # Kết hợp dự đoán bằng cách nhân
    final_predictions = predictions_1 * predictions_2
    mae = mean_absolute_error(y_test, final_predictions)
    mse = mean_squared_error(y_test, final_predictions)
    rmse = calculate_rmse(y_test, final_predictions)
    cv_rmse = calculate_cvrmse(y_test, final_predictions)
    
    # Vẽ biểu đồ so sánh giá trị thực và dự đoán cuối cùng
    buf = plot_final_prediction(y_test, final_predictions, title="Biểu đồ Dự đoán - Mô hình Tuần Tự Nhân")
    
    return final_predictions,mae, mse, rmse, cv_rmse, buf

def combine_predictions_parallel(predictions_1, predictions_2, y_test, input_dim, output_dim):
    omega, omega2 = calculate_square_deviation (y_test,predictions_1,predictions_2)
    print("gia tri omega:", omega)
    print("gia tri omega 2:", omega2)
    final_predict = omega * predictions_1 + omega2 * predictions_2

    # prediction_1_final = omega*predictions_1
    # prediction_2_final = omega2*predictions_2

    mae = mean_absolute_error(y_test, final_predict)
    mse = mean_squared_error(y_test, final_predict)
    rmse = calculate_rmse(y_test, final_predict)
    cv_rmse = calculate_cvrmse(y_test, final_predict)
    
    buf = plot_final_prediction(y_test, final_predict, title="Biểu đồ Dự đoán - Mô hình Song Song")
    
    # Vẽ biểu đồ so sánh giá trị thực và dự đoán cuối cùng
    # print("Actual Test:", actual_test[:input_dim])  # In 10 giá trị đầu tiên
    # print("Final Predict:", final_predict[:output_dim])
    # print("Predictions 1:", predictions_1[:output_dim])
    # print("Predictions 2:", predictions_2[:output_dim]) 
    return final_predict,mae, mse, rmse, cv_rmse, buf
def plot_predictions(actual, predicted, title="Actual vs Predicted Values"):
    plt.figure(figsize=(12, 6))
    if predicted.shape[1] == 1:
        plt.plot(actual, label='Actual')
        plt.plot(predicted, label='Predicted')
    else:
        plt.plot(actual, label='Actual')
        for i in range(predicted.shape[1]):
            plt.plot(predicted[:, i], label=f'Predicted {i+1}')
    plt.legend()
    plt.title(title)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def plot_final_prediction(y_true, y_pred, title):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="Giá trị thực")
    plt.plot(y_pred, label="Dự đoán")
    plt.xlabel("Thời gian")
    plt.ylabel("Giá trị")
    plt.title(title)
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf

def LSTM_Model(input_dim, output_dim, units=32, learning_rate=0.0001, activation='relu'):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(input_dim, 1), activation=activation))
    model.add(LSTM(units=units, activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dense(units=output_dim))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model
def CNN_Model(input_dim, output_dim, units=32, learning_rate=0.0001, activation='relu'): 
    model = Sequential() 
    model.add(Conv1D(filters=units, input_shape=(input_dim, 1), kernel_size=3, strides=1, padding='same', activation=activation)) 
    model.add(Conv1D(filters=units, kernel_size=3, strides=1, padding='same', activation=activation)) 
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same')) 
    model.add(GlobalAveragePooling1D())
    model.add(Dense(64, activation=activation)) 
    model.add(Dense(output_dim)) 
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse') 
    return model

def CNN_LSTM_Model(input_dim=10, output_dim=1, units=32, learning_rate=0.001, activation='relu'):
    model = Sequential()
    model.add(Conv1D(filters=64, input_shape=(input_dim, 1),
                    kernel_size=3, strides=1, padding='same', activation=activation))
    model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation=activation))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(output_dim))
    model.add(LSTM(units=units, activation=activation, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(units=units, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation=activation))
    model.add(Dense(units=output_dim))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def LSTM_CNN_Model(input_dim=10, output_dim=1, units=32, learning_rate=0.001, activation='relu'):
    model = Sequential()
    model.add(LSTM(units=units,return_sequences=True,input_shape=(input_dim, 1),activation=activation))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=64,kernel_size=3,padding='same',activation=activation))
    model.add(AveragePooling1D(pool_size = 2, strides = 2))
    model.add(Conv1D(filters=32,kernel_size=3,padding='same',activation=activation))
    model.add(AveragePooling1D(pool_size = 2, strides = 2))
    model.add(Conv1D(filters=16,kernel_size=3,padding='same',activation=activation))
    model.add(AveragePooling1D(pool_size = 2, strides = 2))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(32, activation=activation))
    model.add(Dense(units=output_dim))
    model.compile(optimizer=Adam(learning_rate=learning_rate),loss='mse')
    return model

def LSTM_CNN_Model(input_dim=10, output_dim=1, units=32, learning_rate=0.001, activation='relu'):
    model = Sequential()
    model.add(LSTM(units=128,return_sequences=True,input_shape=(input_dim, 1),activation=activation))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation=activation))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    model.add(Conv1D(filters=units,kernel_size=2,padding='same',activation=activation))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    model.add(Conv1D(filters=units,kernel_size=2,padding='same',activation=activation))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    model.add(Flatten())
    model.add(Dense(32, activation=activation))
    model.add(Dropout(0.2))
    model.add(Dense(units=output_dim))
    model.compile(optimizer=Adam(learning_rate=learning_rate),loss='mse')
    return model
    
def Optimize_RandomsearchCV(build_fn,param_dist,X_val,y_val):
    model_regressor = KerasRegressor(build_fn=build_fn,input_dim = 10 , output_dim=1, units =32, learning_rate=0.0001, activation = 'relu')
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

def Optimize_Optuna (trial,X_train,y_train,X_val,y_val,input_dim,output_dim,model_chose):
    batch_size = trial.suggest_categorical("batch_size",[16,32,64,128])
    epochs = trial.suggest_int("epochs" , 10, 100)
    units = trial.suggest_categorical("units",[16,32,64,128]) 
    learning_rate = trial.suggest_categorical("learning_rate",[0.0001, 0.001, 0.01]) 
    result,history = train_and_evaluate(
        X_train, y_train, X_val, y_val,  
        input_dim,
        output_dim, 
        model_chose, 
        epochs, 
        batch_size, 
        learning_rate, 
        return_history=True  )
    val_loss = history.history["val_loss"][-1]  
    return val_loss  
def Visualize(data_old, selected_column_name):
    if data_old is not None:
        # Tạo bản sao của dữ liệu
        data = data_old.copy()
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
    


   