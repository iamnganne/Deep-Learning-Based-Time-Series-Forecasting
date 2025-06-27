import streamlit as st
import numpy as np
import pandas as pd
import optuna
from model import normalize_data, get_data, get_X_y, split_data, train_and_evaluate, calculate_cvrmse, calculate_rmse, predict, \
                LSTM_Model, CNN_Model, CNN_LSTM_Model, LSTM_CNN_Model,Optimize_Optuna,calculate_omega,combine_predictions_parallel,combine_predictions_mul,rescale_data,train_sequential_additive,train_sequential_multi, \
                Optimize_RandomsearchCV, Visualize,Optimize_GridsearchCV,combine_predictions_add, save_optimize_results, load_optimize_results
import plotly.graph_objs as go

from scipy.stats import randint
from sklearn.preprocessing import MinMaxScaler
import time
from skopt.space import Real, Integer,Categorical
import matplotlib.pyplot as plt
st.title('Dự báo chuỗi thời gian bằng mô hình lai ghép')
st.info('This app builds a hybrid model')
scaler = MinMaxScaler()
# Initialize session_state variables
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "data" not in st.session_state:
    st.session_state.data = None
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
if "trained_model1" not in st.session_state:
    st.session_state.trained_model1 = None
if "trained_model_2" not in st.session_state:
    st.session_state.trained_model_2 = None
if "tested_model" not in st.session_state:
    st.session_state.tested_model = None
if "mae" not in st.session_state:
    st.session_state.mae = None

if "ts_model" not in st.session_state:
    st.session_state.ts_model = None
if 'optimize_configs' not in st.session_state:
    st.session_state['optimize_configs'] = []
if 'best_params' not in st.session_state:
    st.session_state.best_params = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
# Ensure Visualize is properly assigned
if 'Visualize' not in st.session_state:
    st.session_state.Visualize = Visualize
if 'get_data' not in st.session_state:
    st.session_state.get_data = get_data
if 'get_X_y' not in st.session_state:
    st.session_state.get_X_y = get_X_y    
if 'split_data' not in st.session_state:
    st.session_state.split_data = split_data    
if 'train_and_evaluate' not in st.session_state:
    st.session_state.train_and_evaluate = train_and_evaluate
if 'train_sequential_multi' not in st.session_state:
    st.session_state.train_sequential_multi = train_sequential_multi
if 'train_sequential_additive' not in st.session_state:
    st.session_state.train_sequential_additive = train_sequential_additive
if 'predict' not in st.session_state:
    st.session_state.predict = predict 
if 'rescale_data' not in st.session_state:
    st.session_state.rescale_data = rescale_data        
if 'LSTM_Model' not in st.session_state:
    st.session_state.LSTM_Model = LSTM_Model 
if 'CNN_Model' not in st.session_state:
    st.session_state.CNN_Model = CNN_Model
if 'calculate_omega' not in st.session_state:
    st.session_state.calculate_omega = calculate_omega
if 'combine_predictions_parallel' not in st.session_state:
    st.session_state.combine_predictions_parallel = combine_predictions_parallel    
if 'Optimize_RandomsearchCV' not in st.session_state:
    st.session_state.Optimize_RandomsearchCV = Optimize_RandomsearchCV 
if 'Optimize_GridsearchCV' not in st.session_state:
    st.session_state.Optimize_GridsearchCV = Optimize_GridsearchCV 
if 'Optimize_Optuna' not in st.session_state:
    st.session_state.Optimize_Optuna = Optimize_Optuna 
if 'combine_predictions_add' not in st.session_state:
    st.session_state.combine_predictions_add = combine_predictions_add
if 'combine_predictions_mul' not in st.session_state:
    st.session_state.combine_predictions_mul = combine_predictions_mul
if 'optimize_button_clicked' not in st.session_state:
    st.session_state.optimize_button_clicked = False
if 'predict_button_clicked' not in st.session_state:
    st.session_state.predict_button_clicked = False
if 'training_button_clicked' not in st.session_state:
    st.session_state.training_button_clicked = False
if 'testing_button_clicked' not in st.session_state:
    st.session_state.testing_button_clicked = False

MODEL_CHOSES = {
    "Mô hình CNN": "CNN",
    "Mô hình LSTM": "LSTM", 
}
MODEL_AVAILABLE = {
    "Mô hình CNN": "CNN",
    "Mô hình LSTM": "LSTM", 
}
MODEL_TYPES = {
    "Mô hình đơn": "Mô hình đơn",
    "Mô hình tuần tự": "Mô hình tuần tự",
    "Mô hình tuần tự cộng": "Mô hình tuần tự cộng",
    "Mô hình tuần tự nhân": "Mô hình tuần tự nhân",
    "Mô hình song song": "Mô hình song song"
}
MODEL_SEQUENTIAL = {
    "Mô hình CNN-LSTM": "CNN-LSTM",
    "Mô hình LSTM-CNN": "LSTM-CNN",
}

# Khởi tạo trạng thái mở của các expander trong session_state
if 'optimize_expanded' not in st.session_state:
    st.session_state.optimize_expanded = False
if 'training_expanded' not in st.session_state:
    st.session_state.training_expanded = False
if 'testing_expanded' not in st.session_state:
    st.session_state.testing_expanded = False

with st.expander('Data'):
    
    st.header("CHỌN FILE DỮ LIỆU ĐỂ KIỂM TRA MÔ HÌNH")
    
    uploaded_file = st.file_uploader("", type=['csv'])
    
    # Kiểm tra nu file đã được tải lên
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        try:
            # Đọc dữ liệu từ file CSV
            data = pd.read_csv(uploaded_file)
            if len(data.columns) < 2:
                st.error("File CSV phải có ít nhất 2 cột")
            else:
                st.session_state.data = data
                
                # Chọn cột để dự đoán
                target_column = st.selectbox(
                    "Chọn cột dữ liệu để dự đoán",
                    data.select_dtypes(include=[np.number]).columns
                )
                
                # Kiểm tra và xác nhận Visualize là một hàm hợp lệ
                if callable(st.session_state.Visualize):
                    # Hiển thị biểu đồ ban đầu
                    Visualize(data, target_column)
                    
                else:
                    st.error("Hàm Visualize không hợp lệ.")

                # Lưu cột mục tiêu vào session state
                st.session_state.target_column = target_column
                
                st.success("File đã được tải lên thành công!")
                st.write("Dữ liệu mẫu:")
                st.write(data.head())
        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi đọc file: {e}")


with st.sidebar: 
    
    st.header('Features')
    model_type = st.selectbox("Chọn loại mô hình",list(MODEL_TYPES.keys()))
    
    if model_type == "Mô hình tuần tự":
        model_chose = st.selectbox("Chọn mô hình", list(MODEL_SEQUENTIAL.keys()))
    else:
        model_chose = st.selectbox("Chọn mô hình", list(MODEL_CHOSES.keys()))
        if model_type != "Mô hình đơn":
            model_chose1 = [model for model in MODEL_CHOSES.keys() if MODEL_CHOSES[model] != model_chose]
            combined_model = st.selectbox('Chọn mô hình kết hợp',model_chose1, key=f"combined_model_{model_type}" )
        
    optimize_model = st.selectbox('Mô hình Optimize',('RandomizedSearchCV','GridSearchCV','Optuna Optimization'))
    col1, col2 = st.columns(2)
    with col1:
        input_date = st.number_input("Chọn số ngày dùng dự đoán:", min_value=0, max_value=100, value=0, step=1)
        train_size = st.slider("Tỷ lệ Training %:", min_value=0, max_value=100, value=70, step=10)
        val_percent = st.slider("Tỷ lệ Validation %:", min_value=0, max_value=100, value=20, step=5)
    with col2:
        output_date = st.number_input("Chọn số ngày muốn dự đoán:", min_value=0, max_value=100, value=0, step=1)
        test_size = st.slider("Tỷ lệ Testing %:", min_value=0, max_value=100, value=30, step=10)

    val_size = (train_size * val_percent) / 100
    actual_train_size = train_size - val_size
    
    st.write(f"Tỷ lệ cuối cùng: Train={actual_train_size}%, Val={val_size}%, Test={test_size}%")

    col1, col2 = st.columns(2)
    with col1:
        optimize_button = st.button('Optimize')
        training_button = st.button('Training')
    with col2:
        testing_button = st.button('Testing')
        predict_button = st.button('Predict')

    # Update session state for all buttons
    if optimize_button:
        st.session_state.optimize_button_clicked = True
    if training_button:
        st.session_state.training_button_clicked = True
    if testing_button:
        st.session_state.testing_button_clicked = True
    if predict_button:
        st.session_state.predict_button_clicked = True


# Optimize expander
with st.expander("Optimize value", expanded=True): 
    if st.session_state.optimize_button_clicked:    
        with st.spinner('Đang tối ưu hóa mô hình...'):
            start_train_time = time.time()
            # Định nghĩa tham số cho RandomizedSearchCV
            param_dist = {
                'units': [32, 64, 128],
                'learning_rate': [0.0001, 0.001, 0.01],
                'epochs': randint(10, 100),
                'batch_size': [16, 32, 64, 128, 256]
            }
            param_grid = {
                'units': [32, 64, 128],
                'learning_rate': [0.0001, 0.001, 0.01],
                'epochs': [10,20,30,40,50], 
                'batch_size': [16, 32, 64, 128, 256]
            }

            # Xác định đúng tên mô hình để lưu kết quả
            if model_type == "Mô hình tuần tự":
                model_name = MODEL_SEQUENTIAL[model_chose]  # Lấy tên mô hình từ MODEL_SEQUENTIAL
            else:
                model_name = MODEL_CHOSES[model_chose]      # Lấy tên mô hình từ MODEL_CHOSES

            if target_column is not None and data is not None and split_data is not None:
                X, y = st.session_state.get_data(data,input_dim = input_date, output_dim = output_date, target_column=st.session_state.target_column)
                X_train, X_val, X_test, y_train, y_val, y_test = st.session_state.split_data(
                    X, y, 
                    test_ratio = test_size / 100,
                    val_ratio = val_size / 100
                )

                if model_type == "Mô hình tuần tự":
                    if MODEL_SEQUENTIAL[model_chose] == "CNN-LSTM":
                        model_fn = CNN_LSTM_Model
                    else: 
                        model_fn = LSTM_CNN_Model

                else:
                    if MODEL_CHOSES[model_chose] == "LSTM":
                        model_fn = LSTM_Model
                    elif MODEL_CHOSES[model_chose] == "CNN":
                        model_fn = CNN_Model
                # Optimize model
                if optimize_model == 'RandomizedSearchCV':
                    results, best_params = st.session_state.Optimize_RandomsearchCV(
                        build_fn=model_fn,
                        param_dist=param_dist,
                        X_val=X_val,
                        y_val=y_val
                    )
                elif optimize_model == 'GridSearchCV':
                    results, best_params = st.session_state.Optimize_GridsearchCV(
                        build_fn=model_fn,
                        param_grid=param_grid,
                        X_val=X_val,
                        y_val=y_val
                    )

                elif optimize_model == "Optuna Optimization":
                    if model_type == "Mô hình đơn":                    
                        study = optuna.create_study(direction = "minimize")
                        study.optimize(lambda trial: Optimize_Optuna(trial,X_train, y_train, X_val, y_val,
                                                                    input_date,output_date, model_chose = MODEL_CHOSES.get(model_chose, model_chose)),
                                                                    n_trials=50)
                        best_params = study.best_params
                        best_loss = study.best_value
                        results = study.trials_dataframe()
                    else: 
                        study = optuna.create_study(direction = "minimize")
                        study.optimize(lambda trial: Optimize_Optuna(trial,X_train, y_train, X_val, y_val,
                                                                    input_date,output_date, model_chose = MODEL_SEQUENTIAL.get(model_chose, model_chose)),
                                                                    n_trials=50)
                        best_params = study.best_params
                        best_loss = study.best_value
                        results = study.trials_dataframe()

                # Hiển thị kết quả tối ưu hóa sau khi hoàn thành
                if results is not None and best_params is not None:
                    # Lưu kết quả optimize
                    filename = save_optimize_results(
                        results, 
                        best_params, 
                        model_name, 
                        optimize_model
                    )
                    end_train_time = time.time()
                    optimize_time = end_train_time - start_train_time
                    st.success("Tối ưu hóa hoàn tất!")
                    st.write (f"Thời gian tối ưu hóa: {optimize_time}")
                            
                    # Hiển thị best parameters
                    st.subheader("Tham số tốt nhất:")
                    for param, value in best_params.items():
                        st.write(f"- {param}: {value}")

                    st.session_state['optimize_configs'].append({
                        'model_type': model_chose,
                        'best_params': best_params}
                    )        
                    st.session_state.best_params = best_params
                    st.session_state.optimize_button_clicked = False 
                else:
                    st.error("Quá trình tối ưu hóa không thành công. Vui lòng thử lại với các tham số khác.")

            # Thêm checkbox để chọn sử dụng kết quả đã lưu
            use_saved_results = st.checkbox("Sử dụng kết quả optimize đã lưu (nếu có)")

            if use_saved_results:
                # Xác định đúng tên mô hình để load kết quả
                if model_type == "Mô hình tuần tự":
                    model_name = MODEL_SEQUENTIAL[model_chose]
                else:
                    model_name = MODEL_CHOSES[model_chose]
                    
                saved_results, saved_params = load_optimize_results(
                    model_name,  # Sử dụng model_name đã xác định
                    optimize_model
                )
                if saved_results is not None and saved_params is not None:
                    st.success("Đã tải kết quả optimize từ file!")
                    # Hiển thị kết quả đã lưu
                    st.subheader("Tham số tốt nhất đã lưu:")
                    for param, value in saved_params.items():
                        st.write(f"- {param}: {value}")
                else:
                    st.warning("Không tìm thấy kết quả optimize đã lưu")

# Training expander
with st.expander("Training Model", expanded=True):  # Luôn mở
    if st.session_state.training_button_clicked:
        try:
                    # Xác định tên mô hình cho việc load optimize results
            if model_type == "Mô hình tuần tự":
                model_name = MODEL_SEQUENTIAL[model_chose]
                saved_results, saved_params = load_optimize_results(
                    model_name, 
                    optimize_model
                )
            elif model_type in ["Mô hình tuần tự cộng", "Mô hình tuần tự nhân", "Mô hình song song"]:
                model_name = MODEL_CHOSES[model_chose]
                model_name1 = MODEL_CHOSES[combined_model]
                
                # Load tối ưu hóa cho mô hình đầu tiên
                saved_results, saved_params = load_optimize_results(
                    model_name, optimize_model
                )
                # Load tối ưu hóa cho mô hình thứ hai
                saved_results_1, saved_params_1 = load_optimize_results(
                    model_name1, optimize_model
                )

            else:
                model_name = MODEL_CHOSES[model_chose]
                saved_results, saved_params = load_optimize_results(
                    model_name,
                    optimize_model
                )

            if saved_params is not None :
                with st.spinner('Đang huấn luyện mô hình với tham số tối ưu...'):
                    # Prepare data
                    data = st.session_state.data
                    target_column = st.session_state.target_column
                    X, y = st.session_state.get_data(data, input_dim = input_date, output_dim = output_date,target_column=target_column)
                    X_train, X_val, X_test, y_train, y_val, y_test = st.session_state.split_data(
                        X, y, 
                        test_ratio = test_size / 100,
                        val_ratio = val_size / 100
                    )

                    start_train_time = time.time()
                    
                    # Hiển thị tham số tối ưu được sử dụng

                    # st.subheader(f"Tham số tối ưu cho quá trình huấn luyện - {model_name}:")
                    # for param, value in saved_params.items():
                    #     st.write(f"- {param}: {value}")
                    # if model_type in [ "Mô hình song song"]:
                    #     # Kiểm tra sự tồn tại của tham số tối ưu cho mô hình thứ hai
                    #     if saved_params_1:
                    #         st.subheader(f"Tham số tối ưu cho quá trình huấn luyện - {model_name1}:")
                    #         for param, value in saved_params_1.items():
                    #             st.write(f"- {param}: {value}")
                    #     else:
                    #         st.warning("Không tìm thấy tham số tối ưu cho mô hình thứ hai!")
                    # Train và lưu history
                    # Truyền trực tiếp tên mô hình từ MODEL_SEQUENTIAL hoặc MODEL_CHOSES
                    if model_type == "Mô hình đơn" or model_type == "Mô hình tuần tự":
                        model, history = train_and_evaluate(
                            X_train, y_train,
                            X_val, y_val,
                            input_dim = input_date,
                            output_dim = output_date,
                            model_chose=model_name,  # Truyền trực tiếp tên mô hình
                            epochs=saved_params.get('epochs', 50),
                            batch_size=saved_params.get('batch_size', 32),
                            learning_rate=saved_params.get('learning_rate', 0.001),
                            return_history=True
                        )
                        
                        end_train_time = time.time()
                        train_time = end_train_time - start_train_time
                        
                        # Vẽ biểu đồ loss/accuracy
                        fig_metrics = go.Figure()
                        fig_metrics.add_trace(go.Scatter(y=history.history['loss'], name='Train Loss'))
                        fig_metrics.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))
                        fig_metrics.update_layout(title='Loss theo epoch',
                                            xaxis_title='Epoch',
                                            yaxis_title='Loss')
                        st.plotly_chart(fig_metrics)
                        
                        # Hiển thị kết quả cuối cùng
                        st.subheader("Kết quả huấn luyện:")
                        st.write(f"- Thời gian huấn luyện: {train_time:.2f} giây")
                        st.write(f"- Số epoch hoàn thành: {saved_params['epochs']}")
                        st.write(f"- Loss cuối cùng: {history.history['loss'][-1]:.4f}")
                        st.write(f"- Validation Loss cuối cùng: {history.history['val_loss'][-1]:.4f}")
                        
                        # Save model to session state
                        st.session_state.trained_model = model
                    elif model_type == "Mô hình song song":
                        model, history = st.session_state.train_and_evaluate(
                                            X_train, y_train,
                                            X_val, y_val,
                                            input_dim = input_date,
                                            output_dim = output_date,
                                            model_chose=model_name,  # Mô hình đầu tiên
                                            epochs=saved_params.get('epochs', 50),
                                            batch_size=saved_params.get('batch_size', 32),
                                            learning_rate=saved_params.get('learning_rate', 0.001),
                                            return_history=True
                                            )
                        model1, history1 = st.session_state.train_and_evaluate(
                                            X_train, y_train,
                                            X_val, y_val,
                                            input_dim = input_date,
                                            output_dim = output_date,
                                            model_chose=model_name1,  # Mô hình thứ hai
                                            epochs=saved_params_1.get('epochs', 50),
                                            batch_size=saved_params_1.get('batch_size', 32),
                                            learning_rate=saved_params_1.get('learning_rate', 0.001),
                                            return_history=True
                                            )
                        end_train_time = time.time()
                        train_time = end_train_time - start_train_time
                        
                        # Vẽ biểu đồ loss/accuracy
                        fig_metrics = go.Figure()
                        fig_metrics.add_trace(go.Scatter(y=history.history['loss'], name=f'Train Loss {model_name}'))
                        fig_metrics.add_trace(go.Scatter(y=history1.history['loss'], name=f'Train Loss {model_name1} '))
                        fig_metrics.add_trace(go.Scatter(y=history.history['val_loss'], name=f'Validation Loss {model_name}'))
                        fig_metrics.add_trace(go.Scatter(y=history1.history['val_loss'], name=f'Validation Loss {model_name1}'))
                        fig_metrics.update_layout(title='Loss theo epoch',
                                            xaxis_title='Epoch',
                                            yaxis_title='Loss')
                        st.plotly_chart(fig_metrics)
                        
                        # Hiển thị kết quả cuối cùng
                        st.subheader("Kết quả huấn luyện:")
                        st.write(f"- Thời gian huấn luyện: {train_time:.2f} giây")
                        st.write(f"- Số epoch hoàn thành: {saved_params['epochs']}")
                        st.write(f"- Final Loss: {history.history['loss'][-1]:.4f}")
                        st.write(f"- Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
                        
                        # Save model to session state
                        st.session_state.trained_model = model
                        st.session_state.trained_model1 = model1

                    elif model_type == "Mô hình tuần tự cộng":
                        model1, model2, history1, history2 = st.session_state.train_sequential_additive(
                                            X_train, y_train,
                                            X_val, y_val,
                                            input_dim = input_date,
                                            output_dim = output_date,
                                            first_model_name=model_name,
                                            second_model_name = model_name1, # Mô hình đầu tiên
                                            epochs=saved_params.get('epochs', 50),
                                            batch_size=saved_params.get('batch_size', 32),
                                            learning_rate=saved_params.get('learning_rate', 0.001),
                                            return_history=True
                                            )
                        
                        end_train_time = time.time()
                        train_time = end_train_time - start_train_time
                        
                        # Vẽ biểu đồ loss/accuracy
                        fig_metrics = go.Figure()
                        fig_metrics.add_trace(go.Scatter(y=history1.history['loss'], name=f'Train Loss {model_name}'))
                        fig_metrics.add_trace(go.Scatter(y=history2.history['loss'], name=f'Train Loss {model_name1} '))
                        fig_metrics.add_trace(go.Scatter(y=history1.history['val_loss'], name=f'Validation Loss {model_name}'))
                        fig_metrics.add_trace(go.Scatter(y=history2.history['val_loss'], name=f'Validation Loss {model_name1}'))

                        fig_metrics.update_layout(title='Loss theo epoch',
                                            xaxis_title='Epoch',
                                            yaxis_title='Loss')
                        st.plotly_chart(fig_metrics)
                        
                        # Hiển thị kết quả cuối cùng
                        st.subheader("Kết quả huấn luyện:")
                        st.write(f"- Thời gian huấn luyện: {train_time:.2f} giây")
                        st.write(f"- Số epoch hoàn thành: {saved_params['epochs']}")
                        st.write(f"- Final Loss mô hình {model_name}: {history1.history['loss'][-1]:.4f}")
                        st.write(f"- Final Validation Loss mô hình {model_name}: {history1.history['val_loss'][-1]:.4f}")
                        st.write(f"- Final Loss mô hình {model_name1}: {history2.history['loss'][-1]:.4f}")
                        st.write(f"- Final Validation Loss mô hình {model_name1}: {history2.history['val_loss'][-1]:.4f}")

                        st.session_state.trained_model = model1
                        st.session_state.trained_model1 = model2
                    elif model_type == "Mô hình tuần tự nhân":
                        model1, model2, history1, history2 = st.session_state.train_sequential_multi(
                                            X_train, y_train,
                                            X_val, y_val,
                                            input_dim = input_date,
                                            output_dim = output_date,
                                            first_model_name=model_name,
                                            second_model_name = model_name1, # Mô hình đầu tiên
                                            epochs=saved_params.get('epochs', 50),
                                            batch_size=saved_params.get('batch_size', 32),
                                            learning_rate=saved_params.get('learning_rate', 0.001),
                                            return_history=True
                                            )
                        
                        end_train_time = time.time()
                        train_time = end_train_time - start_train_time
                        
                        # Vẽ biểu đồ loss/accuracy
                        fig_metrics = go.Figure()
                        fig_metrics.add_trace(go.Scatter(y=history1.history['loss'], name=f'Train Loss {model_name}'))
                        fig_metrics.add_trace(go.Scatter(y=history2.history['loss'], name=f'Train Loss {model_name1} '))
                        fig_metrics.add_trace(go.Scatter(y=history1.history['val_loss'], name=f'Validation Loss {model_name}'))
                        fig_metrics.add_trace(go.Scatter(y=history2.history['val_loss'], name=f'Validation Loss {model_name1}'))

                        fig_metrics.update_layout(title='Loss theo epoch',
                                            xaxis_title='Epoch',
                                            yaxis_title='Loss')
                        st.plotly_chart(fig_metrics)
                        
                        # Hiển thị kết quả cuối cùng
                        st.subheader("Kết quả huấn luyện:")
                        st.write(f"- Thời gian huấn luyện: {train_time:.2f} giây")
                        st.write(f"- Số epoch hoàn thành: {saved_params['epochs']}")
                        st.write(f"- Final Loss mô hình {model_name}: {history1.history['loss'][-1]:.4f}")
                        st.write(f"- Final Validation Loss mô hình {model_name}: {history1.history['val_loss'][-1]:.4f}")
                        st.write(f"- Final Loss mô hình {model_name1}: {history2.history['loss'][-1]:.4f}")
                        st.write(f"- Final Validation Loss mô hình {model_name1}: {history2.history['val_loss'][-1]:.4f}")

                        st.session_state.trained_model = model1
                        st.session_state.trained_model1 = model2
                        
            else:
                st.error("Không tìm thấy tham số tối ưu đã lưu. Vui lòng chạy optimize trước.")
                    
            st.session_state.training_button_clicked = False
            
        except Exception as e:
            st.error(f"Lỗi trong quá trình huấn luyện: {str(e)}")
            st.session_state.training_button_clicked = False

# Testing expander
with st.expander("Testing Model", expanded=True):  # Luôn mở
    if st.session_state.optimize_expanded:
        st.subheader("Tham số tối ưu đã tìm được:")
        for param, value in st.session_state.optimize_results.items():
            st.write(f"- {param}: {value}")
    
    # Hiển thị kết quả training nếu có
    # if hasattr(st.session_state, 'training_results'):
    #     st.subheader("Kết quả training:")
    #     tr = st.session_state.training_results
    #     st.write(f"- Thời gian huấn luyện: {tr['train_time']:.2f} giây")
    #     st.write(f"- Số epoch: {tr['epochs']}")
    #     st.write(f"- Loss cuối cùng: {tr['final_loss']:.4f}")
    #     st.write(f"- Validation Loss cuối cùng: {tr['final_val_loss']:.4f}")
    
    if st.session_state.testing_button_clicked:
        try:
            if st.session_state.trained_model is not None:
                with st.spinner('Đang đánh giá mô hình trên tập test...'):
                    # Lấy dữ liệu test
                    data = st.session_state.data
                    target_column = st.session_state.target_column
                    X, y = st.session_state.get_data(data,input_dim = input_date, output_dim = output_date, target_column=target_column)
                    X_train, X_val, X_test, y_train, y_val, y_test = st.session_state.split_data(
                        X, y, 
                        # train_ratio=actual_train_size,
                        # test_ratio=test_size
                        test_ratio = test_size / 100,
                        val_ratio = val_size / 100
                    )
                    if model_type == "Mô hình đơn":
                    # Dự đoán và tính các metrics
                        start_test_time = time.time()
                        model = st.session_state.trained_model
                        predictions, actual_test, mae, mse, rmse, cv_rmse, buf = st.session_state.predict(model, X_test, y_test)
                        end_test_time = time.time()
                        test_time = end_test_time - start_test_time
                        st.write (f'Testing time: {test_time:.2f}s ')
                        st.subheader("Các chỉ số đánh giá:")
                        metrics_cols = st.columns(4)
                        with metrics_cols[0]:
                            st.metric("MAE", f"{mae:.6f}")
                        with metrics_cols[1]:
                            st.metric("MSE", f"{mse:.6f}")
                        with metrics_cols[2]:
                            st.metric("RMSE", f"{rmse:.6f}")
                        with metrics_cols[3]:
                            st.metric("CV_RMSE", f"{cv_rmse:.4f}%")
                            
                    elif model_type == "Mô hình tuần tự":
                        start_test_time = time.time()
                        model = st.session_state.trained_model
                        predictions, actual_test, mae, mse, rmse, cv_rmse, buf = st.session_state.predict(model,X_test, y_test)
                        end_test_time = time.time()
                        test_time = end_test_time - start_test_time
                        st.write (f'Testing time: {test_time:.2f}s ')
                        st.subheader("Các chỉ số đánh giá:")
                        metrics_cols = st.columns(4)
                        with metrics_cols[0]:
                            st.metric("MAE", f"{mae:.6f}")
                        with metrics_cols[1]:
                            st.metric("MSE", f"{mse:.6f}")
                        with metrics_cols[2]:
                            st.metric("RMSE", f"{rmse:.6f}")
                        with metrics_cols[3]:
                            st.metric("CV_RMSE", f"{cv_rmse:.4f}%")
                    elif model_type == "Mô hình song song":
                        start_test_time = time.time()
                        model = st.session_state.trained_model
                        model1 = st.session_state.trained_model1
                        predictions,actual_test, mae, mse, rmse, cv_rmse, buf = st.session_state.predict (model,X_test, y_test)
                        predictions1,actual_test1, mae1, mse1, rmse1, cv_rmse1, buf1 = st.session_state.predict(model1,X_test, y_test)
                        final_predict,mae, mse, rmse, cv_rmse, buf = combine_predictions_parallel(predictions,predictions1,actual_test, input_dim = input_date, output_dim = output_date)
                    # Hiển thị metrics
                        end_test_time = time.time()
                        test_time = end_test_time - start_test_time
                        st.write (f'Testing time: {test_time:.2f}s ')
                        st.subheader("Các chỉ số đánh giá:")
                        metrics_cols = st.columns(4)
                        with metrics_cols[0]:
                            st.metric("MAE", f"{mae:.6f}")
                        with metrics_cols[1]:
                            st.metric("MSE", f"{mse:.6f}")
                        with metrics_cols[2]:
                            st.metric("RMSE", f"{rmse:.6f}")
                        with metrics_cols[3]:
                            st.metric("CV_RMSE", f"{cv_rmse:.4f}%")

                    elif model_type == "Mô hình tuần tự nhân":
                        start_test_time = time.time()
                        model = st.session_state.trained_model
                        model1 = st.session_state.trained_model1
                        predictions,actual_test, mae, mse, rmse, cv_rmse, buf = predict(model,X_test, y_test)
                        predictions1,actual_test1, mae1, mse1, rmse1, cv_rmse1, buf1 = predict(model1,X_test, y_test)
                        final_predict,mae, mse, rmse, cv_rmse, buf = combine_predictions_mul(predictions,predictions1,actual_test)
                        end_test_time = time.time()
                        test_time = end_test_time - start_test_time
                        st.write (f'Testing time: {test_time:.2f}s ')
                        st.subheader("Các chỉ số đánh giá:")
                        metrics_cols = st.columns(4)
                        with metrics_cols[0]:
                            st.metric("MAE", f"{mae:.6f}")
                        with metrics_cols[1]:
                            st.metric("MSE", f"{mse:.6f}")
                        with metrics_cols[2]:
                            st.metric("RMSE", f"{rmse:.6f}")
                        with metrics_cols[3]:
                            st.metric("CV_RMSE", f"{cv_rmse:.4f}%")

                    elif model_type == "Mô hình tuần tự cộng":
                        start_test_time = time.time()
                        model = st.session_state.trained_model
                        model1 = st.session_state.trained_model1
                        predictions,actual_test, mae, mse, rmse, cv_rmse, buf = predict(model,X_test, y_test)
                        predictions1,actual_test1, mae1, mse1, rmse1, cv_rmse1, buf1 = predict(model1,X_test, y_test)
                        final_predict,mae, mse, rmse, cv_rmse, buf = combine_predictions_add(predictions,predictions1,actual_test)
                        end_test_time = time.time()
                        test_time = end_test_time - start_test_time
                        st.write (f'Testing time: {test_time:.2f}s ')
                        st.subheader("Các chỉ số đánh giá:")
                        metrics_cols = st.columns(4)
                        with metrics_cols[0]:
                            st.metric("MAE", f"{mae:.6f}")
                        with metrics_cols[1]:
                            st.metric("MSE", f"{mse:.6f}")
                        with metrics_cols[2]:
                            st.metric("RMSE", f"{rmse:.6f}")
                        with metrics_cols[3]:
                            st.metric("CV_RMSE", f"{cv_rmse:.4f}%")
                        st.success("Đánh giá mô hình hoàn tất!")
                    else: 
                        st.success("Không thể đánh giá mô hình")
            
            else:
                st.error("Vui lòng huấn luyện mô hình trước khi testing!")
                
            st.session_state.testing_button_clicked = False
            
        except Exception as e:
            st.error(f"Lỗi trong quá trình testing: {str(e)}")
            st.session_state.testing_button_clicked = False

# Train and Predict expander
with st.expander("Predict Actual Value"):
    if st.session_state.predict_button_clicked:
        try:
            if st.session_state.trained_model is not None:
                with st.spinner('Đang dự đoán...'):
                    data = st.session_state.data
                    target_column = st.session_state.target_column
                    X, y = st.session_state.get_data(data,input_dim = input_date, output_dim = output_date, target_column=target_column)
                    X_train, X_val, X_test, y_train, y_val, y_test = st.session_state.split_data(
                        X, y, 
                        test_ratio=test_size/100,
                        val_ratio=val_size/100
                    )

                    if model_type == "Mô hình đơn":
                        model = st.session_state.trained_model
                        predictions, actual_test, mae, mse, rmse, cv_rmse, buf = st.session_state.predict(
                            model, 
                            X_test, 
                            y_test
                        )
                        st.image(buf, caption="Biểu đồ dự đoán", use_container_width=True)

                    elif model_type == "Mô hình tuần tự":
                        model = st.session_state.trained_model
                        predictions, actual_test, mae, mse, rmse, cv_rmse, buf = st.session_state.predict(model,X_test, y_test)
                        st.image(buf, caption="Biểu đồ dự đoán", use_container_width=True)
                    elif model_type == "Mô hình song song":
                        model = st.session_state.trained_model
                        model1 = st.session_state.trained_model1
                        predictions,actual_test, mae, mse, rmse, cv_rmse, buf = st.session_state.predict (model,X_test, y_test)
                        predictions1,actual_test1, mae1, mse1, rmse1, cv_rmse1, buf1 = st.session_state.predict(model1,X_test, y_test)

                        final_predict,mae, mse, rmse, cv_rmse, buf_final = combine_predictions_parallel(predictions,predictions1,actual_test, input_dim = input_date, output_dim = output_date)
                        final_predict_scale,actual_test_scale = rescale_data(final_predict, actual_test)
                        st.image(buf_final, caption="Biểu đồ dự đoán", use_container_width=True)

                    elif model_type == "Mô hình tuần tự nhân":
                        model = st.session_state.trained_model
                        model1 = st.session_state.trained_model1
                        predictions,actual_test, mae, mse, rmse, cv_rmse, buf = st.session_state.predict (model,X_test, y_test)
                        predictions1,actual_test1, mae1, mse1, rmse1, cv_rmse1, buf1 = st.session_state.predict(model1,X_test, y_test)

                        final_predict,mae, mse, rmse, cv_rmse, buf = combine_predictions_mul(predictions,predictions1,actual_test)
                        final_predict_scale,actual_test_scale = rescale_data(final_predict, actual_test)
                        st.image(buf, caption="Biểu đồ dự đoán", use_container_width=True)

                    elif model_type == "Mô hình tuần tự cộng":
                        model = st.session_state.trained_model
                        model1 = st.session_state.trained_model1
                        predictions,actual_test, mae, mse, rmse, cv_rmse, buf = st.session_state.predict (model,X_test, y_test)
                        predictions1,actual_test1, mae1, mse1, rmse1, cv_rmse1, buf1 = st.session_state.predict(model1,X_test, y_test)

                        final_predict,mae, mse, rmse, cv_rmse, buf = combine_predictions_mul(predictions,predictions1,actual_test)
                        final_predict_scale,actual_test_scale = rescale_data(final_predict, actual_test)
                        st.image(buf, caption="Biểu đồ dự đoán", use_container_width=True)

            else:
                st.error("Vui lòng huấn luyện mô hình trước khi d đoán!")
                
            st.session_state.predict_button_clicked = False
            
        except Exception as e:
            st.error(f"Lỗi trong quá trình dự đoán: {str(e)}")
            st.session_state.predict_button_clicked = False
