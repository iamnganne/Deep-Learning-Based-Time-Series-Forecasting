# Deep Learning Based Time Series Forecasting
Time series forecasting is a critical task in numerous fields, including finance, supply chain management, and energy demand planning. To enhance forecasting accuracy, hybrid models that combine the strengths of multiple architectures have gained significant attention.
## 📌 Objectives
- This project explores a hybrid approach by integrating Long Short-Term Memory (LSTM) networks, known for their ability to capture sequential dependencies, with Convolutional Neural Networks (CNNs), renowned for feature extraction capabilities.
- The model leverages CNNs to extract meaningful patterns from input data and LSTMs to model temporal relationships, resulting in improved predictive performance.
- Compare models based on metrics such as RMSE,MSE,MAE
- Build an interactive GUI using Streamlit
## 🧠 Model Implemented 
- LSTM (Long Short Term Memory)
- CNN (1D Convolutional Neural Network)
- CNN-LSTM, LSTM-CNN (Sequential)
- Hybrid models with additive/multiplicative fusion
- CNN-LSTM (Parallel)
## ⚙️ Requirements
Python 3.8+
TensorFlow / Keras
NumPy, Pandas, Scikit-learn
Optuna
Matplotlib, Seaborn
Streamlit (for dashboard)
Install dependencies: 
```bash
pip install -r requirements.txt
```
## Sample Visualization
![image](https://github.com/user-attachments/assets/c0c3e92d-343e-494a-a4f9-6a9fc5464a3a)
## Comparison Results 
## 📚 References
- TensorFlow Time Series Tutorial.
- Hands-On Time Series Forecasting with Python.
- Academic papers on CNN-LSTM and hybrid models.
