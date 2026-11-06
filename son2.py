import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_absolute_error
from scipy import stats
from tqdm import tqdm
import time
import gc
import logging
import os
from sktime.forecasting.croston import Croston
from sktime.forecasting.base import ForecastingHorizon

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Output directory (HPC uyumlu) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'Veri')
os.makedirs(OUT_DIR, exist_ok=True)

# Load data and get unique product IDs
all_intermittent_products = pd.read_csv(os.path.join(BASE_DIR, 'Veri', 'filtered_data.csv'))
unique_product_ids = all_intermittent_products.URUNID.unique()
logger.info(f"Loaded data with {len(unique_product_ids)} unique products")

# Calculate sparsity for each product
def calculate_sparsity(df, product_col='URUNID', demand_col='Satis Adet'):
    sparsity_dict = {}
    for pid in df[product_col].unique():
        product_data = df[df[product_col] == pid][demand_col]
        total_periods = len(product_data)
        zero_periods = (product_data == 0).sum()
        sparsity = zero_periods / total_periods if total_periods > 0 else 0
        sparsity_dict[pid] = sparsity
    return sparsity_dict

# Define sparsity levels (8 bins)
sparsity_bins = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
sparsity_labels = ['[0.0, 0.2)', '[0.2, 0.3)', '[0.3, 0.4)', '[0.4, 0.5)', 
                   '[0.5, 0.6)', '[0.6, 0.7)', '[0.7, 0.8)', '[0.8, 1.0)']

# Compute sparsity and assign levels
sparsity_dict = calculate_sparsity(all_intermittent_products)
sparsity_df = pd.DataFrame.from_dict(sparsity_dict, orient='index', columns=['Sparsity'])
sparsity_df['Sparsity_Level'] = pd.cut(sparsity_df['Sparsity'], bins=sparsity_bins, 
                                       labels=sparsity_labels, include_lowest=True, right=False)
sparsity_df.index.name = 'URUNID'

# Dictionaries for results and best thresholds
all_results = {}
best_alpha_per_product = {}
best_threshold_rnn_per_product = {}

# Scoreboard (based on MAE)
scoreboard = {
    'CTR': 0,
    'CTR_Regression': 0,
    'Feedforward Neural Network (FNN)': 0,
    'RNN Model': 0,
    'LSTM Model': 0,
    'Simple Exponential Smoothing': 0,
    'Croston\'s Method': 0,
    'SBA Method': 0
}

# Prediction storage
predictions_ctr = []
predictions_ctr_regression = []
predictions_fnn = []
predictions_rnn = []
predictions_lstm = []
predictions_ses = []
predictions_croston = []
predictions_sba = []

# Error metric functions
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.inf if np.sum(y_pred != 0) > 0 else 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mean_absolute_scaled_error(y_true, y_pred, y_train):
    naive_errors = np.abs(y_train[1:] - y_train[:-1])
    scale = np.mean(naive_errors) if len(naive_errors) > 0 and np.mean(naive_errors) != 0 else 1e-10
    return np.mean(np.abs(y_true - y_pred)) / scale

def smape(y_true, y_pred):
    # Convert inputs to NumPy arrays
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    
    # Check for valid inputs
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must have the same non-empty shape")
    
    # Compute absolute values and differences
    abs_true = np.abs(y_true)
    abs_pred = np.abs(y_pred)
    abs_diff = np.abs(y_true - y_pred)
    denominator = abs_true + abs_pred
    
    # Handle zero denominators
    mask = denominator == 0
    if np.all(mask):
        # If all denominators are zero, check if predictions are correct
        return 0.0 if np.all(abs_diff == 0) else 100.0  # Max error if y_true != y_pred
    
    # Avoid division by zero by setting denominator to 1 where zero (only affects terms where abs_diff = 0)
    denominator = np.where(mask, 1, denominator)
    
    # Compute SMAPE: 2 * |y_true - y_pred| / (|y_true| + |y_pred|)
    smape_values = np.where(mask, 0, 2 * abs_diff / denominator)
    
    # Return mean SMAPE as percentage
    return np.mean(smape_values) * 100

def rmsse(y_true, y_pred, y_train):
    mse = np.mean((y_true - y_pred) ** 2)
    naive_errors = np.mean((y_train[1:] - y_train[:-1]) ** 2)
    scale = naive_errors if naive_errors != 0 else 1e-10
    return np.sqrt(mse / scale)


def sba(y, extra_periods=0, alpha=0.4, beta=0.4):
    # Initialize arrays
    demand = []
    intervals = []
    last_demand_idx = -1

    # Separate non-zero demands and intervals
    for t, value in enumerate(y):
        if value > 0:
            demand.append(value)
            if last_demand_idx >= 0:
                intervals.append(t - last_demand_idx)
            last_demand_idx = t

    if not demand:  # No non-zero demands
        return np.zeros(extra_periods)

    # Initialize smoothed values
    z = demand[0]  # Smoothed demand
    p = intervals[0] if intervals else 1  # Smoothed interval
    forecasts = []

    # Apply exponential smoothing
    for i in range(1, len(demand)):
        z = alpha * demand[i] + (1 - alpha) * z
        p = beta * intervals[i-1] + (1 - beta) * p
        forecast = (1 - beta / 2) * z / p
        forecasts.append(forecast)

    # Extend forecast for extra_periods
    final_forecast = (1 - beta / 2) * z / p
    return np.array([final_forecast] * extra_periods)

def create_sequences(X, y, time_steps=7):
    Xs, ys = [], []
    for i in range(len(X) - time_steps + 1):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps - 1])
    return np.array(Xs), np.array(ys)

start_time = time.time()

def custom_ctr_loss(y_true, y_pred):
    mask = tf.cast(y_true > 0, tf.float32)
    return tf.reduce_mean(tf.abs(y_true - y_pred) * (1 + 2 * mask)) + 0.005 * tf.reduce_mean(tf.square(y_pred))

for idx, product_id in enumerate(tqdm(unique_product_ids, desc="Processing Products")):
    logger.info(f"Processing Product ID {product_id} (#{idx + 1})")
    
    # Preprocess data
    product_rows = all_intermittent_products[all_intermittent_products['URUNID'] == product_id].copy()
    product_rows['Date'] = pd.to_datetime(product_rows['Date'])
    product_rows['Day'] = product_rows['Date'].dt.day
    product_rows['Month'] = product_rows['Date'].dt.month
    product_rows['Year'] = product_rows['Date'].dt.year
    product_rows['DayOfWeek'] = product_rows['Date'].dt.dayofweek
    product_rows['IsWeekend'] = product_rows['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    product_rows['Lag_1'] = product_rows['Satis Adet'].shift(1)
    product_rows['Lag_2'] = product_rows['Satis Adet'].shift(2)
    product_rows['Lag_3'] = product_rows['Satis Adet'].shift(3)
    product_rows['Rolling_Mean_3'] = product_rows['Satis Adet'].shift(1).rolling(window=3).mean()
    product_rows['Rolling_Mean_7'] = product_rows['Satis Adet'].shift(1).rolling(window=7).mean()
    product_rows.dropna(inplace=True)
    
    time_steps = 7
    if len(product_rows) < time_steps:
        logger.warning(f"Skipping Product ID {product_id}: Insufficient data after preprocessing ({len(product_rows)} rows)")
        continue
    
    X = product_rows[['URUNID', 'Day', 'Month', 'Year', 'DayOfWeek', 'IsWeekend', 
                      'Lag_1', 'Lag_2', 'Lag_3', 'Rolling_Mean_3', 'Rolling_Mean_7']].values
    y = product_rows['Satis Adet'].values
    
    test_size = int(len(X) * 0.15)
    val_size = int(len(X) * 0.15)
    if test_size == 0 or val_size == 0 or len(X) - test_size - val_size < 1:
        logger.warning(f"Skipping Product ID {product_id}: Insufficient data for train/val/test split")
        continue
    
    scaler = StandardScaler()
    scaler.fit(X[:len(X) - test_size - val_size])
    X = scaler.transform(X)
    
    X_seq, y_seq = create_sequences(X, y, time_steps)
    if X_seq.size == 0:
        logger.warning(f"Skipping Product ID {product_id}: No sequences created")
        continue
    
    logger.info(f"Product ID {product_id}: X_seq shape {X_seq.shape}, y_seq shape {y_seq.shape}")
    
    X_train_seq = X_seq[:len(X_seq) - test_size - val_size]
    X_val_seq = X_seq[len(X_seq) - test_size - val_size : len(X_seq) - test_size]
    X_test_seq = X_seq[-test_size:]
    y_train_seq = y_seq[:len(X_seq) - test_size - val_size]
    y_val_seq = y_seq[len(X_seq) - test_size - val_size : len(X_seq) - test_size]
    y_test_seq = y_seq[-test_size:]
    
    X_train_scaled = X_train_seq.reshape(-1, X_train_seq.shape[-1] * time_steps)
    X_val_scaled = X_val_seq.reshape(-1, X_val_seq.shape[-1] * time_steps)
    X_test_scaled = X_test_seq.reshape(-1, X_test_seq.shape[-1] * time_steps)

    # CTR Model with Timing
    start_ctr = time.time()
    input_layer = Input(shape=(time_steps, X_train_seq.shape[-1]))
    lstm_layer = LSTM(32, activation='relu', return_sequences=False)(input_layer)
    shared_layer = Dense(64, activation='relu')(lstm_layer)
    shared_layer = Dropout(0.1)(shared_layer)
    shared_layer = Dense(32, activation='relu')(shared_layer)
    classification_output = Dense(1, activation='sigmoid', name='classification_output')(shared_layer)
    regression_output = Dense(1, name='regression_output')(shared_layer)
    model_ctr = Model(inputs=input_layer, outputs=[classification_output, regression_output])
    model_ctr.compile(optimizer=Adam(learning_rate=0.0001),
                      loss={'classification_output': 'binary_crossentropy', 'regression_output': custom_ctr_loss},
                      loss_weights={'classification_output': 0.5, 'regression_output': 0.5},
                      metrics={'classification_output': 'accuracy', 'regression_output': 'mae'})
    early_stopping = EarlyStopping(monitor='val_regression_output_loss', patience=15, restore_best_weights=True, mode='min')
    model_ctr.fit(X_train_seq, 
                  {'classification_output': (y_train_seq > 0).astype(int), 'regression_output': y_train_seq},
                  validation_data=(X_val_seq, 
                                   {'classification_output': (y_val_seq > 0).astype(int), 'regression_output': y_val_seq}),
                  epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0)
    ctr_val_predictions = model_ctr.predict(X_val_seq, verbose=0)
    classification_val_preds = ctr_val_predictions[0].flatten()
    regression_val_preds = ctr_val_predictions[1].flatten()
    results_df_val_ctr = pd.DataFrame({
        'Real Values': y_val_seq,
        'Classification Predictions': classification_val_preds,
        'Regression Predictions': regression_val_preds
    })
    alpha_levels = np.linspace(0.01, 0.99, 50)
    mae_val_results_ctr = []
    for alpha in alpha_levels:
        results_df_val_ctr['Final Prediction'] = results_df_val_ctr.apply(
            lambda row: row['Regression Predictions'] if row['Classification Predictions'] > alpha else 0, axis=1)
        mae_val = mean_absolute_error(results_df_val_ctr['Real Values'], results_df_val_ctr['Final Prediction'])
        mae_val_results_ctr.append((alpha, mae_val))
    best_alpha_ctr, _ = min(mae_val_results_ctr, key=lambda x: x[1])
    best_alpha_per_product[product_id] = best_alpha_ctr
    ctr_test_predictions = model_ctr.predict(X_test_seq, verbose=0)
    classification_test_preds = ctr_test_predictions[0].flatten()
    regression_test_preds = ctr_test_predictions[1].flatten()
    results_df_test_ctr = pd.DataFrame({
        'Real Values': y_test_seq,
        'Classification Predictions': classification_test_preds,
        'Regression Predictions': regression_test_preds
    })
    results_df_test_ctr['Final Prediction'] = results_df_test_ctr.apply(
        lambda row: row['Regression Predictions'] if row['Classification Predictions'] > best_alpha_ctr else 0, axis=1)
    ctr_preds = results_df_test_ctr['Final Prediction'].values
    end_ctr = time.time()
    logger.info(f"CTR Model for Product ID {product_id} took {end_ctr - start_ctr:.2f} seconds")
    all_results[product_id] = {
        'CTR': {
            'MAE': mean_absolute_error(y_test_seq, ctr_preds),
            'MAPE': mean_absolute_percentage_error(y_test_seq, ctr_preds),
            'RMSE': root_mean_squared_error(y_test_seq, ctr_preds),
            'MASE': mean_absolute_scaled_error(y_test_seq, ctr_preds, y_train_seq),
            'sMAPE': smape(y_test_seq, ctr_preds),
            'RMSSE': rmsse(y_test_seq, ctr_preds, y_train_seq)
        }
    }
    predictions_ctr.append(pd.DataFrame({
        'Product_ID': [product_id] * len(y_test_seq),
        'y_test': y_test_seq,
        'CTR_Predictions': ctr_preds
    }))

    # CTR_Regression with Timing
    start_ctr_reg = time.time()
    input_layer_reg = Input(shape=(X_train_scaled.shape[1],))
    shared_layer_reg = Dense(64, activation='relu')(input_layer_reg)
    shared_layer_reg = Dense(64, activation='relu')(shared_layer_reg)
    regression_output_reg = Dense(1, name='regression_output')(shared_layer_reg)
    model_ctr_regression = Model(inputs=input_layer_reg, outputs=regression_output_reg)
    model_ctr_regression.compile(optimizer='adam', loss='mean_absolute_error')
    model_ctr_regression.fit(X_train_scaled, y_train_seq, validation_data=(X_val_scaled, y_val_seq), epochs=50, verbose=0)
    ctr_regression_preds = model_ctr_regression.predict(X_test_scaled, verbose=0).flatten()
    end_ctr_reg = time.time()
    logger.info(f"CTR_Regression Model for Product ID {product_id} took {end_ctr_reg - start_ctr_reg:.2f} seconds")
    all_results[product_id]['CTR_Regression'] = {
        'MAE': mean_absolute_error(y_test_seq, ctr_regression_preds),
        'MAPE': mean_absolute_percentage_error(y_test_seq, ctr_regression_preds),
        'RMSE': root_mean_squared_error(y_test_seq, ctr_regression_preds),
        'MASE': mean_absolute_scaled_error(y_test_seq, ctr_regression_preds, y_train_seq),
        'sMAPE': smape(y_test_seq, ctr_regression_preds),
        'RMSSE': rmsse(y_test_seq, ctr_regression_preds, y_train_seq)
    }
    predictions_ctr_regression.append(pd.DataFrame({
        'Product_ID': [product_id] * len(y_test_seq),
        'y_test': y_test_seq,
        'CTR_Regression_Predictions': ctr_regression_preds
    }))

    # FNN with Timing
    start_fnn = time.time()
    inputs_fnn = Input(shape=(X_train_scaled.shape[1],))
    x = Dense(10, activation='relu')(inputs_fnn)
    x = Dense(3, activation='relu')(x)
    outputs_fnn = Dense(1)(x)
    model_fnn = Model(inputs=inputs_fnn, outputs=outputs_fnn)
    model_fnn.compile(optimizer='adam', loss='mean_absolute_error')
    model_fnn.fit(X_train_scaled, y_train_seq, epochs=50, verbose=0)
    fnn_preds = model_fnn.predict(X_test_scaled, verbose=0).flatten()
    end_fnn = time.time()
    logger.info(f"FNN Model for Product ID {product_id} took {end_fnn - start_fnn:.2f} seconds")
    all_results[product_id]['Feedforward Neural Network (FNN)'] = {
        'MAE': mean_absolute_error(y_test_seq, fnn_preds),
        'MAPE': mean_absolute_percentage_error(y_test_seq, fnn_preds),
        'RMSE': root_mean_squared_error(y_test_seq, fnn_preds),
        'MASE': mean_absolute_scaled_error(y_test_seq, fnn_preds, y_train_seq),
        'sMAPE': smape(y_test_seq, fnn_preds),
        'RMSSE': rmsse(y_test_seq, fnn_preds, y_train_seq)
    }
    predictions_fnn.append(pd.DataFrame({
        'Product_ID': [product_id] * len(y_test_seq),
        'y_test': y_test_seq,
        'FNN_Predictions': fnn_preds
    }))

    # RNN with Timing
    start_rnn = time.time()
    inputs_rnn = Input(shape=(time_steps, X_train_seq.shape[-1]))
    lstm_out = LSTM(50, activation='relu')(inputs_rnn)
    outputs_rnn = Dense(1)(lstm_out)
    model_rnn = Model(inputs=inputs_rnn, outputs=outputs_rnn)
    model_rnn.compile(optimizer='adam', loss='mean_absolute_error')
    model_rnn.fit(X_train_seq, y_train_seq, validation_data=(X_val_seq, y_val_seq), epochs=50, verbose=0)
    rnn_val_preds = model_rnn.predict(X_val_seq, verbose=0).flatten()
    rnn_val_classifier = (rnn_val_preds > np.mean(rnn_val_preds)).astype(int)
    results_df_val_rnn = pd.DataFrame({
        'Real Values': y_val_seq,
        'Classification Predictions': rnn_val_classifier,
        'Regression Predictions': rnn_val_preds
    })
    mae_val_results_rnn = []
    for threshold in alpha_levels:
        results_df_val_rnn['Final Prediction'] = results_df_val_rnn.apply(
            lambda row: row['Regression Predictions'] if row['Classification Predictions'] > threshold else 0, axis=1)
        mae_val = mean_absolute_error(results_df_val_rnn['Real Values'], results_df_val_rnn['Final Prediction'])
        mae_val_results_rnn.append((threshold, mae_val))
    best_threshold_rnn, _ = min(mae_val_results_rnn, key=lambda x: x[1])
    best_threshold_rnn_per_product[product_id] = best_threshold_rnn
    rnn_test_preds = model_rnn.predict(X_test_seq, verbose=0).flatten()
    rnn_test_classifier = (rnn_test_preds > np.mean(rnn_test_preds)).astype(int)
    results_df_test_rnn = pd.DataFrame({
        'Real Values': y_test_seq,
        'Classification Predictions': rnn_test_classifier,
        'Regression Predictions': rnn_test_preds
    })
    results_df_test_rnn['Final Prediction'] = results_df_test_rnn.apply(
        lambda row: row['Regression Predictions'] if row['Classification Predictions'] > best_threshold_rnn else 0, axis=1)
    rnn_preds = results_df_test_rnn['Final Prediction'].values
    end_rnn = time.time()
    logger.info(f"RNN Model for Product ID {product_id} took {end_rnn - start_rnn:.2f} seconds")
    all_results[product_id]['RNN Model'] = {
        'MAE': mean_absolute_error(y_test_seq, rnn_preds),
        'MAPE': mean_absolute_percentage_error(y_test_seq, rnn_preds),
        'RMSE': root_mean_squared_error(y_test_seq, rnn_preds),
        'MASE': mean_absolute_scaled_error(y_test_seq, rnn_preds, y_train_seq),
        'sMAPE': smape(y_test_seq, rnn_preds),
        'RMSSE': rmsse(y_test_seq, rnn_preds, y_train_seq)
    }
    predictions_rnn.append(pd.DataFrame({
        'Product_ID': [product_id] * len(y_test_seq),
        'y_test': y_test_seq,
        'RNN_Predictions': rnn_preds
    }))

    # LSTM with Timing
    start_lstm = time.time()
    inputs_lstm = Input(shape=(time_steps, X_train_seq.shape[-1]))
    lstm_out = LSTM(50, activation='relu')(inputs_lstm)
    outputs_lstm = Dense(1)(lstm_out)
    model_lstm = Model(inputs=inputs_lstm, outputs=outputs_lstm)
    model_lstm.compile(optimizer='adam', loss='mean_absolute_error')
    model_lstm.fit(X_train_seq, y_train_seq, validation_data=(X_val_seq, y_val_seq), epochs=50, verbose=0)
    lstm_test_preds = model_lstm.predict(X_test_seq, verbose=0).flatten()
    end_lstm = time.time()
    logger.info(f"LSTM Model for Product ID {product_id} took {end_lstm - start_lstm:.2f} seconds")
    all_results[product_id]['LSTM Model'] = {
        'MAE': mean_absolute_error(y_test_seq, lstm_test_preds),
        'MAPE': mean_absolute_percentage_error(y_test_seq, lstm_test_preds),
        'RMSE': root_mean_squared_error(y_test_seq, lstm_test_preds),
        'MASE': mean_absolute_scaled_error(y_test_seq, lstm_test_preds, y_train_seq),
        'sMAPE': smape(y_test_seq, lstm_test_preds),
        'RMSSE': rmsse(y_test_seq, lstm_test_preds, y_train_seq)
    }
    predictions_lstm.append(pd.DataFrame({
        'Product_ID': [product_id] * len(y_test_seq),
        'y_test': y_test_seq,
        'LSTM_Predictions': lstm_test_preds
    }))

    # SES with Timing
    start_ses = time.time()
    model_ses = SimpleExpSmoothing(y_train_seq).fit(optimized=True)
    ses_forecast = model_ses.forecast(len(y_test_seq))
    end_ses = time.time()
    logger.info(f"SES Model for Product ID {product_id} took {end_ses - start_ses:.2f} seconds")
    all_results[product_id]['Simple Exponential Smoothing'] = {
        'MAE': mean_absolute_error(y_test_seq, ses_forecast),
        'MAPE': mean_absolute_percentage_error(y_test_seq, ses_forecast),
        'RMSE': root_mean_squared_error(y_test_seq, ses_forecast),
        'MASE': mean_absolute_scaled_error(y_test_seq, ses_forecast, y_train_seq),
        'sMAPE': smape(y_test_seq, ses_forecast),
        'RMSSE': rmsse(y_test_seq, ses_forecast, y_train_seq)
    }
    predictions_ses.append(pd.DataFrame({
        'Product_ID': [product_id] * len(y_test_seq),
        'y_test': y_test_seq,
        'SES_Predictions': ses_forecast
    }))

    

    # Croston's Method with Timing (using sktime)
    y_train = y[:len(y) - test_size]
    y_test = y[-test_size:]

    start_croston = time.time()
    y_train_series = pd.Series(y_train)
    model_croston = Croston(smoothing=0.4)
    model_croston.fit(y_train_series)
    fh = ForecastingHorizon(list(range(1, len(y_test) + 1)), is_relative=True)  # Convert range to list
    croston_forecast = model_croston.predict(fh).values
    end_croston = time.time()

    logger.info(f"Croston's Method for Product ID {product_id} took {end_croston - start_croston:.2f} seconds")
    all_results[product_id]['Croston\'s Method'] = {
        'MAE': mean_absolute_error(y_test_seq, croston_forecast),
        'MAPE': mean_absolute_percentage_error(y_test_seq, croston_forecast),
        'RMSE': root_mean_squared_error(y_test_seq, croston_forecast),
        'MASE': mean_absolute_scaled_error(y_test_seq, croston_forecast, y_train_seq),
        'sMAPE': smape(y_test_seq, croston_forecast),
        'RMSSE': rmsse(y_test_seq, croston_forecast, y_train_seq)
    }
    predictions_croston.append(pd.DataFrame({
        'Product_ID': [product_id] * len(y_test_seq),
        'y_test': y_test_seq,
        'Croston_Predictions': croston_forecast
    }))

    # SBA Method with Timing
    start_sba = time.time()
    sba_forecast = sba(y_train_seq, extra_periods=len(y_test_seq))
    end_sba = time.time()
    logger.info(f"SBA Method for Product ID {product_id} took {end_sba - start_sba:.2f} seconds")
    all_results[product_id]['SBA Method'] = {
        'MAE': mean_absolute_error(y_test_seq, sba_forecast),
        'MAPE': mean_absolute_percentage_error(y_test_seq, sba_forecast),
        'RMSE': root_mean_squared_error(y_test_seq, sba_forecast),
        'MASE': mean_absolute_scaled_error(y_test_seq, sba_forecast, y_train_seq),
        'sMAPE': smape(y_test_seq, sba_forecast),
        'RMSSE': rmsse(y_test_seq, sba_forecast, y_train_seq)
    }
    predictions_sba.append(pd.DataFrame({
        'Product_ID': [product_id] * len(y_test_seq),
        'y_test': y_test_seq,
        'SBA_Predictions': sba_forecast
    }))

    min_mae_model = min(all_results[product_id], key=lambda x: all_results[product_id][x]['MAE'])
    scoreboard[min_mae_model] += 1

    tf.keras.backend.clear_session()
    gc.collect()
    logger.info(f"Completed Product ID {product_id}")

# Save results
forecast_data = []
models = list(all_results[next(iter(all_results))].keys())
metrics = ['MAE', 'MAPE', 'RMSE', 'MASE', 'sMAPE', 'RMSSE']

for product_id in all_results:
    for metric in metrics:
        row = {'Product_ID': product_id, 'Metric': metric}
        for model in models:
            row[model] = all_results[product_id][model][metric]
        forecast_data.append(row)

forecast_df = pd.DataFrame(forecast_data)
forecast_df.to_csv(os.path.join(OUT_DIR, 'intermittent_demand_forecasts.csv'), index=False)

combined_data = []
for product_id in all_results:
    for model, metrics_dict in all_results[product_id].items():
        row = {
            'Product_ID': product_id,
            'Model': model,
            'MAE': metrics_dict['MAE'],
            'MAPE': metrics_dict['MAPE'],
            'RMSE': metrics_dict['RMSE'],
            'MASE': metrics_dict['MASE'],
            'sMAPE': metrics_dict['sMAPE'],
            'RMSSE': metrics_dict['RMSSE'],
            'Best_Alpha_CTR': best_alpha_per_product.get(product_id)
        }
        combined_data.append(row)

combined_df = pd.DataFrame(combined_data)
combined_df = combined_df.merge(sparsity_df[['Sparsity_Level']], 
                               left_on='Product_ID', right_index=True, how='left')
combined_df.to_csv(os.path.join(OUT_DIR, 'combined_results.csv'), index=False)

scoreboard_df = pd.DataFrame(list(scoreboard.items()), columns=['Model', 'Wins'])
scoreboard_df.to_csv(os.path.join(OUT_DIR, 'scoreboard.csv'), index=False)

pd.concat(predictions_ctr).to_csv(os.path.join(OUT_DIR, 'ctr_predictions.csv'), index=False)
pd.concat(predictions_ctr_regression).to_csv(os.path.join(OUT_DIR, 'ctr_regression_predictions.csv'), index=False)
pd.concat(predictions_fnn).to_csv(os.path.join(OUT_DIR, 'fnn_predictions.csv'), index=False)
pd.concat(predictions_rnn).to_csv(os.path.join(OUT_DIR, 'rnn_predictions.csv'), index=False)
pd.concat(predictions_lstm).to_csv(os.path.join(OUT_DIR, 'lstm_predictions.csv'), index=False)
pd.concat(predictions_ses).to_csv(os.path.join(OUT_DIR, 'ses_predictions.csv'), index=False)
pd.concat(predictions_croston).to_csv(os.path.join(OUT_DIR, 'croston_predictions.csv'), index=False)
pd.concat(predictions_sba).to_csv(os.path.join(OUT_DIR, 'sba_predictions.csv'), index=False)

# Compute mean and median metrics by model
mean_table_data = []
median_table_data = []
for model in models:
    mean_row = {'Model': model}
    median_row = {'Model': model}
    for metric in metrics:
        metric_values = [all_results[pid][model][metric] for pid in all_results]
        mean_row[metric] = np.mean(metric_values)
        median_row[metric] = np.median(metric_values)
    mean_table_data.append(mean_row)
    median_table_data.append(median_row)

mean_table_df = pd.DataFrame(mean_table_data)
median_table_df = pd.DataFrame(median_table_data)

mean_table_df.to_csv(os.path.join(OUT_DIR, 'mean_metrics.csv'), index=False)
median_table_df.to_csv(os.path.join(OUT_DIR, 'median_metrics.csv'), index=False)

# Aggregate metrics by sparsity level
sparsity_agg_data = []
for model in models:
    model_data = combined_df[combined_df['Model'] == model]
    grouped = model_data.groupby('Sparsity_Level')[metrics].mean().reset_index()
    for _, row in grouped.iterrows():
        agg_row = {'Model': model, 'Sparsity_Level': row['Sparsity_Level']}
        for metric in metrics:
            agg_row[metric] = row[metric]
        sparsity_agg_data.append(agg_row)

sparsity_agg_df = pd.DataFrame(sparsity_agg_data)
sparsity_agg_df.to_csv(os.path.join(OUT_DIR, 'sparsity_metrics.csv'), index=False)

# -------------------------------------------------------
# Mann-Whitney U testi: CTR vs diger modeller (MAE, sMAPE, MASE)
# Istatistiksel anlamlilik icin (makale icin gerekli)
# -------------------------------------------------------
mw_results = []
ctr_mae_vals  = [all_results[pid]['CTR']['MAE']   for pid in all_results]
ctr_smape_vals= [all_results[pid]['CTR']['sMAPE'] for pid in all_results]
ctr_mase_vals = [all_results[pid]['CTR']['MASE']  for pid in all_results]

competitors = [m for m in models if m != 'CTR']
for comp in competitors:
    comp_mae   = [all_results[pid][comp]['MAE']   for pid in all_results]
    comp_smape = [all_results[pid][comp]['sMAPE'] for pid in all_results]
    comp_mase  = [all_results[pid][comp]['MASE']  for pid in all_results]

    stat_mae,   p_mae   = stats.mannwhitneyu(ctr_mae_vals,   comp_mae,   alternative='less')
    stat_smape, p_smape = stats.mannwhitneyu(ctr_smape_vals, comp_smape, alternative='less')
    stat_mase,  p_mase  = stats.mannwhitneyu(ctr_mase_vals,  comp_mase,  alternative='less')

    mw_results.append({
        'Comparison': f'CTR vs {comp}',
        'MAE_statistic': stat_mae,   'MAE_pvalue': p_mae,   'MAE_significant': p_mae < 0.05,
        'sMAPE_statistic': stat_smape, 'sMAPE_pvalue': p_smape, 'sMAPE_significant': p_smape < 0.05,
        'MASE_statistic': stat_mase,  'MASE_pvalue': p_mase,  'MASE_significant': p_mase < 0.05,
    })

mw_df = pd.DataFrame(mw_results)
mw_df.to_csv(os.path.join(OUT_DIR, 'mann_whitney_results.csv'), index=False)
print("\nMann-Whitney U Test Results (CTR vs others, alternative='less' = CTR better):")
print(mw_df[['Comparison','MAE_pvalue','MAE_significant','sMAPE_pvalue','sMAPE_significant','MASE_pvalue','MASE_significant']].to_string())

end_time = time.time()
total_time = end_time - start_time

print(f"\nTotal time elapsed: {total_time:.2f} seconds")
print("\nMedian Metrics Table (PRIMARY):")
print(median_table_df[['Model','MAE','sMAPE','MASE','RMSSE']].to_string())
print("\nMean Metrics Table (reference):")
print(mean_table_df[['Model','MAE','sMAPE','MASE','RMSSE']].to_string())