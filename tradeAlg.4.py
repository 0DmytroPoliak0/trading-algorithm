import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load historical Bitcoin data from Yahoo Finance
btc_data = yf.download('BTC-USD', start='2015-01-01', end='2024-06-15', interval='1d')
btc_data.reset_index(inplace=True)

# Convert the date column to datetime and set as index
btc_data['Date'] = pd.to_datetime(btc_data['Date'])
btc_data.set_index('Date', inplace=True)

# Use only the 'Close' column for prediction
btc_prices = btc_data['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_btc_prices = scaler.fit_transform(btc_prices)

# Create sequences for training
sequence_length = 60
X_btc, y_btc = [], []
for i in range(len(scaled_btc_prices) - sequence_length):
    X_btc.append(scaled_btc_prices[i:i + sequence_length])
    y_btc.append(scaled_btc_prices[i + sequence_length])
X_btc, y_btc = np.array(X_btc), np.array(y_btc)

# Split the data into training and test sets
split = int(len(X_btc) * 0.8)
X_train_btc, X_test_btc = X_btc[:split], X_btc[split:]
y_train_btc, y_test_btc = y_btc[:split], y_btc[split:]

# Build the LSTM model with dropout layers
btc_model = Sequential()
btc_model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train_btc.shape[1], 1)))
btc_model.add(Dropout(0.2))
btc_model.add(LSTM(units=100))
btc_model.add(Dropout(0.2))
btc_model.add(Dense(units=1))

# Compile the model
optimizer = Adam(learning_rate=0.001)
btc_model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
btc_model.fit(X_train_btc, y_train_btc, epochs=50, batch_size=32)

# Function to predict future prices
def predict_future_prices(model, data, scaler, sequence_length, steps):
    last_sequence = data[-sequence_length:]
    future_predictions = []
    for _ in range(steps):
        last_sequence_scaled = scaler.transform(last_sequence)
        last_sequence_scaled = np.expand_dims(last_sequence_scaled, axis=0)
        next_prediction = model.predict(last_sequence_scaled)
        next_prediction_inverse = scaler.inverse_transform(next_prediction)
        future_predictions.append(next_prediction_inverse[0, 0])
        last_sequence = np.append(last_sequence[1:], next_prediction_inverse)
        last_sequence = last_sequence.reshape(-1, 1)
    return future_predictions

# Predict future prices for Bitcoin for the next 6 months (180 days)
future_steps = 180
btc_future_predictions = predict_future_prices(btc_model, btc_prices, scaler, sequence_length, future_steps)

# Load ETF data from Yahoo Finance
etfs = ['BITI', 'SH', 'GLD']
etf_data = {}

for etf in etfs:
    etf_data[etf] = yf.download(etf, start='2020-01-01', end='2024-06-13', interval='1d')
    etf_data[etf].reset_index(inplace=True)
    etf_data[etf].set_index('Date', inplace=True)

# Function to preprocess ETF data
def preprocess_data(data, sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length])
    
    X, y = np.array(X), np.array(y)
    return X, y, scaler

# Preprocess each ETF's adjusted close prices
etf_X, etf_y, etf_scalers = {}, {}, {}

for etf in etfs:
    etf_close = etf_data[etf]['Adj Close'].values.reshape(-1, 1)
    X, y, scaler = preprocess_data(etf_close, sequence_length)
    etf_X[etf], etf_y[etf], etf_scalers[etf] = X, y, scaler

# Function to predict future prices for ETFs
def predict_etf_prices(model, X, scaler, sequence_length, steps=180):
    future_predictions = []
    last_sequence = X[-1]
    
    for _ in range(steps):
        next_prediction = model.predict(np.expand_dims(last_sequence, axis=0))
        future_predictions.append(next_prediction)
        last_sequence = np.append(last_sequence[1:], next_prediction)
        last_sequence = last_sequence.reshape(-1, 1)
    
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
    return future_predictions

# Predict future prices for each ETF based on Bitcoin predictions
etf_future_predictions = {}

for etf in etfs:
    etf_future_predictions[etf] = predict_etf_prices(btc_model, etf_X[etf], etf_scalers[etf], sequence_length, future_steps)

# Analyze and plot future predictions for each ETF
plt.figure(figsize=(14, 7))

for etf in etfs:
    future_dates = pd.date_range(start=etf_data[etf].index[-1] + pd.Timedelta(days=1), periods=future_steps, freq='D')
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted': etf_future_predictions[etf].flatten()})
    future_df.set_index('Date', inplace=True)
    
    full_df = pd.concat([etf_data[etf]['Adj Close'], future_df['Predicted']], axis=1)
    full_df.columns = ['Actual', 'Predicted']
    
    plt.plot(full_df.index, full_df['Predicted'], label=f'{etf} Predicted')

plt.title('Predicted ETF Performance Based on Bitcoin Price Predictions')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Calculate potential returns
returns = {}

for etf in etfs:
    start_price = etf_data[etf]['Adj Close'].iloc[-1]
    end_price = etf_future_predictions[etf][-1]
    returns[etf] = (end_price - start_price) / start_price

best_etf = max(returns, key=returns.get)
print(f"The best ETF to purchase based on predictions is: {best_etf} with an expected return of {returns[best_etf] * 100:.2f}%")

