import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load historical Bitcoin data from Yahoo Finance
data = yf.download('BTC-USD', start='2015-01-01', end='2024-06-15', interval='1d')
data.reset_index(inplace=True)

# Convert the date column to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Use only the 'Close' column for prediction
prices = data['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Create sequences for training
sequence_length = 60
X, y = [], []
for i in range(len(scaled_prices) - sequence_length):
    X.append(scaled_prices[i:i + sequence_length])
    y.append(scaled_prices[i + sequence_length])
X, y = np.array(X), np.array(y)

# Split the data into training and test sets
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build the LSTM model with dropout layers
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=100))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Make predictions on the test set
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Prepare data for plotting
data['Predicted'] = np.nan
data.iloc[-len(predicted_prices):, data.columns.get_loc('Predicted')] = predicted_prices.flatten()

# Plot the actual and predicted prices
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Close Price', alpha=0.5)
plt.plot(data['Predicted'], label='Predicted Price', alpha=0.75)
plt.title('Bitcoin Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Function to predict future prices
def predict_future_prices(model, data, scaler, steps):
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

# Predict future prices for the next 6 months (180 days)
future_steps = 180
future_predictions = predict_future_prices(model, prices, scaler, future_steps)

# Create a new DataFrame to hold the future predictions
last_date = data.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps, freq='D')
future_df = pd.DataFrame({'Date': future_dates, 'Predicted': future_predictions})
future_df.set_index('Date', inplace=True)

# Concatenate the historical data with the future predictions
full_df = pd.concat([data, future_df])

# Define buy and sell signals based on predictions
full_df['Signal'] = 0
full_df['Signal'][-(len(predicted_prices) + future_steps):] = np.where(full_df['Predicted'][-(len(predicted_prices) + future_steps):] > full_df['Predicted'].shift(1)[-(len(predicted_prices) + future_steps):], 1, 0)
full_df['Position'] = full_df['Signal'].diff()

# Plot buy and sell signals
plt.figure(figsize=(14, 7))
plt.plot(full_df['Close'], label='Close Price', alpha=0.5)
plt.plot(full_df['Predicted'], label='Predicted Price', alpha=0.75)
plt.plot(full_df[full_df['Position'] == 1].index, full_df['Predicted'][full_df['Position'] == 1], '^', markersize=10, color='g', lw=0, label='Buy Signal')
plt.plot(full_df[full_df['Position'] == -1].index, full_df['Predicted'][full_df['Position'] == -1], 'v', markersize=10, color='r', lw=0, label='Sell Signal')
plt.title('Bitcoin Price Prediction and Buy/Sell Signals')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Simulate leveraged inverse ETF returns based on predicted price movements
leveraged_inverse_returns = -2 * full_df['Predicted'].pct_change().dropna()
leveraged_inverse_returns[full_df['Position'] == 1] = leveraged_inverse_returns[full_df['Position'] == 1] * -1  # Inverse for buy signals

# Calculate cumulative returns
cumulative_returns = (leveraged_inverse_returns + 1).cumprod() - 1

# Plot cumulative returns
plt.figure(figsize=(14, 7))
plt.plot(cumulative_returns, label='Leveraged Inverse ETF Returns')
plt.title('Simulated Leveraged Inverse ETF Returns Based on Predicted Bitcoin Prices')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()
