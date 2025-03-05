import os
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
from datetime import datetime, timedelta

# Constants
SEQUENCE_LENGTH = 24  # More data points for each sequence
FUTURE_STEPS = 3
EPOCHS = 20
DATA_INTERVAL = "1h"
DATA_PERIOD = "30d"
DATA_FOLDER = 'temp_data'

# Logging setup
logging.basicConfig(level=logging.INFO)

# Ensure data folder exists
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)


def fetch_historical_data(ticker):
    """Fetch historical data for the given ticker."""
    data = yf.download(ticker, period="1mo", interval=DATA_INTERVAL)  # Use '1mo' for one month
    if data.empty:
        raise ValueError("No data fetched. Check the ticker or data availability.")

    # Ensure there is enough data for the given SEQUENCE_LENGTH and FUTURE_STEPS
    if len(data) < SEQUENCE_LENGTH + FUTURE_STEPS:
        raise ValueError(
            f"Insufficient data: Need at least {SEQUENCE_LENGTH + FUTURE_STEPS} entries, but got {len(data)}."
        )

    # Keep only the 'Close' column and drop any missing values
    data = data[['Close']].dropna()
    logging.info(f"Fetched data for {ticker} with {DATA_INTERVAL} interval over 1 month")
    return data



def add_features(data):
    """Add technical indicators to the dataset."""
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = compute_rsi(data['Close'], window=14)
    data.dropna(inplace=True)
    return data


def compute_rsi(series, window=14):
    """Compute the Relative Strength Index (RSI)."""
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def preprocess_data(data, sequence_length=SEQUENCE_LENGTH):
    """Preprocess the data for the model."""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length, 0])

    return np.array(X), np.array(y), scaler


def build_model(input_shape):
    """Build and compile the GRU model."""
    model = Sequential([
        Bidirectional(GRU(64, return_sequences=True, input_shape=input_shape)),
        Dropout(0.2),
        Bidirectional(GRU(64)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
    return model


def predict_future_prices(model, last_sequence, scaler, sequence_length, steps):
    future_predictions = []
    for _ in range(steps):
        # Scale the input sequence
        last_sequence_scaled = scaler.transform(last_sequence)
        last_sequence_scaled = last_sequence_scaled.reshape(1, sequence_length, -1)

        # Predict the next value
        next_prediction = model.predict(last_sequence_scaled, verbose=0)[0, 0]
        future_predictions.append(next_prediction)

        # Update the last sequence for the next prediction
        new_row = np.array([next_prediction] + [0] * (last_sequence.shape[1] - 1)).reshape(1, -1)
        last_sequence = np.vstack([last_sequence[1:], new_row])

    # Inverse transform the predictions
    future_predictions_array = np.array(future_predictions).reshape(-1, 1)
    padded_predictions = np.hstack([future_predictions_array, np.zeros((len(future_predictions), scaler.n_features_in_ - 1))])
    return scaler.inverse_transform(padded_predictions)[:, 0]



def plot_predictions(dates, actual, predictions, signals, mae, mse):
    """Plot actual and predicted prices with buy/sell signals."""
    plt.figure(figsize=(14, 7))
    plt.plot(dates[-len(actual):], actual, label="Actual Prices", color="blue", marker='o')

    future_dates = [dates[-1] + timedelta(hours=i + 1) for i in range(len(predictions))]
    plt.plot(future_dates, predictions, label="Predicted Prices", color="orange", marker='o')

    for i, signal in enumerate(signals):
        if signal == "Buy":
            plt.scatter(future_dates[i], predictions[i], label="Buy Signal" if i == 0 else "", color="green", marker="^")
        elif signal == "Sell":
            plt.scatter(future_dates[i], predictions[i], label="Sell Signal" if i == 0 else "", color="red", marker="v")

    plt.title(f"Bitcoin Price Predictions with Buy/Sell Signals\nMAE: {mae:.2f}, MSE: {mse:.2f}")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid()
    plt.show()


def main():
    # Fetch and preprocess data
    btc_data = fetch_historical_data('BTC-USD')
    btc_data = add_features(btc_data)

    X, y, scaler = preprocess_data(btc_data)
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Build and train the model
    model = build_model(input_shape=(SEQUENCE_LENGTH, X.shape[2]))
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=32, verbose=1)

    # Predict future prices
    last_sequence = X_val[-1]
    predictions = predict_future_prices(model, last_sequence, scaler, SEQUENCE_LENGTH, FUTURE_STEPS)

    # Calculate metrics
    mae = mean_absolute_error(y_val[-FUTURE_STEPS:], predictions)
    mse = mean_squared_error(y_val[-FUTURE_STEPS:], predictions)

    # Generate buy/sell signals
    signals = ["Buy" if predictions[i] > predictions[i - 1] else "Sell" for i in range(1, len(predictions))]

    # Plot results
    plot_predictions(btc_data.index, btc_data['Close'].values, predictions, signals, mae, mse)


if __name__ == "__main__":
    main()
