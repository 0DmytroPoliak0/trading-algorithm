import os
import yfinance as yf
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from keras_tuner.tuners import RandomSearch
from keras_tuner import HyperModel
import logging
import time
import shutil

# Constants
SEQUENCE_LENGTH = 10
FUTURE_STEPS = 5
MAX_TRIALS = 3
EPOCHS = 5
DATA_FOLDER = 'temp_data'

# Set up logging
logging.basicConfig(level=logging.INFO)

# Ensure the data folder exists
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)


def fetch_historical_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
        data.to_csv(f'{DATA_FOLDER}/{ticker}.csv')
        logging.info(f"Fetched data for {ticker} from {start_date} to {end_date}")
        return data
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return None


def fetch_news_sentiment(query):
    try:
        api_key = "3ea7f60856f544559111f37671ed2dda"
        url = f"https://newsapi.org/v2/everything?q={query}&language=en&apiKey={api_key}"
        response = requests.get(url)
        articles = response.json()['articles']
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = [analyzer.polarity_scores(article['title'])['compound'] for article in articles]
        sentiment = np.mean(sentiment_scores)
        logging.info(f"Fetched sentiment for {query}: {sentiment}")
        return sentiment
    except Exception as e:
        logging.error(f"Error fetching sentiment for {query}: {e}")
        return 0


def add_technical_indicators(data):
    try:
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        data['Volume_SMA_20'] = data['Volume'].rolling(window=20).mean()
        data = data.dropna()
        logging.info("Added technical indicators")
        return data
    except Exception as e:
        logging.error(f"Error adding technical indicators: {e}")
        return None


def preprocess_data(data, sequence_length=SEQUENCE_LENGTH):
    try:
        features = ['Close', 'SMA_20', 'SMA_50', 'EMA_20', 'Volume_SMA_20']
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[features].values)

        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:i + sequence_length])
            y.append(scaled_data[i + sequence_length, 0])  # Predicting 'Close' price

        X, y = np.array(X), np.array(y)
        logging.info("Preprocessed data")
        return X, y, scaler
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        return None, None, None


class CryptoPriceHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Input(shape=(hp.Int('sequence_length', min_value=5, max_value=10, step=5), 5)))  # 5 features
        for i in range(hp.Int('num_layers', 1, 2)):
            model.add(Bidirectional(GRU(units=hp.Int(f'units_{i}', min_value=50, max_value=100, step=50),
                                        return_sequences=True if i < hp.Int('num_layers', 1, 2) - 1 else False)))
            model.add(Dropout(rate=hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.3, step=0.1)))
        model.add(Dense(units=1))
        model.compile(optimizer=Adam(learning_rate=hp.Float('lr', min_value=1e-4, max_value=1e-3, sampling='LOG')),
                      loss='mean_squared_error')
        return model


def tune_hyperparameters(X_train, y_train, X_val, y_val):
    tuner = RandomSearch(
        CryptoPriceHyperModel(),
        objective='val_loss',
        max_trials=MAX_TRIALS,
        executions_per_trial=1,
        directory='hypermodel_dir',
        project_name='crypto_price_prediction'
    )

    tuner.search(X_train, y_train, epochs=EPOCHS, validation_data=(X_val, y_val))
    best_model = tuner.get_best_models(num_models=1)[0]
    logging.info("Performed hyperparameter tuning")
    return best_model


def predict_future_prices(model, data, scaler, sequence_length, steps, sentiment):
    last_sequence = data[-sequence_length:]
    future_predictions = []
    for _ in range(steps):
        last_sequence_scaled = scaler.transform(last_sequence)  # Scale the sequence
        last_sequence_scaled = last_sequence_scaled.reshape(1, sequence_length, -1)  # Reshape for model input
        next_prediction = model.predict(last_sequence_scaled)
        next_prediction += sentiment
        next_prediction_inverse = scaler.inverse_transform(
            np.concatenate((next_prediction, np.zeros((next_prediction.shape[0], 4))), axis=1))
        future_predictions.append(next_prediction_inverse[0, 0])
        last_sequence = np.append(last_sequence[1:], [next_prediction_inverse[0, :5]], axis=0)
    logging.info("Predicted future prices")
    return future_predictions


def backtest(model, data, scaler, sequence_length, steps, sentiment):
    try:
        features = ['Close', 'SMA_20', 'SMA_50', 'EMA_20', 'Volume_SMA_20']
        test_data = data[-(steps + sequence_length):]
        scaled_test_data = scaler.transform(test_data[features].values)

        X_test, y_test = [], []
        for i in range(len(scaled_test_data) - sequence_length):
            X_test.append(scaled_test_data[i:i + sequence_length])
            y_test.append(scaled_test_data[i + sequence_length, 0])  # Predicting 'Close' price

        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = X_test.reshape((X_test.shape[0], sequence_length, 5))  # Ensure consistent input shape
        predictions = predict_future_prices(model, test_data[features].values, scaler, sequence_length, steps,
                                            sentiment)
        actual = y_test[:steps]
        logging.info("Performed backtesting")
        return actual, predictions
    except Exception as e:
        logging.error(f"Error in backtesting: {e}")
        return [], []


def generate_signals(prices, predictions):
    signals = np.zeros_like(predictions)
    signals[1:] = np.where(predictions[1:] > predictions[:-1], 1, -1)
    logging.info("Generated buy/sell signals")
    return signals


def plot_predictions_and_signals(dates, actual, predictions, signals, title):
    try:
        plt.figure(figsize=(14, 7))
        plt.plot(dates[-len(actual):], actual, label=f'{title} Actual')
        plt.plot(dates[-len(predictions):], predictions, label=f'{title} Predicted')
        buy_signals = dates[-len(predictions):][signals == 1]
        sell_signals = dates[-len(predictions):][signals == -1]
        plt.plot(buy_signals, [predictions[i] for i in range(len(signals)) if signals[i] == 1], '^', markersize=10,
                 color='g', lw=0, label='Buy Signal')
        plt.plot(sell_signals, [predictions[i] for i in range(len(signals)) if signals[i] == -1], 'v', markersize=10,
                 color='r', lw=0, label='Sell Signal')
        plt.title(f'Predicted and Actual Prices with Buy/Sell Signals for {title}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        logging.info(f"Plotted predictions and signals for {title}")
    except Exception as e:
        logging.error(f"Error in plotting: {e}")


def main():
    start_time = time.time()

    btc_data = fetch_historical_data('BTC-USD', '2015-01-01', '2024-06-20')
    eth_data = fetch_historical_data('ETH-USD', '2015-01-01', '2024-06-20')
    btt_data = fetch_historical_data('BTT-USD', '2019-01-01', '2024-06-20')

    btc_sentiment = fetch_news_sentiment('Bitcoin')
    eth_sentiment = fetch_news_sentiment('Ethereum')
    btt_sentiment = fetch_news_sentiment('BitTorrent')

    btc_data = add_technical_indicators(btc_data)
    eth_data = add_technical_indicators(eth_data)
    btt_data = add_technical_indicators(btt_data)

    btc_X, btc_y, btc_scaler = preprocess_data(btc_data)
    eth_X, eth_y, eth_scaler = preprocess_data(eth_data)
    btt_X, btt_y, btt_scaler = preprocess_data(btt_data)

    btc_train_size = int(len(btc_X) * 0.8)
    eth_train_size = int(len(eth_X) * 0.8)
    btt_train_size = int(len(btt_X) * 0.8)

    btc_X_train, btc_X_val = btc_X[:btc_train_size], btc_X[btc_train_size:]
    btc_y_train, btc_y_val = btc_y[:btc_train_size], btc_y[btc_train_size:]

    eth_X_train, eth_X_val = eth_X[:eth_train_size], eth_X[eth_train_size:]
    eth_y_train, eth_y_val = eth_y[:eth_train_size], eth_y[eth_train_size:]

    btt_X_train, btt_X_val = btt_X[:btt_train_size], btt_X[btt_train_size:]
    btt_y_train, btt_y_val = btt_y[:btt_train_size], btt_y[btt_train_size:]

    btc_best_model = tune_hyperparameters(btc_X_train, btc_y_train, btc_X_val, btc_y_val)
    eth_best_model = tune_hyperparameters(eth_X_train, eth_y_train, eth_X_val, eth_y_val)
    btt_best_model = tune_hyperparameters(btt_X_train, btt_y_train, btt_X_val, btt_y_val)

    btc_future_prices = predict_future_prices(btc_best_model, btc_X_val[:, -1, :], btc_scaler, SEQUENCE_LENGTH,
                                              FUTURE_STEPS, btc_sentiment)
    eth_future_prices = predict_future_prices(eth_best_model, eth_X_val[:, -1, :], eth_scaler, SEQUENCE_LENGTH,
                                              FUTURE_STEPS, eth_sentiment)
    btt_future_prices = predict_future_prices(btt_best_model, btt_X_val[:, -1, :], btt_scaler, SEQUENCE_LENGTH,
                                              FUTURE_STEPS, btt_sentiment)

    btc_actual, btc_predictions = backtest(btc_best_model, btc_data, btc_scaler, SEQUENCE_LENGTH, FUTURE_STEPS,
                                           btc_sentiment)
    eth_actual, eth_predictions = backtest(eth_best_model, eth_data, eth_scaler, SEQUENCE_LENGTH, FUTURE_STEPS,
                                           eth_sentiment)
    btt_actual, btt_predictions = backtest(btt_best_model, btt_data, btt_scaler, SEQUENCE_LENGTH, FUTURE_STEPS,
                                           btt_sentiment)

    if len(btc_actual) == 0 or len(btc_predictions) == 0:
        logging.error("BTC backtesting produced empty arrays.")
    if len(eth_actual) == 0 or len(eth_predictions) == 0:
        logging.error("ETH backtesting produced empty arrays.")
    if len(btt_actual) == 0 or len(btt_predictions) == 0:
        logging.error("BTT backtesting produced empty arrays.")

    btc_signals = generate_signals(btc_data['Close'].values[-len(btc_predictions):], btc_predictions)
    eth_signals = generate_signals(eth_data['Close'].values[-len(eth_predictions):], eth_predictions)
    btt_signals = generate_signals(btt_data['Close'].values[-len(btt_predictions):], btt_predictions)

    dates = btc_data.index
    if len(btc_actual) == len(btc_predictions):
        plot_predictions_and_signals(dates, btc_actual, btc_predictions, btc_signals, 'Bitcoin')
    if len(eth_actual) == len(eth_predictions):
        plot_predictions_and_signals(dates, eth_actual, eth_predictions, eth_signals, 'Ethereum')
    if len(btt_actual) == len(btt_predictions):
        plot_predictions_and_signals(dates, btt_actual, btt_predictions, btt_signals, 'BitTorrent')

    shutil.rmtree(DATA_FOLDER)

    end_time = time.time()
    logging.info(f"Total execution time: {end_time - start_time} seconds")


if __name__ == "__main__":
    main()
