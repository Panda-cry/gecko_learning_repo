import os.path

import numpy as np
import pandas as pd
import tensorflow as tf

# load data i posavka dates da bude index colona
data_frame = pd.read_csv('bitcoin_data.csv',
                         # parse_dates=['Date'],
                         # index_col=['Date']
                         )

# Kod time series treba znati da ne mozemo random da delimo podatke
# sa train_test_split treba da pazimo da uzimamo podatke za trening iz proslosti
# a za future tj test nest iz buducnosti
prices = data_frame['Closing Price (USD)'].to_numpy()
times = data_frame['Date'].to_numpy()
split_size = int(0.8 * len(prices))

# data_frame kao takav je multivariant jer bitcoin predstavlja vise kolona

bitcoin_unvariant = pd.DataFrame(data_frame['Closing Price (USD)']).rename(columns={'Closing Price (USD)': "Price"})
# bitcoin_unvariant je predstavljen samo jednom kolonom

# Create train data splits (everything before the split)
X_train, y_train = times[:split_size], prices[:split_size]

# Create test data splits (everything after the split)
X_test, y_test = times[split_size:], prices[split_size:]


# window / horizon koliko vremenski cemo uzeti iz proslosti podataka / vreme koje treba da pogodimo


def evaluate_predictions(y_predicted, y_true):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_predicted, dtype=tf.float32)

    mae = tf.metrics.mean_absolute_error(y_true, y_pred)
    mse = tf.metrics.mean_squared_error(y_true, y_pred)
    rmse = tf.sqrt(mse)

    mape = tf.metrics.mean_absolute_percentage_error(y_true, y_pred)

    print(f"mae : {mae.numpy()}, mse : {mse.numpy()}, rmse : {rmse.numpy()}, mape : {mape.numpy()}")


HORIZON = 1
WINDOW_SIZE = 7


def get_labelled_data(x, horizon=1):
    return x[:, :-horizon], x[:, -horizon:]


def make_window(x, window_size=WINDOW_SIZE, horizon=HORIZON):
    window_step = np.expand_dims(np.arange(window_size + horizon), axis=0)
    window_index = window_step + np.expand_dims(np.arange(len(x) - (window_size + horizon - 1)), axis=0).T

    windowed_array = x[window_index]

    # 4. Get the labelled windows
    windows, labels = get_labelled_data(windowed_array, horizon=horizon)
    return windows, labels


full_windows, full_labels = make_window(prices, window_size=WINDOW_SIZE, horizon=HORIZON)


def make_train_test_splits(windows, labels, test_split=0.2):
    """
    Splits matching pairs of windows and labels into train and test splits.
    """
    split_size = int(len(windows) * (1 - test_split))  # this will default to 80% train/20% test
    train_windows = windows[:split_size]
    train_labels = labels[:split_size]
    test_windows = windows[split_size:]
    test_labels = labels[split_size:]
    return train_windows, test_windows, train_labels, test_labels


train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)


def create_model_checkpoint(model_name, save_path="model_experiments"):
    return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name), verbose=0,
                                              save_best_only=True)

#model je proban sa horizon=1/7 window= 7/30
def model_one():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(HORIZON, activation="linear")
    ])

    model.compile(loss="mae",
                  optimizer="Adam",
                  metrics="mae")

    model.fit(train_windows,train_labels,epochs=100,batch_size=128, validation_data=(test_windows,test_labels),
              #callbacks=[create_model_checkpoint("model_one")]
    )



def model_two():
    model = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x : tf.expand_dims(x, axis=1)),
        tf.keras.layers.Conv1D(filters=128,kernel_size=5,padding="causal", activation="relu"),
        tf.keras.layers.Dense(HORIZON)
    ])

    model.compile(loss="mae",
                  optimizer="Adam",
                  metrics="mae")

    model.fit(train_windows, train_labels, epochs=100, batch_size=128, validation_data=(test_windows, test_labels),
              verbose=0
              # callbacks=[create_model_checkpoint("model_one")]
              )
    model.evaluate(test_windows,test_labels)


def model_three_rnn():
    model = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1)),
        tf.keras.layers.LSTM(128, activation="relu"),
        tf.keras.layers.Dense(HORIZON)
    ])

    model.compile(loss="mae",
                  optimizer="Adam",
                  metrics="mae")

    model.fit(train_windows, train_labels, epochs=100, batch_size=128, validation_data=(test_windows, test_labels),
              verbose=0
              # callbacks=[create_model_checkpoint("model_one")]
              )
    model.evaluate(test_windows, test_labels)


# Block reward values
block_reward_1 = 50 # 3 January 2009 (2009-01-03) - this block reward isn't in our dataset (it starts from 01 October 2013)
block_reward_2 = 25 # 28 November 2012
block_reward_3 = 12.5 # 9 July 2016
block_reward_4 = 6.25 # 11 May 2020


block_reward_2_datetime = np.datetime64("2012-11-28")
block_reward_3_datetime = np.datetime64("2016-07-09")
block_reward_4_datetime = np.datetime64("2020-05-11")
# Add block_reward column
bitcoin_prices_block = prices.copy()
bitcoin_prices_block["block_reward"] = 0

bitcoin_prices_block.iloc[:1012, -1] = block_reward_2
bitcoin_prices_block.iloc[1012:2414, -1] = block_reward_3
bitcoin_prices_block.iloc[2414:, -1] = block_reward_4

bitcoin_prices_windowed = bitcoin_prices_block.copy()

# Add windowed columns
for i in range(WINDOW_SIZE): # Shift values for each step in WINDOW_SIZE
  bitcoin_prices_windowed[f"Price+{i+1}"] = bitcoin_prices_windowed["Price"].shift(periods=i+1)
bitcoin_prices_windowed.head(10)


X = bitcoin_prices_windowed.dropna().drop("Price", axis=1).astype(np.float32)
y = bitcoin_prices_windowed.dropna()["Price"].astype(np.float32)

X_train, y_train = X[:split_size], y[:split_size]
X_test, y_test = X[split_size:], y[split_size:]
len(X_train), len(y_train), len(X_test), len(y_test)

def model_four():
    model = tf.keras.Sequential([
        # tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1)),
        # tf.keras.layers.LSTM(128, activation="relu"),
        tf.keras.layers.Dense(128,activation="relu"),
        tf.keras.layers.Dense(HORIZON)
    ])

    model.compile(loss="mae",
                  optimizer="Adam",
                  metrics="mae")

    model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_test, y_test),
              verbose=0
              # callbacks=[create_model_checkpoint("model_one")]
              )
    model.evaluate(test_windows, test_labels)

#Napomena podatke treba nekako upakovati tensorflow ima fju batch i prefatch da iskoritimo maksimalno resurase racunara
#Podaci za forecasting ne bi trebali da budu podeljeni na test i train jer je ceo taj vremenski opseg train
#nikada se ne zna u buducosti sta ce da se desi i da li ce doci do neke promene
#npr kada se racuna predikcija potrosnje energije tu su praznici/ slave / proslave koje mogu da povecaju potrosnju na nivou grada
#pritom dolazi do pogorsanja prediktovanja
#Naive forecast je predikcija tipa jedan za drugim a*X = Y za ovaj slucaj je 2 nabolja opcija
#Jos bolja je da uzmemo mae , mse , rmse i jos par da napravimo modele i da sve zajedno uporedimo

#Neizvesnost je na nivou podataka i modela
#nekako da olaksamo neizvesnost modela je da upoznamo bolje podatke da dodamo mozda jos nesto da modelu bude jasnije i laske da uci

if __name__ == "__main__":
    model_four()