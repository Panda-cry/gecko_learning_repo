from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class CSVReader:

    def __init__(self, path):
        self.path = Path(path)
        self.data: pd.DataFrame = None
        self.feature_output = None

    def get_data(self):
        try:
            self.data = pd.read_csv(self.path)
        except Exception as ex:
            print(ex.args[0])

    def convert_feature_output(self):
        le = LabelEncoder()
        feature_output = le.fit_transform(self.data['weather'])
        self.data.drop('weather', axis=1, inplace=True)
        self.data.drop('date', axis=1, inplace=True)
        self.feature_output = feature_output

    def normalize_input_data(self):
        scaler = MinMaxScaler()
        scaler = scaler.fit(self.data)
        self.data = scaler.transform(self.data)

    def split_data(self):
        return train_test_split(self.data, self.feature_output, test_size=0.2, shuffle=True, random_state=42)

    def main(self):
        self.get_data()
        self.convert_feature_output()
        #self.normalize_input_data()


class WeatherForecast:

    def __init__(self, train_X, test_X, train_y, test_y):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.model = None

    def crate_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(32,activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(5,activation=tf.keras.activations.softmax),
        ])
        self.model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                           optimizer=tf.optimizers.Adam(),
                           metrics=["accuracy"])

        history = self.model.fit(self.train_X, self.train_y, epochs=100, verbose=0)

        pd.DataFrame(history.history).plot(figsize=(10, 7))
        plt.show()

    def evaluate_model(self):
        self.model.evaluate(self.test_X, self.test_y)

    def get_predictions(self):
        return self.model.predict(self.test_X)




if __name__ == "__main__":
    csv = CSVReader("seattle-weather.csv")
    csv.main()
    X_train, X_test, y_train, y_test = csv.split_data()
    weather = WeatherForecast(train_X=X_train, test_X=X_test, train_y=y_train, test_y=y_test)
    weather.crate_model()
    weather.evaluate_model()

#Trenutno preciznost je 82 posto uvodi se problem jer imamo fog koja remeti predikciju