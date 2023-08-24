import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def read_from_csv():
    housing_price = pd.read_csv('USA_Housing.csv')
    #adresa nam nije od nekog znacaja trenutno
    housing_price = housing_price.drop("Address", axis=1)
    #Radi lakseg razumevanja zaokruzicemo sve na 2 decimale
    #housing_price = housing_price.round(2)
    #Odvajamo zavisne promenljive
    y_values = housing_price['Price'].to_numpy()
    housing_price = housing_price.drop("Price", axis=1)
    #skalirani su podaci izmedju 0,1
    scaler = MinMaxScaler()
    scaled_data = scaler.fit(housing_price)
    scaled_data = scaler.transform(housing_price)
    #Podela podatak na test i train
    X_train, X_test, y_train, y_test = train_test_split(scaled_data,y_values,test_size=0.1,random_state=2)
    return X_train, X_test, y_train, y_test


def  run_model(X_train, X_test, y_train, y_test):

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64,activation="relu"),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(32,activation="relu"),
        tf.keras.layers.Dense(32),
        # tf.keras.layers.Dense(32),
        # tf.keras.layers.Dropout(rate=0.001),
        tf.keras.layers.Dense(16),
        # tf.keras.layers.Dense(16,activation="relu"),
        tf.keras.layers.Dense(8),
        tf.keras.layers.Dense(1),
    ])

    model.compile(loss="huber",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['mape'])

    model.fit(X_train,y_train,epochs=100)

    y_predicted = model.predict(X_test)

    plot_prediction(y_predicted, y_test)

def plot_prediction(predicted, expected):
    import matplotlib.pyplot as plt
    c = [i for i in range(1, 501, 1)]
    plt.plot(c,predicted,color="blue")
    plt.plot(c,expected,color="red")
    plt.show()

#Trenutno model gresi oko 7 posto

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = read_from_csv()
    run_model(X_train, X_test, y_train, y_test)