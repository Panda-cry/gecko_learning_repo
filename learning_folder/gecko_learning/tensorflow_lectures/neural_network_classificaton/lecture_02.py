import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
# Pojam klasifikacije je da imamo na primer ulazni parametre i sada treba da donesemo odluku neku. Npr da od nekih parametara
# odredimo da li je neko bolestan ili ne to je binarna klasifikacija, Multi-class kalsifikacija da od odredjenih stvari
# na primer slika odredimo sta je na njoj, multi- label kalsifikacija sta sve moze da predstavi nesto

from sklearn.datasets import make_circles
from sklearn.preprocessing import MinMaxScaler
tf.random.set_seed(42)


def _plot_decision_boundary(model, X, y):
    """
    Plots the decision boundary created by a model predicting on X.
    This function has been adapted from two phenomenal resources:
     1. CS231n - https://cs231n.github.io/neural-networks-case-study/
     2. Made with ML basics - https://github.com/GokuMohandas/MadeWithML/blob/main/notebooks/08_Neural_Networks.ipynb
    """
    # Define the axis boundaries of the plot and create a meshgrid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Create X values (we're going to predict on all of these)
    x_in = np.c_[
        xx.ravel(), yy.ravel()]  # stack 2D arrays together: https://numpy.org/devdocs/reference/generated/numpy.c_.html

    # Make predictions using the trained model
    y_pred = model.predict(x_in)

    # Check for multi-class
    if model.output_shape[
        -1] > 1:  # checks the final dimension of the model's output shape, if this is > (greater than) 1, it's multi-class
        print("doing multiclass classification...")
        # We have to reshape our predictions to get them ready for plotting
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print("doing binary classifcation...")
        y_pred = np.round(np.max(y_pred, axis=1)).reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

def _first_example():
    n_samples = 1000

    X, y = make_circles(n_samples, noise=0.03, random_state=42)
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4,activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(4,activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(1,activation=tf.keras.activations.sigmoid),
    ])

    model.compile(loss=tf.losses.BinaryCrossentropy(),
                  optimizer=tf.optimizers.Adam(learning_rate=0.01),
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=50, verbose=0)
    #model.evaluate(X,y)
    #_plot_decision_boundary(model,X,y)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Model loss on the test set: {loss}")
    print(f"Model accuracy on the test set: {100 * accuracy:.2f}%")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    _plot_decision_boundary(model, X=X_train, y=y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    _plot_decision_boundary(model, X=X_test, y=y_test)
    plt.show()

def _multiclass_classification():
    (train_data, train_lables), (test_data,test_labels) = fashion_mnist.load_data()

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(4,activation="relu"),
        tf.keras.layers.Dense(4,activation="relu"),
        tf.keras.layers.Dense(10,activation="softmax"),
    ])
    scaler = MinMaxScaler()
    train_data = train_data / 255
    test_data = test_data / 255
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer="Adam",
                  metrics=['accuracy'])

    model.fit(train_data,
              train_lables,
              epochs=10,
              validation_data=(test_data,test_labels))


#Funkcionisanje mreze kada se okida fit tada se ustvari popravljaju tezine u mrezi tj biasi ,
#Nakon svake iteracije sto su epohe kod nas na pocetku se popravlajju inicijalne tezine
#uskaldjivanje podataka tako da lepo pogadja je popravljanje vrednosti biasa
#Kada se desi da podaci koji se treniraju pogrese mnogo tada optimizer se trudi da popravi to i koristi real labels
#Na primer ubacimo podatke prodje kroz sve loss funkija izracuna ali je loss veliki
#optimizer treba da se vrati i da popravi biase

if __name__ == "__main__":
    _first_example()
    #_multiclass_classification()
