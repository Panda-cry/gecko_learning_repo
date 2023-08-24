import tensorflow as tf

class NBeastBlock(tf.keras.layers.Layer):

    def __init__(self,
                 input_size: int,
                 theta_size: int,
                 horizon: int,
                 n_neurons: int,
                 n_layers: int,
                 **kwargs):
        super().__init__(**kwargs)
        self.intput_size = input_size
        self.theta_size = theta_size
        self.horizon = horizon
        self.n_neurons = n_neurons
        self.n_layers = n_layers

        self.hidden = [tf.keras.layers.Dense(n_neurons,activation="relu") for _ in range(n_layers)]
        self.theta = tf.keras.layers.Dense(theta_size, activation="linear")

        def call(self, inputs, *args, **kwargs):
            x = inputs
            for layer in self.hidden:
                x = layer(x)

            theta = self.theta(x)

            backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]
            return backcast, forecast