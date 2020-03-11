import tensorflow as tf


class NeuralNet(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.hidden_layers = [
            tf.keras.layers.Dense(
                units=unit, activation=tf.nn.relu, kernel_initializer="he_normal"
            )
            for unit in kwargs["units"]
        ]
        self.output_layer = tf.keras.layers.Dense(units=kwargs["num_classes"])

    def call(self, features):
        activations = {}
        for index, layer in enumerate(self.hidden_layers):
            if index == 0:
                activations[index] = layer(features)
            else:
                activations[index] = layer(activations[index - 1])
        logits = self.output_layer(activations[len(activations) - 1])
        return logits
