from tensorflow import keras

kl = keras.layers

"""
R2D3 Image Model
https://arxiv.org/abs/1909.01387
"""


class R2D3ImageBlock(keras.Model):
    def __init__(
        self,
        filters: int = 16,
        activation: str = "relu",
        enable_time_distributed_layer: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.res1 = _ResBlock(filters, activation, enable_time_distributed_layer)
        self.res2 = _ResBlock(filters * 2, activation, enable_time_distributed_layer)
        self.res3 = _ResBlock(filters * 2, activation, enable_time_distributed_layer)
        self.act = kl.Activation(activation)

    def call(self, x, training=False):
        x = self.res1(x, training=training)
        x = self.res2(x, training=training)
        x = self.res3(x, training=training)
        x = self.act(x)
        return x

    def build(self, input_shape):
        self.__input_shape = input_shape
        super().build(self.__input_shape)

    def init_model_graph(self, name: str = ""):
        self.res1.init_model_graph()
        self.res2.init_model_graph()
        self.res3.init_model_graph()

        x = kl.Input(shape=self.__input_shape[1:])
        name = self.__class__.__name__ if name == "" else name
        keras.Model(inputs=x, outputs=self.call(x), name=name)


class _ResBlock(keras.Model):
    def __init__(
        self,
        filters,
        activation,
        enable_time_distributed_layer,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.conv = kl.Conv2D(filters, (3, 3), strides=(1, 1), padding="same")
        self.pool = kl.MaxPooling2D((3, 3), strides=(2, 2), padding="same")
        self.res1 = _ResidualBlock(filters, activation, enable_time_distributed_layer)
        self.res2 = _ResidualBlock(filters, activation, enable_time_distributed_layer)

        if enable_time_distributed_layer:
            self.conv = kl.TimeDistributed(self.conv)
            self.pool = kl.TimeDistributed(self.pool)

    def call(self, x, training=False):
        x = self.conv(x, training=training)
        x = self.pool(x, training=training)
        x = self.res1(x, training=training)
        x = self.res2(x, training=training)
        return x

    def build(self, input_shape):
        self.__input_shape = input_shape
        super().build(self.__input_shape)

    def init_model_graph(self, name: str = ""):
        self.res1.init_model_graph()
        self.res2.init_model_graph()

        x = kl.Input(shape=self.__input_shape[1:])
        name = self.__class__.__name__ if name == "" else name
        keras.Model(inputs=x, outputs=self.call(x), name=name)


class _ResidualBlock(keras.Model):
    def __init__(
        self,
        n_filter,
        activation,
        enable_time_distributed_layer,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.act1 = kl.Activation(activation)
        self.conv1 = kl.Conv2D(n_filter, (3, 3), strides=(1, 1), padding="same")
        self.act2 = kl.Activation(activation)
        self.conv2 = kl.Conv2D(n_filter, (3, 3), strides=(1, 1), padding="same")
        self.add = kl.Add()

        if enable_time_distributed_layer:
            self.conv1 = kl.TimeDistributed(self.conv1)
            self.conv2 = kl.TimeDistributed(self.conv2)

    def call(self, x, training=False):
        x1 = self.act1(x)
        x1 = self.conv1(x1, training=training)
        x1 = self.act2(x1)
        x1 = self.conv2(x1, training=training)
        return self.add([x, x1])

    def build(self, input_shape):
        self.__input_shape = input_shape
        super().build(self.__input_shape)

    def init_model_graph(self, name: str = ""):
        x = kl.Input(shape=self.__input_shape[1:])
        name = self.__class__.__name__ if name == "" else name
        keras.Model(inputs=x, outputs=self.call(x), name=name)


if __name__ == "__main__":
    m = R2D3ImageBlock(enable_time_distributed_layer=True)
    m.build((None, None, 96, 72, 3))
    m.init_model_graph()
    m.summary(expand_nested=True)
