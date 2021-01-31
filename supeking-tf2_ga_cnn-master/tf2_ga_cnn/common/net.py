import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, Input, AvgPool2D, GlobalAvgPool2D, Flatten, AveragePooling2D, MaxPooling2D



def fnn2(input_shape=[28, 28]):
    inputs = Input(input_shape)
    x = tf.reshape(inputs, (-1, input_shape[0] * input_shape[1] * input_shape[2]))
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)
    return Model(inputs=inputs, outputs=[x])


def fnn3(input_shape=[28, 28]):
    inputs = Input(input_shape)
    x = tf.reshape(inputs, (-1, input_shape[0] * input_shape[1]* input_shape[2]))
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)
    return Model(inputs=inputs, outputs=[x])



def cnn(input_shape=[28, 28]):
    inputs = Input(input_shape)
    #x = tf.reshape(inputs, [-1]+input_shape+[3])
    x = Conv2D(8, (3, 3), activation='relu', use_bias=False)(inputs)
    x = AvgPool2D(strides=2)(x)
    x = Conv2D(16, (3, 3), activation='relu', use_bias=False)(x)
    x = GlobalAvgPool2D()(x)
    x = Dense(10, activation='softmax', use_bias=False)(x)
    return Model(inputs=inputs, outputs=[x])


def cnn2(input_shape=[28, 28]):
    inputs = Input(input_shape)
    #x = tf.reshape(inputs, [-1] + input_shape + [1])
    x = Conv2D(40, (1, 1), activation='relu', use_bias=False)(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(40, (1, 1), activation='relu', use_bias=False)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(5, (1, 1), activation='relu', use_bias=False)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(1, (1, 1), activation='relu', use_bias=False)(x)
    x = Flatten()(x)
    x = Dense(40, activation='relu', use_bias=False)(x)
    x = Dense(10, activation='softmax', use_bias=False)(x)
    return Model(inputs=inputs, outputs=[x])



def cnn3(input_shape=[32, 32, 3]):
    inputs = Input(input_shape)
    #x = tf.reshape(inputs, [-1] + input_shape + [3])
    x = Conv2D(6, (1, 1), activation='relu', use_bias=False)(inputs)
    x = AveragePooling2D((2, 2))(x)
    x = Conv2D(16, (1, 1), activation='relu', use_bias=False)(x)
    x = AveragePooling2D((2, 2))(x)
    x = Conv2D(120, (1, 1), activation='relu', use_bias=False)(x)
    x = AveragePooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(120, activation='relu', use_bias=False)(x)
    x = Dense(84, activation='relu', use_bias=False)(x)
    x = Dense(10, activation='softmax', use_bias=False)(x)
    return Model(inputs=inputs, outputs=[x])


if __name__ == "__main__":
    model = fnn2([8, 8, 1])
    model.summary()
    print('-')