#!/usr/bin/env python3
""" LeNet-5 (Keras) """
import tensorflow.keras as K


def lenet5(X):

    init = K.initializers.he_normal(seed=None)
    output = K.layers.Conv2D(filters=6,
                             kernel_size=5,
                             padding='same',
                             kernel_initializer=init,
                             activation='relu')(X)

    output2 = K.layers.MaxPool2D(strides=2)(output)

    output3 = K.layers.Conv2D(filters=16,
                              kernel_size=5,
                              padding='valid',
                              kernel_initializer=init,
                              activation='relu')(output2)

    output4 = K.layers.MaxPool2D(strides=2)(output3)

    output5 = K.layers.Flatten()(output4)

    output6 = K.layers.Dense(units=120,
                             kernel_initializer=init,
                             activation='relu')(output5)

    output7 = K.layers.Dense(units=84,
                             kernel_initializer=init,
                             activation='relu')(output6)

    output8 = K.layers.Dense(units=10,
                             kernel_initializer=init,
                             activation='softmax')(output7)

    model = K.models.Model(inputs=X, outputs=output8)

    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
