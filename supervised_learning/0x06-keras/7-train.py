#!/usr/bin/env python3
"Learning Rate Decay"
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):

    callbacks = []
    if validation_data:
        if early_stopping:
            early_stop = K.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience)
            callbacks.append(early_stop)
        if learning_rate_decay:
            def schedule(step):
                '''stepwise inverse time decay function'''
                return alpha * 1 / (1 + decay_rate * step)
            lr_decay = K.callbacks.LearningRateScheduler(schedule=schedule,
                                                         verbose=1)
            callbacks.append(lr_decay)
    return network.fit(x=data,
                       y=labels,
                       batch_size=batch_size,
                       validation_data=validation_data,
                       epochs=epochs,
                       verbose=verbose,
                       callbacks=callbacks,
                       shuffle=shuffle)
