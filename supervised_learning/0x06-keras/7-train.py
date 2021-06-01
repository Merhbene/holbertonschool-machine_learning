#!/usr/bin/env python3
"Learning Rate Decay"
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    "train the model with learning rate decay"
    cb = []
    if validation_data is not None:
        if early_stopping:
            early = K.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=patience)
            cb.append(early)
        if learning_rate_decay:
            def decayed_lr(step):
                """ Inverse time decay schedule function, step == epoch"""
                return alpha / (1 + decay_rate * (step))
            learn = K.callbacks.LearningRateScheduler(schedule=decayed_lr,
                                                      verbose=True)
            cb.append(learn)
    history = network.fit(data, labels, epochs=epochs,
                          batch_size=batch_size,
                          callbacks=cb,
                          validation_data=validation_data,
                          verbose=verbose, shuffle=shuffle)
    return history
