#!/usr/bin/env python3
"Save and Load Configuration"
import tensorflow.keras as K


def save_config(network, filename):
    "saves a model’s configuration in JSON format"

    with open(filename, "w") as json_file:
        json_file.write(network.to_json())


def load_config(filename):
    "loads a model with a specific configuration"
    with open(filename, 'r') as f:
        return K.models.model_from_json(f.read())
