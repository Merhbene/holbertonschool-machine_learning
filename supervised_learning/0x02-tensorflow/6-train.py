#!/usr/bin/env python3
"""
this module contain train function
"""
import tensorflow.compat.v1 as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    builds, trains, and saves a neural network classifier:
    X_train is a numpy.ndarray containing the training input data
    Y_train is a numpy.ndarray containing the training labels
    X_valid is a numpy.ndarray containing the validation input data
    Y_valid is a numpy.ndarray containing the validation labels
        *layer_sizes is a list containing the number of
        nodes in each layer of the network
        *activations is a list containing the
        activation functions for each layer of the network
    alpha is the learning rate
    iterations is the number of iterations to train over
    save_path designates where to save the model
    Add the following to the graph’s collection
    placeholders: x and y
    tensors: y_pred, loss and accuracy
    operation: train_op
    """
    # x: is the placeholder for the input data
    # to the neural network
    # y: is the placeholder for the one-hot labels
    # for the input data
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])

    # TENSORS:

    # y_pred: the prediction of the network in tensor form
    y_pred = forward_prop(x, layer_sizes, activations)
    # tensor containing the decimal accuracy of the prediction (mean)
    accuracy = calculate_accuracy(y, y_pred)
    # tensor containing the loss of the prediction
    loss = calculate_loss(y, y_pred)

    # operation that trains the network using gradient descent
    train_op = create_train_op(loss, alpha)

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    # instance of tf.train.Saver() to save
    saver = tf.train.Saver()

    # allocates the memory for the Variable and sets its initial values.
    init = tf.global_variables_initializer()

    # tf.Session object encapsulates the environment
    # in which Operation objects are executed

    # and Tensor objects are evaluated
    with tf.Session() as session:
        session.run(init)

        for i in range(iterations + 1):
            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(lossTrain))
                print("\tTraining Accuracy: {}".format(accuracyTrain))
                print("\tValidation Cost: {}".format(lossValid))
                print("\tValidation Accuracy: {}".format(accuracyValid))
                session.run(train_op,
                            feed_dict={x: X_train, y: Y_train})
            # foward propagation
            lossTrain = session.run(loss,
                                    feed_dict={x: X_train, y: Y_train})
            lossValid = session.run(loss,
                                    feed_dict={x: X_valid, y: Y_valid})

            # back propagation
            accuracyTrain = session.run(accuracy,
                                        feed_dict={x: X_train, y: Y_train})
            accuracyValid = session.run(accuracy,
                                        feed_dict={x: X_valid, y: Y_valid})


        # This method runs the ops added by the constructor
        # for saving variables.
        # It requires a session in which the graph was launched.
        # The variables to save must also have been initialized.
        return saver.save(session, save_path)
