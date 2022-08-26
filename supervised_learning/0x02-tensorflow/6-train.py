#!/usr/bin/env python3


""" contains train funct"""
import tensorflow.compat.v1 as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


def train(
        X_train,
        Y_train,
        X_valid,
        Y_valid,
        layer_sizes,
        activations,
        alpha,
        iterations,
        save_path="/tmp/model.ckpt"):
    """ train funct"""
    nx = X_train.shape[1]
    classes = Y_train.shape[1]

    x, y = create_placeholders(nx, classes)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations):
            if not i % 100:
                print('After {} iterations:'.format(i))
                train_cost, train_accuracy = sess.run(
                    (loss, accuracy), feed_dict={x: X_train, y: Y_train})
                print('\tTraining Cost: {}'.format(train_cost))
                print('\tTraining Accuracy: {}'.format(train_accuracy))
                valid_cost, valid_accuracy = sess.run(
                    (loss, accuracy), feed_dict={x: X_valid, y: Y_valid})
                print('\tValidation Cost: {}'.format(valid_cost))
                print('\tValidation Accuracy: {}'.format(valid_accuracy))
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})

        print('After {} iterations:'.format(iterations))
        train_cost, train_accuracy = sess.run(
            (loss, accuracy), feed_dict={x: X_train, y: Y_train})
        print('\tTraining Cost: {}'.format(train_cost))
        print('\tTraining Accuracy: {}'.format(train_accuracy))
        valid_cost, valid_accuracy = sess.run(
            (loss, accuracy), feed_dict={x: X_valid, y: Y_valid})
        print('\tValidation Cost: {}'.format(valid_cost))
        print('\tValidation Accuracy: {}'.format(valid_accuracy))

        saver = tf.train.Saver()
        return saver.save(sess, save_path)