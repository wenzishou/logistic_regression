'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
# Parameters
learning_rate = 0.002
training_epochs = 50
batch_size = 100
display_step = 1


def scoreAUC( num_clicks, predicted_ctr ):

    assert len(num_clicks) == len(predicted_ctr) 
    """
    Calculates the area under the ROC curve (AUC) for click rates

    Parameters
    ----------
    num_clicks : a list containing the number of clicks

    predicted_ctr : a list containing the predicted click-through rates

    Returns
    -------
    auc : the area under the ROC curve (AUC) for click rates
    """
    i_sorted = sorted(range(len(predicted_ctr)),key=lambda i: predicted_ctr[i], reverse=True)
    auc_temp = 0.0
    click_sum = 0.0
    old_click_sum = 0.0
    no_click = 0.0
    no_click_sum = 0.0

    # treat all instances with the same predicted_ctr as coming from the
    # same bucket
    last_ctr = predicted_ctr[i_sorted[0]] + 1.0

    for i in range(len(predicted_ctr)):
        if last_ctr != predicted_ctr[i_sorted[i]]:
            auc_temp += (click_sum+old_click_sum) * no_click / 2.0
            old_click_sum = click_sum
            no_click = 0.0
            last_ctr = predicted_ctr[i_sorted[i]]
        no_click += 1 - num_clicks[i_sorted[i]]
        no_click_sum += 1 - num_clicks[i_sorted[i]]
        click_sum += num_clicks[i_sorted[i]]

    auc_temp += (click_sum+old_click_sum) * no_click / 2.0
    total_area = click_sum * no_click_sum
    
    if total_area == 0: 
        auc = 1
    else:
        auc = auc_temp / total_area
    return auc


def cross_entropy(y, y_):
    p = tf.clip_by_value(y_, 10e-30, 1)
    neg_p = 1-p
    neg_p  = tf.clip_by_value(neg_p, 10e-30, 1)
    target = y#tf.select(t>0,tf.ones(tf.shape(y)),tf.zeros(tf.shape(y)))
    err = - target * tf.log(p) - (1 - target) * tf.log(neg_p)
    return tf.reduce_mean(err)


# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 1]))
b = tf.Variable(tf.zeros([1]))

# Construct model
pred = tf.sigmoid(tf.matmul(x, W) + b) # Softmax

#most important here
pred_x = pred * tf.pow(2.71828, 1.8*(pred - 0.1))

# Minimize error using cross entropy
#cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
cost = cross_entropy(y, pred_x)    
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        x_pred_x = []
        x_pred = []
        total_batch =  int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size, shuffle=False)
            labels_train = []
            for i in batch_ys:
                if i == 1:
                    labels_train.append(1)
                else:
                    labels_train.append(0)
            #print(labels_train)
            # Run optimization op (backprop) and cost op (to get loss value)
            #_, c, x_pred, x_pred_x = sess.run([optimizer, cost, pred, pred_x], feed_dict={x: batch_xs,
            _, c, = sess.run([optimizer, cost, ], feed_dict={x: batch_xs,
                                                          y: labels_train})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            #print('x_pred', x_pred[0]) 
            #print('x_pred_x', x_pred_x[0]) 
    print("Optimization Finished!")

    # Test model
    #correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    
    # Calculate accuracy
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    test_eval = pred.eval({x: mnist.test.images})
    test_label = []
    for i in mnist.test.labels:
        if 1 == i:
            test_label.append(1)
        else:
            test_label.append(0)


    test_eval_x = pred_x.eval({x: mnist.test.images})

    print(np.mean(test_eval), np.var(test_eval), test_eval[0]) 
    print(np.mean(test_eval_x), np.var(test_eval_x), test_eval_x[0])
    
    print(scoreAUC(test_label, test_eval))
