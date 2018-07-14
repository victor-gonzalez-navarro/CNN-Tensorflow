import math
import numpy as np
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt


def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def random_mini_batches(X, Y, mini_batch_size=64):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


# --------------------------------

# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
conv_layers = {}

print("Tensorflow version " + tf.__version__)

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)

# neural network structure for this sample:
#
# · · · · · · · · · ·      (input data, 1-deep)                    X [batch, 28, 28, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer +BN 6x6x1=>24 stride 1      W1 [5, 5, 1, 24]        B1 [24]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                              Y1 [batch, 28, 28, 6]
#   @ @ @ @ @ @ @ @     -- conv. layer +BN 5x5x6=>48 stride 2      W2 [5, 5, 6, 48]        B2 [48]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                                Y2 [batch, 14, 14, 12]
#     @ @ @ @ @ @       -- conv. layer +BN 4x4x12=>64 stride 2     W3 [4, 4, 12, 64]       B3 [64]
#     ∶∶∶∶∶∶∶∶∶∶∶                                                  Y3 [batch, 7, 7, 24] => reshaped to YY [batch, 7*7*24]
#      \x/x\x\x/ ✞      -- fully connected layer (relu+dropout+BN) W4 [7*7*24, 200]       B4 [200]
#       · · · ·                                                    Y4 [batch, 200]
#       \x/x\x/         -- fully connected layer (softmax)         W5 [200, 10]           B5 [10]
#        · · ·                                                     Y [batch, 10]


costs = []  # To keep track of the cost

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 64, 64, 3])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 6])
# test flag for batch norm
tst = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.int32)
# dropout probability
pkeep = tf.placeholder(tf.float32)
pkeep_conv = tf.placeholder(tf.float32)

def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_averages

def no_batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    return Ylogits, tf.no_op()

def compatible_convolutional_noise_shape(Y):
    noiseshape = tf.shape(Y)
    noiseshape = noiseshape * tf.constant([1,0,0,1]) + tf.constant([0,1,1,0])
    return noiseshape


##########################
W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable("W3", [2 * 2 * 16, 32], initializer=tf.contrib.layers.xavier_initializer())
W4 = tf.get_variable("W4", [32, 6], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable("b1", shape=(8), initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable("b2", shape=(16), initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable("b3", shape=(32), initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.get_variable("b4", shape=(6), initializer=tf.contrib.layers.xavier_initializer())

Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
Z1bn, update_ema1 = batchnorm(Z1, tst, iter, b1, convolutional=True)
A1 = tf.nn.relu(Z1bn)
A1 = tf.nn.dropout(A1, pkeep_conv, compatible_convolutional_noise_shape(A1))
Y1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')

Z2 = tf.nn.conv2d(Y1, W2, strides=[1, 1, 1, 1], padding='SAME')
Z2bn, update_ema2 = batchnorm(Z2, tst, iter, b2, convolutional=True)
A2 = tf.nn.relu(Z2bn)
A2 = tf.nn.dropout(A2, pkeep_conv, compatible_convolutional_noise_shape(A2))
Y2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

P2 = tf.contrib.layers.flatten(Y2)

fc1 = tf.matmul(P2, W3)
fc1bn, update_ema3 = batchnorm(fc1, tst, iter, b3)
fc1a = tf.nn.relu(fc1bn)
Y4 = tf.nn.dropout(fc1a, pkeep)

Ylogits = tf.matmul(Y4, W4) + b4
##########################

update_ema = tf.group(update_ema1, update_ema2, update_ema3)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# matplotlib visualisation
allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1])], 0)
allbiases  = tf.concat([tf.reshape(b1, [-1]), tf.reshape(b2, [-1]), tf.reshape(b3, [-1]), tf.reshape(b4, [-1])], 0)

# training step
# the learning rate is: # 0.0001 + 0.03 * (1/e)^(step/1000)), i.e. exponential decay from 0.03->0.0001
lr = 0.0001 + tf.train.exponential_decay(0.02, iter, 1600, 1/math.e)
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


num_epochs = 20
minibatch_size=64
m = X_train.shape[0]

# You can call this function in a loop to train the model, 100 images at a time
for i in range(num_epochs):
    # training on batches of 100 images with 100 labels
    minibatch_cost = 0.
    num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
    minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

    for minibatch in minibatches:
        (batch_X, batch_Y) = minibatch

        # the backpropagation training step
        c, l = sess.run([cross_entropy, lr], feed_dict={X: batch_X, Y_: batch_Y, iter: i, tst: False, pkeep: 1.0, pkeep_conv: 1.0})
        minibatch_cost += c / num_minibatches
        #sess.run([train_step,update_ema] , {X: batch_X, Y_: batch_Y, tst: False, iter: i, pkeep: 1.0, pkeep_conv: 1.0})
        sess.run(train_step, {X: batch_X, Y_: batch_Y, tst: False, iter: i, pkeep: 1.0, pkeep_conv: 1.0})
        sess.run(update_ema, {X: batch_X, Y_: batch_Y, tst: False, iter: i, pkeep: 1.0,  pkeep_conv: 1.0})
        # print(sess.run(allbiases[0]))


    print("Epoch " + str(i) + " Loss= " + str(minibatch_cost) + " (lr=" + str(l) + ")")
    costs.append(minibatch_cost)



# plot the cost
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations')
plt.show()


# Calculate accuracy on the test set
correct_prediction = tf.equal(tf.argmax(Ylogits, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_accuracy = sess.run(accuracy, {X: X_train, Y_: Y_train, tst: True, iter: num_epochs, pkeep: 1.0, pkeep_conv: 1.0})
test_accuracy = sess.run(accuracy, {X: X_test, Y_: Y_test, tst: True, iter: num_epochs, pkeep: 1.0, pkeep_conv: 1.0})
print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


print("Study effect dropout")
