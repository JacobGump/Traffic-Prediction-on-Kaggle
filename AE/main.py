from __future__ import division, print_function
import numpy as np
import sklearn.datasets
import tensorflow as tf
from tensorflow.python.framework.ops import reset_default_graph

from tensorflow.contrib.layers import fully_connected
from tensorflow.python.ops.nn import relu, sigmoid

import os
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# TODO: load custom data from traffic dataset
# x_train[n,12,228] x_valid[m,12,228] x_test[k,12,228]
# targets_train[n,3,228] targets_valid[m,3,228] targets_test[k,3,228]
def feature_loader(flag, index, time):
    key = [-7,-4,-1]
    path = "./tp_data/" + flag + '/' + str(index) + ".csv"
    f = open(path)
    num = 0
    feature = np.zeros(21 * 228).reshape(21, 228)
    for lines in f:
        if(num >= time and num < time + 21):
            words = lines.split(',') #len(words) = 228; observations at the same time at 228 different stations
            for i in range(len(words)):
                feature[num-time][i] = float(words[i])
        num += 1
    x = feature[:12]
    y = feature[key]
    return x.T, y.T
print("loading data...")
x_train = []
x_valid = []
targets_train = []
targets_valid = []
for i in range(34):
    for t in range(268):
        x, y = feature_loader("train", i, t)
        if(i<34):
            x_train.append(x.flatten())
            targets_train.append(y.flatten())
        else:
            x_valid.append(x.flatten())
            targets_valid.append(y.flatten())

x_train = np.array(x_train)
x_valid = np.array(x_valid)
targets_train = np.array(targets_train)
targets_valid = np.array(targets_valid)

x_test = []
for i in range(80):
    path = "./tp_data/test/" + str(i) + ".csv"
    f = open(path)
    num = 0
    feature = np.zeros(12 * 228).reshape(12, 228)
    for lines in f:
        words = lines.split(',') #len(words) = 228; observations at the same time at 228 different stations
        for j in range(len(words)):
            feature[num][j] = float(words[j])
        num += 1
    x_test.append(feature.T.flatten())
    f.close()
x_test = np.array(x_test)


# define in/output size
num_features = x_train.shape[1]
num_predictions = targets_train.shape[1]
print("features number x:", num_features)
print("features number y:", num_predictions)
print("building graph...")
# reset graph
reset_default_graph()

x_pl = tf.placeholder(tf.float32, [None, num_features], 'x_pl')
y_pl = tf.placeholder(tf.float32, [None, num_predictions], 'y_pl')
l_enc1 = fully_connected(inputs=x_pl, num_outputs=4096, activation_fn=relu, scope='l_enc1')
l_enc2 = fully_connected(inputs=l_enc1, num_outputs=2048, activation_fn=relu, scope='l_enc2')
l_enc3 = fully_connected(inputs=l_enc2, num_outputs=1024, activation_fn=relu, scope='l_enc3')
l_enc4 = fully_connected(inputs = l_enc3, num_outputs = 512, activation_fn = relu, scope = 'l_enc4')
l_z = fully_connected(inputs=l_enc4, num_outputs=32, activation_fn=relu, scope='l_z') # None indicates a linear output.
l_dec1 = fully_connected(inputs=l_z, num_outputs=256, activation_fn=relu, scope='l_dec1')
l_dec2 = fully_connected(inputs=l_dec1, num_outputs=512, activation_fn=relu, scope='l_dec2')
l_dec3 = fully_connected(inputs=l_dec2, num_outputs=1024, activation_fn=relu, scope='l_dec3')
l_dec4 = fully_connected(inputs = l_dec3, num_outputs = 2048, activation_fn = relu, scope='l_dec4')
l_out = fully_connected(inputs=l_dec4, num_outputs=num_predictions, activation_fn=relu) # iid pixel intensities between 0 and 1.

loss_per_pixel = tf.square(tf.subtract(l_out, y_pl))
loss = tf.reduce_mean(loss_per_pixel, name="mean_square_error")

optimizer =  tf.train.AdamOptimizer(learning_rate=0.0001)

train_op = optimizer.minimize(loss)


# test the forward pass
_x_test = np.zeros(shape=(32, num_features))
_y_test = np.zeros(shape=(32, num_predictions))
# initialize the Session
sess = tf.Session()
# test the forward pass
sess.run(tf.initialize_all_variables())
feed_dict = {x_pl: _x_test, y_pl: _y_test}
res_forward_pass = sess.run(fetches=[l_out], feed_dict=feed_dict)
print("l_out", res_forward_pass[0].shape)

batch_size = 100
num_epochs = 200
num_samples_train = x_train.shape[0]
num_batches_train = num_samples_train // batch_size
num_samples_valid = x_valid.shape[0]
num_batches_valid = num_samples_valid // batch_size
updates = []

train_loss = []
valid_loss = []
cur_loss = 0
for epoch in range(num_epochs):
    #Forward->Backprob->Update params
    cur_loss = []
    for i in range(num_batches_train):
        idxs = np.random.choice(range(x_train.shape[0]), size=(batch_size), replace=False)    
        x_batch = x_train[idxs]
        y_batch = targets_train[idxs]
        # setup what to fetch, notice l
        fetches_train = [train_op, loss, l_out, l_z]
        feed_dict_train = {x_pl: x_batch, y_pl: y_batch}
        # do the complete backprob pass
        res_train = sess.run(fetches_train, feed_dict_train)
        _, batch_loss, train_out, train_z = tuple(res_train)
        cur_loss += [batch_loss]
        #if(epoch%100 == 0):
        #   print(train_out[:5])
    train_loss += [np.mean(cur_loss)]
    updates += [batch_size*num_batches_train*(epoch+1)]
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(np.mean(cur_loss)))

'''
    # evaluate
    fetches_eval = [loss, l_out, l_z]
    feed_dict_eval = {x_pl: x_valid}
    res_valid = sess.run(fetches_eval, feed_dict_eval)
    eval_loss, eval_out, eval_z = tuple(res_valid)
    valid_loss += [eval_loss]

'''
# predict
csvFile = open("result_AE_1.csv", "w")
writer = csv.writer(csvFile)
writer.writerow(["id", "Expected"])
for i in range(80):
    fetch_test = [l_out]
    batch_x = np.array(x_test[i]).reshape(1,2736)
    feed_dict_test = {x_pl:batch_x}
    y_test = sess.run(fetch_test,feed_dict_test)
    y_test = np.array(y_test)
    result = y_test.reshape(3,228)
    for j in range(3):
        for k in range(228):
            name = str(i)+"_"+str(15*j+15)+"_"+str(k)
            writer.writerow([name, result[j][k]])
csvFile.close()
print("Finished prediction!")

