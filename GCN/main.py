from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from models import GCN
from scipy import sparse
import os
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
tf.app.flags.DEFINE_string('f', '', 'kernel') # for usage on jupyter notebook

# load features
# time in range[0,23]
def feature_loader(flag, index, time):
    key = [-7,-4,-1]
    path = "../tp_data/" + flag + '/' + str(index) + ".csv"
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
    x_m = np.zeros(228)
    x_std = np.zeros(228)
    for i in range(228):
        x_m[i] = x[:,i].mean()
        x_std[i] = x[:,i].std()
        y[:,i]-=y[:,i].mean()
        if(y[:,i].std()!=0):
            y[:,i]/=y[:,i].std()
    return x.T, y.T, x_m, x_std

# load adj matrix
import numpy as np
adj_f = open('./distance.csv')
adj = []
dis = []

for i, lines in enumerate(adj_f):
    words = lines.split(",")
    for j in range(len(words)):
        d = float(words[j])
        if(d<5000):
            adj.append([i, j])
            if(d>0):
                dis.append(1/d)
            else:
                dis.append(1)
support = [(np.array(adj),np.array(dis),[228, 228])]

features, y_train, m, std = feature_loader("train", 0, 0)
f = sparse.csr_matrix(features).tolil()
f = preprocess_features(f)
# Some preprocessing
# features = preprocess_features(features)
'''
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))
'''
model_func = GCN
# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(1)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(f[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}


# Create model
model = model_func(placeholders, input_dim= f[2][1], logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):
    train_mask =np.zeros(228)
    val_mask=np.zeros(228)
    test_mask = np.zeros(228)
    t = time.time()
    for i in range(228):
        train_mask[i] = 1
    loss = 0
    for i in range(33):
        for j in range(268):
            train, gt, x_m, x_std = feature_loader("train", i, j)
            train = sparse.csr_matrix(train).tolil()
            train = preprocess_features(train)
            feed_dict = construct_feed_dict(train, support, gt, train_mask, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
            loss += outs[1]
            '''
            print(j, loss)
            if(j%10 == 0):
                outs = sess.run([model.outputs],feed_dict = feed_dict)
                for k in range(5):
                    print(k, outs[0][k][0]*x_std[k]+x_m[k])
            '''
    for j in range()
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss/(34*268)),
            "time=", "{:.5f}".format(time.time() - t))
                # Validation
                #cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
                #cost_val.append(cost)

        # Print results


print("Optimization Finished!")
train_mask =np.zeros(228)
val_mask=np.zeros(228)
test_mask = np.zeros(228)
for i in range(228):
    test_mask[i] = 1
csvFile = open("result_gcn.csv", "w")
writer = csv.writer(csvFile)
writer.writerow(["id", "Expected"])
for i in range(80):
    features, y_train, x_m, x_std = feature_loader("test", i, 0)
    y_val = y_test = y_train
    features = sparse.csr_matrix(features).tolil()
    features = preprocess_features(features)
    #print(features[0][0,:5], y_train[0,:5])
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: 0.0})
    outs = sess.run([model.outputs],feed_dict = feed_dict)
    for j in range(3):
        for k in range(228):
            name = str(i)+"_" + str(15*j+15)+"_"+str(k)
            writer.writerow([name, outs[0][k][j]*x_std[k]+x_m[k]])
csvFile.close()
print("Finished prediction!")