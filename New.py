import sklearn
import sklearn as sk
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"


# constant variables or hyperparamaters
feature_no = 6
n_dim = feature_no
n_class = 2  # change it to 1
n_hidden_1 = 50
n_hidden_2 = 40
n_hidden_3 = 30
n_hidden_4 = 20
learning_rate = 0.05
training_epoch = 150


def thesis(file_path, name, layer_no):

    # data processing

    def read_dataset(file_path):
        # Reading the dataset using panda's dataframe
        df = pd.read_csv(file_path)
        X = df[df.columns[0:6]].values
        y = df[df.columns[6]]
        # Encode the dependent variable
        Y = one_hot_encode(y)
        return X, Y

    # Define the encoder function.
    def one_hot_encode(labels):
        n_labels = len(labels)
        n_unique_labels = len(np.unique(labels))
        one_hot_encode = np.zeros((n_labels, n_unique_labels))
        one_hot_encode[np.arange(n_labels), labels] = 1
        return one_hot_encode

    X, Y = read_dataset(file_path)
    X, Y = shuffle(X, Y, random_state=20)
    final_train_x, final_test_x, final_train_y, final_test_y = train_test_split(X, Y, test_size=0.20, random_state=415)

    # model parameters
    if layer_no == 2:
        weights = {
            'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
            'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.truncated_normal([n_hidden_2, n_class]))
        }
        biases = {
            'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
            'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
            'out': tf.Variable(tf.truncated_normal([n_class]))
        }
    elif layer_no == 3:
        weights = {
            'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
            'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
            'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
            'out': tf.Variable(tf.truncated_normal([n_hidden_3, n_class]))
        }
        biases = {
            'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
            'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
            'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
            'out': tf.Variable(tf.truncated_normal([n_class]))
        }
    else:
        weights = {
            'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
            'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
            'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
            'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
            'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_class]))
        }
        biases = {
            'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
            'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
            'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
            'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
            'out': tf.Variable(tf.truncated_normal([n_class]))
        }

    # inputs and outputs
    x = tf.placeholder(tf.float32, [None, n_dim])
    y = tf.placeholder(tf.float32, [None, n_class])


    # model definition

    def multilayer_perceptron(x, weights, biases):

        # Hidden layer with tanh activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.tanh(layer_1)

        # Hidden layer with tanh activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.tanh(layer_2)

        # Hidden layer with tanh activation
        if layer_no > 2:
            layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
            layer_3 = tf.nn.tanh(layer_3)

        # Hidden layer with tanh activation
        if layer_no > 3:
            layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
            layer_4 = tf.nn.tanh(layer_4)

        # Output layer with linear activation
        if layer_no == 2:
            out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
        elif layer_no == 3:
            out_layer = tf.add(tf.matmul(layer_3, weights['out']), biases['out'])
        else:
            out_layer = tf.add(tf.matmul(layer_4, weights['out']), biases['out'])
        out_layer = tf.nn.softmax(out_layer)
        return out_layer

    model = multilayer_perceptron(x, weights, biases)

    # loss function and optimizer
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    cost_function = tf.losses.mean_squared_error(labels=y, predictions=model)
    # cost_function += sum(reg_losses)
    # cost_function = cost_function + 0.01*tf.nn.l2_loss(weights['h1']) + 0.01*tf.nn.l2_loss(weights['h2']) +\
    #                 0.01*tf.nn.l2_loss(weights['h3']) + 0.01*tf.nn.l2_loss(weights['h4'])
    training_step = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)

    # accuracy measurement
    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # initialize global variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)

    # correct weights and biases
    cost_history = []
    accuracy_history = []
    for epoch in range(0, training_epoch):
        sess.run(training_step, {x: final_train_x, y: final_train_y})
        print(name, "Epoch: ", epoch, end=' ')

        accuracy_value = sess.run(accuracy, {x: final_train_x, y: final_train_y})
        accuracy_history.append(accuracy_value)
        print("Accuracy: ", accuracy_value, end=' ')

        cost = sess.run(cost_function, {x: final_train_x, y: final_train_y})
        cost_history.append(cost)
        print("Cost: ", cost)


    '''
    # training accuracy and cost graph plot
    fig, ax = plt.subplots()
    plt.plot(accuracy_history, label='%s' % name)
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    fig.savefig('/home/ashiq/Pictures/filename1.eps', format='eps')
    plt.show()

    fig, ax = plt.subplots()
    plt.plot(cost_history, label='%s' % name)
    plt.xlabel('Epoch')
    plt.ylabel('Training Error')
    fig.savefig('/home/ashiq/Pictures/filename2.eps', format='eps')
    plt.show()
    '''

    # test accuracy & f1-score calculation
    print(name)
    print("Accuracy: ", sess.run(accuracy, {x: final_test_x, y: final_test_y}))
    final_y = sess.run(model, feed_dict={x: final_test_x})
    actual_y = np.argmax(final_test_y, 1)  # 1 0 -> 0 , 0 1 -> 1
    predicted_y = np.argmax(final_y, 1)  # max min -> 0 , min max -> 1
    f1_score = sk.metrics.f1_score(actual_y, predicted_y)
    precision = sk.metrics.precision_score(actual_y, predicted_y)
    recall = sk.metrics.recall_score(actual_y, predicted_y)
    print("F1 score: ", f1_score)
    # print("Recall: ", recall, "Precision: ", precision)

    # auc_score calculation
    predicted_y = np.array(final_y)[:, 1]
    y_true = []
    for i in range(0, len(final_test_y)):
        if final_test_y[i][0] == 1:
            y_true.append(0)
        else:
            y_true.append(1)
    y_true = np.array(y_true)
    # print(y_true)
    fpr, tpr, thresholds = roc_curve(y_true, predicted_y)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    # print(fpr, tpr, thresholds)
    print("AUC score: ", roc_auc)
    sess.close()
    return fpr, tpr, thresholds, roc_auc, accuracy_history, cost_history


# open file
fpr_2, tpr_2, threshholds_2, roc_auc_2, \
accuracy_graph_2, cost_graph_2 = thesis("input_csv_files/other/conflicker.csv", '500KL3', 2)
fpr_3, tpr_3, threshholds_3, roc_auc_3, \
accuracy_graph_3, cost_graph_3 = thesis("input_csv_files/other/conflicker.csv", '500KL3', 3)
fpr_4, tpr_4, threshholds_4, roc_auc_4, \
accuracy_graph_4, cost_graph_4 = thesis("input_csv_files/other/conflicker.csv", '500KL3', 4)

# ROC graph drawing
'''
fig, ax = plt.subplots()
plt.title('Receiver Operating Characteristic for PCFG BOTNETs')
plt.plot(pcfg_dict_fpr, pcfg_dict_tpr, color='darkgrey', label='pcfg_dict, AUC = %0.2f' % pcfg_dict_roc_auc)
plt.plot(pcfg_dict_num_fpr, pcfg_dict_num_tpr, 'g', label='pcfg_dict_num, AUC = %0.2f' % pcfg_dict_num_roc_auc)
plt.plot(pcfg_ipv4_fpr, pcfg_ipv4_tpr, 'b', label='pcfg_ipv4, AUC = %0.2f' % pcfg_ipv4_roc_auc)
plt.plot(pcfg_ipv4_num_fpr, pcfg_ipv4_num_tpr, 'm', label='pcfg_ipv4_num, AUC = %0.2f' % pcfg_ipv4_num_roc_auc)

# common part for all files
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
fig.savefig('/home/ashiq/Pictures/filename.eps', format='eps')
plt.show()
'''

fig, ax = plt.subplots()
plt.title('For conflicker botnet')
plt.plot(accuracy_graph_2, label='%s' % 'number of hidden layers = 2')
plt.plot(accuracy_graph_3, label='%s' % 'number of hidden layers = 3')
plt.plot(accuracy_graph_4, label='%s' % 'number of hidden layers = 4')
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
plt.legend(loc='lower right')
fig.savefig('/home/ashiq/Pictures/Thesis_image/hidden_layer_variation.eps', format='eps')
plt.show()

'''
fig, ax = plt.subplots()
plt.plot(pcfg_dict_cost_graph, label='%s' % 'pcfg_dict')
plt.plot(pcfg_dict_num_cost_graph, label='%s' % 'pcfg_dict_num')
plt.plot(pcfg_ipv4_cost_graph, label='%s' % 'pcfg_ipv4')
plt.plot(pcfg_ipv4_num_cost_graph, label='%s' % 'pcfg_ipv4_num')
plt.xlabel('Epoch')
plt.ylabel('Training Error')
fig.savefig('/home/ashiq/Pictures/error.eps', format='eps')
plt.show()
'''
