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

n_class = 2  # change it to 1
n_hidden_1 = 50
n_hidden_2 = 40
n_hidden_3 = 30
n_hidden_4 = 20
learning_rate = 0.05
training_epoch = 150


def thesis(file_path, name, fn):
    feature_no = fn
    n_dim = feature_no
    # data processing

    def read_dataset(file_path):
        # Reading the dataset using panda's dataframe
        df = pd.read_csv(file_path)
        X = df[df.columns[0: feature_no]].values
        y = df[df.columns[feature_no]]
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

        # Hidden layer with leaky relu activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.tanh(layer_1)

        # Hidden layer with leaky relu activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.tanh(layer_2)

        # Hidden layer with leaky relu activation
        layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
        layer_3 = tf.nn.tanh(layer_3)

        # Hidden layer with leaky relu activation
        layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
        layer_4 = tf.nn.tanh(layer_4)

        # Output layer with linear activation
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
    mse = sess.run(cost_function, {x: final_test_x, y: final_test_y})
    print("Mean Squared Error: ", mse)
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

    with open('/home/ashiq/Pictures/Thesis_data/other.txt', 'a') as f:
        f.write(name + '\n')
        f.write('Accuracy: ' + str(sess.run(accuracy, {x: final_test_x, y: final_test_y})) + '\n')
        f.write('F1_Score: ' + str(f1_score) + '\n')
        f.write('Mean Squared Error: ' + str(mse) + '\n')
        f.write('AUC Score: ' + str(roc_auc) + '\n')

    sess.close()
    return fpr, tpr, thresholds, roc_auc, accuracy_history, cost_history


# open file
KL3_fpr, KL3_tpr, KL3_threshholds, KL3_roc_auc, \
KL3_accuracy_graph, KL3_cost_graph = thesis("input_csv_files/hmm_dga/500KL3.csv", '500KL3', 9)
kwyjibo_fpr, kwyjibo_tpr, kwyjibo_threshholds, kwyjibo_roc_auc, \
kwyjibo_accuracy_graph, kwyjibo_cost_graph = thesis("input_csv_files/other/kwyjibo.csv", 'kwyjibo', 9)
zeus_fpr, zeus_tpr, zeus_threshholds, zeus_roc_auc, \
zeus_accuracy_graph, zeus_cost_graph= thesis("input_csv_files/other/zeus.csv", 'zeus', 8)
srizbi_fpr, srizbi_tpr, srizbi_threshholds, srizbi_roc_auc, \
srizbi_accuracy_graph, srizbi_cost_graph= thesis("input_csv_files/other/srizbi.csv", 'srizbi', 8)
kraken_fpr, kraken_tpr, kraken_threshholds, kraken_roc_auc, \
kraken_accuracy_graph, kraken_cost_graph= thesis("input_csv_files/other/kraken.csv", 'kraken', 9)
conflicker_fpr, conflicker_tpr, conflicker_threshholds, conflicker_roc_auc, \
conflicker_accuracy_graph, conflicker_cost_graph= thesis("input_csv_files/other/conflicker.csv", 'conflicker', 8)
pcfg_ipv4_num_fpr, pcfg_ipv4_num_tpr, pcfg_ipv4_num_threshholds, pcfg_ipv4_num_roc_auc, \
pcfg_ipv4_num_accuracy_graph, pcfg_ipv4_num_cost_graph= thesis("input_csv_files/pcfg_dga/pcfg_ipv4_num.csv", 'pcfg_ipv4_num', 9)
torpig_fpr, torpig_tpr, torpig_threshholds, torpig_roc_auc, \
torpig_accuracy_graph, torpig_cost_graph = thesis("input_csv_files/other/torpig.csv", 'torpig', 8)

# ROC graph drawing
fig, ax = plt.subplots()
plt.title('ROC curve for OTHER BOTNETs, 20 times, 2000 data')
plt.plot(KL3_fpr, KL3_tpr, color='b', label='500KL3, AUC = %0.2f' % KL3_roc_auc)
plt.plot(kwyjibo_fpr, kwyjibo_tpr, color='g', label='kwyjibo, AUC = %0.2f' % kwyjibo_roc_auc)
plt.plot(zeus_fpr, zeus_tpr, color='r', label='zeus, AUC = %0.2f' % zeus_roc_auc)
plt.plot(srizbi_fpr, srizbi_tpr, color='c', label='srizbi, AUC = %0.2f' % srizbi_roc_auc)
plt.plot(kraken_fpr, kraken_tpr, color='m', label='kraken, AUC = %0.2f' % kraken_roc_auc)
plt.plot(conflicker_fpr, conflicker_tpr, color='y', label='conflicker, AUC = %0.2f' % conflicker_roc_auc)
plt.plot(pcfg_ipv4_num_fpr, pcfg_ipv4_num_tpr, color='k', label='pcfg_ipv4_num, AUC = %0.2f' % pcfg_ipv4_num_roc_auc)
plt.plot(torpig_fpr, torpig_tpr, color='grey', label='torpig, AUC = %0.2f' % torpig_roc_auc)

# common part for all files
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
fig.savefig('/home/ashiq/Pictures/Thesis_image/OTHER_US.eps', format='eps')
plt.show()

'''
fig, ax = plt.subplots()
plt.plot(KL3_accuracy_graph, 'b', label='%s' % '500KL3')
plt.plot(kwyjibo_accuracy_graph, 'g', label='%s' % 'kwyjibo')
plt.plot(zeus_accuracy_graph, 'r', label='%s' % 'zeus')
plt.plot(srizbi_accuracy_graph, 'c', label='%s' % 'srizbi')
plt.plot(kraken_accuracy_graph, 'm', label='%s' % 'kraken')
plt.plot(conflicker_accuracy_graph, 'y', label='%s' % 'conflicker')
plt.plot(pcfg_ipv4_num_accuracy_graph, 'k', label='%s' % 'pcfg_ipv4_num')
plt.plot(torpig_accuracy_graph, 'grey', label='%s' % 'torpig')
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
plt.legend(loc='lower right')
fig.savefig('/home/ashiq/Pictures/Thesis_image/other_accuracy.eps', format='eps')
plt.show()

fig, ax = plt.subplots()
plt.plot(KL3_cost_graph, 'b', label='%s' % '500KL3')
plt.plot(kwyjibo_cost_graph, 'g', label='%s' % 'kwyjibo')
plt.plot(zeus_cost_graph, 'r', label='%s' % 'zeus')
plt.plot(srizbi_cost_graph, 'c', label='%s' % 'srizbi')
plt.plot(kraken_cost_graph, 'm', label='%s' % 'kraken')
plt.plot(conflicker_cost_graph, 'y', label='%s' % 'conflicker')
plt.plot(pcfg_ipv4_num_cost_graph, 'k', label='%s' % 'pcfg_ipv4_num')
plt.plot(torpig_cost_graph, 'grey', label='%s' % 'torpig')
plt.xlabel('Epoch')
plt.ylabel('Training Error')
plt.legend(loc='lower right')
fig.savefig('/home/ashiq/Pictures/Thesis_image/other_error.eps', format='eps')
plt.show()
'''

