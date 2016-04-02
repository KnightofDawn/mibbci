

import params
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt




########################################################################################################################


def load_data(data_filename_list):
    X_list = []
    labels_list = []

    for data_filename in data_filename_list:
        print 'Loading data from', data_filename, '...'
        data_loaded_train = np.loadtxt(fname=data_filename, delimiter=',', skiprows=1);
        print 'data_loaded.shape:', data_loaded_train.shape
        X_list.append(data_loaded_train[:, 1:(1 + params.NUM_CHANNELS)])
        labels_list.append(data_loaded_train[:, params.LABEL_ID_RED:(params.LABEL_ID_RED+params.NUM_EVENT_TYPES)])
        # print 'X_raw.shape', X_train_raw.shape

    #X = np.concatenate((X_list[0], X_list[1], X_list[2]), axis=0)
    #labels = np.concatenate((labels_list[0], labels_list[1], labels_list[2]), axis=0)
    X = np.concatenate(X_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    return X, labels


########################################################################################################################


def calculate_auroc(labels_groundtruth, labels_predictions, labels_names, tpr_target_arr, plot=True):
    print 'labels_groundtruth.shape, labels_predictions.shape:', labels_groundtruth.shape, labels_predictions.shape

    fpr_arr_list = []
    tpr_arr_list = []
    threshold_arr_list = []
    auroc_val_list = []
    for i_event in range(labels_groundtruth.shape[1]):
        # fpr_arr, tpr_arr, _ = roc_curve(label_feed_test, pred_feed)
        fpr_arr_temp, tpr_arr_temp, threshold_arr_temp = roc_curve(labels_groundtruth[:, i_event], labels_predictions[:, i_event])
        auroc_val = auc(fpr_arr_temp, tpr_arr_temp)
        fpr_arr_list.append(fpr_arr_temp)
        tpr_arr_list.append(tpr_arr_temp)
        threshold_arr_list.append(threshold_arr_temp)
        auroc_val_list.append(auroc_val)

    # Get the threshold values for the acceptable TPR-s per event
    threshold_target_arr = np.zeros(labels_groundtruth.shape[1])
    print 'tpr_arr_list[0].shape:', tpr_arr_list[0].shape
    print 'labels_groundtruth.shape:', labels_groundtruth.shape
    for i_event in range(labels_groundtruth.shape[1]):
        for i_tpr in range(tpr_arr_list[i_event].shape[0] - 1):
            if tpr_target_arr[i_event] >= tpr_arr_list[i_event][i_tpr] and tpr_target_arr[i_event] < tpr_arr_list[i_event][i_tpr+1]:
                threshold_target_arr[i_event] = threshold_arr_list[i_event][i_tpr]

    if plot:
        # Plot TPR vs. thresholds
        plt.plot(threshold_arr_list[0], tpr_arr_list[0], 'r-', label='rh - tpr')
        plt.plot(threshold_arr_list[0], fpr_arr_list[0], 'r--', label='rh - fpr')
        plt.plot(threshold_arr_list[1], tpr_arr_list[1], 'b-', label='lh - tpr')
        plt.plot(threshold_arr_list[1], fpr_arr_list[1], 'b--', label='lh - fpr')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.title('TPR vs. thresholds')
        plt.legend(loc='upper right')
        plt.show()

        # Plot ROC
        for i_event in range(labels_groundtruth.shape[1]):
            plt.plot(fpr_arr_list[i_event], tpr_arr_list[i_event],
                     label='{0} (auc={1:0.4f})'.format(labels_names[i_event], auroc_val_list[i_event]))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC')
        plt.legend(loc='lower right')
        plt.show()

    return threshold_target_arr