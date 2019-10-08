import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as pyplot
pyplot.rcParams['savefig.dpi'] = 300  # pixel
pyplot.rcParams['figure.dpi'] = 300  # resolution
pyplot.rcParams["figure.figsize"] = [5,4] # figure size

def results(y_real, y_pred):
    prec = metrics.precision_score(y_real, y_pred)
    recall = metrics.recall_score(y_real, y_pred)
    num_alarms = len(np.where(y_real == 1)[0])
    num_alarms_correct = len(np.where((y_real == 1) & (y_pred == 1))[0])
    num_pos_preds = len(np.where(y_pred == 1)[0])

    mylist = [y_real.shape[0], num_alarms, num_alarms_correct, num_pos_preds, prec, recall]
    df = pd.DataFrame([mylist], columns = ['num examples','num alarms', 'num_alarms_correct', 'num pos predictions', 'precision', 'recall'])
    return df

def predict_avoid_OOM(model, data, output_dim, *args):
    proba = np.empty([0, output_dim])
    for i in range(int(np.floor(data.shape[0]/10000)+1)):
        proba = np.concatenate([proba, model.get_proba(data[i*10000:i*10000+10000], *args)])
        print(i)
    print('Done')
    return proba

def auc_roc(y_pred, y_test):
    auc = roc_auc_score(y_true=y_test, y_score=y_pred)
    fprs, tprs, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
    return auc, fprs, tprs, thresholds

def precision_recall(y_pred, y_test):
    precisions, recalls, thresholds = precision_recall_curve(y_true=y_test, probas_pred=y_pred)
    area = auc(recalls, precisions)
    return area, precisions, recalls, thresholds

def plot_roc_curve(fprs, tprs, auc, x_axis = 1, plt = pyplot):

    plt.plot(fprs, tprs, color="darkorange", label='ROC curve (area = %0.3f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, x_axis])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

def plot_pr_curve(recalls, precisions, auc, x_axis = 1, plt = pyplot):

    plt.plot(recalls, precisions, color="darkorange", label='Precision-Recall curve (area = %0.3f)' % auc)
    plt.plot([1, 0], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, x_axis])
    plt.ylim([0.0, 1.05])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")
    plt.show()