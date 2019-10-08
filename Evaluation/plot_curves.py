import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

def precision_recall(y_pred, y_test):
    precisions, recalls, thresholds = precision_recall_curve(y_true=y_test, probas_pred=y_pred)
    area = auc(recalls, precisions)
    return area, precisions, recalls, thresholds

def plot_pr_curve(recalls, precisions, auc, x_axis = 1):
    plt.rcParams['savefig.dpi'] = 300  # pixel
    plt.rcParams['figure.dpi'] = 300  # resolution
    plt.rcParams["figure.figsize"] = [5, 4]  # figure size

    plt.plot(recalls, precisions, color="darkorange", label='Precision-Recall curve (area = %0.3f)' % auc)
    plt.plot([1, 0], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, x_axis])
    plt.ylim([0.0, 1.05])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")
    plt.show()