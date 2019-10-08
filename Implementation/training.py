from Models.models.present_fc import PresentSimple
from Implementation.dataset import Dataset_anomaly_detection

model = PresentSimple('models/', 'logs/', input_dim=211, num_classes=1, encode_dim=64, dev_num=31, batch_size=1000,
                  lr=0.001, regression=True, threshold=0.9, max_to_keep=50)
dataset = Dataset_anomaly_detection('../data/allpm_anomaly/')

model.initialize_variables()
# model.train_ae(dataset)
model.train(dataset)



proba = model.get_proba(dataset.test_set)

def cut(data, threshold):
    data = data >= threshold
    data = data.astype(int)
    return data
# # results
# hackathon_format = results(dataset.test_set['y'], prediction)
#
# # plot roc curve
# auc, fprs, tprs, thresholds = auc_roc(y_pred=proba, y_test=dataset.test_set['y'])
#
# plot_roc_curve(fprs, tprs, auc, x_axis=0.05)
#
# # plot precision recall curve
# auc, precisions, recalls, thresholds = precision_recall(y_pred=proba, y_test=dataset.test_set['y'])
#
# plot_pr_curve(recalls, precisions, auc)
#
# # get attention matrix