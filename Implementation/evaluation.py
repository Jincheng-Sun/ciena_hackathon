from Models.models.present_fc import PresentFc
from Implementation import Dataset_anomaly_detection
from Evaluation.results import results, \
    precision_recall, plot_pr_curve, auc_roc, plot_roc_curve


model = PresentFc('models/', 'logs', input_dim=211, num_classes=1, encode_dim=64, dev_num=31, batch_size=1000,
                  lr=0.001, regression=True, threshold=0.9)

dataset = Dataset_anomaly_detection('../../data/allpm_anomaly/')

model.restore_checkpoint(14000)

# get proba
# proba = predict_avoid_OOM(model, dataset.test_set, output_dim=1)

# get prediction
prediction = model.get_prediction(dataset.test_set)

proba = model.get_proba(dataset.test_set)
# results
hackathon_format = results(dataset.test_set['y'], prediction)

# plot roc curve
auc, fprs, tprs, thresholds = auc_roc(y_pred=proba, y_test=dataset.test_set['y'])

plot_roc_curve(fprs, tprs, auc, x_axis=0.05)

# plot precision recall curve
auc, precisions, recalls, thresholds = precision_recall(y_pred=proba, y_test=dataset.test_set['y'])

plot_pr_curve(recalls, precisions, auc)

# get attention matrix