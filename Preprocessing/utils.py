# This file contains useful functions for manipulating data

from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


def get_daily_count(df, timecol='pk_timestamp'):
    """Compute number of rows per day

    Arguments:
        df {pandas DataFrame} -- must contain a time column (usually pk_timestamp)
    Returns:
        pandas DataFrame -- contains number of rows per day
    """
    newdf = df.copy()
    newdf.index = newdf[timecol]
    daily_alarm_count = newdf.resample('d').count()

    # fill in missing dates with 0
    idx = pd.date_range(min(daily_alarm_count.index), max(daily_alarm_count.index))
    daily_counts = daily_alarm_count.reindex(idx, fill_value=0)

    return daily_counts


def plot_daily_PMs_alarms(daily_counts, daily_counts_alarms, savefile=None):
    """ Makes a pretty plot of the number of rows per day

    Arguments:
        daily_counts {pandas DataFrame} -- must contain pk_id column
        daily_counts_alarms {pandas DataFrame} -- same but for alarm days
    """
    fig, ax = plt.subplots(figsize=(20, 5))
    xplot = pd.to_datetime(daily_counts.index)
    yplot = daily_counts['pk_id']
    ax.plot(xplot, np.log10(yplot), '.', label='PMs')

    # plot red for gap days
    these = np.where(yplot == 0)[0]
    ax.plot(xplot[these], yplot[these] - 0.1, '.r', label='no data')

    xplot = pd.to_datetime(daily_counts_alarms.index)
    yplot = daily_counts_alarms['pk_id']
    ax.plot(xplot, np.log10(yplot), 'o', label='alarms')

    # plot red for gap days
    these = np.where(yplot == 0)[0]
    ax.plot(xplot[these], yplot[these] - 0.3, 'or', label='no alarm')

    ax.set_ylabel('$\log_{10}$ num data per day')
    ax.set_xlabel('date')
    ax.legend()
    if savefile:
        fig.savefig(savefile)
    return


def one_hot_encode_alarms(df):
    """Take a pandas dataframe with multi-labels in a "category" column and one-hot encode the labels

    Arguments:
        df {pandas DataFrame} -- dataframe with single column "category" with various string classes. Examples with multiple labels have duplicated rows.
    Returns:
        pandas DataFrame -- dataframe with one-hot-encoded labels, and duplicated rows removed
    """

    print('computing one-hot-encoded alarms')
    # take all rows with alarms (others are assumed to be None)
    df_alarms = df[df['category'].notnull()]

    # get a list of the unique alarm names
    alarm_names = list(df_alarms['category'].unique())

    # Create MultiLabelBinarizer object - include None (i.e. no alarm) as well
    one_hot = MultiLabelBinarizer(classes=[None] + alarm_names)

    # group the category labels that share the same pk_id and pk_timestamp (i.e. a device experienced multiple alarms simultaneously)
    labels = df_alarms.groupby(['pk_id', 'pk_timestamp'])['category'].apply(list)

    # One-hot encode the alarms
    labels_ohe = one_hot.fit_transform(labels)

    # drop the category column (no longer needed) and remove resulting duplicates
    df_alarms.drop(columns=['category'], inplace=True)
    df_alarms.drop_duplicates(inplace=True)

    # add "alarm " to the alarm columns
    alarm_colnames = ['alarm ' + str(alarm) for alarm in one_hot.classes_]

    labels_ohe_df = pd.DataFrame(labels_ohe, columns=alarm_colnames, index=df_alarms.index)

    # drop the categories column
    print('preparing dataframe for merging with one-hot-encoded alarms')
    df.drop(columns=['category'], inplace=True)
    df.drop_duplicates(inplace=True)

    # add the labels columns for the rest of the No Alarm device
    for colname in alarm_colnames:
        if 'None' in colname:
            df[colname] = 1
        else:
            df[colname] = 0
    print('adding one-hot-encoded alarms to dataframe')
    df.update(labels_ohe_df)

    return df


def calc_acc_metrics(preds, teY):
    """Given the predictions of a model, and the ground truth,
    calculate accuracy metrics to evaluate the performance of the model on
    unseen data

    Keyword Arguments:
    preds -- the predictions of the model on the unseen test set (teX)
    teY -- the ground truth (the classes that correspond to the unseen data)

    Returns:
    prec -- the precision score     == tp / (tp + fp)
    rec  -- the recall score        == tp / (tp + fn)
    acc  -- the accuracy score      == (tp + tn) / (tp + tn + fp + fn)
    f1   -- the f1 score            == 2 * (prec + rec) / (prec + rec)
    fpr  -- the false positive rate == fp / (fp + tn)
    """

    prec = precision_score(teY, preds)
    rec = recall_score(teY, preds)
    acc = accuracy_score(teY, preds)
    f1 = f1_score(teY, preds)

    conf_matrix = confusion_matrix(teY, preds)
    fpr = conf_matrix[0][1] / (conf_matrix[0][1] + conf_matrix[0][0])

    return prec, rec, acc, f1, fpr