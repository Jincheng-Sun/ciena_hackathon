import pandas as pd
import numpy as np
from Preprocessing import utils


def data_pipline(data):
    '''
    Argument:
        data {pandas DataFrame}-- raw dataset

    Returns:
        data {pandas DataFrame}-- without CHMON, sorted and alarm-renamed dataset
    '''

    # drop CHMON, NMCMON
    data = data[~data['meta_facility'].isin(['CHMON', 'NMCMON'])]
    data = data.drop_duplicates(subset=['meta_nhpId', 'pk_timestamp'], keep='first')
    data = data.set_index(['meta_nhpId', 'pk_timestamp']).sort_index().reset_index()
    data.category = data.category.str.replace(" unplanned", '')
    data.category = data.category.str.replace(" planned", '')
    return data


def anomaly_detection_pipline(data):
    # convert to datetime
    data['pk_timestamp'] = pd.to_datetime(data['pk_timestamp'])
    # Step 1: Drop data before this date
    data = data[data['pk_timestamp'] > pd.to_datetime('2018-12-25')]
    # Step 2: Drop all NaN rows
    feature_names = [col for col in data.columns if
                     'pk_' not in col and 'meta_' not in col and 'alarm' not in col and 'category' not in col]
    data = data.dropna(axis=0, how='all', subset=feature_names)
    # Step 3: One-Hot encode the alarms
    data = utils.one_hot_encode_alarms(data)
    # Step 4: if it's normal and any UAS is greater than threshold, drop
    # Not considering if it's IS or not
    threshold = 8640
    uas_features = [col for col in data.columns if 'UAS' in col]
    uas_mask = data[uas_features].ge(threshold).any(axis=1)
    # normal_mask = data['alarm None'] == 1
    # mask = uas_mask & normal_mask
    # data = data[~mask]

    # drop all uas
    data = data[~uas_mask]

    # All PM: 211 features
    alarm_names = [col for col in data.columns if 'alarm ' in col]
    X = data[feature_names]
    y = data[alarm_names]
    dev = data['meta_facility']
    return X, y, dev


def anomaly_detection_pipline_new_rule(data):
    # convert to datetime
    data['pk_timestamp'] = pd.to_datetime(data['pk_timestamp'])
    # Step 1: Drop data before this date
    data = data[data['pk_timestamp'] > pd.to_datetime('2018-12-25')]
    # Step 2: Drop all NaN rows
    feature_names = [col for col in data.columns if
                     'pk_' not in col and 'meta_' not in col and 'alarm' not in col and 'category' not in col]
    data = data.dropna(axis=0, how='all', subset=feature_names)
    # Step 3: One-Hot encode the alarms
    data = utils.one_hot_encode_alarms(data)
    # Step 4: if it's normal and any UAS is greater than threshold, drop
    threshold = 8640
    uas_features = [col for col in data.columns if 'UAS' in col]
    uas_mask = data[uas_features].ge(threshold).any(axis=1)

    # New Rule, if OPINAVG
    op_features = [col for col in data.columns if 'OPRAVG' in col] + [col for col in data.columns if
                                                                      'OPINAVG' in col]
    op_features.remove('OPOUTAVG-OTS_OPINAVG-OTS_-')

    op_mask = data[op_features].le(-30).any(axis=1)

    mask = np.array(uas_mask) | np.array(op_mask)
    # drop Unavailable
    data = data[~mask]

    # All PM: 211 features
    alarm_names = [col for col in data.columns if 'alarm ' in col]
    X = data[feature_names]
    y = data[alarm_names]
    dev = data['meta_facility']
    return X, y, dev


# ----------- create dataset ------------
data = pd.read_parquet(
    '/Users/sunjincheng/Desktop/Hackathon/data/hackathon/colt_london_1DAY_allPMs_pivoted_processed_anonymized_20191002.parquet')
data = data.drop(columns=['__index_level_0__'])
data = data_pipline(data)
X, y, dev = anomaly_detection_pipline_new_rule(data)
X.to_parquet('../data/x_allpm_present.parquet')
y.to_parquet('../data/y_allpm_present.parquet')
dev.to_frame().to_parquet('../data/dev_allpm_present.parquet')
