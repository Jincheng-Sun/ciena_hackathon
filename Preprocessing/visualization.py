import pandas as pd
import matplotlib.pyplot as plt

def data_pipline(data):
    '''
    Argument:
        data {pandas DataFrame}-- raw dataset

    Returns:
        data {pandas DataFrame}-- without CHMON, sorted and alarm-renamed dataset
    '''

    # drop CHMON
    data = data[data['meta_facility'] != 'CHMON']
    data = data.drop_duplicates(subset=['meta_nhpId', 'pk_timestamp'], keep='first')
    data = data.set_index(['meta_nhpId', 'pk_timestamp']).sort_index().reset_index()
    data.category = data.category.str.replace(" unplanned", '')
    return data


def plot_per_device(data, device_id, date_range=pd.date_range('20181118', end='20190318', freq='D')):
    '''
    Argument:
        data {pandas DataFrame}-- preprocessed dataset
        device_id {String}-- meta_nhpId, e.g. 'Node1055:OPTMON-2-9-13:NTK554TA'

    '''
    plt.rcParams['figure.figsize'] = [10, 8]
    plt.rcParams['figure.dpi'] = 200

    # get data by device id
    per_device = data[data.meta_nhpId == device_id]
    per_device = per_device.set_index('pk_timestamp')
    per_device.index = pd.to_datetime(per_device.index)

    # get meta facility
    device_type = per_device['meta_facility'].values[0]

    # get anomaly indexes
    anomaly = per_device.loc[per_device['category'].notna()]
    anomaly_index_list = anomaly.index.tolist()

    # get out of sevice indexes
    oos = per_device[~per_device['meta_status'].isin(['IS'])]
    oos_index_list = oos.index.tolist()

    # drop all zero columns and reindex the data for the sake of plotting
    per_device = per_device.dropna('columns', 'all')
    days = date_range
    per_device = per_device.reindex(days)

    # print anomalies and the occurrence time
    print('-------------------- Anomalies --------------------')
    if anomaly.empty:
        print('No anomaly instance found')
    else:
        print(anomaly['category'])

    # print oos
    print('----------------- Out of service ------------------')
    if oos.empty:
        print('No out-of-service instance found')
    else:
        print(oos['meta_status'])

    plt.figure()
    # create a subplot for each PM value
    axes = per_device.plot(subplots=True, grid=True, marker='o', markersize=2)
    # plot anomalies
    for ax in axes:
        ax.legend(loc=2, prop={'size': 5})
        for vline in anomaly_index_list:
            ax.axvline(vline, color='r', lw=1, ls='dashed', label='anomaly')
            # print(anomaly.loc[vline, ['TIME', 'category']])
    # mark out of service cases

    for ax in axes:
        for dot in oos_index_list:
            ax.axvspan(dot - pd.Timedelta(days=1), dot, facecolor='#2ca02c', alpha=0.5)

    plt.xlabel('ID: ' + device_id + '   Device Type: ' + device_type)

    plt.show()

# ---------- turnkey-------------
# data = pd.read_parquet(
#     '/Users/sunjincheng/Desktop/Hackathon/data/hackathon/colt_london_1DAY_tkPMs_pivoted_processed_anonymized.parquet')
#
# data = data_pipline(data)
#
# anomalies = data.loc[data['category'].notna()]
#
# alarm_device_list = anomalies.meta_nhpId.unique()
#
# plot_per_device(data, alarm_device_list[300])

#----------- all PM -------------
data = pd.read_parquet(
    '/Users/sunjincheng/Desktop/Hackathon/data/hackathon/colt_london_1DAY_allPMs_pivoted_processed_anonymized.parquet')

data = data_pipline(data)

anomalies = data.loc[data['category'].notna()]

alarm_device_list = anomalies.meta_nhpId.unique()

plot_per_device(data, alarm_device_list[1010], pd.date_range('20190222', end='20190919', freq='D'))
# 'Node1163:ETTP-4-27-8:NTTP84BA'