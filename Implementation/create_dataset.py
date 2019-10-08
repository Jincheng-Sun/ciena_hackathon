import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

# read data
X = pd.read_parquet('../data/x_allpm_present.parquet')
y = pd.read_parquet('../data/y_allpm_present.parquet')
dev = pd.read_parquet('../data/dev_allpm_present.parquet')

# scaler and encoders
feature_scaler = MinMaxScaler()
facility_label_encoder = LabelEncoder()
facility_onehot_encoder = OneHotEncoder()

# meta_facility
meta_facility = np.array(['OPTMON', 'WAN', 'ETH10G', 'OTM2', 'AMP', 'OTM4', 'FLEX', 'OSC', 'OTM0',
       'PTP', 'ODUCTP', 'ETH100G', 'ETTP', 'OTUTTP', 'NMCMON', 'RAMAN',
       'OC192', 'ETH', 'ODUTTP', 'OTMC2', 'ETH40G', 'OTMFLEX', 'OTM1', 'ODU0',
       'OTM3', 'OTDRCFG', 'STTP', 'ODU1', 'OC12', 'ODUFLEX', 'OC48'])
# fit and transform
X = feature_scaler.fit_transform(X)
facility_onehot_encoder.fit(meta_facility.reshape(-1,1))
dev = facility_onehot_encoder.transform(dev).toarray()

# fill na with 0
X[np.isnan(X)] = 0

# create label, for binary anomaly detection
y = 1 - y['alarm None']
y = y.astype(int)
y = np.expand_dims(y, -1)

# split dataset
X_train, X_test = train_test_split(X, test_size=0.2, random_state=12)
y_train, y_test = train_test_split(y, test_size=0.2, random_state=12)
dev_train, dev_test = train_test_split(dev, test_size=0.2, random_state=12)


# save data in numpy format
np.save('../data/allpm_anomaly/X_train.npy',X_train)
np.save('../data/allpm_anomaly/X_test.npy',X_test)
np.save('../data/allpm_anomaly/y_train.npy',y_train)
np.save('../data/allpm_anomaly/y_test.npy',y_test)
np.save('../data/allpm_anomaly/dev_train.npy',dev_train)
np.save('../data/allpm_anomaly/dev_test.npy',dev_test)