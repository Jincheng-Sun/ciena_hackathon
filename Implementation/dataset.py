from Models.modules.framework import Dataset, series_index
import numpy as np
from sklearn.model_selection import train_test_split


class Dataset_anomaly_detection(Dataset):
    def __init__(self, file_dir):
        super().__init__()
        # load data in numpy format
        X_train = np.load(file_dir + 'X_train.npy')
        X_test = np.load(file_dir + 'X_test.npy')
        y_train = np.load(file_dir + 'y_train.npy')
        y_test = np.load(file_dir + 'y_test.npy')
        dev_train = np.load(file_dir + 'dev_train.npy')
        dev_test = np.load(file_dir + 'dev_test.npy')

        self.train_set = np.zeros(X_train.shape[0], dtype=[
            ('x', np.float32, (X_train.shape[1:])),
            ('dev', np.int32, (dev_train.shape[1:])),
            ('y', np.int32, (y_train.shape[1:]))
        ])
        self.train_set['x'] = X_train
        self.train_set['dev'] = dev_train
        self.train_set['y'] = y_train

        self.test_set = np.zeros(X_test.shape[0], dtype=[
            ('x', np.float32, (X_test.shape[1:])),
            ('dev', np.int32, (dev_test.shape[1:])),
            ('y', np.int32, (y_test.shape[1:]))
        ])
        self.test_set['x'] = X_test
        self.test_set['dev'] = dev_test
        self.test_set['y'] = y_test

        self.train_set, self.eval_set = train_test_split(self.train_set, test_size=10000, random_state=12)

    def training_ae_generator(self, batch_size=100, random=np.random):
        i = 0
        while i <= np.ceil(len(self.train_set) / batch_size):
            yield self.train_set[i * batch_size:(i + 1) * batch_size]
            i+=1

