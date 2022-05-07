import pandas as pd


class HohoDT:

    def __init__(self, split_ratio=0.9):
        self.split_ratio = split_ratio

    def get_training_data(self, path='data.h5'):
        train_df = pd.read_hdf(path, "train")
        valid_df = pd.read_hdf(path, "valid")
        return train_df, valid_df
