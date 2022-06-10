import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

class Preprocessor:
    def __init__(self, version='v1.0'):
        self.version = version
        self.h5 = 'data.h5'
        self.random_state = 1228

    def get_h5_key(self, task):
        """task={'raw', 'train', 'eval', 'test'},
        return key format: version/task"""
        v = self.version.replace(".","-")
        return f"{v}/{task}"

    def preprocess_raw(self):
        if self.version == 'v1.0':
            xlsx_file = "raw_data/training data for IF drug_category_labels 1222_LXT.xlsx"
            df = pd.read_excel(xlsx_file)
            df.rename(columns={'drug_category_content':'text'}, inplace=True)
            df.to_hdf(self.h5, self.get_h5_key('raw'))

    def preprocess_train_test(self, eval_size = 0.1):
        if self.version == 'v1.0':
            df = pd.read_hdf(self.h5, self.get_h5_key('raw')).drop_duplicates()
            df_train, res_eval = train_test_split(df, random_state=self.random_state,
                                                  test_size=eval_size)
            ros = RandomOverSampler(random_state=self.random_state)
            X,y = ros.fit_resample(df_train[['text']].values, df_train['labels'].values)
            res_train = pd.DataFrame(np.hstack((X, y[:,np.newaxis])), columns=['text','labels'])\
                .astype(df_train[['text','labels']].dtypes.to_dict())
            res_train.to_hdf(self.h5, self.get_h5_key('train'))
            res_eval.to_hdf(self.h5, self.get_h5_key('eval'))

    def get_train_test_dataset(self):
        df_train = pd.read_hdf(self.h5, self.get_h5_key('train'))
        df_eval = pd.read_hdf(self.h5, self.get_h5_key('eval'))
        return df_train, df_eval

if __name__ == '__main__':
    preprocessor = Preprocessor(version='v1.0')
    preprocessor.preprocess_raw()
    preprocessor.preprocess_train_test()