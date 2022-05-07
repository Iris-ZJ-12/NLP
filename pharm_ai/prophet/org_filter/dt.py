import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import os.path
import langid

class OrgFilterPreprocessor:
    def __init__(self, version='v1'):
        self.h5 = os.path.join(os.path.dirname(__file__),'data.h5')
        self.excel_file = ["raw_data/机构识别段落提取 -交付版20201112.xlsx",
                           "raw_data/org ner段落提取-20201128加时间.xlsx"]
        self.current_version = version
        self.h5_keys = {'raw': self.current_version+'/raw',
                        'train': self.current_version+'/train',
                        'test': self.current_version+'/test'}
        self.random_state = 1128

    def preprocess_raw(self):
        if self.current_version=='v1':
            df_raw = pd.read_excel(self.excel_file[0])
            res = df_raw.rename(columns={'段落':'text','标注':'labels'})
            res.to_hdf(self.h5, self.h5_keys['raw'])
        elif self.current_version=='v1-1':
            df_raw = pd.read_excel(self.excel_file[1])
            res = df_raw.rename(columns={'时间':'date','段落':'paragraph','标注':'labels'})
            res['date'] = pd.to_datetime(res['date'], format='%Y/%m/%d')
            res['text'] = res[['date','paragraph']].apply(
                (lambda d: self.generate_text_from_date_paragraph(d['date'],d['paragraph'])), axis=1)
            res.to_hdf(self.h5, self.h5_keys['raw'])


    def train_test(self):
        if self.current_version in ['v1', 'v1-1']:
            df = pd.read_hdf(self.h5, self.h5_keys['raw'])
            train_df, test_df = train_test_split(df, test_size=0.1, random_state=self.random_state)
            # print dataset size info
            train_grs = train_df.groupby('labels')
            train_count_info = train_grs.count()
            test_count_info = test_df.groupby('labels').count()
            print("Train dataset size:\n", train_count_info)
            print("Test dataset size:\n", test_count_info)
            # upsampling to balance size of different labels
            max_label_size = train_grs['text'].count().max()
            upsampled_df = pd.concat([resample(d, n_samples=max_label_size, random_state=self.random_state)
                                      if d.shape[0]<max_label_size else d
                                      for _,d in train_grs])
            upsampled_df.sample(frac=1, random_state=self.random_state)
            # save to h5 file
            upsampled_df.to_hdf(self.h5, self.h5_keys['train'])
            test_df.to_hdf(self.h5, self.h5_keys['test'])

    def generate_text_from_date_paragraph(self, date, paragraph):
        """
        :param datetime.datetime date:  Publish date.
        :param str paragraph:  Paragraph string.
        :return: joined text.
        """
        paragraph_res = (str(paragraph)).strip()
        lang = langid.classify(paragraph_res)[0]
        if lang == 'zh':
            if not paragraph_res.endswith('。'):
                paragraph_res += '。'
            paragraph_res = f'本文发布日期为{date.year}年{date.month}月{date.day}日。{paragraph_res}'
        else:
            if not paragraph_res.endswith('.'):
                paragraph_res += '.'
            paragraph_res = f'This article was published on {date.strftime("%Y-%m-%d")}. {paragraph_res}'
        return paragraph_res


if __name__ == '__main__':
    print('s')
    # x = OrgFilterPreprocessor(version='v1-1')
    # x.preprocess_raw()
    # x.train_test()