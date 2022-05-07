# encoding: utf-8
"""
@author: zyl
@file: data_analysis.py
@time: 2021/11/25 10:36
@desc:
"""

import pandas as pd
from zyl_utils.data_utils.analysis import BaseAnalyzer


class Analyzer(BaseAnalyzer):
    def __init__(self):
        super(Analyzer, self).__init__()
        pass

    def run(self):
        # self.test()
        self.prepare_data_to_be_analyzed_v4_2()

    def prepare_data_to_be_analyzed_v4_2(self):
        df = pd.read_excel(
            "/home/zyl/disk/PharmAI/pharm_ai/panel/data/v2.4.c.2/processed_1124_raw.xlsx") # type: pd.DataFrame

        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        from tqdm import tqdm
        tqdm.pandas(desc="my bar！")
        print('1')
        df['input_text_tokens'] = df['input_text'].apply(self.get_text_token_length,args=(tokenizer,))
        df['target_text_tokens'] = df['target_text'].apply(self.get_text_token_length,args=(tokenizer,))
        df.to_excel('./data/v4/analysis_v4_2_t5.xlsx')

    def test(self):
        df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/data/v2.4.c.2/processed_1124_raw.xlsx")
        # df['input_text_tokens_num'] = df['input_text'].apply
        self.show_df_base_info(df)
        # self.draw_box()


if __name__ == '__main__':
    Analyzer().run()
    pass
