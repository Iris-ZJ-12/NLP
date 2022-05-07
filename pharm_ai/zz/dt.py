"""
Author      : Zhang Yuliang
Datetime    : 2021/1/28 下午7:26
Description : 
"""
import PIL.ImageCms
import pandas as pd
import random
from pharm_ai.util.utils import Utilfuncs as U
from pharm_ai.zz.es_utils import ESObject
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

class ZZDataSet2:
    def __init__(self, version, saved, saved_processed_data_file, projects_dict_file, label_yes_dt, label_no_dt,
                 random_rate=234, h5_file='./data/zz.h5'):
        self.h5 = h5_file
        self.es = ESObject()
        self.ver = version
        self.ran = random_rate

        self.saved_xlsx = saved_processed_data_file

        projects_df = pd.read_excel(projects_dict_file)
        self.proj = self.deal_with_project_dict(projects=projects_df, saved=False)

        # label_yes_df = pd.read_excel(label_yes_dt)
        # label_no_df = pd.read_excel(label_no_dt)
        # new_df = pd.concat([label_yes_df, label_no_df], ignore_index=True)
        # raw_data = self.deal_with_new_data(new_df, saved=True)
        # sum_data = self.get_sum_data(raw_data, saved=True)
        # m1_train, m1_eval, m2_train_pos, m2_eval_pos = self.get_m1m2_data(sum_data, saved=True)
        # m2_pos = pd.concat([m2_train_pos, m2_eval_pos])
        # m2_fake = self.create_m2_fake_data(m2_pos_data=m2_pos, stratified_sampling=(0.1, 1, 0.1),
        #                                    choosed_k=4000, query_max_size=80, saved=True)
        # m2_train_neg, m2_eval_neg = ZZDataSet2.split_train_eval(data=m2_fake,
        #                                                         train_size=0.9, test_size=0.1,
        #                                                         random_rate=random_rate)
        # m2_train = pd.concat([m2_train_pos, m2_train_neg])
        # m2_eval = pd.concat([m2_eval_pos, m2_eval_neg])
        # if saved:
        #     U.to_excel(m2_train, self.saved_xlsx, 'm2_train')
        #     U.to_excel(m2_eval, self.saved_xlsx, 'm2_eval')

    def deal_with_project_dict(self, projects: pd.DataFrame, saved):
        projects = projects.loc[:, ["ESID", "项目名称", "项目状态", "年份", "级别", "地区"]]
        projects.dropna(subset=['项目名称'], inplace=True)
        projects.fillna('', inplace=True)
        projects.rename(columns={'项目名称': 'project_name', '地区': 'province', '项目状态': 'status', '年份': 'years',
                                 '级别': 'levels'}, inplace=True)

        projects['years'] = projects['years'].astype('str')
        projects['years'] = projects['years'].apply(lambda x: x[0:-2] if len(x) > 4 else '')  # 2019.0-->2019
        projects['text_b'] = projects['project_name'] + ' ' + projects['status'] + ' ' + projects['years'] + ' ' + \
                             projects['levels'] + ' ' + projects['province']
        projects.drop_duplicates(inplace=True)

        if saved:
            U.to_excel(projects, self.saved_xlsx, 'projects')
        return projects

    def deal_with_new_data(self, new_df, saved):
        df = new_df[["标题", "省份", "文章日期", "是否中标资讯", "中标项目(魔方)"]]
        df.drop_duplicates(inplace=True)
        df.rename(columns={'标题': 'title', '省份': 'province', '文章日期': 'date',
                           '是否中标资讯': 'labels', '中标项目(魔方)': 'project_name'}, inplace=True)
        df['date'] = df['date'].str.split(' ', expand=True)[0]  # 2020-06-29 00:00:00-->2020-06-29
        df.fillna('', inplace=True)
        df['labels'] = df['labels'].map({'是': 1, '否': 0, '': 0})
        df['project_name'] = df['project_name'].apply(lambda x: x[1:-1] if x.startswith('|') else x)
        df['text_a'] = df['province'] + ' ' + df['title'] + ' ' + df['date']

        # merge data and dict
        raw_data = pd.merge(df, self.proj, how='outer', on=['project_name'])

        if saved:
            U.to_excel(raw_data, self.saved_xlsx, 'raw_data')
            raw_data.to_hdf(self.h5, self.ver + '/raw_data')
        return raw_data

    def get_sum_data(self, raw_data: pd.DataFrame, saved):
        raw_data.dropna(subset=['title'], inplace=True)
        raw_data.fillna('', inplace=True)
        sum_data = raw_data[(raw_data['province_x'] == raw_data['province_y']) | (raw_data['province_y'] == '')]
        sum_data.drop_duplicates(inplace=True, ignore_index=True)
        if saved:
            U.to_excel(sum_data, self.saved_xlsx, 'sum_data')
        return sum_data

    def get_m1m2_data(self, sum_data: pd.DataFrame, saved):
        sum_train, sum_eval = ZZDataSet2.split_train_eval(data=sum_data, train_size=0.9, test_size=0.1,
                                                          random_rate=self.ran)
        sum_train.drop_duplicates(inplace=True, ignore_index=True)
        sum_eval.drop_duplicates(inplace=True, ignore_index=True)

        m2_train_pos = sum_train[sum_train['labels'] == 1]
        m2_train_pos = m2_train_pos[['text_a', 'text_b', 'labels']]
        m1_train = sum_train[['title', 'labels']]
        m1_train.rename(columns={'title': 'text'}, inplace=True)
        m1_train.drop_duplicates(inplace=True, ignore_index=True)
        m2_train_pos.drop_duplicates(inplace=True, ignore_index=True)

        m2_eval_pos = sum_eval[sum_eval['labels'] == 1]
        m2_eval_pos = m2_eval_pos[['text_a', 'text_b', 'labels']]
        m1_eval = sum_eval[['title', 'labels']]
        m1_eval.rename(columns={'title': 'text'}, inplace=True)
        m1_eval.drop_duplicates(inplace=True, ignore_index=True)
        m2_eval_pos.drop_duplicates(inplace=True, ignore_index=True)

        if saved:
            U.to_excel(sum_train, self.saved_xlsx, 'sum_train')
            U.to_excel(sum_eval, self.saved_xlsx, 'sum_eval')
            U.to_excel(m1_train, self.saved_xlsx, 'm1_train')
            U.to_excel(m1_eval, self.saved_xlsx, 'm1_eval')
            U.to_excel(m2_train_pos, self.saved_xlsx, 'm2_train_pos')
            U.to_excel(m2_eval_pos, self.saved_xlsx, 'm2_eval_pos')
        return m1_train, m1_eval, m2_train_pos, m2_eval_pos

    def create_m2_fake_data(self, m2_pos_data, stratified_sampling, choosed_k, query_max_size, saved):
        print('create fake data by using stratified sampling:...')

        fake_data = pd.DataFrame(columns=['text_a', 'text_b', 'labels'])
        titles = m2_pos_data['text_a'].tolist()
        text_as = random.choices(titles, k=choosed_k)

        for text_a in text_as:
            similar_text_bs = self.es.fuzzy_match(query_filed_name='text_b', query_filed_value=text_a,
                                                  get_filed='text_b', query_max_size=query_max_size)
            left_text_bs = similar_text_bs[0:int(query_max_size * stratified_sampling[0])]
            left_text_bs = random.choices(left_text_bs, k=int(len(left_text_bs) * stratified_sampling[1]))
            right_text_bs = similar_text_bs[int(query_max_size * stratified_sampling[0]):-1]
            right_text_bs = random.choices(right_text_bs, k=int(len(right_text_bs) * stratified_sampling[2]))
            selected_text_bs = left_text_bs + right_text_bs
            for text_b in selected_text_bs:
                if not ((m2_pos_data['text_a'] == text_a) & (m2_pos_data['text_b'] == text_b)).any():
                    fake_data = fake_data.append([{'text_a': text_a, 'text_b': text_b, 'labels': 0}],
                                                 ignore_index=True)

        fake_data.drop_duplicates(inplace=True, ignore_index=True)
        fake_data = fake_data.reset_index(drop=True)
        if saved:
            U.to_excel(fake_data, self.saved_xlsx, 'm2_fake_data')
        print('Done creating fake data:...')
        return fake_data

    @staticmethod
    def split_train_eval(data, train_size, test_size, random_rate):
        train_dt, eval_dt = train_test_split(data, train_size=train_size, test_size=test_size,
                                             random_state=random_rate, stratify=data['labels'])
        train_dt = train_dt.reset_index(drop=True)
        eval_dt = eval_dt.reset_index(drop=True)
        return train_dt, eval_dt

    # def view_this_version_dt(self):
    #     data = pd.HDFStore(self.h5, mode='r')
    #     # 查看指定h5对象中的所有键
    #     ks = data.keys()
    #     print('#' * 60)
    #     print('data filename:', data.filename)
    #     for key in ks:
    #         if self.ver in key:
    #             print('*' * 20)
    #             d: pd.DataFrame = pd.read_hdf(self.h5, key)
    #             # print(d.info())
    #             # print(d.discribe())
    #             print('data name:', key)
    #             print('data size:', d.shape[0])
    #             if type(d) != pd.core.series.Series:
    #                 if 'labels' in d.columns:
    #                     labels = d['labels'].value_counts()
    #                     print(labels)
    #     data.close()

    def show_all_dt(self):
        data = pd.HDFStore(self.h5, mode='r')
        # 查看指定h5对象中的所有键
        ks = data.keys()
        print('#' * 60)
        print('data filename:', data.filename)
        for key in ks:
            print('*' * 20)
            d = pd.read_hdf(self.h5, key)
            print('data name:', key)
            print('data size:', d.shape[0])
            if type(d) != pd.core.series.Series:
                if 'labels' in d.columns:
                    labels = d['labels'].value_counts()
                    print(labels)
        data.close()


class V620210126(ZZDataSet2):
    def __init__(self):
        super(V620210126, self).__init__(version='v6_0126', saved=True,
                                         projects_dict_file='./data/v6/中标项目(1).xls',
                                         saved_processed_data_file='./data/v6/processed_dt0126.xlsx',
                                         label_yes_dt="./data/v6/是中标资讯且已匹配到对应项目.xls",
                                         label_no_dt="./data/v6/“否”的中标资讯.xls",
                                         )


class V620210128(ZZDataSet2):
    def __init__(self):
        super(V620210128, self).__init__(version='v6_0128', saved=True,
                                         projects_dict_file='./data/v6/中标项目(1).xls',
                                         saved_processed_data_file='./data/v6/processed_dt0128.xlsx',
                                         label_yes_dt="./data/v6/是中标资讯且已匹配到对应项目.xls",
                                         label_no_dt="./data/v6/“否”的中标资讯.xls",
                                         )
        self.view_this_version_dt()

class V720210707(ZZDataSet2):
    def __init__(self):
        super(V720210707, self).__init__(version='v7_0707', saved=True,
                                         projects_dict_file='./data/v7/中标项目(1).xls',
                                         saved_processed_data_file='./data/v6/processed_dt0128.xlsx',
                                         label_yes_dt="./data/v6/是中标资讯且已匹配到对应项目.xls",
                                         label_no_dt="./data/v6/“否”的中标资讯.xls",
                                         )
        self.view_this_version_dt()


class V720210707(ZZDataSet2):
    def __init__(self):
        super(V720210707, self).__init__(version='v7_0707', saved=True,
                                         projects_dict_file='./data/v1.7/项目字典7.8(1).xls',
                                         saved_processed_data_file='./data/v1.7/processed_dt0708.xlsx',
                                         label_yes_dt="./data/v6/是中标资讯且已匹配到对应项目.xls",
                                         label_no_dt="./data/v6/“否”的中标资讯.xls",
                                         )
        new_projects_df = pd.read_excel('./data/v1.7/项目字典7.8(1).xls')
        self.proj = self.deal_with_project_dict(new_projects_df, saved=True)
        self.es = ESObject(index_name='zz_projects_0708', index_type='_doc', hosts='101.201.249.176',
                           user_name='elastic',
                           user_password='Zyl123123', port=9325)

    def run(self):
        self.dt_0708_up()
        # self.test()

    def test(self):
        # self.compare_projs()
        # self.get_old_dt()
        self.dt_0707()

        # o = pd.read_excel(self.saved_xlsx, 'new_data')
        # oss = o['title'].tolist()
        # new_df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/zz/data/v1.7/上半年资讯7.8(1).xls")
        # new_df = new_df[~new_df['标题'].isin(oss)]
        # new_df.to_excel('./data/t2.xlsx')

    def modify_old_project(self, proj: str):
        proj = str(proj)
        if '全国药品集中采购项目' in str(proj):
            proj = str(proj).replace('轮', '批')
            proj = proj.replace('（', '(')
            proj = proj.replace('）', ')')
        if proj == '2019年全国药品集中采购项目(国采第二批-新疆含新疆生产建设兵团)':
            return '2019年全国药品集中采购项目(国采第二批-新疆)'
        if proj == '2020年青海省部分药品挂网采购项目(增加)':
            return '2020年青海省部分药品挂网采购项目'
        if proj == '2020年广西壮族自治区第一批常用药品联合带量采购项目':
            return '2020年广西壮族自治区药品集团采购项目（第二批）'
        if proj == '2019年广西壮族自治区药品集团采购项目':
            return '2019年广西壮族自治区药品集团采购项目（第一批）'
        if proj == '2020年江苏省药品集中带量采购项目':
            return '2020年江苏省第一轮药品集中带量采购项目'
        if proj == '2020年山东省药品集中带量采购项目':
            return '2020年山东省第一批药品集中带量采购项目'
        if proj == '2020年湖北省药品集中带量采购项目':
            return '2020年湖北省首批药品集中带量采购项目'
        if ('2020年六省二区省际联盟药品集中带量采购项目' in proj) or ('2020年六省二区省级联盟药品集中带量采购项目' in proj):
            return '2020年六省二区省际联盟药品集中带量采购项目'
        return proj

    def compare_projs(self):
        old_projs = pd.read_excel('./data/v1.6/processed_dt0128.xlsx', 'projects')
        old_projs['project_name'] = old_projs['project_name'].apply(self.modify_old_project)
        old_projs = old_projs['project_name'].tolist()
        new_projs = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/zz/data/v1.7/项目字典7.8(1).xls")
        new_projs = new_projs['项目名称'].tolist()
        l = []
        for i in old_projs:
            if i not in new_projs:
                l.append(i)
        print(list(set(l)))

    def get_old_dt(self, saved=False):
        old_df = pd.read_excel('./data/v1.6/processed_dt0128.xlsx', 'sum_data')
        old_df = old_df[['title', 'province_x', 'date', 'labels', 'project_name', 'text_a']]
        old_df['project_name'] = old_df['project_name'].apply(self.modify_old_project)
        raw_data = pd.merge(old_df, self.proj, how='outer', on=['project_name'])
        raw_data = raw_data[~((raw_data['labels'] == 1) & (raw_data['ESID'].isna()))]
        raw_data.rename(columns={'province': 'province_y'}, inplace=True)
        raw_data = self.get_sum_data(raw_data, saved=False)
        if saved:
            U.to_excel(raw_data, self.saved_xlsx, 'old_data')
        return raw_data

    def dt_0707(self, saved=True):
        new_df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/zz/data/v1.7/上半年资讯7.8(1).xls")
        new_df = new_df[new_df['是否中标资讯'].isin(["是", '否'])]
        new_df['中标项目(魔方)'] = new_df['中标项目(魔方)'].apply(self.modify_old_project)
        raw_data = self.deal_with_new_data(new_df, saved=False)
        raw_data = raw_data[~((raw_data['labels'] == 1) & (raw_data['ESID'].isna()))]
        new_sum_data = self.get_sum_data(raw_data, saved=False)
        if saved:
            U.to_excel(raw_data, self.saved_xlsx, 'new_data')
        olf_sum_data = self.get_old_dt(saved=True)
        sum_data = pd.concat([new_sum_data, olf_sum_data], ignore_index=True)
        sum_data.drop_duplicates(inplace=True, ignore_index=True)

        if saved:
            U.to_excel(sum_data, self.saved_xlsx, 'sum_data')
        m1_train, m1_eval, m2_train_pos, m2_eval_pos = self.get_m1m2_data(sum_data, saved=True)
        m2_pos = pd.concat([m2_train_pos, m2_eval_pos])
        m2_fake = self.create_m2_fake_data(m2_pos_data=m2_pos, stratified_sampling=(0.1, 1, 0.1),
                                           choosed_k=8000, query_max_size=100, saved=True)
        m2_train_neg, m2_eval_neg = ZZDataSet2.split_train_eval(data=m2_fake,
                                                                train_size=0.9, test_size=0.1,
                                                                random_rate=2345)
        m2_train = pd.concat([m2_train_pos, m2_train_neg])
        m2_eval = pd.concat([m2_eval_pos, m2_eval_neg])

        U.to_excel(m2_train, self.saved_xlsx, 'm2_train')
        U.to_excel(m2_eval, self.saved_xlsx, 'm2_eval')

        return raw_data


    def dt_0708_up(self):
        df1 = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/zz/data/v1.7/processed_dt0708.xlsx",'m1_train')
        df1 = self.up_sampling(df1)
        df2 = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/zz/data/v1.7/processed_dt0708.xlsx",'m2_train')
        df2 = self.up_sampling(df2)
        U.to_excel(df1, './data/v1.7/processed_dt0708_up.xlsx', 'm1_train_up')
        U.to_excel(df2, './data/v1.7/processed_dt0708_up.xlsx', 'm2_train_up')

    @staticmethod
    def up_sampling(train_df: pd.DataFrame):
        """
        Args:
            train_df: ['prefix', 'input_text', 'target_text']
        Returns:
        """
        print('use up sampling!')
        pos_df = train_df[train_df['labels'] == 1]
        neg_df = train_df[train_df['labels'] == 0]

        up_sampling_num = max([len(pos_df), len(neg_df)])

        if len(pos_df) < up_sampling_num:
            up_sampling_pos_df = resample(pos_df, replace=True, n_samples=(up_sampling_num - len(pos_df)),
                                         random_state=3242)
        else:
            up_sampling_pos_df = pd.DataFrame()

        if len(neg_df) < up_sampling_num:
            up_sampling_neg_df = resample(neg_df, replace=True, n_samples=(up_sampling_num - len(neg_df)),
                                          random_state=3242)
        else:
            up_sampling_neg_df = pd.DataFrame()

        return pd.concat([pos_df, neg_df, up_sampling_pos_df, up_sampling_neg_df],
                         ignore_index=True)


if __name__ == '__main__':
    # dataset = V620210128()
    V720210707().run()
