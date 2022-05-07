import pandas as pd
from loguru import logger
from pharm_ai.util.ESUtils7 import get_page, Query, QueryType
import json
from pathlib import Path
from pharm_ai.util.utils import Utilfuncs as u
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


class PreprocessorBase:
    version='v0'
    path_root = Path(__file__).parent
    data_path = {
        'raw': path_root / 'dataset' / 'raw',
        'train': path_root / 'dataset' / 'train',
        'eval': path_root / 'dataset' / 'eval'
    }
    random_state = 68


    @classmethod
    def save_dataset(cls, df, item='raw'):
        saving_path = cls.get_dataset_path(item)
        if not saving_path.parent.exists():
            saving_path.parent.mkdir(parents=True)
        df.to_csv(saving_path, index_label='index_')
        logger.info('{} dataset saved to "{}".', item, saving_path)

    @classmethod
    def get_dataset_path(cls, item='raw'):
        result_path = cls.data_path[item]/cls.version/(item+'_dataset.csv')
        return result_path

    @classmethod
    def get_dataset(cls, item='raw'):
        loading_path = cls.get_dataset_path(item)
        df = pd.read_csv(loading_path, index_col='index_', lineterminator='\n')
        logger.info('{} dataset loaded from "{}".', item, loading_path)
        return df

    def preprocess_train_eval_dataset(self):
        df_raw = self.get_dataset()
        df_train, df_eval = train_test_split(df_raw, test_size=0.1, random_state=self.random_state)
        self.save_dataset(df_train, 'train')
        self.save_dataset(df_eval, 'eval')


class PreprocessorUtils:
    possible_indexes = ['news_library', 'invest_news', 'news']
    host_map = {'online': None, 'test': ('test155.cubees.com',)}

    @staticmethod
    def get_content_by_esids(esids, host='online', verbose=False):
        if verbose:
            logger.level('DEBUG')

        MAX_STEP = 2000
        if len(esids)> MAX_STEP:
            # split to multiple batches
            esid_batches = [esids[start_ind: start_ind + MAX_STEP]
                            if start_ind + MAX_STEP <len(esids) else esids[start_ind:]
                            for start_ind in range(0, len(esids), MAX_STEP)]
            result_batch = [PreprocessorUtils.get_content_by_esids(batch, host=host, verbose=verbose)
                            for batch in esid_batches]
            results = [r for rb in result_batch for r in rb]
        else:
            results = [None]*len(esids)
            for index_name in PreprocessorUtils.possible_indexes:
                if all(r is not None for r in results):
                    logger.debug('All data get, break.')
                    break
                queries = Query.queries(*[Query(QueryType.EQ, 'esid', esid)
                                          for esid, cur_res in zip(esids, results) if not cur_res],
                                        and_perator=False)
                res_raw = get_page(index_name, queries=queries, page_size=-1,
                                   show_fields=['content'],
                                   host=PreprocessorUtils.host_map[host])
                logger.debug("{} data get from index '{}'",
                             len(res_raw) if res_raw else 0, index_name)
                if res_raw:
                    for res_raw_each in res_raw:
                        update_ind = esids.index(res_raw_each['esid'])
                        results[update_ind] = res_raw_each.get('content')
            if any(r is None for r in results):
                logger.warning("{} data not get.", sum(1 for r in results if r is None))
            logger.level('INFO')
        return results



class BertPreprocessor(PreprocessorBase):
    def __init__(self):
        super(BertPreprocessor, self).__init__()

    def get_num_labels(self):
        label_map = self.get_dataset('label_map')
        return len(label_map)

    @classmethod
    def get_dataset_path(cls, item='raw'):
        """item: [raw, train, eval, label_map]"""
        if item=='label_map':
            result_path = cls.path_root / 'dataset' / 'raw' / cls.version / 'label_map.json'
        else:
            result_path = super().get_dataset_path(item=item)
        return result_path

    @classmethod
    def save_dataset(cls, df, item='raw'):
        if item=='label_map':
            label_list = df['labels'].unique().tolist()
            label_map = {i: ls_ for i, ls_ in enumerate(label_list)}
            saving_path = cls.get_dataset_path('label_map')
            if not saving_path.parent.exists():
                saving_path.parent.mkdir(parents=True)
            with open(saving_path, 'w') as f:
                json.dump(label_map, f, ensure_ascii=False, indent=4)
            logger.info('Label map saved to "{}".', saving_path)
            return label_map
        else:
            super().save_dataset(df, item)

    @classmethod
    def get_dataset(cls, item='raw', label_map_reverse=False):
        if item=='label_map':
            load_path = cls.get_dataset_path('label_map')
            with open(load_path, 'r') as f:
                label_map_raw = json.load(f)
            logger.info('Label map loaded from "{}".', load_path)
            if label_map_reverse:
                result = {v:int(i) for i,v in label_map_raw.items()}
            else:
                result = {int(i):v for i,v in label_map_raw.items()}
            return result
        elif item=='train':
            df = super().get_dataset(item)
            return cls.upsample_train_dataset(df)
        else:
            return super().get_dataset(item=item)

    @classmethod
    def validate_dataset(cls, item='raw'):
        df = cls.get_dataset(item)
        assert df['text'].notna().all(), f"df[text] has {df['text'].isna().sum()} NA values"
        empty_texts = df['text'].eq('')
        assert not empty_texts.any(), f"df[text] has {empty_texts.sum()} empty items."
        label_types = df['labels'].map(lambda x:isinstance(x, int))
        assert label_types.all(), f"df[labels] not int: {(~label_types).sum()}"
        logger.success('{} dataset validate success!', item)

    @classmethod
    def upsample_train_dataset(cls, df=None):
        if df is None:
            df = cls.get_dataset('train')
        upsample_size = df['labels'].value_counts().max()
        res_df = pd.concat(
            resample(d, n_samples=upsample_size, random_state=cls.random_state)
            if d.shape[0]<upsample_size else d
            for label_, d in df.groupby('labels')
        )
        logger.info('Training dataset upsampled: from {} to {}.', df.shape[0], res_df.shape[0])
        return res_df.sample(frac=1, random_state=cls.random_state)

    @classmethod
    def describe_dataset(cls, item='raw'):
        df = cls.get_dataset(item)
        label_map = cls.get_dataset('label_map')
        df['labels'] = df['labels'].map(label_map)
        if item=='train':
            df = df.drop_duplicates()
        print('-'*50)
        print(f'{item} dataset:')
        res_size = df['labels'].value_counts().to_frame().rename(columns={'labels':'count'})
        res_size['percent'] = res_size/df.shape[0]
        print(res_size)
        print('-'*50)
        print(f'# Total: {item} dataset', df.shape[0])



class PreprocessorV1(BertPreprocessor):
    version = 'v1'
    def __init__(self):
        super(PreprocessorV1, self).__init__()
        self.raw_data_xlsx = "raw_data/20210602_wok_data_examplar_v1.xlsx"


    def preprocess_raw(self, esid_files=['results/all_esids_v1.json', "results/wok_null_content_esids.json"],
                       content_files=["results/contents_v1.json", "results/refind_contents_v1.json"],
                       return_df=False):
        df_raw = pd.read_excel(self.raw_data_xlsx, usecols=['esid','title','actual_labels'])

        # load contents

        esid_content_map = self.get_esid_content_pair(esid_files[0], content_files[0])
        esid_content_map.update(
            self.get_esid_content_pair(esid_files[1], content_files[1])
        )

        # convert contents to fulltexts
        esid_fulltext_map = {esid: u.remove_html_tags(content) if content else content
                             for esid,content in esid_content_map.items()}

        # insert fulltext
        df_raw['fulltext'] = df_raw['esid'].map(esid_fulltext_map.get)

        # save
        if not return_df:
            self.save_dataset(df_raw)
        else:
            return df_raw

    def save_esids(self, saving_json='results/all_esids_v1.json'):
        """Prepare ESIDs to be queried from es."""
        df_raw = pd.read_excel(self.raw_data_xlsx, usecols=['esid'])
        esids = df_raw['esid'].tolist()
        with open(saving_json, 'w') as saving_f:
            json.dump(esids, saving_f, ensure_ascii=False, indent=4)
        logger.info('{} ESIDs saved to "{}"', len(esids), saving_json)

    def get_content_from_esids(self, load_esids_json='results/all_esids_v1.json',
                               saving_json='results/contents_v1.json', host='online'):
        with open(load_esids_json, 'r') as load_f:
            esids = json.load(load_f)
        logger.info('{} ESIDs loaded from "{}"', len(esids), load_esids_json)
        contents = PreprocessorUtils.get_content_by_esids(esids, host)
        with open(saving_json, 'w') as saving_f:
            json.dump(contents, saving_f, ensure_ascii=False, indent=4)
        logger.info('{} contents saved to "{}"', len(contents), saving_json)

    def export_esids_of_empty_contents(
            self,
            load_esids_json='results/all_esids_v1.json',
            load_content_json='results/contents_v1.json',
            saving_excel='results/wok_null_content_esids.xlsx',
            saving_json='results/wok_null_content_esids.json'):
        with open(load_esids_json, 'r') as load_esid_f:
            esids = json.load(load_esid_f)
        with open(load_content_json, 'r') as load_content_f:
            contents = json.load(load_content_f)
        result_esids = [esid for esid, content in zip(esids, contents) if not content]
        if saving_excel:
            result_df = pd.DataFrame({'esid': result_esids})
            result_df.to_excel(saving_excel, index=False)
            logger.info('{} ESIDs of null content saved to "{}"', result_df.shape[0], saving_excel)
        if saving_json:
            with open(saving_json, 'w') as saving_json_f:
                json.dump(result_esids, saving_json_f, ensure_ascii=False, indent=4)
            logger.info('{} ESIDs of null content saved to "{}"', len(result_esids), saving_json)

    def get_esid_content_pair(self, load_esids_json='results/all_esids_v1.json',
                              load_content_json='results/contents_v1.json'):
        with open(load_esids_json, 'r') as esid_f:
            esids = json.load(esid_f)
        with open(load_content_json, 'r') as content_f:
            contents = json.load(content_f)
        return dict(zip(esids, contents))


class PreprocessorV1_0(PreprocessorV1):
    """Only use title to train bert model."""
    version = 'v1.0'

    def preprocess_raw(self, df=None, return_df=False):
        if df is None:
            df = PreprocessorV1.get_dataset()

        # bert dataset format
        select_cols = ['esid', 'title', 'actual_labels']
        col_map = {'title': 'text', 'actual_labels': 'labels'}
        df_res = df[select_cols].rename(columns=col_map)

        # save label map
        label_map=self.save_dataset(df_res, item='label_map')
        df_res['labels'] = df_res['labels'].map(dict(zip(label_map.values(), label_map.keys())))

        # saving
        if not return_df:
            self.save_dataset(df_res)
        else:
            return df_res



class PreprocessorV1_1(PreprocessorV1):
    """text=title+fulltext"""
    version = 'v1.1'

    def preprocess_raw(self, df=None, return_df=False):
        if df is None:
            df = PreprocessorV1.get_dataset()
        df = df[df['fulltext'].notna()]

        # bert dataset format
        df['text'] = df['title']+' '+df['fulltext']
        col_map = {'actual_labels': 'labels'}
        df_res = df[['esid', 'text', 'actual_labels']].rename(columns=col_map)

        # save label map
        label_map = self.save_dataset(df_res, item='label_map')
        df_res['labels'] = df_res['labels'].map(dict(zip(label_map.values(), label_map.keys())))

        # saving
        if return_df:
            return df_res
        else:
            self.save_dataset(df_res)


class PreprocessorV1_2(PreprocessorV1_0):
    version = 'v1.2'

    def __init__(self):
        super().__init__()
        self.raw_data_xlsx = "raw_data/20210608_wok_data_examplar_v2.xlsx"

    def preprocess_raw(self):
        res_df = self.apply_append_dataset()
        res_df = PreprocessorV1_0.preprocess_raw(self, res_df, return_df=True)
        self.save_dataset(res_df)

    def apply_append_dataset(self):
        previous_df = PreprocessorV1.get_dataset()
        new_df = PreprocessorV1.preprocess_raw(
            self, esid_files=['results/all_esids_v1.3.json', 'results/null_content_esids_v1.3.json'],
            content_files=['results/contents_v1.3.json', 'results/refind_null_contents_v1.3.json'],
            return_df=True
        )
        res_df = pd.concat([previous_df, new_df], ignore_index=True)
        return res_df


class PreprocessorV1_3(PreprocessorV1_1, PreprocessorV1_2):
    """Title+fulltext, append data"""
    version = 'v1.3'

    def __init__(self):
        PreprocessorV1_2.__init__(self)

    def preprocess_raw(self):
        df = self.apply_append_dataset()
        res_df = PreprocessorV1_1.preprocess_raw(self, df, return_df=True)
        self.save_dataset(res_df)


class Preprocessor:
    preprocessor_versions = {
        'v1': PreprocessorV1,
        'v1.0': PreprocessorV1_0,
        'v1.1': PreprocessorV1_1,
        'v1.2': PreprocessorV1_2,
        'v1.3': PreprocessorV1_3
    }
    MNC_path = PreprocessorBase.path_root/'dataset'/'MNC_list.csv'

    @classmethod
    def get_preprocessor_class(cls, version):
        return cls.preprocessor_versions.get(version)

    @classmethod
    def process_MNC_list(cls):
        raw_path = PreprocessorBase.path_root/'raw_data/MNC_list.xlsx'
        df = pd.read_excel(raw_path, names=['cn', 'en', 'abb', 'alias'])
        df.to_csv(cls.MNC_path, index_label='index_')

    @classmethod
    def get_MNC_list(cls, verbose=False):
        df = pd.read_csv(cls.MNC_path, index_col='index_', lineterminator='\n')
        if verbose:
            logger.info('{} MNC list loaded from "{}"', df.shape[0], cls.MNC_path)
        return df




if __name__ == '__main__':
    preprocessor = PreprocessorV1_3()
    # df=preprocessor.get_dataset('train')
    # print(df.head())
    preprocessor.describe_dataset('eval')