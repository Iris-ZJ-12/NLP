from pharm_ai.prophet.utils import (get_one_log, get_news_by_esids_and_save,
                                    get_recent_data_save, migrate_data_to_test_es, get_es)
from pathlib import Path
import json
import pytest

class TestGetNews:
    result_dir = Path(__file__).parent/'data'
    index = 'invest_news'
    es = get_es('test')

    prophet_data = [
        (['52b5a332ba29a507832387a1bf40b570', '33242eeca66741ea67c9ef92c17769b6'], 'data1'),
        (['c2220e35bf78e1d6669f92853d0447ed', 'cef0d1e18ac24086a8cb0dba46fc215f',
          '9a945bad2a9bdaf9bc5e6e504b0b9136', '3c953eb420adee39bdf64b7c8e544633',
          '132f4bc44b063b41823c903680fdfd34', 'c81ff1072eda75993cf3ad116752ee16',
          'a4f015a1e970cec4896b9b0add41ddb1', 'ba100c32c76be312b13280a0baebb326'], 'data6'),
        (['b880e06154966762b0a1948c613f9d70', '2a7b65a8f8c7ca025a9a3e28a269552'], 'data14'),
        (['8181a8569fe1a47f9659801b7eb50530', 'bd8f5a63e606c9fac74927cac747a451'], 'data15'),
        (['cdea92dc47b43e1d9b4d87ffda942f2f', 'a7339691f771ca4986712db89db58796',
          '43c3a0a610e5cbc73946e585507ebccd', '5a84f638cbaee56dee684ecaf8a6d0aa',
          '0cada30caac8ba559d02caccc88c28a1'], 'data18'),
        (['d89d2567cd25c7be732d1557d61e8b1c'], 'data19'),
        (['9e2d3eab807e32d7b5f96f329892ff74'], 'data21')
    ]

    old_log_data = [
        ('2021-08-18 11:20:47.001', 'data2'),
        ('2021-08-18 11:20:47.001', 'data3'),
        ('2021-08-19 00:10:43.189', 'data4'),
        ('2021-08-25 11:19:17.630', 'data7')
    ]

    folding_data = [
        ([
             '62e7ccb2f9b8a48153d6ef514a7f211d', 'b4970381ff1204a60ee9fa6fe21988b6',
             'b0ed65eadbe807c8c44292d8026dc347',
             '0f704a6f3469110526da25afccb30ced', 'b1875bed9123d61e861a38687947448e',
             '66169964f84ea812d2c5e64f21ac514f', '54a6e648f10706839d1af86b71295fd8'
         ], 'data9'),
        ([
             'bac53e7b3f7b2e810ff639d0496d8b0f', 'b5d744eae1f67e04c4a4ceb1474232c6',
             'eff0eca2dff2ec9bfd7c421fc3a07fa6',
             '6b07f63ca5983e826d5833f364acab20', '7a861d62f307eaee03ab6f56b7868e73',
             '9b50a2d8e05574bf68898e11e2b72aa7', 'fc121c6e11c6e25e4cf72bb35716df61',
             'a4f5821acb69de8cd5d0a230082efa55', '725f017a2952ca1e7871e4ba24a19b00',
             'e82a31d8069b7381cc2b23c4beafcbf6', '13bd8d59ee573858c3e5599fb08e6227',
             '15acdd09c2472faf8708d3f7f945b063', 'c4b56b3bde439ea0d351301dd43f4ce7',
             '91368ee884f2ec72b045cf52d2ee0b5d'
         ], 'data10'),
        ([
             '8c712e03a7476fded37c4f440427112a', '91d9fc074dcb4be0f46cf7c00a7d98aa',
             '03799b511da5ec054bbaaaa9b99fed5e', '2b23ff775c125fa10e10ef2413cde669',
             '39256a9a6d4bba7a1a6182a1e1737309'
         ], 'data11'),
        ([
             '57ab44d7af20c3f0abce44b03a896679', '4596b715ffacb052a1bedcb53b355473',
             'a5b5b3566469e4355265b6e7245cff18'
         ], 'data12'),
        ([
             'c6c96fafd6fae65a4ac02d1dee58e494', 'da07667f5a5fe0fa02a1ec191850e25e'
         ], 'data13'),
        ([
             '7e4c97bb17c47d5056572becfc037c66', '1a730a55ee490633cd35a3908ea6afaa',
             '44cf89c1bee506bb1f8433d7b5e49ac0'
         ], 'data16'),
        ([
             '8952f1bc463e8cbdc7303aa6a5476448', '3629639fc069a1fa8febf9493544fb58'
         ], 'data17'),
        ([
             'e5bea05cc245b6c42ad8ec346fae36e9', 'f438426c146a743d8ce9c71d544961ac'
         ], 'data20'),
        ([
             '4c450f088995febe9f69d3740ae29925', '8f124d8ecb6fabbaf2cd887962e71866'
         ], 'data22'),
        ([
             '6365913c548af24a8f6a8a7d57e5ae22', '0297bc34ac7c6293623264748c160741'
         ], 'data23'),
        ([
             '0569bd56990f28b73fef2f3e2d7fea07', '02f13833734d673a2349c4ab55265b02',
             '410f1577029a63fe2a803a7e916da1df'
         ], 'data24'),
        ([
             '520fa37e67c7d3f879dc5f0ecceeaf39', 'ff958e960c6865efb30ed05dfe644326',
             'dc2dd4d64994ec7dd5abf1bb03d87d77'
         ], 'data25'),
        ([
             '081dd6a4b16035657b5e2ad159c6d034', '862cb4a4b679579b4499103090f8132f',
             'db18bc2924ec70a590860fe650f77eea'
         ], 'data26'),
        ([
             '6c86fa27782bd08771cc1fd912183cf6', 'dab6b5dd6e54ce4735971784d7407a22'
         ], 'data27'),
        ([
             '2b2496e1deee8b47d9f3f02fdb599e06', '9b5dd69ca845f7f71747d96c9e9494d1',
             '519243334dd8387b9eccff5b45af0cf0', 'c319075db8f50ab34c9147277f9d532f'
         ], 'data28'),
        ([
             '9dc1a48339ea9f579ee05e89cfde43bb', '35e0a692d36972c82ca405fc89c3b895'
         ], 'data29'),
        ([
             '3f8120357bff8926940d1677b5a43d4b', '937cc57a1db21db24ae5b8936a6741a5'
         ], 'data30'),
        ([
            'd622edc16209115452b7dce2acd28cac', '532654c9168a60ee5268ba724eb52651'
        ], 'data32'),
        ([
            '817d6fccacb03df753d524dedff0d04f', 'f0002b572b04bffb657e637661fda863'
        ], 'data33'),
        ([
            'fb40381346d0d548380a61c5f32851c4', '208c016846d4ab80321a315b7649530f'
        ], 'data34'),
        ([
            '6e4805a0d6248b77ac63402467c32a94', 'e580142940f42345fe4fa32938582f8c'
        ], 'data35')
    ]

    @pytest.mark.parametrize('esids, saving_file_name', prophet_data)
    def test_get_news_by_esid(self, esids, saving_file_name):
        """Get online articles by esid, tidy formats and save to json file (run at gpu176)"""
        saving_json_path = self.result_dir/(saving_file_name+'.json')
        ls = get_news_by_esids_and_save('online', esids, saving_json_path)
        assert all(k in esids for k in ls)

    @pytest.mark.skip(reason="Old logger is no longer used.")
    @pytest.mark.parametrize("start_string, saving_file_name", old_log_data)
    def test_get_data_from_log(self, start_string, saving_file_name):
        saving_json_path = self.result_dir/(saving_file_name+'.json')
        log_path = Path(".").absolute().parent.parent/'pharm_ai'/'prophet'/'result.log'
        data = get_one_log(saving_json_path, start_string, log_path)
        assert len(data)==1
        esid = list(data.keys())[0]
        assert all(k in ['title', 'paragraphs', 'publish_date', 'news_source'] for k in data[esid])
        with saving_json_path.open('w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        assert saving_json_path.exists()

    def test_get_recent_data(self):
        """Get recent batch data."""
        num_data = 300
        saving_file_name = 'data5'
        saving_json_path = self.result_dir/(saving_file_name+'.json')
        res = get_recent_data_save(num_data, saving_json_path)
        assert len(res) == num_data
        assert saving_json_path.exists()

    @pytest.mark.parametrize("esids, saving_file_name", folding_data)
    def test_migrate_data(self, esids, saving_file_name):
        """Get online raw articles (no changes) and dump to json file as intermediary."""
        json_path = self.result_dir/(saving_file_name+'.json')
        res = migrate_data_to_test_es(esids, json_path)
        if not json_path.exists():
            assert all(r['esid'] in esids for r in res)
        else:
            es = get_es('online')
            assert all(es.exists(self.index, esid) for esid in esids)