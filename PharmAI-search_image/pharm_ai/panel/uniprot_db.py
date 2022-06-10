# -*- coding: utf-8 -*-
'''
@author: zyl
@file: uniprot_db.py
@time: 2021/9/6 9:47
@desc: deal with uniprot-dataset
'''
import gc
import gzip
import hashlib
import time
from io import StringIO, BytesIO

import pandas as pd
from lxml import etree
from tqdm import tqdm


class UniprotDB:
    def __init__(self):
        pass

    def run(self):
        self.run_0916()

    @staticmethod
    def process_entry(elem, name_space='{http://uniprot.org/uniprot}'):
        all_name = []
        all_accession = elem.xpath('./*[local-name()="accession"]')
        if all_accession:
            accession = all_accession[0].text
            esid = hashlib.md5(accession.encode('utf-8')).hexdigest()
            if len(esid) < 32:
                esid = '0' * (32 - len(esid)) + esid

            accession_secondary = []
            if len(all_accession) > 1:
                for a in all_accession[1:]:
                    accession_secondary.append(a.text)
        else:
            return None

        # base info
        dataset = elem.get('dataset')
        create_time_array = time.strptime(elem.get('created'), "%Y-%m-%d")
        create_time_stamp = int(round(time.mktime(create_time_array) * 1000))
        dataset_create_time = create_time_stamp
        modify_time_array = time.strptime(elem.get('modified'), "%Y-%m-%d")
        modify_time_stamp = int(round(time.mktime(modify_time_array) * 1000))
        dataset_modify_time = modify_time_stamp

        # protein
        protein = elem.xpath('./*[local-name()="protein"][1]')
        if protein:
            # protein_recommend --protein_name_recom,protein_name_recom_short
            protein_recommend = protein[0].xpath('./*[local-name()="recommendedName"][1]')
            if protein_recommend:
                protein_recommend_full = protein_recommend[0].xpath('./*[local-name()="fullName"][1]')
                if protein_recommend_full:
                    protein_name_recom = protein_recommend_full[0].text
                    all_name.append(protein_name_recom)
                else:
                    protein_name_recom = ''

                protein_name_recom_short = protein_recommend[0].xpath('./*[local-name()="shortName"]/text()')
                if protein_name_recom_short:
                    all_name.extend(protein_name_recom_short)
            else:
                protein_name_recom = ''
                protein_name_recom_short = []

            # alternativeName - protein_name_alter
            protein_name_alter = protein[0].xpath(
                './*[local-name()="alternativeName"]/*[local-name()="fullName"]/text()')
            if protein_name_alter:
                all_name.extend(protein_name_alter)

            # alternativeName - protein_name_alter_short
            protein_name_alter_short = protein[0].xpath(
                './*[local-name()="alternativeName"]/*[local-name()="shortName"]/text()')
            if protein_name_alter_short:
                all_name.extend(protein_name_alter_short)

            # cdAntigenName
            protein_name_alter_cd = protein[0].xpath(
                './*[local-name()="cdAntigenName"]/text()')
            if protein_name_alter_cd:
                all_name.extend(protein_name_alter_cd)

            # submittedName-fullName
            protein_name_submit = protein[0].xpath(
                './*[local-name()="submittedName"]/*[local-name()="fullName"]/text()')
            if protein_name_submit:
                all_name.extend(protein_name_submit)
        else:
            protein_name_recom = ''
            protein_name_recom_short = []
            protein_name_alter = []
            protein_name_alter_short = []
            protein_name_alter_cd = []
            protein_name_submit = []

        # gene
        all_gene = elem.xpath('.//*[local-name()="gene"]//*[local-name()="name"]')
        gene = []
        if all_gene:
            for g in all_gene:
                gene.append({'type': g.get('type'), 'name': g.text})
                all_name.append(g.text)

        # organism
        all_organism = elem.xpath('.//*[local-name()="organism"]//*[local-name()="name"]')
        organism = []
        if all_organism:
            for o in all_organism:
                organism.append({'type': o.get('type'), 'name': o.text})

        # evidence_level
        evidence_level = elem.xpath('.//*[local-name()="proteinExistence"][1]')
        if evidence_level:
            evidence_level = evidence_level[0].get('type')
        else:
            evidence_level = ''

        all_name = set(all_name)
        if None in all_name:
            all_name.remove(None)
        all_name = list(all_name)

        return {'esid': esid, 'accession': accession, 'accession_secondary': accession_secondary,
                'dataset': dataset, 'dataset_create_time': dataset_create_time,
                'dataset_modify_time': dataset_modify_time, 'organism': organism, 'gene': gene,
                'protein_name_recom': protein_name_recom,
                'protein_name_recom_short': protein_name_recom_short,
                'protein_name_alter': protein_name_alter,
                'protein_name_alter_short': protein_name_alter_short,
                'protein_name_alter_cd': protein_name_alter_cd,
                'all_name': all_name,
                'protein_name_submit': protein_name_submit, 'evidence_level': evidence_level}

    @staticmethod
    def parse_big_xml_gz(file, process_element_func, tag='{http://uniprot.org/uniprot}entry', save_count=2500000,
                         save_dir='/large_files/5T/uniprot/trembl_db3/dt', continue_element=0,
                         *args, **kwargs):
        """
        Deal with a big .xml.gz file (>100G) which can't be loaded into the memory. The amount of memory occupied depends on the
        parameter--save-count.
        Args:
            file: a big .xml.gz file, str
            process_element_func: the function for processing the element in each dt
            tag: xml tag,usually： namespace+element_node
            save_count: how many pieces of data are stored in a file
            save_dir: save path
            *args: the process_element_func's parameters
            **kwargs:the process_element_func's parameters

        Returns:
            store the processed data in multiple files
        """
        data = gzip.GzipFile(file)
        # etree.iterparse(data, events=('start', 'end')) ---not recommended
        data = etree.iterparse(data, events=('end',), encoding='UTF-8', tag=tag)  # iterator
        entries_list = []
        count = 0
        c = 0
        for event, elem in tqdm(data):
            try:
                if c < continue_element:
                    c += 1
                    # del nodes ---must have
                    elem.clear()
                    for ancestor in elem.xpath('ancestor-or-self::*'):
                        while ancestor.getprevious() is not None:
                            del ancestor.getparent()[0]
                    continue
                else:
                    res = process_element_func(elem, *args, **kwargs)  # your function to process element
                    if res:
                        entries_list.append(res)

                    if len(entries_list) > save_count:
                        count += 1
                        df = pd.DataFrame(entries_list)  # type:pd.DataFrame
                        entries_list.clear()
                        df.to_csv(f'{save_dir}_{count}_{time.strftime("%Y%m%d", time.localtime())}.csv.gz',
                                  compression='gzip')
                        print(f'done and save:{count}')
                        df.drop(index=df.index, inplace=True)
                        del df
                        gc.collect()

            except Exception:
                all_accession = elem.xpath('./*[local-name()="accession"]')
                if all_accession:
                    print(f'error: {all_accession[0]}')

            # del nodes ---must have
            elem.clear()
            for ancestor in elem.xpath('ancestor-or-self::*'):
                while ancestor.getprevious() is not None:
                    del ancestor.getparent()[0]

        count += 1
        df = pd.DataFrame(entries_list)  # type:pd.DataFrame
        entries_list.clear()
        df.to_csv(f'{save_dir}_{count}_{time.strftime("%Y%m%d", time.localtime())}.csv.gz',
                  compression='gzip')
        print(f'done and save:{count}')
        df.drop(index=df.index, inplace=True)
        del df
        gc.collect()
        del data

    def run_0916(self):
        # f = "/large_files/bio_seq/db/uniprot_trembl.xml.gz"
        f = "/large_files/bio_seq/db/uniprot/uniprot_sprot.xml.gz"

        UniprotDB.parse_big_xml_gz(file=f, process_element_func=UniprotDB.process_entry,
                                   tag='{http://uniprot.org/uniprot}entry',
                                   save_count=2500000,
                                   save_dir='/large_files/5T/uniprot/sprot/sprot',
                                   name_space='{http://uniprot.org/uniprot}',
                                   continue_element=0)

        # parse xml file

    def method_1(self):
        f = "xxx.xml"
        etree.parse(f)
        # parse xml text

    def method_2(self):
        # parse bytes text
        some_file_or_file_like_bject = b"<root>data</root>"
        tree = etree.parse(BytesIO(some_file_or_file_like_bject))

        # parse string text
        xml = '<a xmlns="test"><b xmlns="test"/></a>'
        tree = etree.parse(StringIO(xml))

        # parse some text from the middle of the all text--v1

    def method_3(self):
        file = "xxx.xml.gz"
        data = gzip.GzipFile(file)
        middle_place = 500000
        data.seek(middle_place, 0)
        s = data.read(500000)
        s = s.split(b'<entry')
        s = b'<Document>\n' + b'<entry' + b'<entry'.join(s[1:-1]) + b'</Document>'
        tree = etree.parse(BytesIO(s))

        # parse some text from the middle of the all text--v2

    def method_4(self):
        file = "xxx.xml.gz"
        g = gzip.GzipFile(file)
        from itertools import islice
        all_data = etree.iterparse(g, events=('start', 'end'))
        start_line = 1000
        end_line = 40000
        all_data = islice(all_data, start_line, end_line)  # middle iterator
        next(all_data)


if __name__ == '__main__':
    UniprotDB().run()
