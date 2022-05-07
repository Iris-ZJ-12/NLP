import uuid
import time
import datetime

from typing import Dict, Optional


class TemplateGenerator:

    @staticmethod
    def internal_error(error_key=''):
        res = {'api_status': 'internal_error', 'error_message': error_key}
        return res

    @staticmethod
    def update_error(res, error_msg):
        res['api_status'] = 'internal_error'
        if 'error_message' not in res:
            res['error_message'] = [error_msg]
        else:
            res['error_message'].append(error_msg)

    @staticmethod
    def pdf_error():
        res = {'api_status': 'pdf_error', 'error_message': 'unable to open pdf'}
        return res

    @staticmethod
    def no_relationship():
        res = {'api_status': 'no_relationship'}
        return res

    @staticmethod
    def exist_relationship():
        res = {'api_status': 'normal', 'relationship_info': []}
        return res

    @staticmethod
    def relationship_info_dict():
        standard_dict = {
            'sub_id': '',
            'piece': '',
            'inid_code': '',
            'keyword': '',
            'type': '',
            'application_num': '',
            'application_docdb_comb': '',
            'publication_num': '',
            'publication_docdb_comb': '',
            'date_1_str': '',
            'date_1': None,
            'date_2_str': '',
            'date_2': None,
        }
        return standard_dict

    @staticmethod
    def td_info_dict():
        d = TemplateGenerator.relationship_info_dict()
        d['inid_code'] = '*'
        d['keyword'] = 'terminal disclaimer'
        d['type'] = 'TD'
        return d


class BaseProcessor:

    def error_type_check(self, pipe_res: Dict) -> Dict:
        if pipe_res['error'] == 'error when converting to image':
            return TemplateGenerator.pdf_error()
        else:
            return TemplateGenerator.internal_error(pipe_res['error'])

    def date_to_timestamp(self, date: Optional[str]) -> Optional[int]:
        return int(time.mktime(datetime.datetime.strptime(date, "%Y-%m-%d").timetuple()) * 1000) if date else None

    def application_combo_pre(self):
        raise NotImplementedError

    def add_sub_id_and_merge_error(self, res: Dict) -> None:
        # add sub id for res['relationship_info'] in_place
        for _, d in enumerate(res['relationship_info']):
            d['sub_id'] = str(uuid.uuid4())
        if 'error_message' in res:
            res['error_message'] = ' | '.join(res['error_message'])

    def make_combo(self, nation: str, application_num: str) -> str:
        middle_num = self.application_combo_pre(application_num)
        suffix = 'W' if 'pct' in application_num else 'A'
        combo = nation + middle_num + suffix
        return combo

    def check_and_add_date(
            self,
            res: Dict,
            info_dict: Dict,
            date: str,
            date_key: str,
            additional_key: str = '',
            piece: str = ''
    ) -> None:
        if self.date_validation(date):
            info_dict[date_key] = date
            info_dict[date_key[:-4]] = self.date_to_timestamp(date)
        else:
            piece = date if not piece else piece
            error_msg = f'incorrect date format {additional_key}: {piece}'
            TemplateGenerator.update_error(res, error_msg)

    def date_validation(self, date_str: str) -> bool:
        year, month, day = map(int, date_str.split('-'))
        current_year = datetime.datetime.now().year
        return 1900 < year <= current_year and 1 <= month <= 12 and 1 <= day <= 31


class CNProcessor(BaseProcessor):

    def application_combo_pre(self, application_num: str) -> str:
        return application_num[:-1]

    def standardize_result(self, pipe_res):
        if 'error' in pipe_res:
            return self.error_type_check(pipe_res)

        if pipe_res['inid22_date'] == 'inid22 keyword not found' \
                and 'patent_family' not in pipe_res \
                and not pipe_res['inid45_date'].startswith('inid'):
            return TemplateGenerator.no_relationship()

        res = TemplateGenerator.exist_relationship()
        if pipe_res['inid22_date'] != 'inid22 keyword not found':
            d = TemplateGenerator.relationship_info_dict()
            d['inid_code'] = '22'
            d['keyword'] = '申请日'
            d['type'] = '实际申请日'
            self.check_and_add_date(res, d, pipe_res['inid22_date'], 'date_1_str', 'inid 22 date')
            res['relationship_info'].append(d)

        if pipe_res['inid45_date'] == 'inid 45 date not found':
            return TemplateGenerator.internal_error("incorrect inid 45 date format")
        elif not pipe_res['inid45_date'].startswith('inid code'):
            d = TemplateGenerator.relationship_info_dict()
            d['inid_code'] = '45'
            d['keyword'] = '授权日'
            d['type'] = '授权日'
            self.check_and_add_date(res, d, pipe_res['inid45_date'], 'date_1_str', 'inid 45 date')
            res['relationship_info'].append(d)

        if 'patent_family' in pipe_res:
            d = TemplateGenerator.relationship_info_dict()
            d['piece'] = pipe_res['patent_family']['paragraph']
            if pipe_res['patent_family']['inid_code']:
                d['inid_code'] = pipe_res['patent_family']['inid_code']
            d['keyword'] = '分案'
            d['type'] = '分案'
            d['application_num'] = pipe_res['patent_family']['application_num']
            d['application_docdb_comb'] = self.make_combo('CN', d['application_num'])
            if pipe_res['patent_family']['date']:
                self.check_and_add_date(res, d, pipe_res['patent_family']['date'], 'date_1_str', 'patent family date')
            res['relationship_info'].append(d)

        self.add_sub_id_and_merge_error(res)
        return res


class EPProcessor(BaseProcessor):

    def application_combo_pre(self, application_num: str) -> str:
        return application_num[:-1]

    def standardize_result(self, pipe_res):
        if 'error' in pipe_res:
            self.error_type_check(pipe_res)

        if len(pipe_res['patent_family']) == 1 and pipe_res['patent_family'][0]['paragraph'] == '.':
            pipe_res.pop('patent_family')

        if pipe_res['inid22_date'] == 'inid_code_22_not_found':
            return TemplateGenerator.no_relationship()

        res = TemplateGenerator.exist_relationship()
        if not pipe_res['inid22_date']:
            TemplateGenerator.update_error(res, 'inid 22 incorrect date format')
        else:
            d = TemplateGenerator.relationship_info_dict()
            d['inid_code'] = '22'
            d['keyword'] = 'date of filing'
            d['type'] = '实际申请日'
            self.check_and_add_date(res, d, pipe_res['inid22_date'], 'date_1_str', 'inid 22 date')
            res['relationship_info'].append(d)

        if pipe_res['inid45_date'] != 'inid_code_45_not_found':
            if not pipe_res['inid45_date']:
                TemplateGenerator.update_error(res, 'inid 45 incorrect date format')
            else:
                d = TemplateGenerator.relationship_info_dict()
                d['inid_code'] = '45'
                d['keyword'] = '45'
                d['type'] = '授权日'
                self.check_and_add_date(res, d, pipe_res['inid45_date'], 'date_1_str', 'inid 45 date')
                res['relationship_info'].append(d)

        if 'patent_family' in pipe_res:
            for piece_res in pipe_res['patent_family']:

                # validality check
                if len(piece_res['application_number']) == 1:
                    if len(piece_res['publication_number']) == 0:
                        piece_res['publication_number'] = ['']
                    elif len(piece_res['publication_number']) > 1:
                        TemplateGenerator.update_error(res, 'incorrect amount of application or publication number')
                        break
                elif len(piece_res['application_number']) > 1:
                    if len(piece_res['application_number']) != len(piece_res['publication_number']):
                        TemplateGenerator.update_error(res, 'incorrect amount of application or publication number')
                        break
                else:
                    TemplateGenerator.update_error(res, 'incorrect amount of application or publication number')
                    break

                for i in range(len(piece_res['application_number'])):
                    d = TemplateGenerator.relationship_info_dict()
                    d['piece'] = piece_res['paragraph']
                    d['inid_code'] = piece_res['inid_code']
                    d['keyword'] = piece_res['keyword']
                    d['type'] = '分案'
                    d['application_num'] = piece_res['application_number'][i]
                    d['application_docdb_comb'] = self.make_combo('EP', d['application_num'])
                    d['publication_num'] = piece_res['publication_number'][i]
                    d['publication_docdb_comb'] = d['publication_num']
                    if piece_res['date']:
                        self.check_and_add_date(res, d, piece_res['date'], 'date_1_str', 'patent family date')
                    res['relationship_info'].append(d)

        self.add_sub_id_and_merge_error(res)
        return res


class JPProcessor(BaseProcessor):

    def application_combo_pre(self, application_num: str) -> str:
        p1, p2 = application_num.split('-')
        if len(p1) == 4:  # normal case
            if len(p2) < 6:
                p2 = '0' * (6 - len(p2)) + p2
            revised_num = p1 + p2
        else:  # early phase
            dyna, dyna_year = p1[0], p1[1:]
            if dyna == '昭':
                ac = 1925 + int(dyna_year)
            elif dyna == '平':
                ac = 1988 + int(dyna_year)
            else:
                raise ValueError(f'Unknown Japan year: {dyna}')
            revised_num = p2 + str(ac)[-2:]
        return revised_num

    def standardize_result(self, pipe_res):
        if 'error' in pipe_res:
            self.error_type_check(pipe_res)

        if not pipe_res['inid22_date']['date'] and not pipe_res['patent_family']['application_num']:
            return TemplateGenerator.no_relationship()

        res = TemplateGenerator.exist_relationship()
        if pipe_res['inid22_date']['date']:
            if pipe_res['inid22_date']['date'] == 'something wrong':
                TemplateGenerator.update_error(res, 'inid 22 date incorrect format')
            else:
                d = TemplateGenerator.relationship_info_dict()
                d['piece'] = pipe_res['inid22_date']['paragraph']
                d['inid_code'] = '22'
                d['keyword'] = '出願日'
                d['type'] = '实际申请日'
                self.check_and_add_date(res, d, pipe_res['inid22_date']['date'], 'date_1_str', 'inid 22 date')
                res['relationship_info'].append(d)

        if pipe_res['inid45_date']['date']:
            if pipe_res['inid45_date']['date'] == 'something wrong':
                TemplateGenerator.update_error(res, 'inid 45 date incorrect format')
            else:
                d = TemplateGenerator.relationship_info_dict()
                d['piece'] = pipe_res['inid45_date']['paragraph']
                d['inid_code'] = '45'
                d['keyword'] = '発行日'
                d['type'] = '授权日'
                self.check_and_add_date(res, d, pipe_res['inid45_date']['date'], 'date_1_str', 'inid 45 date')
                res['relationship_info'].append(d)

        if pipe_res['patent_family']['application_num']:
            if pipe_res['patent_family']['application_num'] == 'something wrong':
                TemplateGenerator.update_error(res, 'application_num')
            else:
                d = TemplateGenerator.relationship_info_dict()
                d['piece'] = pipe_res['patent_family']['paragraph']
                d['inid_code'] = pipe_res['patent_family']['inid_code']
                d['keyword'] = '分割'
                d['type'] = '分案'
                d['application_num'] = pipe_res['patent_family']['application_num']
                d['application_docdb_comb'] = self.make_combo('JP', d['application_num'])
                if pipe_res['patent_family']['date']:
                    self.check_and_add_date(res, d, pipe_res['patent_family']['date'], 'date_1_str',
                                            'patent family date')
                res['relationship_info'].append(d)

        self.add_sub_id_and_merge_error(res)
        return res


class USProcessor(BaseProcessor):

    def standardize_result(self, pipe_res):
        if 'error' in pipe_res:
            self.error_type_check(pipe_res)

        res = TemplateGenerator.exist_relationship()

        if pipe_res['reissue']['paragraph'] == '-1,-1' and \
                pipe_res['patent_family'][0]['paragraph'] == '-1' and \
                pipe_res['pta_td']['paragraph'] == '-1,-1':
            if pipe_res['inid22_date']['date'] == 'inid_code_22_not_found' and \
                    not pipe_res['patent_date']['date']:
                return TemplateGenerator.no_relationship()
        else:
            if pipe_res['inid22_date']['date'] == 'inid_code_22_not_found':
                TemplateGenerator.update_error(res, 'inid code 22 date not found')

        if pipe_res['inid22_date']['date'] != 'inid_code_22_not_found':
            d = TemplateGenerator.relationship_info_dict()
            d['piece'] = pipe_res['inid22_date']['piece']
            d['inid_code'] = '22'
            d['keyword'] = 'filed'
            d['type'] = '实际申请日'
            self.check_and_add_date(res, d, pipe_res['inid22_date']['date'], 'date_1_str', 'inid 22 date')
            res['relationship_info'].append(d)

        if pipe_res['patent_date']['inid_code'] == '45':
            if not pipe_res['patent_date']['date']:
                TemplateGenerator.update_error(res, 'inid 45 date not found')
            else:
                d = TemplateGenerator.relationship_info_dict()
                d['piece'] = pipe_res['patent_date']['piece']
                d['inid_code'] = '45'
                d['keyword'] = pipe_res['patent_date']['key_word']
                d['type'] = '授权日' if 'reissue' not in d['keyword'] else '再版授权日'
                self.check_and_add_date(res, d, pipe_res['patent_date']['date'], 'date_1_str', 'inid 45 date')
                res['relationship_info'].append(d)

        if pipe_res['reissue']['paragraph'] != '-1,-1':
            if 'error' in pipe_res['reissue']:
                TemplateGenerator.update_error(res, 'something wrong with reissue extraction')
            else:
                d = TemplateGenerator.relationship_info_dict()
                d['piece'] = pipe_res['reissue']['paragraph']
                d['inid_code'] = '64'
                d['keyword'] = 'reissue'
                d['type'] = '再版日期'
                d['application_num'] = pipe_res['reissue']['application_number']
                d['publication_num'] = pipe_res['reissue']['patent_number']
                d['publication_docdb_comb'] = 'US' + d['publication_num']
                self.check_and_add_date(res, d, pipe_res['reissue']['filed_date'], 'date_1_str', 'reissue date_1')
                self.check_and_add_date(res, d, pipe_res['reissue']['issue_date'], 'date_2_str', 'reissue date_2')
                res['relationship_info'].append(d)

        if pipe_res['pta_td']['paragraph'] != '-1,-1':
            if 'date' in pipe_res['pta_td']:
                d = TemplateGenerator.td_info_dict()
                d['piece'] = pipe_res['pta_td']['paragraph']
                self.check_and_add_date(res, d, pipe_res['pta_td']['date'], 'date_1_str', 'pta date')
                res['relationship_info'].append(d)
            else:
                if 'terminal_disclaimer' in pipe_res['pta_td']:
                    d = TemplateGenerator.td_info_dict()
                    d['piece'] = pipe_res['pta_td']['paragraph']
                    if 'publication_number' in pipe_res['pta_td']:
                        d['publication_num'] = pipe_res['pta_td']['publication_number']
                    res['relationship_info'].append(d)
                if '154' in pipe_res['pta_td']:
                    d = TemplateGenerator.relationship_info_dict()
                    d['piece'] = pipe_res['pta_td']['paragraph']
                    d['inid_code'] = '*'
                    d['keyword'] = 'notice/154(b)/days'
                    d['type'] = 'PTA'
                    if 'bydays' in pipe_res['pta_td']:
                        d['date_1_str'] = pipe_res['pta_td']['bydays']
                        res['relationship_info'].append(d)

        if pipe_res['patent_family'][0]['paragraph'] != '-1':
            # track used application num to remove duplicate
            apps = set()

            for piece_res in pipe_res['patent_family']:
                for tup in piece_res['tuples']:
                    d = TemplateGenerator.relationship_info_dict()
                    d['piece'] = tup['piece']
                    d['inid_code'] = piece_res['inid_code']

                    d['keyword'] = tup['keyword']
                    if d['keyword'] == 'continuation':
                        d['type'] = '继续申请'
                    elif d['keyword'] == 'continuation-in-part':
                        d['type'] = '部分继续申请'
                    elif d['keyword'] == 'division':
                        d['type'] = '分案'
                    else:
                        continue

                    if not tup['application_number']:
                        TemplateGenerator.update_error(res, f"patent family no app num: {d['piece']}")
                        continue

                    d['application_num'] = tup['application_number']
                    if '|'.join(d['application_num']) in apps:
                        continue
                    apps.add('|'.join(d['application_num']))

                    if tup['publication_number']:
                        d['publication_num'] = tup['publication_number']
                        d['publication_docdb_comb'] = ['US' + n for n in d['publication_num']]

                    if len(tup['date']) == 0:
                        TemplateGenerator.update_error(res, f"patent family no date: {d['piece']}")
                        continue
                    tup['date'] = sorted(tup['date'], key=lambda x: list(map(int, x.split('-'))))
                    self.check_and_add_date(res, d, tup['date'][0], 'date_1_str', 'patent family date_1', d['piece'])
                    if len(tup['date']) > 1:
                        self.check_and_add_date(res, d, tup['date'][1], 'date_2_str', 'patent family date_2',
                                                d['piece'])
                    res['relationship_info'].append(d)

        self.add_sub_id_and_merge_error(res)
        return res
