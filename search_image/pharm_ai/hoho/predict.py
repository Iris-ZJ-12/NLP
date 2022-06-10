import os
import json
import numpy as np

from elasticsearch import Elasticsearch
from pharm_ai.hoho.model import HohoModel
from simpletransformers.classification import ClassificationArgs


class ESObject:

    def __init__(
        self,
        index_name='base_hospital_all_name',
        index_type='_doc',
        # hosts="172.17.108.112",
        hosts='esnode8.cubees.com',
        es_http_auth="esjava:esjava123abc",
        port=9200
    ):
        self.index_name = index_name
        self.index_type = index_type

        self.es = Elasticsearch(
            hosts=hosts,
            http_auth=es_http_auth,
            port=port,
            use_ssl=True,
            verify_certs=False,
            ssl_show_warn=False
        )

        if self.es.indices.exists(index=self.index_name) is not True:
            print('no index:', index_name)
            raise Exception('Invalid index name.')

    def fuzzy_match(
            self,
            query_field_name,
            query_field_value,
            get_fields=None,
            query_max_size=100
    ):
        query_doc = {
            "query": {
                "match": {
                    query_field_name: query_field_value
                },
            },
            'size': query_max_size
        }
        re = self.es.search(index=self.index_name, doc_type=self.index_type, body=query_doc, request_timeout=60)
        re = re['hits']['hits']
        res = []
        for i in re:
            if get_fields:
                res.append([i['_source'][f] for f in get_fields])
            else:
                res.append(i['_source'])
        return res


class AreaRuleEngine:
    """
    Filter hospital name pairs that the location dont match.
    """

    def __init__(self):
        path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(path, "area_dict.json"), "r") as f:
            self.area = json.load(f)
        self.province = list(self.area.keys())
        self.city = [city for province in self.area.keys() for city in self.area[province].keys()]
        self.district = []
        for p in self.province:
            for c in self.area[p]:
                self.area[p][c] = list(set(self.area[p][c]))
                self.district += self.area[p][c]

        # dealing with overlapping and duplicate district names
        name_list = []
        rm_list = []
        for i in range(len(self.district)):
            for j in range(len(self.district)):
                if i == j:
                    continue
                d1 = self.district[i]
                d2 = self.district[j]
                if d1 == d2:
                    rm_list.append(d1)
                if d1 != d2 and d1 in d2 and d1 + "县" != d2:
                    name_list.append(d1)
        for name in rm_list:
            if name in self.district:
                self.district.remove(name)
        # if a name is contained in another name, move the name to last
        for name in name_list:
            self.district.append(name)
            self.district.remove(name)

        # back-pointers
        self.city2province = {}
        self.district2city = {}
        for province in self.area:
            for city in self.area[province]:
                self.city2province[city] = province
                for district in self.area[province][city]:
                    if district in self.district:
                        self.district2city[district] = city

    def get_area(self, area_list, name, key=None):
        if key and key in name:
            name = name[key]
        for area in area_list:
            if area in name:
                if area + "大学" in name:
                    return self.get_area(area_list, name.replace(area + "大学", ""))
                else:
                    return area
        return None

    def check(self, a1, a2):
        """
        Check whether two areas are same
        Args:
            a1: area1
            a2: area2
        Returns:
            True if same or dont know
            False if different
        """
        return not a1 or not a2 or a1 == a2

    def infer_area_names(self, p, c, d):
        if d:
            if not c:
                c = self.district2city[d]
            if not p:
                p = self.city2province[c]
        else:
            if c and not p:
                p = self.city2province[c]
        return p, c, d

    def filter(self, input_name, area, candidates, topk):
        p, c, d = area
        p_name = p if p else input_name
        c_name = c if c else input_name
        d_name = d if d else input_name

        p = self.get_area(self.province, p_name)
        if p:
            c = self.get_area(self.area[p], c_name)
        else:
            c = self.get_area(self.city, c_name)
        if p and c:
            d = self.get_area(self.area[p][c], d_name)
        else:
            d = self.get_area(self.district, d_name)
        p, c, d = self.infer_area_names(p, c, d)
        result = []
        for i in range(len(candidates)):
            candidate = candidates[i]
            cp = self.get_area(self.province, candidate, 'province')
            city_l = self.area[cp].keys() if cp else self.city
            cc = self.get_area(city_l, candidate, 'city')
            district_l = self.area[cp][cc] if cp and cc else self.district
            cd = self.get_area(district_l, candidate, 'district')
            cp, cc, cd = self.infer_area_names(cp, cc, cd)
            if self.check(p, cp) and self.check(c, cc) and self.check(d, cd):
                result.append(candidates[i])
                if len(result) == topk:
                    break
        return result


class HohoPredictor:

    def __init__(self):
        args = ClassificationArgs(
            silent=True,
            use_multiprocessing=False,
            use_multiprocessing_for_evaluation=False,
            onnx=True,
            quantized_model=True
        )
        self.rule = AreaRuleEngine()
        self.model = HohoModel(args).get_model(train=False, use_cuda=False)
        self.es = ESObject()

    def predict(self, inputs, area, topk=10):
        candidates = self.es.fuzzy_match('alias_name', inputs)
        candidates = self.rule.filter(inputs, area, candidates, topk)
        actual_length = len(candidates)
        if actual_length == 0:
            return []
        model_inputs = [[inputs, candidates[i]['alias_name']] for i in range(actual_length)]
        predictions, outputs = self.model.predict(model_inputs)
        exp_logits = np.exp(outputs)
        prob = exp_logits[:, 1] / np.sum(exp_logits, axis=1)
        return [[
            candidates[i]['alias_name'],
            candidates[i]['hospital_id'],
            candidates[i]['hospital_name'],
            predictions[i],
            prob[i]
        ] for i in range(actual_length)]
