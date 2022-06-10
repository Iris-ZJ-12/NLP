import json
import os
from datetime import datetime

import pandas as pd
from loguru import logger
from pandas.api.types import CategoricalDtype
from typing import Union

from pharm_ai.prophet.utils import get_es, update_bulk, scroll_search

logger.add('folding.log', level='DEBUG', filter=lambda record: record["extra"].get("task") == "folding")
folding_logger = logger.bind(task="folding")


class NewsFolding:
    company_name_file = os.path.join(os.path.dirname(__file__), "company_names.json")
    es_index = "invest_news"
    fields = ['esid', 'publish_date', 'resource', 'company', 'round', 'amount',
              'is_spider', 'similar_esid', 'is_publish', 'is_new', 'label', 'label_prediction']

    def __init__(self, es_host=None, convert_company_id=False):
        folding_logger.debug('initializing NewsFolding...')
        self.today = datetime.today().strftime("%Y-%m-%d")
        self.es_host = es_host
        self.es = get_es(es_host or 'test')
        folding_logger.debug('es host: {}', self.es_host)
        # definate priority of news source, NaN means other categories
        self.source_type = CategoricalDtype(
            categories=['医药魔方', '商业资讯', '美通社', '全球通', 'Biospace', 'FinSMEs', 'Seekingalpha', 'Xconomy', '36氪', '投资界',
                        '动脉网', '智通财经', '亿欧', '猎云网'], ordered=True)
        self.convert_company_id = convert_company_id
        folding_logger.debug('Convert company id: {}', convert_company_id)
        folding_logger.debug('NewsFolding initialized.')

    def predict(self, article_id, date, selected_investee: str, selected_round: str,
                selected_amount: str, news_filter_label='医药', return_changes=False, time_logger=None):
        """
        :param str article_id: Article ID.
        :param str date: Publish date.
        :param str selected_investee: Selected inverstee.
        :param str selected_amount: Selected amount.
        :param str selected_round: Selected round.
        :param str news_filter_label: "医药"(default), "非医药", "非相关".
        :return: Result whether the news is head, and the folding news ID, and changes of folding_head.
        :rtype: tuple[bool, str]
        Predict if an article is folding head, by precise match of investee, round and amount,
        and according to priority of news sources.
        """
        if time_logger:
            time_logger.log_event_start_processing_times()
        folding_logger.debug('article_id={}, date={}, investee={}, round={}, amount={}, news_filter_label={}',
                             article_id, date, selected_investee, selected_round, selected_amount, news_filter_label)
        if news_filter_label != '非相关':
            historical_data = self.get_historical_news(date, investee=selected_investee, round=selected_round,
                                                       amount=selected_amount, exclude_is_new="-1",
                                                       exclude_label_prediction="非相关")
            folding_logger.debug('Got {} historical_data:\n{}', historical_data.shape[0], historical_data.to_dict('records'))
            if article_id in historical_data['esid'].tolist():
                # add query filed into historical data
                historical_data.loc[
                    historical_data['esid'] == article_id,
                    ['company', 'round', 'amount', 'similar_esid']
                ] = [selected_investee, selected_round, selected_amount, article_id]
                folding_logger.debug('Set value of {} to historical data.', article_id)
            if not historical_data.empty:
                head_news = historical_data.sort_values(['is_publish', 'is_new', 'resource', 'publish_date'],
                                                        ascending=[False, True, True, True])
                returned_result = (head_news['esid'].values[0] == article_id, head_news['esid'].values[0]) \
                    if not head_news[head_news['esid'].ne(article_id)].empty else (True, article_id)
                folding_logger.debug('Folding result: {}', returned_result)
            else:
                returned_result = True, article_id
                folding_logger.debug('Historical data is empty, returned self.')
        else:
            returned_result = True, article_id
            folding_logger.debug('Irrelevant article, returned self.')
        if returned_result[0] and news_filter_label != '非相关':
            changes = self.change_to_non_head(article_id, date, selected_investee, selected_round, selected_amount)
        else:
            changes = dict()
        if time_logger:
            time_logger.log_event_processing_finished_timestamp()
        if return_changes:
            return *returned_result, changes
        else:
            return returned_result

    def get_historical_news(self, date: Union[datetime, str], to_next_week: bool = False, investee: str = None,
                            round: str = None, amount: str = None, is_new: str = None, exclude_is_new: str = None,
                            exclude_label_prediction: str = None, exclude_esid: str = None, filter_head=False,
                            exclude_is_publish: str = None):
        if isinstance(date, datetime):
            date_ms = int(date.timestamp()) * 1000
        else:
            date_ms = int(datetime.strptime(date, "%Y-%m-%d").timestamp()) * 1000

        end_date_ms = date_ms + 604800000 if to_next_week else date_ms + 86400000  # next 1w / 2d
        body = {
            "query": {
                "bool": {
                    "filter": {
                        "range": {
                            "publish_date": {"gte": date_ms - 604800000, "lte": end_date_ms}
                        }
                    }
                }
            }
        }
        must_clauses, must_not_clauses = [], []
        investee_id_name = (investee,)
        if investee:
            if not self.convert_company_id:
                must_clauses.append({"term": {"company": investee}})
            else:
                if investee.isdigit():
                    investee_id_name = self.get_company_id_name(id_=investee)
                else:
                    investee_id_name = self.get_company_id_name(name=investee)
                must_clauses.append({"terms": {"company": investee_id_name}})
        if round:
            must_clauses.append({"term": {"round": round}})
        if amount:
            must_clauses.append({"term": {"amount": amount}})
        if is_new:
            must_clauses.append({"term": {"is_new": is_new}})
        if exclude_is_new:
            must_not_clauses.append({"term": {"is_new": exclude_is_new}})
        if exclude_label_prediction:
            must_not_clauses.append({"term": {"label_prediction": exclude_label_prediction}})
        if exclude_esid:
            must_not_clauses.append({"term": {"_id": exclude_esid}})
        if filter_head:
            must_clauses.append(
                {"script": {"script": "doc['_id'] == doc['similar_esid'] || doc['similar_esid'].size() == 0"}}
            )
        if exclude_is_publish:
            must_not_clauses.append({"term": {"is_publish": exclude_is_publish}})
        if must_clauses:
            body['query']['bool']['must'] = must_clauses
        if must_not_clauses:
            body['query']['bool']['must_not'] = must_not_clauses
        raw = self.es.search(body=body, index=self.es_index, size=100, _source_includes=self.fields)
        ls = [{'esid': r['_id'], **r['_source']} for r in raw['hits']['hits']]
        res = pd.DataFrame(ls)
        res = res.mask(res.applymap(lambda x: x == ''))
        if not res.empty:
            for c in self.fields:
                if c not in res.columns:
                    res[c] = None
            if 'publish_date' not in res.columns:
                res['publish_date'] = None
            else:
                res['publish_date'] = pd.to_datetime(res['publish_date'], unit='ms')
            res['resource'] = res['resource'].astype(self.source_type)
            res['is_publish'] = res['is_publish'].fillna(0).astype(int)
            res['is_new'] = res['is_new'].fillna(1).astype(int)
            if self.convert_company_id and len(investee_id_name) == 2:
                res['company'] = res['company'].replace(investee_id_name[0], investee_id_name[1])
            return res
        else:
            return pd.DataFrame(columns=self.fields)

    def get_company_id_name(self, *, id_: str = None, name: str = None):
        """Given either company ID or company short name, return both.
        :param str id_: company_id
        :param str name: company short name
        :return (id, name) or only (id,) or only (name,).
        """
        if id_:
            origin = (id_,)
            body = {"query": {"term": {"id": id_}}}
            folding_logger.debug('Get company name of {}...', id_)
        elif name:
            origin = (name,)
            body = {"query": {"term": {"short_name": name}}}
            folding_logger.debug('Get company id of {}...', name)
        else:
            raise ValueError('Any of _id and name should be input.')
        raw = self.es.search(body=body, index='base_company', _source_includes=['id', 'short_name'])
        hits = raw['hits']['hits']
        if hits:
            source = hits[0]['_source']
            result = source['id'], source['short_name']
            folding_logger.debug('Fetched company id and name: {}', result)
            return result
        else:
            folding_logger.debug('Failed to fetch company id and name, return origin: {}.', origin)
            return origin



    def change_to_non_head(self, article_id, date, investee, round, amount):
        """If there are head articles match current head article, set those as non-head.
        """
        folding_logger.debug('Changing other articles belonging to "{}".', article_id)
        selected_data = self.get_historical_news(date, to_next_week=True, investee=investee, round=round, amount=amount,
                                                 is_new="1", exclude_esid=article_id, filter_head=True,
                                                 exclude_is_publish="1", exclude_label_prediction='非相关')
        folding_logger.debug('Got {} historical_data:\n{}', selected_data.shape[0], selected_data.to_dict('records'))
        change_result = dict()
        if not selected_data.empty:
            to_update = [{'esid': id_, 'similar_esid': article_id}
                         for id_ in selected_data['esid']]
            response = update_bulk(self.es, to_update)
            if not response['errors']:
                change_result = {id_: article_id for id_ in selected_data['esid']}
                folding_logger.debug('Folding head of {} changed to {}', selected_data['esid'].tolist(), article_id)
            else:
                folding_logger.error('Folding head of {} changed to {} error.', selected_data['esid'].tolist(), article_id)
        else:
            folding_logger.debug("No historical data fetched, not change any head.")
        return change_result

    @staticmethod
    def cache_company_name(host=None):
        """Download all data linking company_id to company_name to local json file."""
        if host:
            logger.info('Current host: {}', host)
        es = get_es('online')
        body = {"query": {"match_all": {}}}
        r = scroll_search(es, 'base_company', body=body, desc="Fetching company names",
                          _source_includes=["id", "short_name"])
        res = dict((rr['id'], rr['short_name']) for rr in r if rr.get('short_name') and rr.get('id'))
        with open(NewsFolding.company_name_file, 'w') as f:
            json.dump(res, f, indent=4, ensure_ascii=False)
        logger.success("{} company names saved to '{}'.", len(res), NewsFolding.company_name_file)

    @staticmethod
    def get_cached_company_dictionary():
        try:
            with open(NewsFolding.company_name_file, 'r') as f:
                res = json.load(f)
            return res
        except:
            logger.error('"{}" not exists.', NewsFolding.company_name_file)


if __name__ == '__main__':
    test_folder = NewsFolding()
    res = test_folder.predict('7bda3b053e1fe5be6c7016b6ca25ded8', '2022-01-13',
                              '84517', '并购', '未披露')
    print(res)
