#!/usr/bin/python3.5
# -*- coding:utf-8 -*-
import re
from collections import OrderedDict
from typing import Sequence
from copy import copy
from datetime import datetime
from enum import Enum
from functools import reduce
from uuid import uuid1
from math import ceil
from datetime import date as dt_date, datetime as dt_datetime
from dateutil.relativedelta import relativedelta
from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import NotFoundError

# es地址
# __PRIVATE_DICT = {"es_host": ("192.168.1.150",)}
__PRIVATE_DICT = {"es_host": ("esnode8.cubees.com",)}
es_http_auth="esjava:esjava123abc"
#获取最大值
__hits_max=2147483647
__hits_default=10000
__require_alias = True
_KEY_ENUM_ES = "ES"
_KEY_ENUM_DGRAPH = "DGRAPH"
__ALIAS_SUFFIX = "alias"
__ES_SOURCE_FIELD = "_source"
__ES_HITS_FIELD = "hits"
__ES_EXPLAIN_FIELD = "_explanation"
__ES_TOTAL_FIELD = "total"
__ES_AFTER_FIELD = "after"
__ES_SEARCH_AFTER_FIELD = "search_after"
__ES_SCROLL_ID_FIELD = "_scroll_id"
__doc_type_value = "_all"
# 过期時间
__ES_SCROLL_TIME_2M = "2m"
__ES_SCROLL_TIME_5M = "10m"
# 分组时最大条数
SIZE_GROUP = 2000
MAX_SIZE_GROUP = 10000
MAX_SIZE_GROUP2 = 200000
MAX_SIZE_CARDINALITY = 40000
# esid的原始字段名
__ES_ID_FIELD = "_id"
# 返回对象时,esid的字段名
ESID_FIELD = "esid"
DOC_FIELD = "_doc"
SCORE_FIELD = "es_score"
# 评分的字段名
ES_SCORE_FIELD = "_score"
# 查询时不返回其它字段只返回esid时设置的字段
NULL_VALUE = "None"
__FIELDS_ESID_OR_NONE = {__ES_ID_FIELD, ESID_FIELD, NULL_VALUE}
# 分页获取最大条数
SCORE_SIZE_1000 = 1000
MAX_SIZE_PAGE = 10000
__d_host = {}
re_wkz_special_chars = re.compile("[{}]".format(" \\n\\r\\t!\"#$%&*\\\\'()\\[\\]+,-./:;<=>?@^_`{|}~"))


class AggregationType(Enum):
	# 复合中的分组
	CHI_GROUP = "chi_group"
	# 复合
	COMPOSITE = "composite"
	# 分组
	TERMS = "terms"
	# 单字段分组
	SINGLE_GROUP = "single_group"
	# 多字段分组
	MULTIPLE_GROUP = "multiple_group"
	# 多字段分组(复合方式)
	MULTIPLE_GROUP_COMPOSITE = "multiple_group_composite"
	# 单字段分组聚合
	SINGLE_GROUP_AGGREGATE = "single_group_aggregate"
	# 多字段分组聚合
	MULTIPLE_GROUP_AGGREGATE = "multiple_group_aggregate"
	# 多字段分组聚合(复合方式)
	MULTIPLE_GROUP_COMPOSITE_AGGREGATE = "multiple_group_composite_aggregate"
	# 聚合
	AGGREGATE = "aggregate"
	# 子查询
	NESTED = "nested"
	# 反向子查询
	REVERSE_NESTED = "reverse_nested"
	# 过滤
	FILTER = "filter"
	# 時间分组聚合
	DATE = "date"
	DATEHISTOGRAM = "date_histogram"


class AggregateType(Enum):
	# region 通用
	# 平均值
	AVG = {_KEY_ENUM_ES: "avg", _KEY_ENUM_DGRAPH: "avg"}
	# 最大值【如果该字段不存在会返回Double.NEGATIVE_INFINITY】
	MAX = {_KEY_ENUM_ES: "max", _KEY_ENUM_DGRAPH: "max"}
	# 最小值【如果该字段不存在会返回Double.POSITIVE_INFINITY】
	MIN = {_KEY_ENUM_ES: "min", _KEY_ENUM_DGRAPH: "min"}
	# 和值
	SUM = {_KEY_ENUM_ES: "sum", _KEY_ENUM_DGRAPH: "sum"}
	# 统计数量(比COUNT_DISTINCT效率高)【如果被统计字段是数组类型或者count(*),field必须设置成"_index",才能完美统计数量】
	COUNT = {_KEY_ENUM_ES: "value_count", _KEY_ENUM_DGRAPH: "count"}
	# endregion

	# 下面es专用
	# 统计去重的数量【通过Cardinality实现】【误差】
	COUNT_DISTINCT = {_KEY_ENUM_ES: "cardinality"}
	# 底下3个禁止排序
	# 百分数
	PERCENTILES = {_KEY_ENUM_ES: "percentiles"}
	# 中值(中位数)【通过Percentiles实现】【误差】
	MEDIAN = {_KEY_ENUM_ES: "median"}
	# 当A字段最大值时,此时B字段的值(A代表MAX,B代表Other,得到的是B)【如果B值为空,则会找有值的B,且A字段是相对最大的.通过Terms和Max实现】
	MAX_OTHER = {_KEY_ENUM_ES: "max_other"}
	# 当A字段最小值时,此时B字段的值(A代表MIN,B代表Other,得到的是B)【如果B值为空,则会找有值的B,且A字段是相对最大的.通过Terms和Min实现】
	MIN_OTHER = {_KEY_ENUM_ES: "min_other"}


class RefreshPolicy(Enum):
	NONE = "false"
	IMMEDIATE = "true"
	WAIT_UNTIL = "wait_for"


class OpType(Enum):
	INDEX = "index"
	UPDATE = "update"
	DELETE = "delete"


class Operator(Enum):
	OR = "OR"
	AND = "AND"


class ZeroTermsQuery(Enum):
	NONE = "NONE"
	ALL = "ALL"


class QueryType(Enum):
	# region 通用
	# region 後面带1个参数
	# 大于
	GT = "gt"
	# 小于
	LT = "lt"
	# 大于等于
	GE = "ge"
	# 小于等于
	LE = "le"
	# endregion

	# 不相等
	NE = "not eq"
	# 等于(後面带1或n个参数)
	EQ = "eq"
	# endregion

	# region 图专用,後面带1个参数
	ALLOFTERMS = "allofterms"
	ANYOFTERMS = "anyofterms"
	ALLOFTEXT = "alloftext"
	ANYOFTEXT = "anyoftext"
	REGEXP = "regexp"
	# endregion

	# region es专用
	# region 後面带1+个参数
	# 范围查询[左右都闭合](Between)
	BN = 5
	# 类似于sql中的not in关键字[都是Or的关系=值都是精确匹配]
	NI = 6
	# 类似于sql中的in关键字[都是Or的关系=值都是精确匹配]
	IN = 7
	# 类似于sql中的in关键字[都是Or的关系=但是值是模糊匹配]
	IN_LIKE = 8
	# 通配符查询
	IN_WILDCARD = 8
	# endregion
	# 後面带(0,1)个参数:
	# 类似于sql模糊查询[不对查询字符串进行分词]
	NL = 9
	# 类似于sql模糊查询[不对查询字符串进行分词]
	LIKE = 10
	# 通配符查询
	WILDCARD = 13
	NW = 15
	# 前缀查询
	PREFIX = 14
	# endregion
	ERROR = "error"


class ScoreMode(Enum):
	NONE = "none"
	AVG = "avg"
	MAX = "max"
	TOTAL = "sum"
	MIN = "min"


class ToXContent(object):
	def to_x_content(self, *args, **kwargs):
		pass


class Order(ToXContent):
	"""
	全部使用类方法进行初始化
	"""

	def __init__(self, type_value: int, asc: bool, *, name: str = None):
		self.typeValue = type_value
		self.asc = asc
		self.orders = None
		self.name = name

	@classmethod
	def orderCount(cls, asc: bool):
		me = cls(1, asc, name="_count")
		return me

	@classmethod
	def orderField(cls, asc: bool):
		me = cls(2, asc, name="_key")
		return me

	@classmethod
	def orderAggregation(cls, asc: bool, name: str):
		me = cls(3, asc, name=name)
		return me

	@classmethod
	def orderCompound(cls, *orders):
		if len(orders) < 2:
			raise_runtime_error("orders长度必须大于1")
		me = cls(4, None)
		me.orders = orders
		return me

	def get_asc(self):
		return self.asc and "asc" or "desc"

	def to_x_content(self, *args, **kwargs):
		if self.typeValue == 4:
			return [o.to_x_content() for o in self.orders]
		return {self.name: self.get_asc()}


def raise_runtime_error(*args):
	raise RuntimeError(*args)


def get_max_size(size, max_value, *, min_value=0):
	if min_value <= size and size <= max_value:
		return size
	return max_value


def _add_order(d, order):
	if order:
		d["order"] = order
	return d


def get_ceil(total, *, size=None):
	if size is None:
		return ceil(total)
	return (total + size - 1) // size


def _get_alias(alias, field):
	if alias:
		return alias
	return field


def get_timedelta(dt1: [dt_datetime, dt_date], dt2: [dt_datetime, dt_date]):
	"""
	获取時间差(dt1-dt2)
	:param dt1: 必须dt_datetime/dt_date类型,其它类型抛异常
	:param dt2: 必须dt_datetime/dt_date类型,其它类型抛异常
	:return: relativedelta对象(属性:years, months, weeks, days, hours, minutes, seconds, microseconds)
	"""
	return relativedelta(dt1=dt1, dt2=dt2)


def print_datetime_now(*coll, dt_first=None, sep=" "):
	dt = dt_datetime.now()
	if coll:
		if dt_first:
			print("{}:\t{}".format(dt, sep.join(map(str, coll))), "\t耗时:", get_timedelta(dt, dt_first))
		else:
			print("{}:\t{}".format(dt, sep.join(map(str, coll))))
	else:
		if dt_first:
			print(dt, "\t耗时:", get_timedelta(dt, dt_first))
		else:
			print(dt)
	return dt


def dict_get_by_key(d: dict, *args, keys=None, default=None) -> dict:
	if args:
		keys = args
	return dict(zip(keys, map(lambda o: d.get(o, default), keys)))


class QueryBuilder(ToXContent):
	def _add_query(self, ls, query):
		if isinstance(query, ToXContent):
			ls.append(query.to_x_content())
		elif isinstance(query, dict):
			ls.append(query)
		return self


class BoolQueryBuilder(QueryBuilder):
	def __init__(self):
		self._should_ls = []
		self._must_ls = []
		self._must_not_ls = []
		self._filter_ls = []

	def should(self, query):
		return self._add_query(self._should_ls, query)

	def must(self, query):
		return self._add_query(self._must_ls, query)

	def must_not(self, query):
		return self._add_query(self._must_not_ls, query)

	def filter(self, filters):
		return self._add_query(self._filter_ls, filters)

	def to_x_content(self, *args, boost=1.0, **kwargs):
		"""
		将对象转化为es查询的dict对象
		:return: dict对象或者None
		"""

		def _get_dict_core(bool_dict, ls, name):
			if ls:
				bool_dict[name] = ls

		result_dict = {}
		_get_dict_core(result_dict, self._should_ls, "should")
		_get_dict_core(result_dict, self._must_ls, "must")
		_get_dict_core(result_dict, self._must_not_ls, "must_not")
		_get_dict_core(result_dict, self._filter_ls, "filter")
		if result_dict:
			result_dict["boost"] = boost
			return {"bool": result_dict}
		return None


class InnerHitBuilder(ToXContent):
	def __init__(self, name: str, *, size=100):
		self.name = name
		self.size = size
		self.show_fields_childrens = None

	def set_show_fields_childrens(self, show_fields):
		if show_fields:
			if isinstance(show_fields, str):
				if show_fields.startswith(self.name + "."):
					self.show_fields_childrens = [show_fields]
			else:
				self.show_fields_childrens = list(filter(lambda o: o.startswith(self.name + "."), show_fields))
		return self

	def to_x_content(self, *args, **kwargs):
		result_dict = {"name": self.name, "ignore_unmapped": False, "from": 0, "size": self.size, "version": False, "explain": False, "track_scores": False}
		if self.show_fields_childrens:
			result_dict["_source"] = {"includes": self.show_fields_childrens}
		return result_dict


class NestedQueryBuilder(ToXContent):
	def __init__(self, path: str, query, score_mode: ScoreMode):
		self.path = path
		self.query = query
		self.score_mode = score_mode
		self.__inner_hit_builder = None

	def set_inner_hit_builder(self, inner_hit_builder: InnerHitBuilder):
		self.__inner_hit_builder = inner_hit_builder
		return self

	def to_x_content(self, *args, **kwargs):
		result_dict = {"path": self.path, "score_mode": self.score_mode.value, "ignore_unmapped": False, "boost": 1.0}
		if self.__inner_hit_builder:
			result_dict["inner_hits"] = self.__inner_hit_builder.to_x_content()
		if isinstance(self.query, QueryBuilder):
			result_dict["query"] = self.query.to_x_content()
		elif isinstance(self.query, dict):
			result_dict["query"] = self.query
		else:
			raise_runtime_error("query不可以为空对象.")
		return {"nested": result_dict}


class AggBuilderService(ToXContent):
	DEFAULT_SIZE = -1

	def to_x_content(self, *args, **kwargs):
		pass

	def getType(self):
		pass

	def getAlias(self):
		return self.alias


AGGREGATETYPES = (AggregateType.AVG, AggregateType.MAX, AggregateType.MIN, AggregateType.SUM, AggregateType.COUNT)


def get_AggregateType_value_es(agg: AggregateType):
	return agg.value[_KEY_ENUM_ES]


class AggAggregate(AggBuilderService):
	def __init__(self, aggregateType: AggregateType, field: str, *, alias: str = None, other=None):
		# 聚合字段
		self.field = field
		if alias:
			self.alias = alias
		else:
			self.alias = field
		# 大小, 默认获取全部
		self.size = AggBuilderService.DEFAULT_SIZE
		# 聚合函数
		self.aggregateType = aggregateType
		# 在MAX_OTHER和MIN_OTHER聚合函数有效
		self.fieldOther = None
		# 在MAX_OTHER和MIN_OTHER聚合函数有效
		self.aliasOther = None
		# 是否使用最大size
		self.useMaxSize = False
		if AggregateType.MEDIAN == aggregateType:
			other = [50.0]
		# 有需要時候使用,任意类型【PERCENTILES:list,元素是float,0<other<100】
		if other is None or isinstance(other, list):
			self.other = other
		else:
			raise_runtime_error("AggregateType.PERCENTILES中的other参数必须是list.")

	def setFieldOther(self, fieldOther, *, aliasOther=None):
		self.fieldOther = fieldOther
		if aliasOther:
			self.aliasOther = aliasOther
		else:
			self.aliasOther = fieldOther
		return self

	def getType(self):
		return AggregationType.AGGREGATE

	def to_x_content(self, *args, **kwargs):
		aggregate_type = self.aggregateType
		aggregate_type_value = get_AggregateType_value_es(aggregate_type)
		# (AggregateType.AVG, AggregateType.MAX, AggregateType.MIN, AggregateType.SUM, AggregateType.COUNT)
		if aggregate_type in AGGREGATETYPES:
			return {self.alias: {aggregate_type_value: {"field": self.field}}}
		elif aggregate_type == AggregateType.COUNT_DISTINCT:
			if self.size > 0:
				return {self.alias: {aggregate_type_value: {"field": self.field, "precision_threshold": get_max_size(self.size, MAX_SIZE_CARDINALITY)}}}
			return {self.alias: {aggregate_type_value: {"field": self.field}}}
		elif AggregateType.PERCENTILES == aggregate_type or AggregateType.MEDIAN == aggregate_type:
			# 压缩值(compression):默认100,压缩值越大,精确度越高
			return {self.alias: {"percentiles": {"field": self.field, "percents": self.other, "keyed": True, "tdigest": {"compression": 100.0}}}}
		elif AggregateType.MAX_OTHER == aggregate_type:
			return {self.alias: {"terms": {"field": self.fieldOther, "size": 1, "min_doc_count": 1, "shard_min_doc_count": 0, "show_term_doc_count_error": False, "order": [{self.alias: "desc"}, {"_key": "asc"}]}, "aggregations": {self.alias: {"max": {"field": self.field}}}}}
		elif AggregateType.MIN_OTHER == aggregate_type:
			return {self.alias: {"terms": {"field": self.fieldOther, "size": 1, "min_doc_count": 1, "shard_min_doc_count": 0, "show_term_doc_count_error": False, "order": [{self.alias: "asc"}, {"_key": "asc"}]}, "aggregations": {self.alias: {"min": {"field": self.field}}}}}
		raise_runtime_error("aggregateType错误!")


def _terms_to_x_content(alias, field, size, include, order, aggregations, exclude=None, missing=None, execution_hint=None):
	d1 = {"field": field}
	# "order": {"last_access": "asc" }
	# "order": "asc"
	_add_order(d1, order)
	if size:
		d1["size"] = size
		d1.update({"min_doc_count": 1, "shard_min_doc_count": 0, "show_term_doc_count_error": False})
	if include:
		# "include": ["mazda", "honda"]
		# "include": {"partition": 0, "num_partitions": 20}
		# "include" : ".*sport.*"
		d1["include"] = include
	if exclude:
		d1["exclude"] = exclude
	if missing:
		d1["missing"] = missing
	if execution_hint:
		d1["execution_hint"] = execution_hint
	d2 = {"terms": d1}
	add_aggregations(d2, aggregations)
	return {alias: d2}


def add_aggregations(d, aggregations: dict):
	if aggregations:
		d["aggregations"] = aggregations
	return d


def _composite_to_x_content(alias, size, ls, aggregations, after=None, *, dict_after=False):
	"""
	复合方式多字段分组使用
	:param alias:
	:param size:
	:param ls:
	:param aggregations:
	:param after:{ "date": 1494288000000, "product": "mad max" }
	:return:
	"""
	d1 = {"size": size, "sources": list(map(lambda o: isinstance(o, dict) and o or o.to_x_content(), ls))}
	if after:
		d1["after"] = after
	d2 = {"composite": d1}
	add_aggregations(d2, aggregations)
	d = {alias: d2}
	if dict_after:
		return d, d1
	return d


def _terms2_to_x_content(alias, field, order: Order):
	"""
	复合方式多字段分组使用
	:param alias:
	:param field:
	:param order:
	:return:
	"""
	if order and order.typeValue == 2:
		return {alias: {"terms": {"field": field, "order": order.get_asc()}}}
	return {alias: {"terms": {"field": field}}}


def getAggBuilderServices_to_x_content(aggBuilderServices):
	if aggBuilderServices:
		if len(aggBuilderServices) > 1:
			return dict_merge(coll=map(lambda o: o.to_x_content(), aggBuilderServices), func_replace=lambda hm, k, v: raise_runtime_error("aggBuilderServices中的alias禁止重复名字"))
		for agg in aggBuilderServices:
			return agg.to_x_content()


class AggTerms(AggBuilderService):
	def __init__(self, field, *, alias=None, include=None, order: Order = None, aggregations=None, size=None, useMaxSize=False):
		self.field = field
		self.alias = _get_alias(alias, field)
		self.order = order
		self.aggregations = aggregations
		self.include = include
		self.size = size
		self.useMaxSize = useMaxSize

	def getType(self) -> AggregationType:
		return AggregationType.TERMS

	def to_x_content(self, *args, **kwargs):
		size = self.size
		if size:
			size = get_max_size(size, self.useMaxSize and MAX_SIZE_GROUP2 or SIZE_GROUP)
		return _terms_to_x_content(self.alias, self.field, size, self.include, getOrder_to_x_content(self.order), self.aggregations)


class AggComposite(AggBuilderService):
	def __init__(self, alias, sources, *, size=AggBuilderService.DEFAULT_SIZE, aggBuilderServices=None, after=None):
		# terms,histogram,date_histogram
		self.alias = alias
		# Collection类型,用于分页获取全量数据
		self.sources = sources
		# count distinc str结果都为0(int结果是正常的)
		self.aggBuilderServices = aggBuilderServices
		self.after = after
		self.size = size

	def getType(self) -> AggregationType:
		if self.aggBuilderServices:
			return AggregationType.MULTIPLE_GROUP_COMPOSITE_AGGREGATE
		return AggregationType.MULTIPLE_GROUP_COMPOSITE

	def to_x_content(self, *args, **kwargs):
		total = kwargs.get("total", -1)
		dict_after = kwargs.get("dict_after", False)
		size = get_max_size(self.size, SIZE_GROUP)
		size = get_max_size(total, size)
		aggregations = getAggBuilderServices_to_x_content(self.aggBuilderServices)
		return _composite_to_x_content(self.alias, size, self.sources, aggregations, self.after, dict_after=dict_after)


class AggGroup(AggBuilderService):
	def __init__(self, field, *, alias=None):
		self.field = field
		if alias:
			self.alias = alias
		else:
			self.alias = field
		# 排序方式:Order类型【仅适用于单字段分组和原始多字段分组】
		self.order = None
		# 大小, 默认获取全部
		self.size = AggBuilderService.DEFAULT_SIZE
		# 多字段分组对象:集合类型,元素是AggGroup
		self.aggGroups = None
		# 必须包含的分组值:list类型,元素是str
		self.includeFields = None
		# 默认多字段分组使用复合方式,True:原始多字段分组
		self.useMulTerms = False
		# 是否使用最大size(目前在单字段分组和多字段分组(复合查询)中使用)
		self.useMaxSize = False
		# 聚合函数:集合类型,元素是AggBuilderService
		self.aggBuilderServices = None

	@classmethod
	def groups(cls, *args, aggGroups: [list, tuple] = None):
		if args:
			aggGroups = args
		if len(aggGroups) < 2:
			raise_runtime_error("aggGroups长度必须大于1")
		me = cls(None)
		me.aggGroups = aggGroups
		return me

	def addAggBuilderServices(self, agg):
		if self.aggBuilderServices is None:
			self.aggBuilderServices = []
		self.aggBuilderServices.append(agg)
		return self

	def getOrder_to_x_content(self):
		if self.order:
			return self.order.to_x_content()

	def __dict_merge(self, coll, func_replace):
		def func(hm, d):
			if d and d != hm:
				for k, v in d.items():
					if k in hm:
						func_replace(hm, k, v)
					else:
						hm[k] = v
			return hm

		return reduce(func, coll)

	def getAggBuilderServices_to_x_content(self):
		if self.aggBuilderServices:
			if len(self.aggBuilderServices) > 1:
				# return collection_utils.dict_merge({}, coll=(o.to_x_content() for o in self.aggBuilderServices), func_replace=lambda hm, k, v: raise_runtime_error("self.aggBuilderServices中的alias禁止重复名字"))
				return self.__dict_merge((o.to_x_content() for o in self.aggBuilderServices),
										 lambda hm, k, v: raise_runtime_error("self.aggBuilderServices中的alias禁止重复名字"))
			for agg in self.aggBuilderServices:
				return agg.to_x_content()

	def getType(self):
		if self.aggBuilderServices:
			if self.aggGroups:
				if self.useMulTerms:
					return AggregationType.MULTIPLE_GROUP_AGGREGATE
				return AggregationType.MULTIPLE_GROUP_COMPOSITE_AGGREGATE
			return AggregationType.SINGLE_GROUP_AGGREGATE
		if self.aggGroups:
			if self.useMulTerms:
				return AggregationType.MULTIPLE_GROUP
			return AggregationType.MULTIPLE_GROUP_COMPOSITE
		return AggregationType.SINGLE_GROUP

	def to_x_content(self, *args, **kwargs):
		total = kwargs.get("total", -1)
		agg_groups = self.aggGroups
		if agg_groups:
			judge_iter_len2(agg_groups, func=lambda o: not o is None)
			# region aggGroupsSortSize
			aggGroupsSortSize = []
			aggGroupsSort = []
			aggGroupsSize = []
			aggGroupsNo = []
			for agg in agg_groups:
				if agg.order:
					if agg.size > 0:
						aggGroupsSortSize.append(agg)
					else:
						aggGroupsSort.append(agg)
				else:
					if agg.size > 0:
						aggGroupsSize.append(agg)
					else:
						aggGroupsNo.append(agg)
			aggGroupsSortSize += aggGroupsSort
			aggGroupsSortSize += aggGroupsSize
			aggGroupsSortSize += aggGroupsNo
			del aggGroupsSort, aggGroupsSize, aggGroupsNo
			# endregion
			# region 设置别名
			# 多字段分组/多字段分组聚合的時候, 如果已经有设置总别名, 则将总别名设置到第多字段分组的第1个字段
			# 如果没有设置总别名, 则获取多字段分组的第1个字段别名, 并设置成总别名.因为子查询多字段分组的時候要用到.
			aggGroupFirst = aggGroupsSortSize[0]
			if self.alias:
				aggGroupFirst.alias = self.alias
			else:
				self.alias = aggGroupFirst.alias
			# endregion
			size = get_max_size(total, self.useMaxSize and MAX_SIZE_GROUP2 or SIZE_GROUP)
			# 原始多字段分组
			if self.useMulTerms:
				lastAgg = aggGroupsSortSize[-1]
				lastAgg = _terms_to_x_content(lastAgg.alias, lastAgg.field, get_max_size(lastAgg.size, size), lastAgg.includeFields, self.getOrder_to_x_content(), self.getAggBuilderServices_to_x_content())
				for i in range(len(aggGroupsSortSize) - 2, 0, -1):
					curTermsBuilder = aggGroupsSortSize[i]
					lastAgg = _terms_to_x_content(curTermsBuilder.alias, curTermsBuilder.field, get_max_size(curTermsBuilder.size, size), curTermsBuilder.includeFields, curTermsBuilder.getOrder_to_x_content(), lastAgg)
				curTermsBuilder = aggGroupsSortSize[0]
				return _terms_to_x_content(curTermsBuilder.alias, curTermsBuilder.field, get_max_size(curTermsBuilder.size, size), curTermsBuilder.includeFields, curTermsBuilder.getOrder_to_x_content(), lastAgg)
			else:
				return _composite_to_x_content(self.alias, size, [_terms2_to_x_content(o.alias, o.field, o.order) for o in agg_groups], self.getAggBuilderServices_to_x_content())
		else:
			size = get_max_size(self.size, self.useMaxSize and MAX_SIZE_GROUP2 or MAX_SIZE_GROUP)
			size = get_max_size(total, size)
			return _terms_to_x_content(self.alias, self.field, size, self.includeFields, self.getOrder_to_x_content(), self.getAggBuilderServices_to_x_content())


class SearchModel(object):
	def __init__(self, index, *, queries=None, page_index=1, page_size=10, orders: [str, OrderedDict] = None, show_fields=None, esid_field: [bool, str] = ESID_FIELD, score_field: str = None, explain_field: str = None, return_total=False):
		"""

		:param index:
		:param queries:
		:param page_index:
		:param page_size:
		:param orders:
		:param show_fields:
		:param esid_field:
		:param score_field:
		:param explain_field:
		:param return_total: 最多10000条
		"""
		self.index = index
		self.queries = queries
		self.page_index = page_index
		self.page_size = get_max_size(page_size, MAX_SIZE_PAGE)
		self.orders = orders
		self.show_fields = show_fields
		self.esid_field = esid_field
		self.score_field = score_field
		self.explain_field = explain_field
		self.return_total = return_total

	def __str__(self):
		s = "SearchModel(\"{}\"".format(self.index)
		if self.queries:
			s += ", queries={}".format(self.queries)
		if self.page_index != 1:
			s += ", page_index={}".format(self.page_index)
		if self.page_size != 10:
			if MAX_SIZE_PAGE == self.page_size:
				s += ", page_size=es_utils.MAX_SIZE_PAGE"
			else:
				s += ", page_size={}".format(self.page_size)
		if self.orders:
			s += ", orders={}".format(self.orders)
		if self.show_fields:
			if NULL_VALUE == self.show_fields:
				s += ", show_fields=es_utils.NULL_VALUE"
			else:
				if isinstance(self.show_fields, str):
					s += ", show_fields=\"{}\"".format(self.show_fields)
				else:
					s += ", show_fields={}".format(self.show_fields)
		if self.esid_field and isinstance(self.esid_field, str):
			s += ', esid_field="{}"'.format(self.esid_field)
		else:
			s += ", esid_field={}".format(self.esid_field)
		if self.score_field:
			s += ", score_field=\"{}\"".format(self.score_field)
		if self.explain_field:
			s += ", explain_field=\"{}\"".format(self.explain_field)
		if self.return_total:
			s += ", return_total={}".format(self.return_total)
		return s + ")"


class Query(object):
	def set_values(self, *args, values: tuple = None):
		# 这里self.__values必须tuple类型,所以判断values是否存在
		if values:
			if not isinstance(values, tuple):
				raise_runtime_error("values必须是tuple类型!")
			args = values
		if self.__field == ESID_FIELD:
			args = tuple(str(o) for o in args if not (o is None or o == ""))
			if not args:
				raise_runtime_error("{}的值不可以为空!".format(ESID_FIELD))
		if args and len(args) > 1000:
			raise_runtime_error("values个数不超过1000!")
		self.__values = args
		return self

	def __init__(self, query_type: QueryType, field: str, *args, values: tuple = None):
		self.__query_type = query_type
		self.__field = field
		self.set_values(*args, values=values)
		self.__queries = None
		self.__nested = False
		self.__and_perator = False
		self.__only_match_childrens = False
		self.__score_mode = ScoreMode.AVG
		self.__minimum_should_match = None

	def __distinct_values(self):
		"""
		对值进行去重排序
		:return:
		"""
		values = self.__values
		if values:
			if judge_iter_len2(values, raise_errors=False):
				hs = set(values)
				try:
					self.__values = tuple(sorted(hs))
				except:
					d = {hash(o): o for o in hs}
					self.__values = tuple(d[k] for k in sorted(d.keys()))
		else:
			self.__values = None
		return self

	@classmethod
	def queries(cls, *args, queries=None, and_perator=True, nested_field=None):
		"""

		:param args:
		:param queries: tuple/list
		:param and_perator:
		:param nested_field:
		:return:
		"""
		me = cls(None, nested_field)
		if nested_field:
			me.__nested = True
		me.__and_perator = and_perator
		if args:
			queries = args
		me.__queries = [o.__distinct_values() for o in queries]
		return me

	def __hash__(self):
		return hash(str(self.__dict__))

	def __eq__(self, other):
		return self.__dict__ == other.__dict__

	def __str__(self):
		if self.__queries:
			s = "Query.queries(" + ", ".join(map(str, self.__queries))
			if not self.__and_perator:
				s += ", and_perator={}".format(self.__and_perator)
			if self.__field:
				s += ", nested_field={}".format(self.__field)
		else:
			if self.__query_type in _QUERYTYPE_VALUE_SINGLE:
				if isinstance(self.__values[0], str):
					s = "Query({}, \"{}\", \"{}\"".format(self.__query_type, self.__field, self.__values[0])
				else:
					s = "Query({}, \"{}\", {}".format(self.__query_type, self.__field, self.__values[0])
			elif self.__query_type in _QUERYTYPE_VALUE_SINGLE_PLUS:
				if isinstance(self.__values[0], str):
					s = "Query({}, \"{}\", {}".format(self.__query_type, self.__field, ", ".join(map(lambda o: "\"{}\"".format(o), self.__values)))
				else:
					s = "Query({}, \"{}\", {}".format(self.__query_type, self.__field, ", ".join(map(str, self.__values)))
			else:
				if self.__values:
					if isinstance(self.__values[0], str):
						s = "Query({}, \"{}\", \"{}\"".format(self.__query_type, self.__field, self.__values[0])
					else:
						s = "Query({}, \"{}\", {}".format(self.__query_type, self.__field, self.__values[0])
				else:
					s = "Query({}, \"{}\"".format(self.__query_type, self.__field)
			if self.__nested:
				s += ", nested={}".format(self.__nested)
		s += ")"
		if self.__only_match_childrens:
			s += ".set_only_match_childrens(True)"
		if self.__minimum_should_match:
			if isinstance(self.__minimum_should_match, str):
				s += ".set_minimum_should_match(\"{}\")".format(self.__minimum_should_match)
			else:
				s += ".set_minimum_should_match({})".format(self.__minimum_should_match)
		return s

	def add_queries(self, *args, queries=None):
		"""
		Query.queries初始化才能使用
		:param queries:
		:return:
		"""
		if args:
			queries = args
		for q in queries:
			if not q.__distinct_values() in self.__queries:
				self.__queries.append(q)
		return self

	def set_score_mode(self, score_mode: ScoreMode):
		"""
		子查询计算分数方式
		:param score_mode: 默认:ScoreMode.AVG
		:return:
		"""
		self.__score_mode = score_mode
		return self

	def set_minimum_should_match(self, minimum_should_match):
		"""

		:param minimum_should_match: 3,-2,75%,-25%,3-90%,2<-25%
		:return:
		"""
		self.__minimum_should_match = minimum_should_match
		return self

	def set_only_match_childrens(self, only_match_childrens: bool):
		"""
		子查询字段只返回设置的字段
		:param only_match_childrens: False:返回所有子对象中所有字段,True:返回子对象中设置的字段
		:return:
		"""
		self.__only_match_childrens = only_match_childrens
		return self

	def get_only_match_childrens_size(self):
		return 100

	def get_query_type(self):
		return self.__query_type

	def get_field(self):
		return self.__field

	def get_values(self):
		"""
		__values是tuple
		:return:
		"""
		return self.__values

	def get_queries(self):
		return self.__queries

	def get_nested(self):
		return self.__nested

	def get_and_perator(self):
		return self.__and_perator

	def get_only_match_childrens(self):
		return self.__only_match_childrens

	def get_score_mode(self):
		return self.__score_mode

	def get_minimum_should_match(self):
		return self.__minimum_should_match


_QUERYTYPE_VALUE_SINGLE = (QueryType.GT, QueryType.LT, QueryType.GE, QueryType.LE)
# 带1+个参数(用2个下划线,则Query.__str__无法访问)
_QUERYTYPE_VALUE_SINGLE_PLUS = (QueryType.BN, QueryType.NI, QueryType.IN, QueryType.IN_LIKE)

AGGREGATETYPES = (AggregateType.AVG, AggregateType.MAX, AggregateType.MIN, AggregateType.SUM, AggregateType.COUNT)


# def __delete_client(sleep_times=5):
# 	__PRIVATE_DICT.pop(__PRIVATE_DICT["host"])
# 	if sleep_times > 0:
# 		time.sleep(sleep_times)
# 	else:
# 		time.sleep(5)


def __get_client(host=None) -> Elasticsearch:
	if not host:
		host = __PRIVATE_DICT["es_host"]
	if not host in __d_host:
		print("current es host:{}".format(host))
		es = Elasticsearch(
			hosts=[{"host": ip, "port": "9200", "http_auth": es_http_auth} for ip in host],  timeout=60, retry_on_timeout=True, http_compress=True,
			use_ssl=True,
			verify_certs=False,
			ssl_show_warn=False
		)
		# print(es)
		__d_host[host] = es
	return __d_host[host]

# a=__get_client()
# print(a)
# exit()


def __match_all_query(boost=1.0):
	return {"match_all": {"boost": boost}}


def __wildcard_query(field, value, *, boost=1.0):
	return {"wildcard": {field: {"wildcard": "*{}*".format(value), "boost": boost}}}


def __match_query(field, value, operator=Operator.OR, prefix_length=0, max_expansions=50, minimum_should_match=None,
				  fuzzy_transpositions=True, lenient=False, zero_terms_query=ZeroTermsQuery.NONE,
				  auto_generate_synonyms_phrase_query=True, boost=1.0):
	"""

	:param field:
	:param value:
	:param operator: Operator类型
	:return:
	"""
	if minimum_should_match:
		return {"match": {field: {"query": value, "operator": operator.value, "prefix_length": prefix_length,
								  "max_expansions": max_expansions, "minimum_should_match": minimum_should_match,
								  "fuzzy_transpositions": fuzzy_transpositions, "lenient": lenient,
								  "zero_terms_query": zero_terms_query.value,
								  "auto_generate_synonyms_phrase_query": auto_generate_synonyms_phrase_query,
								  "boost": boost}}}
	return {"match": {field: {"query": value, "operator": operator.value, "prefix_length": prefix_length,
							  "max_expansions": max_expansions, "fuzzy_transpositions": fuzzy_transpositions,
							  "lenient": lenient, "zero_terms_query": zero_terms_query.value,
							  "auto_generate_synonyms_phrase_query": auto_generate_synonyms_phrase_query,
							  "boost": boost}}}


def __match_phrase_query(field, value, *, boost=1.0, slop=0):
	return {"match_phrase": {field: {"query": value, "slop": slop, "boost": boost}}}


def __exists_query(field, boost=1.0):
	return {"exists": {"field": field, "boost": boost}}


def __range_query(field, from_value=None, to_value=None, include_lower=True, include_upper=True, boost=1.0):
	return {"range": {
		field: {"from": from_value, "to": to_value, "include_lower": include_lower, "include_upper": include_upper,
				"boost": boost}}}


def __terms_query(field, *values, boost=1.0):
	"""

	:param field:
	:param values: 必须是list/tuple类型,不可以是set类型,这边传进來是tuple类型
	:return:
	"""
	return {"terms": {field: values, "boost": boost}}


def __term_query(field, value, boost=1.0):
	return {"term": {field: {"value": value, "boost": boost}}}


def __ids_query(*values, boost=1.0):
	"""

	:param values: 必须是list/tuple类型,不可以是set类型,这边传进來是tuple类型
	:return:
	"""
	return {"ids": {"type": [], "values": values, "boost": boost}}


def __contain_special_chars(txt):
	"""
	判断是否包含特殊字符和空格
	:param txt:
	:return: True包含
	"""
	return re_wkz_special_chars.search(txt)


def __get_like_query(field, value, *, minimum_should_match=None):
	query = BoolQueryBuilder()
	if minimum_should_match:
		query.should(__match_query(field, value, minimum_should_match=minimum_should_match))
		query.should(__match_query("{}.raw".format(field), value, minimum_should_match=minimum_should_match))
		query.should(__match_query("{}.raw2".format(field), value, minimum_should_match=minimum_should_match))
		query.should(__match_query("{}.raw3".format(field), value, minimum_should_match=minimum_should_match))
		query.should(__match_query("{}.raw4".format(field), value, minimum_should_match=minimum_should_match))
	else:
		if __contain_special_chars(value):
			query.should(__match_query(field, value, operator=Operator.AND))
			query.should(__match_query("{}.raw".format(field), value, operator=Operator.AND))
			query.should(__match_query("{}.raw2".format(field), value, operator=Operator.AND))
			query.should(__match_query("{}.raw3".format(field), value, operator=Operator.AND))
			query.should(__match_query("{}.raw4".format(field), value, operator=Operator.AND))
		else:
			v = value.lower()
			query.should(__match_phrase_query(field, v))
			query.should(__match_phrase_query("{}.raw".format(field), v))
			query.should(__match_phrase_query("{}.raw2".format(field), v))
			query.should(__match_phrase_query("{}.raw3".format(field), v))
			query.should(__match_phrase_query("{}.raw4".format(field), v, boost=0.5))
	return query


def __get_in_xxx(query_builder, filter_builder, field, values, func, **kwargs):
	if values:
		bool_query = BoolQueryBuilder()
		i = 0
		for value in values:
			if value:
				i += 1
				# bool_query.should(__get_like_query(field, value, minimum_should_match=minimum_should_match))
				bool_query.should(func(field, value, **kwargs))
		if i > 250:
			filter_builder.must(bool_query)
			return True
		else:
			query_builder.must(bool_query)
	else:
		filter_builder.must_not(__exists_query(field))
		return True
	return False


def __get_core_query(query_builder: BoolQueryBuilder, filter_builder: BoolQueryBuilder, query_type, field, values,
					 minimum_should_match):
	isfilter = False
	if query_type:
		if QueryType.GT == query_type:
			filter_builder.must(__range_query(field, from_value=values[0], include_lower=False))
			isfilter = True
		elif QueryType.LT == query_type:
			filter_builder.must(__range_query(field, to_value=values[0], include_upper=False))
			isfilter = True
		elif QueryType.GE == query_type:
			filter_builder.must(__range_query(field, from_value=values[0]))
			isfilter = True
		elif QueryType.LE == query_type:
			filter_builder.must(__range_query(field, to_value=values[0]))
			isfilter = True
		elif QueryType.BN == query_type:
			min_value = values[0]
			max_value = values[1]
			if min_value > max_value:
				min_value, max_value = max_value, min_value
			filter_builder.must(__range_query(field, from_value=min_value, to_value=max_value))
			isfilter = True
		elif QueryType.NI == query_type:
			if values:
				if None in values:
					values = [o for o in values if not o is None]
					if values:
						filter_builder.must_not(
							BoolQueryBuilder().should(BoolQueryBuilder().must_not(__exists_query(field))).should(
								__terms_query(field, *values)))
					else:
						filter_builder.must(__exists_query(field))
				else:
					filter_builder.must_not(__terms_query(field, *values))
			else:
				filter_builder.must(__exists_query(field))
			isfilter = True
		elif QueryType.IN == query_type:
			if values:
				if None in values:
					values = [o for o in values if not o is None]
					if values:
						filter_builder.must(
							BoolQueryBuilder().should(BoolQueryBuilder().must_not(__exists_query(field))).should(
								__terms_query(field, *values)))
					else:
						filter_builder.must_not(__exists_query(field))
				else:
					filter_builder.must(__terms_query(field, *values))
			else:
				filter_builder.must_not(__exists_query(field))
			isfilter = True
		elif QueryType.IN_LIKE == query_type:
			isfilter = __get_in_xxx(query_builder, filter_builder, field, values, __get_like_query, minimum_should_match=minimum_should_match)
		elif QueryType.IN_WILDCARD == query_type:
			isfilter = __get_in_xxx(query_builder, filter_builder, field, values, __wildcard_query)
		elif QueryType.NL == query_type:
			if not values or values[0] is None:
				filter_builder.must(__exists_query(field))
				isfilter = True
			else:
				query_builder.must_not(__get_like_query(field, values[0], minimum_should_match=minimum_should_match))
		elif QueryType.LIKE == query_type:
			if not values or values[0] is None:
				filter_builder.must_not(__exists_query(field))
				isfilter = True
			else:
				query_builder.must(__get_like_query(field, values[0], minimum_should_match=minimum_should_match))
		elif QueryType.NE == query_type:
			if not values or values[0] is None:
				filter_builder.must(__exists_query(field))
			else:
				if NULL_VALUE == values[0]:
					# filter_builder.must_not(BoolQueryBuilder().should(BoolQueryBuilder().must_not(__exists_query(field))).should(__term_query(field, "")))
					filter_builder.must(__exists_query(field)).must_not(__term_query(field, ""))
				else:
					filter_builder.must_not(__term_query(field, values[0]))
			isfilter = True
		elif QueryType.EQ == query_type:
			if not values or values[0] is None:
				filter_builder.must_not(__exists_query(field))
			else:
				if NULL_VALUE == values[0]:
					filter_builder.must(
						BoolQueryBuilder().should(BoolQueryBuilder().must_not(__exists_query(field))).should(
							__term_query(field, "")))
				else:
					filter_builder.must(__term_query(field, values[0]))
			isfilter = True
		elif QueryType.WILDCARD == query_type:
			if not values or values[0] is None:
				filter_builder.must_not(__exists_query(field))
			else:
				filter_builder.must(__wildcard_query(field, values[0]))
			isfilter = True
		elif QueryType.NW == query_type:
			if not values or values[0] is None:
				filter_builder.must(__exists_query(field))
			else:
				filter_builder.must_not(__wildcard_query(field, values[0]))
			isfilter = True
		elif QueryType.PREFIX == query_type:
			pass
		else:
			raise_runtime_error("{}错误!".format(query_type))
	return isfilter


def judge_iter_len(coll, length, *, func=None, raise_errors=True):
	"""
	判断是否不为空,且里面元素个数>=length
	:param coll: tuple/list/set/dict
	:param length: length>0
	:param func: 判断元素是否全部符合条件,则返回True.eg:func=lambda o:not o is None,ls中元素有为None,则返回False
	:param raise_errors: 不符合是否抛异常,默认:抛
	:return: bool
	"""
	if length < 1:
		raise_runtime_error("length参数错误,length必须>=1!")
	if coll and len(coll) >= length and (not func or all(map(func, coll))):
		return True
	if raise_errors:
		raise raise_runtime_error("{}中元素个数不够,或者元素不符合条件".format(coll))
	return False


def judge_iter_len2(coll, *, func=None, raise_errors=True):
	"""
	判断是否不为空,且里面元素个数>=2
	:param coll: tuple/list/set/dict
	:param func: 判断元素是否全部符合条件,则返回True.eg:func=lambda o:not o is None,ls中元素有为None,则返回False
	:param raise_errors: 不符合是否抛异常,默认:抛
	:return: bool
	"""
	return judge_iter_len(coll, 2, func=func, raise_errors=raise_errors)


def __make_query_builder(q: Query, query_builder: BoolQueryBuilder, filter_builder: BoolQueryBuilder, show_fields):
	isfilter = False
	if q.get_queries():
		if q.get_nested():
			query_cur = BoolQueryBuilder()
			if q.get_and_perator():
				for chi_q in q.get_queries():
					__make_query_builder(chi_q, query_cur, query_cur, show_fields)
			else:
				for chi_q in q.get_queries():
					query_chi = BoolQueryBuilder()
					__make_query_builder(chi_q, query_chi, query_chi, show_fields)
					query_cur.should(query_chi)
			field = q.get_field()
			nested_query_builder = NestedQueryBuilder(field, query_cur, q.get_score_mode())
			if q.get_only_match_childrens():
				nested_query_builder.set_inner_hit_builder(
					InnerHitBuilder(field, size=q.get_only_match_childrens_size()).set_show_fields_childrens(
						show_fields))
			query_builder.must(nested_query_builder)
		else:
			if q.get_and_perator():
				for chi_q in q.get_queries():
					temp = __make_query_builder(chi_q, query_builder, filter_builder, show_fields)
					isfilter = isfilter or temp
			else:
				query_cur = BoolQueryBuilder()
				for chi_q in q.get_queries():
					query_chi = BoolQueryBuilder()
					__make_query_builder(chi_q, query_chi, query_chi, show_fields)
					query_cur.should(query_chi)
				query_builder.must(query_cur)
	else:
		field = q.get_field()
		values = q.get_values()
		query_type = q.get_query_type()

		# region 判断参数个数非法性
		if query_type in _QUERYTYPE_VALUE_SINGLE:
			if not values:
				raise_runtime_error("{}字段做{}操作,\t必须有1个值!".format(field, query_type))
		elif query_type in _QUERYTYPE_VALUE_SINGLE_PLUS:
			if not values:
				raise_runtime_error("{}字段做{}操作,\t至少有1个值!".format(field, query_type))
			if query_type == QueryType.BN:
				# 不可以为None和""
				judge_iter_len2(values, func=bool)
			elif not judge_iter_len2(values, raise_errors=False):
				if query_type == QueryType.NI:
					query_type = QueryType.NE
				elif query_type == QueryType.IN:
					query_type = QueryType.EQ
				elif query_type == QueryType.IN_LIKE:
					query_type = QueryType.LIKE
		# endregion

		if ESID_FIELD == field:
			if QueryType.NI == query_type or QueryType.NL == query_type or QueryType.NE == query_type:
				filter_builder.must_not(__ids_query(*values))
			else:
				filter_builder.must(__ids_query(*values))
			isfilter = True
		else:
			temp = __get_core_query(query_builder, filter_builder, query_type, field, values,
									q.get_minimum_should_match())
			isfilter = isfilter or temp
	return isfilter


def getOrder_to_x_content(order):
	if order:
		return order.to_x_content()


def __get_query(queries, *, show_fields=None):
	result = None
	if queries:
		query_builder = BoolQueryBuilder()
		filter_builder = BoolQueryBuilder()
		isfilter = False
		if isinstance(queries, Query):
			temp = __make_query_builder(queries, query_builder, filter_builder, show_fields)
			isfilter = temp or isfilter
		else:
			for q in queries:
				temp = __make_query_builder(q, query_builder, filter_builder, show_fields)
				isfilter = isfilter or temp
		if isfilter:
			query_builder.filter(filter_builder)
		result = query_builder.to_x_content()
	return result


# 根据分页参数获取起始位置
def __get_from_index(page_index, page_size):
	"""
	根据分页参数获取起始位置
	:param page_index: 当前页码,<2直接返回0
	:param page_size: 每页条数
	:return:
	"""
	return page_index > 1 and (page_index - 1) * page_size or 0


# 获取排序对象
def __get_sort(field, value, *, missing="_last"):
	"""
	获取排序对象
	:param field: 排序字段
	:param value: 排序方式:真:升序;其它:降序
	:param value: 空值排序位置:_first/_last
	:return:
	"""
	return {ESID_FIELD == field and __ES_ID_FIELD or field: {"missing": missing, "order": value and "asc" or "desc"}}


# 获取排序对象
def __get_sorts(orders: [str, dict, OrderedDict]):
	"""
	获取排序对象
	:param orders:
	:return:
	"""
	if orders:
		if isinstance(orders, str):
			return [__get_sort(orders, True)]
		return [__get_sort(field, direction) for field, direction in orders.items()]


__sort_doc = [DOC_FIELD]


def __add_sort_doc(sorts, body):
	if sorts:
		print("警报警报警报!!!\nget_page的page_size<0时推荐不排序!!!")
	else:
		if body:
			body["sort"] = __sort_doc
		else:
			body = {"sort": __sort_doc}
	return body


def __get_body(*, query: dict = None, orders: list = None, aggregations: dict = None, after_values: list = None):
	"""
	获取es查询dsl(由query,分组聚合,排序方式组成)
	:param query: dict对象
	:param aggregations: dict对象
	:param orders: OrderedDict()/str【默认升序】
	:return: None/dict对象
	"""
	if query:
		body = {"query": query}
	else:
		body = None
	# 有聚合分组
	if aggregations:
		if not body:
			body = {}
		body["aggregations"] = aggregations
	# 有排序字段
	if orders:
		if not body:
			body = {}
		body["sort"] = orders
		if after_values:
			body[__ES_SEARCH_AFTER_FIELD] = after_values
	return body


def __get_body_mul(body_ls: list, search_model: SearchModel):
	body_ls.append({"index": search_model.index})
	if search_model.queries:
		query = __get_query(search_model.queries, show_fields=search_model.show_fields)
		sorts = __get_sorts(search_model.orders)
		body = __get_body(query=query, orders=sorts)
	elif search_model.orders:
		sorts = __get_sorts(search_model.orders)
		body = __get_body(orders=sorts)
	else:
		body = {}
	body["size"] = search_model.page_size
	# 查询列表时,才设置下面属性
	if search_model.page_size:
		body["from"] = __get_from_index(search_model.page_index, search_model.page_size)
		body["explain"] = bool(search_model.explain_field)
		# "_source": {"includes": ["drug_id", "submission_type_worm_revised"], "excludes": []}
		# 如果要不返回_source中字段的值,则_source=False,如果要返回字段则必须是str或者list类型,设置[]则返回全部
		show_fields = __get_show_fields(search_model.show_fields)
		if show_fields is None:
			body["_source"] = False
		else:
			body["_source"] = {"includes": show_fields}
	body_ls.append(body)


# 根据search/scroll/index的特定响应内容,返回单个对象
def __get_one(response: dict, fetch_source: bool, esid_field: [bool, str], explain_field: str, score_field: str):
	"""
	根据search/scroll/index的特定响应内容,返回对象
	:param response: search/scroll特定响应内容,index的响应内容
	:param esid_field: 是否要返回esid字段/返回的esid字段名[None/False则不返回esid字段,True返回字段名为esid/"xxx"返回字段名为xxx]
	:param score_field:
	:param explain_field:
	:return: dict对象
	"""
	if __ES_SOURCE_FIELD in response:
		model = response[__ES_SOURCE_FIELD]
	else:
		model = {}
	if explain_field:
		model[explain_field] = response.get(__ES_EXPLAIN_FIELD)
	if score_field:
		model[score_field] = response.get(ES_SCORE_FIELD)
	if fetch_source:
		inner_hits_dict = response.get("inner_hits")
		if inner_hits_dict:
			for k, v_d in inner_hits_dict.items():
				if k in model:
					hits_chi = v_d.get("hits")
					v = None
					if hits_chi:
						hits_chi2 = hits_chi.get("hits")
						if hits_chi2:
							v = [i.get("_source") for i in hits_chi2]
					model[k] = v
	if esid_field:
		model[isinstance(esid_field, bool) and ESID_FIELD or esid_field] = response[__ES_ID_FIELD]
	if model:
		return model


# 根据response,返回集合对象
def __get_list_from_search_response(response, fetch_source: bool, esid_field: [bool, str], score_field: str, *,
									explain_field: str = None, ls=None):
	"""
	# 根据response,返回集合对象
	:param response:
	:param esid_field: 是否要返回esid字段/返回的esid字段名[None/False则不返回esid字段,True返回字段名为esid/"xxx"返回字段名为xxx]
	:param score_field:
	:param explain_field:
	:param ls: 返回的集合对象
	:return:
	"""
	coll = filter(bool, map(lambda d: __get_one(d, fetch_source, esid_field, explain_field, score_field), response[__ES_HITS_FIELD][__ES_HITS_FIELD]))
	if ls is None:
		ls = list(coll)
		if ls:
			return ls
		return
	ls.extend(coll)
	return ls


# 根据response,返回集合对象
def __get_list_from_search_response2(response, *, ls=None):
	"""
	# 根据response,返回集合对象
	:param response:
	:param esid_field: 是否要返回esid字段/返回的esid字段名[None/False则不返回esid字段,True返回字段名为esid/"xxx"返回字段名为xxx]
	:param score_field:
	:param explain_field:
	:param ls: 返回的集合对象
	:return:
	"""
	hits = response[__ES_HITS_FIELD][__ES_HITS_FIELD]
	if ls is None:
		ls = [d[__ES_ID_FIELD] for d in hits]
		if ls:
			return ls
		return
	for d in hits:
		ls.append(d[__ES_ID_FIELD])
	return ls


# 获取索引别名
def __get_index_new(index):
	"""
	获取索引别名
	:param index:
	:return:
	"""
	return index.endswith(__ALIAS_SUFFIX) and index or "{}{}".format(index, __ALIAS_SUFFIX)


# 获取search/scroll的响应对象的总条数
def __get_es_total(response):
	"""
	获取search/scroll的响应对象的总条数
	:param response: search/scroll的响应对象,dict类型
	:return:
	"""
	return response[__ES_HITS_FIELD][__ES_TOTAL_FIELD]["value"]


def __get_es_scroll_id(response):
	"""
	获取search/scroll响应对象的scroll_id
	:param response: search/scroll的响应对象,dict类型
	:return:
	"""
	return response[__ES_SCROLL_ID_FIELD]


def __get_scroll_response(es: Elasticsearch, scroll_id, scroll_time):
	scroll_response = es.scroll(scroll_id=scroll_id, scroll=scroll_time)
	return scroll_response


def __get_esid_str(esid):
	if esid:
		return isinstance(esid, str) and esid or str(esid)
	raise_runtime_error("esid禁止为空!")


def __get_show_fields(show_fields):
	"""

	:param show_fields: 单个字段:str,多个字段:list/tuple
	:return:
	"""
	if show_fields:
		if isinstance(show_fields, str):
			if show_fields == NULL_VALUE:
				return None
			elif "," in show_fields:
				return show_fields.split(",")
		return show_fields
	return []


def __get_fetct_source(show_fields):
	"""

	:param show_fields: 单个字段:str,多个字段:list/tuple
	:return:
	"""
	# if show_fields and show_fields == NULL_VALUE:
	# 	return False
	# return True
	if show_fields:
		if isinstance(show_fields, str) and show_fields in __FIELDS_ESID_OR_NONE:
			return False
		if isinstance(show_fields, Sequence) and len(show_fields) < 2 and show_fields[0] in __FIELDS_ESID_OR_NONE:
			return False
	return True


def __get_update_body(body: dict, upsert):
	return {"doc": body, "doc_as_upsert": upsert}

def __is_success(respose: dict, *, detect_noop=False):
	if detect_noop:
		return respose["_shards"]["successful"] > 0 or respose["result"] == "noop"
	return respose["_shards"]["successful"] > 0


def __get_esid(d, *, raise_errors=True):
	if ESID_FIELD in d:
		return d.pop(ESID_FIELD)
	if __ES_ID_FIELD in d:
		return d.pop(__ES_ID_FIELD)
	if raise_errors:
		raise_runtime_error("{}中不存在{},且不存在{}!".format(d, ESID_FIELD, __ES_ID_FIELD))


def get_new_esid():
	"""
	获取新esid
	:return:
	"""
	return uuid1().hex


def get_datetime_now():
	return datetime.now()


def refresh(indexes: [str, tuple, list], *, print_run_msg=True, es=None, host=None):
	"""
	刷新索引
	:param indexes: str/list/tuple类型,str类型可以是1个,也可以是多个,多个用','分割
	:param print_run_msg:
	:param es:
	:param host:
	:return:
	"""
	if not es:
		es = __get_client(host=host)
	r = es.indices.refresh(indexes)
	if print_run_msg:
		print()
		print("{}\trefresh:{}索引,结果:{}".format(get_datetime_now(), indexes, r))
		print()


def clear_scroll_id(scroll_id, print_run_msg, es=None, host=None):
	"""
	删除scroll_id
	:param es:
	:param scroll_id:
	:return:
	"""
	if not es:
		es = __get_client(host=host)
	if print_run_msg:
		print("clear_scroll_id:{}".format(scroll_id))
	try:
		es.clear_scroll(scroll_id=scroll_id)
	except Exception as e:
		print(e)
		print("clear_scroll_id【{}】失败!".format(scroll_id))


def insert_or_replace(index: str, d: dict, *, es_dict=False, refresh=RefreshPolicy.IMMEDIATE, host=None):
	"""
	插入/替换

	需要bulk线程池(9578)
	:param index:
	:param d:
	:param refresh:
	:param host:
	:return:
	"""
	if es_dict:
		d2 = d["_source"]
		esid = d[__ES_ID_FIELD]
	else:
		d2 = d
		esid = __get_esid(d)
	es = __get_client(host=host)
	index = __get_index_new(index)
	dd = es.index(index, d2, id=esid, refresh=refresh.value, require_alias=__require_alias)
	return __is_success(dd)


def delete(index, esid, *, refresh=RefreshPolicy.IMMEDIATE, host=None):
	"""
	根据esid删除

	需要bulk线程池(11712)
	:param index:
	:param esid:
	:param refresh:
	:param host:
	:return:
	"""
	es = __get_client(host=host)
	# {"_index": "atest", "_shards": {"successful": 1, "failed": 0, "total": 1}, "_version": 2, "_type": "atest", "found": True, "_id": "AWFk0pot5H9Ilyn8QlBG"}
	try:
		return __is_success(es.delete(index, esid, refresh=refresh.value))
	except:
		return False


def __update_core(index: str, d: dict, *, upsert=False, refresh=RefreshPolicy.IMMEDIATE, retry_on_conflict=False):
	index = __get_index_new(index)
	esid = __get_esid(d)
	es = __get_client()
	retry = retry_on_conflict and 2 or 0
	a = es.update(index, esid, body=__get_update_body(d, upsert), refresh=refresh.value, retry_on_conflict=retry, require_alias=__require_alias)
	return __is_success(a, detect_noop=True)


def update(index: str, d: dict, *, refresh=RefreshPolicy.IMMEDIATE, retry_on_conflict=False, host=None):
	return __update_core(index, d, upsert=False, refresh=refresh, retry_on_conflict=retry_on_conflict)


def update_or_insert(index: str, d: dict, *, refresh=RefreshPolicy.IMMEDIATE, retry_on_conflict=False):
	return __update_core(index, d, upsert=True, refresh=refresh, retry_on_conflict=retry_on_conflict)


def bulk(index: str, op_type: OpType, coll, *, es_dict=False, refresh=RefreshPolicy.NONE, upsert=True, ignore_errors=False, es=None, desc=False, host=None):
	"""
	OpType.INDEX:[{"esid": str(i), "address": "yymf5", "name": "wkz"}]
	OpType.UPDATE:[{"esid": str(i), "address": "yymf5", "name": "wkz"}]
	OpType.DELETE:[1,2,3,4,5]
	:param index: 索引名
	:param op_type: 操作方式
	:param coll: tuple/list/set/生成器,里面是dict/str对象,其它可选参数:("_op_type", "_index", "_parent", "_percolate", "_routing", "_timestamp", "_version", "_version_type", "_id", "_retry_on_conflict", "pipeline")
	:param upsert: 修改时不存在的esid是否新增
	:return:
	"""
	index = __get_index_new(index)
	if not es:
		es = __get_client(host=host)
	op_type_value = op_type.value
	if op_type == OpType.DELETE:
		func_transfer = lambda id: {"_op_type": op_type_value, "_index": index, __ES_ID_FIELD: __get_esid_str(id)}
	else:
		if op_type == OpType.UPDATE:
			def f2(d):
				return {"_op_type": op_type_value, "_index": index, __ES_ID_FIELD: __get_esid_str(__get_esid(d)), "doc": d, "doc_as_upsert": upsert}
		elif op_type == OpType.INDEX:
			def f2(d):
				d["_op_type"] = op_type_value
				d["_index"] = index
				d[__ES_ID_FIELD] = __get_esid_str(__get_esid(d))
				return d
		else:
			raise_runtime_error("op_type参数错误")
		if es_dict:
			def f1(d):
				d2 = d["_source"]
				d2[__ES_ID_FIELD] = d[__ES_ID_FIELD]
				return d2

			func_transfer = lambda d: f2(f1(d))
		else:
			func_transfer = f2

	response = helpers.streaming_bulk(es, map(func_transfer, coll), refresh=refresh.value, raise_on_error=not ignore_errors, raise_on_exception=not ignore_errors, require_alias=__require_alias)
	result = [t for t in response if not t or not t[0]]
	# print(t)
	# (True, {'index': {'_primary_term': 1, '_version': 1, '_type': 'atest', '_seq_no': 84, '_shards': {'total': 1, 'successful': 1, 'failed': 0}, 'result': 'created', '_id': '11', 'status': 201, 'forced_refresh': True, '_index': 'atest'}})
	# (True, {'index': {'forced_refresh': True, '_shards': {'total': 1, 'successful': 1, 'failed': 0}, '_primary_term': 1, '_id': '1', '_index': 'atest', '_seq_no': 69, '_version': 2, 'result': 'updated', '_type': 'atest', 'status': 200}})
	# (True, {'update': {'_primary_term': 1, '_version': 2, '_type': 'atest', '_seq_no': 86, '_shards': {'total': 1, 'successful': 1, 'failed': 0}, 'result': 'updated', '_id': '11', 'status': 200, 'forced_refresh': True, '_index': 'atest'}})
	# (True, {'delete': {'forced_refresh': True, '_seq_no': 73, '_version': 4, 'result': 'deleted', '_type': 'atest', 'status': 200, '_index': 'atest', '_shards': {'total': 1, 'successful': 1, 'failed': 0}, '_primary_term': 1, '_id': '1'}})
	if result:
		return result


def __get_total_hits(b):
	return b and __hits_max or __hits_default


def get_page(index, *, queries=None, page_size=10, scroll_size=SCORE_SIZE_1000, show_fields: [str, tuple, list] = None,
			 orders: [str, dict, OrderedDict] = None, esid_field: [bool, str] = True, search_values: list = None, is_scroll=True, result=None,
			 score_field: [bool, str] = None, explain_field: [bool, str] = None, return_total=False,
			 print_run_msg=False, page_index=1, host=None):
	"""

	:param index: 单个索引名,str类型
	:param queries: 查询对象,单个Query对象或者list/tuple/set对象【里面元素是Query】
	:param page_size: 每页条数【<0:获取全部数据,且page_index无效;=0只获取总条数;>0获取分页数据】
	:param orders: 排序参数,key是排序字段,value是排序方式【真:升序;其它:降序】
	:param show_fields: 返回字段名,单个字段:str,多个字段:list/tuple
	:param is_scroll: 索引数据量小的時候好用,数据量大的话,直接爆炸!!!
	:param score_field: 分数字段,如果有,则包含分数
	:param explain_field: 分数计算细节字段,,如果有,则返回
	:param page_index: 分页参数
	:return:
	"""
	es = __get_client(host=host)
	index = __get_index_new(index)
	query = __get_query(queries, show_fields=show_fields)
	sorts = __get_sorts(orders)
	body = __get_body(query=query, orders=sorts, after_values=search_values)
	fetct_source = __get_fetct_source(show_fields)
	show_fields = __get_show_fields(show_fields)
	track_total_hits = __get_total_hits(return_total or is_scroll)
	if print_run_msg:
		dt1 = print_datetime_now("开始获取{}索引数据".format(index))
	if page_size < 0 or page_index * page_size > MAX_SIZE_PAGE:
		if is_scroll:
			body = __add_sort_doc(sorts, body)
			search_response = es.search(index=index, body=body, size=scroll_size, _source=fetct_source,
										_source_includes=show_fields, scroll=__ES_SCROLL_TIME_2M, explain=False, track_total_hits=track_total_hits)
			result = __get_list_from_search_response(search_response, fetct_source, esid_field, score_field, ls=result)
			if not result:
				return
			if page_size < 0:
				total = __get_es_total(search_response)
			else:
				total = page_index * page_size
			if print_run_msg:
				current_size = len(result)
				print_datetime_now("总条数:{},当前次数:{},已获取条数:{}".format(total, 0, current_size))
			scroll_id = __get_es_scroll_id(search_response)
			if total > scroll_size:
				times = get_ceil(total, size=scroll_size) - 1
				if print_run_msg:
					for i in range(1, times + 1):
						scroll_response = __get_scroll_response(es, scroll_id, __ES_SCROLL_TIME_2M)
						__get_list_from_search_response(scroll_response, fetct_source, esid_field, score_field, ls=result)
						current_size = len(result)
						print_datetime_now("总条数:{},当前次数:{}/{},已获取条数:{}".format(total, i, times, current_size))
				else:
					for i in range(1, times + 1):
						scroll_response = __get_scroll_response(es, scroll_id, __ES_SCROLL_TIME_2M)
						__get_list_from_search_response(scroll_response, fetct_source, esid_field, score_field, ls=result)
			clear_scroll_id(scroll_id, print_run_msg, es=es)
		else:
			if orders:
				field_after = isinstance(orders, str) and (orders,) or tuple(orders.keys())
			else:
				# field_after = (ESID_FIELD,)
				# if not body:
				# 	body = {}
				# body["sort"] = __sort_esid
				raise_runtime_error("必须要有排序字段!!!")
			i = -1
			current_size = 0
			is_stop = False
			result = []
			while True:
				if page_size > 0:
					t = page_size - current_size
					if scroll_size >= t:
						scroll_size = t
						is_stop = True
				search_response = es.search(index=index, body=body, size=scroll_size, _source=fetct_source, _source_includes=show_fields, track_total_hits=track_total_hits)
				__get_list_from_search_response(search_response, fetct_source, esid_field, score_field, ls=result)
				if result and len(result) != current_size:
					current_size = len(result)
					if print_run_msg:
						i += 1
						total = __get_es_total(search_response)
						times = get_ceil(total, size=scroll_size) - 1
						print_datetime_now("总条数:{},当前次数:{}/{},已获取条数:{}".format(total, i, times, current_size))
					is_end = len(result) % scroll_size
					if not is_end:
						d = result[-1]
						search_values = [d.get(o) for o in field_after]
						body[__ES_SEARCH_AFTER_FIELD] = search_values
					if not is_end and not is_stop:
						continue
				break
	else:
		search_response = es.search(index=index, body=body, from_=__get_from_index(page_index, page_size),
									size=page_size, _source=fetct_source, _source_includes=show_fields,
									explain=bool(explain_field), track_total_hits=track_total_hits)
		result = __get_list_from_search_response(search_response, fetct_source, esid_field, score_field,
												 explain_field=explain_field, ls=result)
		if print_run_msg:
			current_size = result and len(result) or 0
			print_datetime_now("总条数:{},当前次数:{},已获取条数:{}".format(current_size, 0, current_size))
	if print_run_msg:
		print_datetime_now("结束获取{}索引{}数据".format(index, current_size), dt_first=dt1)
	if return_total:
		return result, __get_es_total(search_response)
	return result


def dict_merge(*args: dict, coll=None, func_replace=None) -> dict:
	if func_replace:
		def func(hm, d):
			if d and d != hm:
				for k, v in d.items():
					if k in hm:
						func_replace(hm, k, v)
					else:
						hm[k] = v
			return hm
	else:
		def func(hm, d):
			if d and d != hm:
				hm.update(d)
			return hm
	if args:
		coll = args
	return reduce(func, coll)


def __getAbstractAggregationBuilder(total: int, agg_builder_service: AggBuilderService, agg: dict, *, dict_after=False):
	if isinstance(agg_builder_service, AggAggregate):
		dict_merge(agg, agg_builder_service.to_x_content())
	elif isinstance(agg_builder_service, AggGroup):
		dict_merge(agg, agg_builder_service.to_x_content(total=total))
	elif isinstance(agg_builder_service, AggComposite):
		if dict_after:
			t = agg_builder_service.to_x_content(total=total, dict_after=dict_after)
			dict_merge(agg, t[0])
			return t[1]
		else:
			dict_merge(agg, agg_builder_service.to_x_content(total=total))


def __get_aggregate_value(v):
	if v == "NaN":
		return None
	return v


def __getAggregationsValue_aggregate(aggregations: dict, agg: AggAggregate, result: dict):
	alias = agg.getAlias()
	aggregation = aggregations.pop(alias, None)
	aggregate_type = agg.aggregateType
	if aggregation:
		if aggregate_type in AGGREGATETYPES or aggregate_type == AggregateType.COUNT_DISTINCT:
			result[alias] = __get_aggregate_value(aggregation.get("value"))
		elif aggregate_type == AggregateType.MEDIAN:
			values = aggregation.get("values")
			if values:
				result[alias] = __get_aggregate_value(values.get("50.0"))
		elif aggregate_type == AggregateType.PERCENTILES:
			values = aggregation.get("values")
			if values:
				for k1 in agg.other:
					flag = False
					for k, v in values.items():
						if float(k) == k1:
							flag = True
							break
					if flag:
						values.pop(k)
						values[k1] = __get_aggregate_value(v)
					else:
						# values[k]=__get_aggregate_value(v)
						pass
				result[alias] = values
	else:
		if aggregate_type == AggregateType.MAX_OTHER or aggregate_type == AggregateType.MIN_OTHER:
			alias_other = agg.aliasOther
			aggregation = aggregations.pop(alias_other, None)
			if aggregation:
				buckets = aggregation.get("buckets")
				if buckets:
					bucket = buckets[0]
					result[alias_other] = __get_aggregate_value(bucket.get("key"))
					alias_value = bucket.get(alias)
					if alias_value:
						result[alias] = __get_aggregate_value(alias_value.get("value"))


def __getAggregationsValue_aggregate_single_group(alias, aggregations: dict):
	return [o.get("key") for o in aggregations[alias].get("buckets")]


def __getAggregationsValue_group_aggregate_core(alias, aggregations, agg: AggGroup, func):
	ls = []
	agg_builder_services = agg.aggBuilderServices
	for d in aggregations[alias].get("buckets"):
		d2 = func(d.get("key"), alias)
		ls.append(d2)
		for agg2 in agg_builder_services:
			__getAggregationsValue_aggregate(d, agg2, d2)
	return ls


def __getAggregationsValue_single_group_aggregate(alias, aggregations, agg: AggGroup):
	return __getAggregationsValue_group_aggregate_core(alias, aggregations, agg, lambda val, alias: {alias: val})


def __getAggregationsValue_multiple_group_composite_aggregate(alias, aggregations, agg):
	return __getAggregationsValue_group_aggregate_core(alias, aggregations, agg, lambda val, alias: val)


def copys(obj):
	return copy(obj)


def __getAggregationsValue_aggregate_multiple_group_core(hm, alias, d, ls):
	if hm is None:
		hm = {}
		ls.append(hm)
	hm[alias] = d.pop("key", None)
	for k, v in d.items():
		if isinstance(v, dict):
			buckets = v.get("buckets")
			if len(buckets) > 1:
				for d2 in buckets[1:]:
					hm2 = copys(hm)
					ls.append(hm2)
					__getAggregationsValue_aggregate_multiple_group_core(hm2, k, d2, ls)
			__getAggregationsValue_aggregate_multiple_group_core(hm, k, buckets[0], ls)
			break


def __getAggregationsValue_aggregate_multiple_group(alias, aggregations):
	ls = []
	for d in aggregations[alias].get("buckets"):
		__getAggregationsValue_aggregate_multiple_group_core(None, alias, d, ls)
	return ls


def __getAggregationsValue_aggregate_multiple_group_aggregate_core(hm: dict, alias, d: dict, ls, aliases: set,
																   agg_builder_services):
	if hm is None:
		hm = {}
		ls.append(hm)
	hm[alias] = d.pop("key", None)
	if aliases.intersection(d.keys()):
		for agg in agg_builder_services:
			__getAggregationsValue_aggregate(d, agg, hm)
	else:
		for k, v in d.items():
			if isinstance(v, dict) and "buckets" in v:
				buckets = v.get("buckets")
				if len(buckets) > 1:
					for d2 in buckets[1:]:
						hm2 = copys(hm)
						ls.append(hm2)
						__getAggregationsValue_aggregate_multiple_group_aggregate_core(hm2, k, d2, ls, aliases,
																					   agg_builder_services)
				__getAggregationsValue_aggregate_multiple_group_aggregate_core(hm, k, buckets[0], ls, aliases,
																			   agg_builder_services)
				break


def __getAggregationsValue_aggregate_multiple_group_aggregate(alias, aggregations, agg):
	agg_builder_services = agg.aggBuilderServices
	ls = []
	aliases = {o.getAlias() for o in agg_builder_services}
	for d in aggregations[alias].get("buckets"):
		__getAggregationsValue_aggregate_multiple_group_aggregate_core(None, alias, d, ls, aliases,
																	   agg_builder_services)
	return ls


# def __get_aggs_from_search_response(response, skip, *agg_builders: AggBuilderService, result=None):
# 	aggregations = response["aggregations"]
# 	b = result is None
# 	if b:
# 		result = {}
# 	if len(agg_builders) > 1:
# 		flag = False
# 	else:
# 		flag = True
# 	for agg in agg_builders:
# 		get_type = agg.getType()
# 		alias = agg.getAlias()
# 		if alias in aggregations:
# 			if get_type == AggregationType.MULTIPLE_GROUP_COMPOSITE:
# 				ls = __getAggregationsValue_aggregate_single_group(alias, aggregations)
# 			elif get_type == AggregationType.MULTIPLE_GROUP_COMPOSITE_AGGREGATE:
# 				ls = __getAggregationsValue_multiple_group_composite_aggregate(alias, aggregations, agg)
# 			if b:
# 				if flag:
# 					if skip:
# 						return ls[skip:]
# 					return ls
# 				result[alias] = ls
# 			else:
# 				if flag:
# 					if skip:
# 						raise_runtime_error("禁止分页!")
# 					result.extend(ls)
# 					return result
# 				if alias in result:
# 					result[alias].extend(ls)
# 				else:
# 					result[alias] = ls
# 	return result
def __get_aggs_from_search_response(response, skip, *agg_builders: AggBuilderService, result=None):
	aggregations = response["aggregations"]
	b = result is None
	if b:
		result = {}
	if len(agg_builders) > 1:
		flag = False
	else:
		flag = True
	for agg in agg_builders:
		get_type = agg.getType()
		alias = agg.getAlias()
		if alias in aggregations:
			if get_type == AggregationType.AGGREGATE:
				__getAggregationsValue_aggregate(aggregations, agg, result)
			else:
				if get_type == AggregationType.SINGLE_GROUP:
					ls = __getAggregationsValue_aggregate_single_group(alias, aggregations)
				elif get_type == AggregationType.SINGLE_GROUP_AGGREGATE:
					ls = __getAggregationsValue_single_group_aggregate(alias, aggregations, agg)
				elif get_type == AggregationType.MULTIPLE_GROUP_COMPOSITE:
					ls = __getAggregationsValue_aggregate_single_group(alias, aggregations)
				elif get_type == AggregationType.MULTIPLE_GROUP_COMPOSITE_AGGREGATE:
					ls = __getAggregationsValue_multiple_group_composite_aggregate(alias, aggregations, agg)
				elif get_type == AggregationType.MULTIPLE_GROUP:
					ls = __getAggregationsValue_aggregate_multiple_group(alias, aggregations)
				elif get_type == AggregationType.MULTIPLE_GROUP_AGGREGATE:
					ls = __getAggregationsValue_aggregate_multiple_group_aggregate(alias, aggregations, agg)
				if b:
					if flag:
						if skip:
							return ls[skip:]
						return ls
					result[alias] = ls
				else:
					if flag:
						if skip:
							raise_runtime_error("禁止分页!")
						result.extend(ls)
						return result
					if alias in result:
						result[alias].extend(ls)
					else:
						result[alias] = ls
	return result


def get_aggregations(index, *aggBuilders: AggBuilderService, queries=None, page_index=1, page_size=10, host=None):
	"""

	:param index:
	:param aggBuilders: 分组聚合函数,多个时,page_index失效,只有page_size有效
	:param queries:
	:param page_index: aggBuilders只有1个时,且page_size>0有效
	:param page_size:
	:param host:
	:return:
	"""
	if not page_size:
		raise_runtime_error("page_size禁止为0!!!")
	agg = {}
	skip = 0
	if len(aggBuilders) > 1:
		for aggBuilder in aggBuilders:
			__getAbstractAggregationBuilder(page_size, aggBuilder, agg)
	else:
		if page_size < 0:
			total = -1
		else:
			if page_index < 2:
				total = page_size
			else:
				total = page_index * page_size
				skip = total - page_size
		__getAbstractAggregationBuilder(total, aggBuilders[0], agg)
	if not agg:
		raise_runtime_error("aggBuilders错误!!!")
	query = __get_query(queries)
	body = __get_body(query=query, aggregations=agg)
	es = __get_client(host=host)
	index = __get_index_new(index)
	search_response = es.search(index=index, body=body, size=0)
	result = __get_aggs_from_search_response(search_response, skip, *aggBuilders)
	return result


def get_aggregations2(index, aggBuilder: AggComposite, *, queries=None, scroll_size=1000, host=None, print_run_msg=True):
	"""

	:param index:
	:param aggBuilder: 分组聚合函数,多个时,page_index失效,只有page_size有效
	:param queries:
	:param page_index: aggBuilders只有1个时,且page_size>0有效
	:param page_size:
	:param host:
	:return:
	"""
	page_size = aggBuilder.size
	if not page_size:
		raise_runtime_error("page_size禁止为0!!!")
	agg = {}
	skip = 0
	total = page_size < 0 and -1 or page_size
	scroll_size = get_max_size(page_size, scroll_size)
	dict_after = __getAbstractAggregationBuilder(scroll_size, aggBuilder, agg, dict_after=True)
	if not agg:
		raise_runtime_error("aggBuilders错误!!!")
	query = __get_query(queries)
	body = __get_body(query=query, aggregations=agg)
	es = __get_client(host=host)
	index = __get_index_new(index)

	field_after = tuple(map(lambda o: o.getAlias(), aggBuilder.sources))
	i = -1
	current_size = 0
	is_stop = False
	result = []
	while True:
		if page_size > 0:
			t = page_size - current_size
			if scroll_size >= t:
				scroll_size = t
				is_stop = True
		search_response = es.search(index=index, body=body, size=0)
		__get_aggs_from_search_response(search_response, skip, aggBuilder, result=result)
		if result and len(result) != current_size:
			current_size = len(result)
			if print_run_msg:
				i += 1
				if total > 0:
					times = get_ceil(total, size=scroll_size) - 1
					print_datetime_now("总条数:{},当前次数:{}/{},已获取条数:{}".format(total, i, times, current_size))
				else:
					print_datetime_now("当前次数:{},已获取条数:{}".format(i, current_size))
			is_end = len(result) % scroll_size
			if not is_end:
				dict_after[__ES_AFTER_FIELD] = dict_get_by_key(result[-1], keys=field_after)
			if not is_end and not is_stop:
				continue
		break
	return result


def get_exist(index: str, esid: str, host=None) -> bool:
	es = __get_client(host=host)
	index = __get_index_new(index)
	return es.exists(index, doc_type=__doc_type_value, id=esid)


def get_count(index, *, queries=None, host=None) -> int:
	"""
	查询条数
	:param index:
	:param queries:
	:param host:
	:return:
	"""
	es = __get_client(host=host)
	index = __get_index_new(index)
	body = __get_body(query=__get_query(queries))
	# {'count': 195434, '_shards': {'total': 5, 'successful': 5, 'skipped': 0, 'failed': 0}}
	return es.count(body=body, index=index)["count"]


def get_one(index, arg, *, show_fields: [str, tuple, list] = None, orders: [str, dict, OrderedDict] = None, esid_field: [bool, str] = ESID_FIELD, score_field: [bool, str] = None,
			explain_field: [bool, str] = None, host=None):
	"""
	根据esid/queries查询
	:param index:
	:param arg: esid/queries
	:param show_fields:
	:param esid_field: 是否要返回esid字段/返回的esid字段名[None/False则不返回esid字段,True返回字段名为esid/"xxx"返回字段名为xxx]
	:param score_field:
	:param explain_field:
	:param host:
	:return: dict
	"""
	es = __get_client(host=host)
	index = __get_index_new(index)
	fetct_source = __get_fetct_source(show_fields)
	show_fields = __get_show_fields(show_fields)

	if isinstance(arg, Query):
		query = __get_query(arg)
		sorts = __get_sorts(orders)
		body = __get_body(query=query, orders=sorts)
		search_response = es.search(index=index, body=body, size=1, _source=fetct_source, _source_includes=show_fields)
		hits = search_response[__ES_HITS_FIELD][__ES_HITS_FIELD]
		if hits:
			return __get_one(hits[0], fetct_source, esid_field, explain_field, score_field)
	else:
		try:
			# esid不存在,则抛异常
			get_response = es.get(index=index, id=arg, _source=fetct_source, _source_includes=show_fields)
			return __get_one(get_response, fetct_source, esid_field, explain_field, score_field)
		except NotFoundError:
			return


def oper_func_by_query(index, func_ls, *, queries=None,
					   show_fields: [str, tuple, list] = None, size=-1, scroll_size=SCORE_SIZE_1000,
					   esid_field: [bool, str] = ESID_FIELD, score_field: [bool, str] = None, print_run_msg=True,
					   host=None):
	"""

	:param index:
	:param func_ls: 对当前获取内容处理的函数,输入参数:(ls,*args),ls:list对象,*args任意类型,返回结果为True,则中止遍历
	:param print_run_msg: 是否显示当前次数
	:param queries:
	:param orders:
	:param show_fields: list/tuple/str
	:param scroll_size:
	:param score_field:
	:return:
	"""
	if not size:
		return
	index = __get_index_new(index)
	es = __get_client(host=host)
	query = __get_query(queries)
	body = __get_body(query=query)
	fetct_source = __get_fetct_source(show_fields)
	show_fields = __get_show_fields(show_fields)
	current_size = 0
	if print_run_msg:
		dt1 = print_datetime_now("开始获取{}索引数据".format(index))
	if size < 0 or size > scroll_size:
		body = __add_sort_doc(None, body)
		search_response = es.search(index=index, body=body, size=scroll_size, _source=fetct_source, _source_includes=show_fields, scroll=__ES_SCROLL_TIME_5M, track_total_hits=__hits_max)
		scroll_id = __get_es_scroll_id(search_response)
		result = __get_list_from_search_response(search_response, fetct_source, esid_field, score_field)
		if result:
			total = __get_es_total(search_response)
			if print_run_msg:
				current_size = len(result)
				print_datetime_now("总条数:{},当前次数:{},已获取条数:{}".format(total, 0, current_size))
			# 判断是否中止
			is_interrupt = func_ls(result)
			if not is_interrupt:
				if total > scroll_size:
					times = get_ceil(total, size=scroll_size) - 1
					for i in range(1, times + 1):
						scroll_response = __get_scroll_response(es, scroll_id, __ES_SCROLL_TIME_5M)
						result = __get_list_from_search_response(scroll_response, fetct_source, esid_field, score_field)
						current_size += len(result)
						if print_run_msg:
							print_datetime_now("总条数:{},当前次数:{}/{},已获取条数:{}".format(total, i, times, current_size))
						# 执行处理函数之後,根据返回值判断是否中止
						is_interrupt = func_ls(result)
						if is_interrupt or (size > 0 and current_size >= size):
							break
		clear_scroll_id(scroll_id, print_run_msg, es=es)
	else:
		search_response = es.search(index=index, body=body, size=size, _source=fetct_source,
									_source_includes=show_fields)
		result = __get_list_from_search_response(search_response, fetct_source, esid_field, score_field)
		if result:
			if print_run_msg:
				current_size = len(result)
				print_datetime_now("当前次数:{},已获取总条数:{}".format(0, current_size))
			func_ls(result)
	if print_run_msg:
		print_datetime_now("结束获取{}索引{}数据".format(index, current_size), dt_first=dt1)


# queries = Query(QueryType.NE, "consistency_evaluation_pass_top", "非一致性评价")
# agg = AggAggregate(AggregateType.COUNT_DISTINCT, "consistency_evaluation_pass_top")
# aggGroup = AggGroup("associated_accepted_nos_inn").addAggBuilderServices(agg)
#
# obj = get_aggregations("drug_register", aggGroup, queries=queries, page_size=100, page_index=1)
# print(obj)

# # 获取全部
# ls = [AggTerms("drug_id"), AggTerms("inn_id")]
# ls = [AggTerms("drug_id")]
# aggs = [AggAggregate(AggregateType.COUNT_DISTINCT, "company_ids")]
# agg_composite = AggComposite("aaa", ls, aggBuilderServices=aggs)
# queries = None
# obj = get_aggregations2("drug_associated_accepted_nos_inn", agg_composite, queries=queries)
# print(obj)

# queries = Query(QueryType.NE, "consistency_evaluation_pass_top", "非一致性评价")
# ls = [AggTerms("associated_accepted_nos_inn")]
# ls = [AggTerms("associated_accepted_nos_inn"), AggTerms("consistency_evaluation_pass_top")]
# aggs = [AggAggregate(AggregateType.COUNT_DISTINCT, "consistency_evaluation_pass_top")]
# agg_composite = AggComposite("aaa", ls, aggBuilderServices=aggs, size=10)
# obj = get_aggregations2("drug_register", agg_composite, queries=queries)
# # print(obj)
# print(obj)


# queries = Query(QueryType.NE, "consistency_evaluation_pass_top", "非一致性评价")
# agg = AggAggregate(AggregateType.COUNT_DISTINCT, "consistency_evaluation_pass_top")
# aggGroup = AggGroup("associated_accepted_nos_inn").addAggBuilderServices(agg)
#
# obj = get_aggregations("drug_register", aggGroup, queries=queries, page_size=100, page_index=1)
# logging.info(obj)

# # 获取全部
# ls = [AggTerms("drug_id"), AggTerms("inn_id")]
# ls = [AggTerms("drug_id")]
# aggs = [AggAggregate(AggregateType.COUNT_DISTINCT, "company_ids")]
# agg_composite = AggComposite("aaa", ls, aggBuilderServices=aggs)
# queries = None
# obj = get_aggregations2("drug_associated_accepted_nos_inn", agg_composite, queries=queries)
# logging.info(obj)

# queries = Query(QueryType.NE, "consistency_evaluation_pass_top", "非一致性评价")
# ls = [AggTerms("associated_accepted_nos_inn")]
# ls = [AggTerms("associated_accepted_nos_inn"),AggTerms("consistency_evaluation_pass_top")]
# aggs = [AggAggregate(AggregateType.COUNT_DISTINCT, "consistency_evaluation_pass_top")]
# agg_composite = AggComposite("aaa", ls, aggBuilderServices=aggs,size=10)
# obj = get_aggregations2("drug_register", agg_composite, queries=queries)
# # logging.info(obj)
# logging.info(obj)
#
# # es 的基本操作样例
# # count = es_utils.get_count("news")
# # str = '{"title":"111111","esid":"2222"}'
# # es_utils.update("news",d=json.loads(str))
# # results = es_utils.get_count("news",queries=Query(QueryType.EQ,'esid','2222'))
# # results = es_utils.get_page("news",queries=Query(QueryType.EQ,'esid','2222'),page_index=-1,show_fields=['title'])
# # results = es_utils.get_page("drug_ct",page_size=-1, show_fields=['registration_no'])
# # or; 默认是and
# # Query.queries(Query(QueryType.EQ, 'channel_name', '医药自媒体'), Query(QueryType.GE, 'spider_wormtime', 1604851200000),and_perator=False)
# # 指定字段排序
# #pages = get_page(index='business_info', show_fields=['name'], orders={'spider_wormtime': True}, page_size=1000)
# #queries = Query.queries(Query(QueryType.EQ, "search_query.source_es_index", "discover_indication"), nested_fiel
...

# print(get_count("news"))
# print(get_count("news",queries=Query(QueryType.EQ,"esid","123")))
# print(get_count("news",queries=Query(QueryType.IN,"esid","c0e2e4b718200ec2e7cbef082dbc4d97","9a3bc9b999b078fc4d614e3eba609648","abc")))