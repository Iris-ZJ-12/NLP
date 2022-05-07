import json
import itertools
import os
import ahocorasick
from typing import List, Union, Optional
from pharm_ai.pom.kwranking import keyword_ranking
from pharm_ai.config import ConfigFilePaths


class Indication:

    def __init__(
            self,
            name: str,
            idx: str,
            parent: Optional[List[str]],
            name_en: Optional[Union[str, List[str]]],
            name_synonyms: Optional[Union[str, List[str]]],
            name_short: Optional[Union[str, List[str]]]
    ):
        self.name = name
        self.id = idx
        self.parent = parent
        self.all_name = [name]
        self.name_2_type = {name: 'standard_name'}
        self.all_parent_ids = []
        self.add_name(name_en, 'name_en')
        self.add_name(name_synonyms, 'name_synonyms')
        self.add_name(name_short, 'name_short')

    def add_name(self, name_list: Optional[Union[str, List[str]]], name_type: str):
        if not name_list:
            return
        if type(name_list) == str:
            name_list = [name_list]
        for name in name_list:
            if name.isdigit():
                continue
            self.all_name.append(name)
            self.name_2_type[name] = name_type


class IndicationMatcher:

    ignore = ['其他', '手术', '营养', '流泪', '手术冲洗', '肠内营养', '消毒', '苔藓']

    def __init__(self):
        self.indications = IndicationMatcher.load_indications()
        self.id_2_ind = {ind.id: ind for ind in self.indications}
        self.all_name = set()
        self.name_2_id = {}
        for ind in self.indications:
            self.all_name |= set(ind.all_name)  # union
            for name in ind.all_name:
                self.name_2_id[name] = ind.id
        self.find_all_parents(self.indications)

    @staticmethod
    def load_indications() -> List[Indication]:
        """
        Load indication objects from file on disk.

        indication file is of format:
        [
            {
                'name': ...,
                'id': ...,
                'parent_indication': ...,
                'name_en': ...,
                'name_short': ...,
                'name_synonyms': ...
            },
            ...
        ]

        Returns:
            List of Indication objects loaded.
        """
        path = os.path.join(ConfigFilePaths.project_dir, 'pom', 'indication.json')
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        inds = []
        for d in data:
            name = d['name']
            if name in IndicationMatcher.ignore or name.isdigit():
                continue
            esid = d['id']
            ind = Indication(
                name,
                esid,
                parent=d.get('parent_indication', None),
                name_en=d.get('name_en', None),
                name_synonyms=d.get('name_synonyms', None),
                name_short=None
            )
            inds.append(ind)
        return inds

    def find_all_parents(self, indications: List[Indication]) -> None:
        """
        Find all parent indications for each indication in one pass.
        Modifies indication objects in place.

        Args:
            indications: List of Indication objects.
        """

        def dfs(ind: Optional[Indication]) -> List[str]:
            if not ind:
                return []
            if ind.id in visited or not ind.parent:
                return ind.all_parent_ids
            all_parent_ids = []
            all_parent_ids.extend(ind.parent)
            for pid in ind.parent:
                all_parent_ids.extend(dfs(self.id_2_ind.get(pid, None)))
            ind.all_parent_ids = all_parent_ids
            return all_parent_ids
            # 可能得到的是名字和id
        visited = set()
        for indication in indications:
            dfs(indication)

    def match(
            self,
            text: str,
            content: str,
            model_outputs: Optional[List[str]] = None,
            allow_string_include: bool = False
    ) -> List[str]:
        """
        Match indications in text, filter parent indication if child exists.
        The candidates are got from text, which will be ranked and filtered based on text and content to get keywords.

        Args:
            text: text to match indication
            model_outputs: model outputs used together to filter results
            allow_string_include: whether allow one indication string included in other strings

        Returns:
            List of indications candidates and keyword results.
        """
        inds_original = []
        inds_normalized = []
        for name in self.all_name:
            inds_original.append(name)
            _name = name.replace('-', '')
            _name = _name.replace(' ', '')
            _name = _name.lower()
            inds_normalized.append(_name)
        _text = text.replace('-', '')
        _text = _text.replace(' ', '')
        _text = _text.lower()

        matched = []
        A = ahocorasick.Automaton()
        for index, word in enumerate(inds_normalized):
            A.add_word(word, (index, word))
        A.make_automaton()

        for item1 in A.iter(_text):
            matched.append(inds_original[item1[1][0]])
            for item2 in A.iter(_text):
                if item1 == item2:
                    continue
                else:
                    e1 = item1[0]
                    e2 = item2[0]
                    s1 = e1 - len(item1[1][1]) + 1
                    s2 = e2 - len(item2[1][1]) + 1
                    if s1 >= s2 and e1 <= e2:
                        if inds_original[item1[1][0]] in matched:
                            matched.remove(inds_original[item1[1][0]])
        matched = set(matched)

        if model_outputs:
            for label in model_outputs:
                if label in self.all_name and label not in matched:
                    matched.add(label)

        # get unique ids for matched indications
        ids = set([self.name_2_id[n] for n in matched])
        # map indications back to ids
        id_2_matched = {esid: [] for esid in ids}
        for m in matched:
            id_2_matched[self.name_2_id[m]].append(m)

        # remove parent indications
        remove = []
        for esid in ids:
            for pid in self.id_2_ind[esid].all_parent_ids:
                if pid in ids:
                    remove.append(pid)
        ids = [esid for esid in ids if esid not in remove]
        refine = list(itertools.chain.from_iterable([id_2_matched[esid] for esid in ids]))
        if model_outputs:
            for label in model_outputs:
                if label not in refine:
                    refine.append(label)

        if not allow_string_include:
            final = []
            for i in range(len(refine)):
                for j in range(len(refine)):
                    if i == j:
                        continue
                    if refine[i] in refine[j]:
                        break
                else:
                    final.append(refine[i])
            refine = final

        if refine:
            a = keyword_ranking.keywords(refine, ids, self.indications)
            refine, ids_kw, scores_kw = a.ranking(text, content)

        return refine


if __name__ == "__main__":
    path1 = os.path.join(ConfigFilePaths.project_dir, 'pom', 'kwranking', 'ran.json')
    with open(path1, 'r+', encoding='UTF-8-sig') as f:
        print("Load str file from {}".format(path1))
        str1 = f.read()
        orig_ar = json.loads(str1)

    for ar in orig_ar:
        if ar.get('abstract') is None:
            text = ar.get('title') + "。"
        else:
            text = ar.get('title') + "。" + ar.get('abstract') + "。"
        content = ar.get('content')
        # text是文章的标题摘要，content是正文

        matcher = IndicationMatcher()
        keywords = matcher.match(text, content, allow_string_include=True)
        # 得到适应症候选词和关键词结果
