# -*- coding: UTF-8 -*-
import os
import pandas as pd

from typing import Optional, Union, List
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


class Task:

    def __init__(self, path: Union[str, os.PathLike]):
        self.name = path.split('/')[-1].split('.')[0]
        self.df = pd.read_excel(path).applymap(str)

    def rename(self, name: str) -> None:
        self.df['prefix'] = self.df['prefix'].replace(self.name, name)
        self.name = name

    def stats(self, split: str = '', group_index: str = 'prefix') -> pd.DataFrame:
        stat = self.df.drop(columns=['input_text'])
        stat = stat.rename(columns={'prefix': 'num'})
        if split:
            stat[['prefix', 'suffix']] = stat['target_text'].str.split(split, 1, expand=True)
            return stat.groupby(group_index).count()
        return stat.groupby('target_text').count()

    def train_test_split(self, test_size: float = 0.15, seed: int = 999) -> (pd.DataFrame, pd.DataFrame):
        train, test = train_test_split(self.df, test_size=test_size, random_state=seed)
        return train, test

    def get_data(self) -> dict:
        return self.df.to_dict('list')


class TaskManager:

    def __init__(self, task_dir: str = 'tasks'):
        self.tasks = {}
        for fn in os.listdir(task_dir):
            if fn.endswith('.xlsx'):
                task = Task(os.path.join(task_dir, fn))
                self.tasks[task.name] = task

    def _get_train_and_eval_df(self, task: str, test_size: float = 0.15, seed: int = 999) \
            -> (pd.DataFrame, pd.DataFrame):
        if task not in self.tasks:
            raise ValueError(f'Invalid task name: {task}, please check again.')
        return self.tasks[task].train_test_split(test_size, seed)

    def get_train_and_eval_df(self, task_name: Optional[Union[str, List[str]]] = None, up_sampling: bool = False) \
            -> (pd.DataFrame, pd.DataFrame):
        if type(task_name) == str:
            return self._get_train_and_eval_df(task_name)

        tasks = task_name
        if task_name is None:
            tasks = self.tasks
        trains, tests = [], []
        for task in tasks:
            train, test = self.get_train_and_eval_df(task)
            trains.append(train)
            tests.append(test)
        train_df = pd.concat(trains, ignore_index=True).sample(frac=1)
        eval_df = pd.concat(tests, ignore_index=True).sample(frac=1)

        if up_sampling:
            prefixes = list(set(train_df['prefix'].tolist()))
            dfs = []
            for prefix in prefixes:
                dfs.append(train_df[train_df['prefix'] == prefix])
            up_sampling_num = max([len(df) for df in dfs])
            up_dfs = []
            for df in dfs:
                if len(df) < up_sampling_num:
                    up_sampling_df = resample(
                        df,
                        replace=True,
                        n_samples=(up_sampling_num - len(df)),
                        random_state=9999
                    )
                else:
                    up_sampling_df = pd.DataFrame()
                up_dfs.append(up_sampling_df)
            dfs.extend(up_dfs)
            train_df = pd.concat(dfs, ignore_index=True).sample(frac=1)
        return train_df, eval_df


if __name__ == '__main__':
    manager = TaskManager()
