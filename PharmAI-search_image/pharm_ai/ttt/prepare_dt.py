# encoding: utf-8
'''
@author: zyl
@file: prepare_dt.py
@time: 2021/11/16 14:51
@desc:
'''
import pandas as pd


class PrepareDT:
    def __init__(self):
        pass

    def run(self):
        self.get_dt()

    def get_dt(self):
        file = "/home/zyl/disk/PharmAI/pharm_ai/ttt/data/v1/processed_1109.xlsx"
        df = pd.read_excel(file, "all")  # type:pd.DataFrame
        li = []
        task_id = 10000
        print(len(df))
        for _, sub_df in df.iterrows():
            task_id += 1
            if pd.isna(sub_df['therapy_labels']):
                therapy_labels = []
            else:
                therapy_labels = [[t] for t in sub_df['therapy_labels'].split(',')]

            for k in ['study_title','criteria']:
                task_id += 1
                if pd.isna(sub_df[k]):
                    continue
                li.append({
                    'id': task_id,
                    'data': {
                        'my_text': sub_df[k],
                        'nct_id': sub_df['nct_id'],
                        'therapy_labels': therapy_labels,
                        'type': k,
                    },
                    "annotations":[
                        {
                            "id": "default",
                            "result": [
                                {
                                    "value": {
                                        "taxonomy": therapy_labels
                                    },
                                    "id": "default",
                                    "from_name": "taxonomy",
                                    "to_name": "text",
                                    "type": "taxonomy"
                                }
                            ],
                            "task": task_id,
                        },
                    ]
                })


        import json
        with open("./data/to_label/to_label_1122.json", 'w') as t:
            json.dump(li, t)


if __name__ == '__main__':
    PrepareDT().run()
