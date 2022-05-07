from multiprocessing.spawn import import_main_path
import pandas
import time
import datetime

import json
import os


import ijson


# 输入毫秒级的时间，转出正常格式的时间
def timeStamp(timeNum):
    timestamp = timeNum/1000
    date = datetime.datetime.fromtimestamp(timestamp)
    format_date = date.strftime('%Y-%m-%d %H:%M:%S')
    return date


obj=[]
#这里输入你的文件路径
file_name = '2022_03_09.json'
with open(file_name, 'r', encoding='utf-8') as f:
    for object in ijson.items(f,"item"):
        obj.append(object)


article=[]
for ar in obj:
# 输入毫秒级的时间，转出正常格式的时间

    print(type(ar))
    if ar.get('publish_time') == None:
        print('没有时间')
    else:

        publish_time=ar.get('publish_time')
        print(publish_time)
        p=timeStamp(publish_time)

        low=datetime.datetime.fromtimestamp(1640966400)
        high=datetime.datetime.fromtimestamp(1646064000)
    
        if p > low and p<high:
            article.append(ar)
print(len(article))
print(type(article))
    
json_file_path = 'keyword ranking/ti2.json'
json_file = open(json_file_path, mode='w',encoding='utf8')

json.dump(article, json_file, indent=4, ensure_ascii=False)  
# json.dump(save_json_content, json_file, ensure_ascii=False, indent=4) # 保存中文
print('近两个月')