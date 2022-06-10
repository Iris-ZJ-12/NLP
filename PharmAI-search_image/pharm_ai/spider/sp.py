# -*- coding: UTF-8 -*-
"""
Description : 
"""
import json
import requests
import datetime

postUrl = 'https://sellercentral.amazon.com/fba/profitabilitycalculator/getafnfee?profitcalcToken=en2kXFaY81m513NydhTZ9sdb6hoj3D'
# payloadData数据
payloadData = {
    'afnPriceStr': 10,
    'currency':'USD',
    'productInfoMapping': {
        'asin': 'B072JW3Z6L',
        'dimensionUnit': 'inches',
    }
}
# 请求头设置
payloadHeader = {
    'Host': 'sellercentral.amazon.com',
    'Content-Type': 'application/json',
}
# 下载超时
timeOut = 25
# 代理
proxy = "183.12.50.118:8080"
proxies = {
    "http": proxy,
    "https": proxy,
}
r = requests.post(postUrl, data=json.dumps(payloadData), headers=payloadHeader)
dumpJsonData = json.dumps(payloadData)
print(f"dumpJsonData = {dumpJsonData}")
res = requests.post(postUrl, data=dumpJsonData, headers=payloadHeader, timeout=timeOut, proxies=proxies, allow_redirects=True)
# 下面这种直接填充json参数的方式也OK
# res = requests.post(postUrl, json=payloadData, headers=header)
print(f"responseTime = {datetime.datetime.now()}, statusCode = {res.status_code}, res text = {res.text}")
