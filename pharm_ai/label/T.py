# encoding: utf-8
'''
@author: zyl
@file: T.py
@time: 2021/7/12 上午12:33
@desc:
'''
import pandas as pd
from pharm_ai.util.utils import Utilfuncs
pdfs_dir = "/home/zyl/disk/PharmAI/pharm_ai/label/pmda/data/pmda-001-pdf/"
s1 = Utilfuncs.list_all_files(pdfs_dir, 'pdf')
pdfs_dir = "/home/zyl/disk/PharmAI/pharm_ai/label/pmda/data/pmda-001-pdf-2/"
s2 = Utilfuncs.list_all_files(pdfs_dir, 'pdf')
pdfs_dir = "/home/zyl/disk/PharmAI/pharm_ai/label/pmda/data/pmda-002-pdf/"
s3 = Utilfuncs.list_all_files(pdfs_dir, 'pdf')
pdfs_dir = "/home/zyl/disk/PharmAI/pharm_ai/label/pmda/data/pmda-002-pdf-2/"
s4 = Utilfuncs.list_all_files(pdfs_dir, 'pdf')
pdfs_dir = "/home/zyl/disk/PharmAI/pharm_ai/label/pmda/data/pmda-003-pdf/"
s5 = Utilfuncs.list_all_files(pdfs_dir, 'pdf')
pdfs_dir = "/home/zyl/disk/PharmAI/pharm_ai/label/pmda/data/pmda-003-pdf-2/"
s6 = Utilfuncs.list_all_files(pdfs_dir, 'pdf')
pdfs_dir = "/home/zyl/disk/PharmAI/pharm_ai/label/pmda/data/pmda-004-pdf/"
s7 = Utilfuncs.list_all_files(pdfs_dir, 'pdf')

print(len(s1+s2+s3+s4+s5+s6+s7))