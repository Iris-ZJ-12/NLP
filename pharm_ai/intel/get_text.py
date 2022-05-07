# encoding: utf-8
'''
@author: zyl
@file: get_text.py
@time: 2021/9/24 9:45
@desc:
'''

import ast
import datetime
import io
import os
import requests

import fitz
from pdf2image import convert_from_path


def pyMuPDF_fitz(pdfPath, imagePath):
    startTime_pdf2img = datetime.datetime.now()  # 开始时间

    print("imagePath=" + imagePath)
    pdfDoc = fitz.open(pdfPath)
    for pg in range(pdfDoc.pageCount):
        page = pdfDoc[pg]
        rotate = int(0)
        # 每个尺寸的缩放系数为1.3，这将为我们生成分辨率提高2.6的图像。
        # 此处若是不做设置，默认图片大小为：792X612, dpi=72
        zoom_x = 1.33333333  # (1.33333333-->1056x816)   (2-->1584x1224)
        zoom_y = 1.33333333
        mat = fitz.Matrix(zoom_x, zoom_y).preRotate(rotate)
        pix = page.getPixmap(matrix=mat, alpha=False)

        if not os.path.exists(imagePath):  # 判断存放图片的文件夹是否存在
            os.makedirs(imagePath)  # 若图片文件夹不存在就创建

        pix.writePNG(imagePath + '/' + 'images_%s.png' % pg)  # 将图片写入指定的文件夹内

    endTime_pdf2img = datetime.datetime.now()  # 结束时间
    print('pdf2img时间=', (endTime_pdf2img - startTime_pdf2img).seconds)


def get_ocr_result(pdf):
    images = convert_from_path(pdf)
    rawtext = []
    for i in range(len(images)):
        newimg = images[i]
        buf = io.BytesIO()
        newimg.save(buf, format='JPEG')
        byte_im = buf.getvalue()
        # ocr_api = 'http://101.201.249.176:1990/2txt'
        ocr_api = 'http://localhost/2txt'
        # ocr_api = 'http://localhost/2txt_CV'
        response = requests.post(ocr_api, files={'file': byte_im})
        removed_header_tail = ast.literal_eval(response.text)
        rawtext.extend(removed_header_tail)
    if rawtext:
        rawtext = '\n'.join(rawtext)
    else:
        rawtext = ''
    return rawtext


if __name__ == "__main__":
    pdf = "/home/zyl/disk/PharmAI/pharm_ai/intel/data/to_label/pdfs/982432712c99c12b5401089e8a479cf4.pdf"
    print(repr(get_ocr_result(pdf)))
