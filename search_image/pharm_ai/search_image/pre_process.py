import os
from PyPDF2 import PdfFileReader
import fitz


def get_pdf_name_file(pdf_path):
    # 获取pdf的名字
    pdf_path = pdf_path[pdf_path.rfind("/") + 1: pdf_path.rfind(".")]
    return pdf_path


def get_h_and_w_from_pdf_path(path):
    # 从pdf的路径获取可解析pdf的宽高
    try:
        pdf = PdfFileReader(open(path, 'rb'))
        page_1 = pdf.getPage(0)
        if page_1.get('/Rotate', 0) in [90, 270]:
            return page_1['/MediaBox'][2], page_1['/MediaBox'][3]
        else:
            return page_1['/MediaBox'][3], page_1['/MediaBox'][2]
    except Exception as e:
        print(e)


def pdf_2_img(pdf_path, image_path):
    # 将pdf切成图片
    idx = 1
    pdf_doc = fitz.open(pdf_path)
    for pg in range(pdf_doc.pageCount):
        page = pdf_doc[pg]
        rotate = int(0)
        zoom_x = 2
        zoom_y = 2
        trans = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
        try:
            pix = page.get_pixmap(matrix=trans, alpha=False)
        except Exception as e:
            print(e)
            print("image_path=" + pdf_path)
            continue
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        pix.save(image_path + '/' + '%s.png' % idx)
        idx += 1
