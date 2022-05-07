import fitz
from PIL import Image
import numpy as np
import cv2
import io
import requests
import ast
import re


def get_ratio(page):
    """
    In order to crop the INN name part from the original pdf page.
    We detect the left-top corner of the main text block and calculate the ratio of page width. (ratio = x/page_width)
    input:
        page(pdf page)
    return:
        ratio(float),
        image(converted image from pdf page)
    """
    # load the pdf page as gray image
    rotate = int(0)
    zoom_x, zoom_y = 2, 2
    mat = fitz.Matrix(zoom_x, zoom_y).preRotate(rotate)
    pix = page.getPixmap(matrix=mat, alpha=False)
    image = pix.tobytes()
    image = Image.open(io.BytesIO(image))
    img = np.asarray(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # thresh hold the image with OTSU
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    # dilation image to connect connect adjacent elements
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
    # find contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    im2 = img.copy()
    # remove one column style blocks. keep two column style blocks.
    # we only want the left column of the two column area
    pre_cnt = (-float('inf'), -float('inf'), -float('inf'), -float('inf'))
    contours_list = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if y == pre_cnt[1]:
            contours_list.append((x, y, w, h))
            contours_list.append(pre_cnt)
        pre_cnt = (x, y, w, h)
    # find the biggest block and return the top left corner
    max_cnt = (0, 0, 0, 0)
    for c in contours_list:
        x, y, w, h = c
        if w * h > max_cnt[2] * max_cnt[3]:
            max_cnt = (x, y, w, h)
    x, y, w, h = max_cnt
    ratio = x / im2.shape[1]
    return ratio, im2


def filter_out(res):
    """
    filter out unnecessary info
        1. numbers
        2. upper letters
        3. length smaller than 4
        4. empty string
    """
    ans = []
    for line in res:
        line = line.strip()
        line = re.sub('[#0-9]', '', line)
        if re.search('[A-Z]', line):
            continue
        if len(line) < 4:
            continue
        ans.append(line)
    return ans


def get_left_names(path):
    """
    get the name from readable pdf pages
    input:
            pdf_path(str)
    return:
            text(list)
    """
    # read from stream memory
    doc = fitz.open(stream=path, filetype="pdf")
    res = []
    for i in range(0, doc.pageCount):
        pdf_page = doc[i]
        if 'AMENDMENTS' in pdf_page.getText():
            print(True)
            break
        ratio, _ = get_ratio(pdf_page)
        if ratio < 0.2:
            ratio = 0.36
        left_clip = fitz.Rect(pdf_page.rect.width * 0.00, pdf_page.rect.height * 0.00,
                              pdf_page.rect.width * ratio, pdf_page.rect.height * 1.0)
        left_text = pdf_page.getText(clip=left_clip).split('\n')[1:]
        left_text = [i for i in left_text if len(i) > 1]
        left_text = filter_out(left_text)
        res.extend(left_text)
    if len(res) != 0:
        return res
    else:
        raise Exception(path, "empty")


def ocr_get_left_names(path):
    """
    get the name from non-readable pdf pages using OCR.
    input:
            pdf_path(str)
    return:
            text(list)
    """

    # read from stream memory
    doc = fitz.open(stream=path, filetype="pdf")
    res = []
    for i in range(0, doc.pageCount):
        pdf_page = doc[i]
        ratio, image = get_ratio(pdf_page)
        # if ratio smaller than 0.2, manually set ratio to 0.36
        if ratio < 0.2:
            ratio = 0.36
        # keep left column image[H,W], right part set 255 (white)
        image[0:image.shape[0], int(image.shape[1] * ratio):image.shape[1]] = 255
        left_column = image.copy()
        # resize the image, if too big.
        height, width = left_column.shape[0], left_column.shape[1]
        if height > 1500:
            height = 1400
            width = width * 1400 / height
        if width > 1100:
            width = 990
            height = height * 990 / width
        left_column = cv2.resize(left_column, (int(width), int(height)))
        is_success, left_column = cv2.imencode(".jpg", left_column)
        left_column = left_column.tobytes()
        ocr_api = 'http://localhost/2txt'
        response = requests.post(ocr_api, files={'file': left_column})
        # string to structure format list.  string_list --> list of string
        left_text = ast.literal_eval(response.text)
        left_text = [i for i in left_text if len(i) > 1]
        left_text = filter_out(left_text)
        res.extend(left_text)

    if len(res) != 0:
        return res
    else:
        raise Exception(path, "empty")
