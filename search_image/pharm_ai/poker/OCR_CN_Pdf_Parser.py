import os
import json
import time
import os
import requests
import io
import ast
from seq_clean import *
from BioClassifier import SeqClassifier
from tqdm import tqdm
from torch.multiprocessing import Pool, Process, set_start_method

SC = SeqClassifier('/home/zb/disk/pycharm/pharm_ai/poker/bst_resnet18_4.pt')


def get_ocr_result(images, page):
    rawtext = []
    if len(images) < page[1] + 1:
        start_page, end_page = page[0], len(images)
    else:
        start_page, end_page = page[0], page[1]
    for i in range(start_page, end_page):
        newimg = padding_resize(images[i], 980, 1400)
        buf = io.BytesIO()
        newimg.save(buf, format='JPEG')
        byte_im = buf.getvalue()
        ocr_api = 'http://localhost/2txt'
        response = requests.post(ocr_api, files={'file': byte_im})
        removed_header_tail = ast.literal_eval(response.text)[2:-1]
        rawtext.extend(removed_header_tail)
    return rawtext


def extract_CN_from_path(combo):
    pdf_path, output_dir = combo
    st = time.time()
    print('processing', pdf_path)
    pdf_name = os.path.basename(pdf_path)
    images, pdf_path, page = SC.detect_sequence_page(pdf_path)
    if page == 'broken':
        return 'broken', pdf_path
    if len(page) == 0:
        return 'no_page_detected', pdf_path
    try:
        raw_text = get_ocr_result(images, page)
        if len(raw_text) == 0:
            return 'no_page_detected', pdf_path
        filename = os.path.join(output_dir, pdf_name + '.txt')
        txt = open(filename, 'w')
        txt.write(pdf_path + '\n')
        for l in raw_text:
            txt.write(l + '\n')
        txt.close()
        ed = time.time()
        t = ed - st
        print('successful finished:', pdf_path, t)
        return 'success', pdf_path
    except Exception as e:
        print("major errors {}".format(pdf_path))
        print(e)
        return 'major_error', pdf_path


if __name__ == '__main__':

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    pdf_dir = '/home/zb/disk/pycharm/pharm_ai/poker/20211122055413992/pdf'
    pdfs = os.listdir(pdf_dir)
    pdf_list = [os.path.join(pdf_dir, p) for p in pdfs]
    output_dir = '/home/zb/disk/pycharm/pharm_ai/poker/20211122055413992/OCR-2021-raw'

    L = len(pdf_list)
    lists = pdf_list[:L]
    outs = [output_dir] * L
    combo = zip(lists, outs)

    with Pool(4) as p:
        r = list(tqdm(p.imap(extract_CN_from_path, combo), total=L))
        p.close()
        p.join()

    data = {'success': [], 'no_page_detected': [], 'broken': [], 'major_error': []}
    for res in r:
        k, v = res
        data[k].append(v)

    with open(os.path.join(output_dir, 'CN_info_list.json'), 'w') as outfile:
        json.dump(data, outfile, indent=4)


