from PIL import Image
import json
from paddleocr import PaddleOCR
import fitz
import numpy as np
import os


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def save_title(args, json_path, pdf_name):
    # 找出图表框上面一行和下面一行的文本，根据字数过滤之后存进Figure_Title.json
    data_json = json.load(open(json_path, encoding='UTF-8'))
    results = []
    for i in range(0, len(data_json)):
        if data_json[i]['page']['type'] == 'TableRegion' or data_json[i]['page']['type'] == 'ImageRegion' or data_json[i]['page']['type'] == 'Figure' or data_json[i]['page']['type'] == 'Table':
            result = {'Picture_name': {}, 'Predict_Title_1': {}, 'Predict_Title_2': {}}
            id = data_json[i]['page']['id']
            if id == 0:
                if i < len(data_json) - 1:
                    if len(data_json[i + 1]['page']['data']) > 30:
                        data_json[i + 1]['page']['data'] = 'TOO Long'
                    result['Picture_name'] = 'page-' + str(data_json[i]['page']['page_number']) + '-blockid-' + str(data_json[i]['page']['id'])
                    result['Predict_Title_1'] = 'None'
                    result['Predict_Title_2'] = data_json[i + 1]['page']['data']
            elif id > 0:
                if i > 0 and i < len(data_json) - 1:
                    if len(data_json[i + 1]['page']['data']) > 30:
                        data_json[i + 1]['page']['data'] = 'TOO Long'
                    if len(data_json[i - 1]['page']['data']) > 30:
                        data_json[i - 1]['page']['data'] = 'TOO Long'
                    result['Picture_name'] = 'page-' + str(data_json[i]['page']['page_number']) + '-blockid-' + str(data_json[i]['page']['id'])
                    result['Predict_Title_1'] = data_json[i + 1]['page']['data']
                    result['Predict_Title_2'] = data_json[i - 1]['page']['data']
            results.append(result)
    json_data = json.dumps(results, indent=1, ensure_ascii=False, cls=NpEncoder)
    json_result_path = args.file_save_path + '/' + pdf_name + '/Figure_Title.json'
    f = open(json_result_path, 'w')
    f.write(json_data)
    f.close()


def get_text(args, name, pdf_path, file_path, picture_path, p_max):
    # 抽取pdf的文本，存进pdf_text.json，对于可解析的pdf用fitz直接抽取文本，对不可解析的pdf用paddleocr提取文本。
    # f1_path = pdf_path + '/' + name + '.pdf'
    f1_path = pdf_path
    contents = []
    p = 0
    # 打开pdf文件
    doc = fitz.open(f1_path)
    for page in doc:
        p += 1
        words = page.get_text_words()
        for w in words:
            # 位置信息：fitz.Rect(w[:4])
            # w[4]：文本信息
            location_t = fitz.Rect(w[:4])
            text = w[4]
            # print(location_t,text)
            content = {}
            content['page_number'] = p
            content['x0_t'] = location_t[0] * 2
            content['y0_t'] = location_t[1] * 2
            content['x1_t'] = location_t[2] * 2
            content['y1_t'] = location_t[3] * 2
            content['width'] = location_t[3] * 2 - location_t[1] * 2
            # 左上角为坐标原点
            content['type'] = 'text'
            content['text'] = text
            contents.append(content)

    if not contents:
        p = 1
        while p <= p_max:
            try:
                f7_path = file_path + '/' + name + '/pdf_image/' + str(p) + '.png'
                f = open(f7_path)
                print(f)
            except FileNotFoundError:
                f7_path = picture_path + '/' + name + '/pdf_image/' + str(p) + '.png'
            ocr = PaddleOCR(use_angle_cls=True, use_gpu=True, lang='ch', rec_char_dict_path=args.rec_char_dict_path)
            ocr_result = ocr.ocr(f7_path, cls=True)
            for line in ocr_result:
                content = {}
                content['page_number'] = p
                content['text'] = line[1][0]
                content['x0_t'] = line[0][0][0]
                content['y0_t'] = line[0][0][1]
                content['x1_t'] = line[0][2][0]
                content['y1_t'] = line[0][2][1]
                content['width'] = line[0][2][1] - line[0][0][1]
                contents.append(content)
            p += 1

    f2_path = file_path + '/' + name + '/pdf_text.json'
    json_file = open(f2_path, mode='w')
    json.dump(contents, json_file, ensure_ascii=False, indent=4)
    return contents


def get_picture(args, name, pdf_path, file_path, picture_path):
    try:
        pic_path = file_path + '/' + name + '/pdf_image/' + '1.png'
        img = Image.open(pic_path)

    except FileNotFoundError:
        pic_path = picture_path + '/' + name + '/pdf_image/' + '1.png'
        img = Image.open(pic_path)

    w = img.width
    h = img.height
    ratio = h / w
    # threshold<1.1 to distinguish ppt from pdf
    if ratio <= 1.1:
        pic = get_title_ppt(args, name, pdf_path, file_path, picture_path, h)
        pass
    if ratio > 1.1:
        pic = get_title_pdf(args, name, pdf_path, file_path, picture_path, h, w)
    return pic


def get_title_pdf(args, name, pdf_path, file_path, picture_path):
    '''
    根据位置坐标获取pdf中截出来的图表的标题，存在pdf_title.json中，再导入Figure_Title.json中的信息，整合后一起写入content_aggregation.json
    一篇pdf的所有标题文本信息整合在一个列表中return传出去
    '''
    f3_path = file_path + '/' + name + '/json_path/' + name + '.json'
    with open(f3_path) as json_file:
        figures = json.load(json_file)

    matches = []
    p_max = figures[-1]['page']['page_number']
    contents = get_text(args, name, pdf_path, file_path, picture_path, p_max)
    for item in figures:
        if item['page']['type'] == 'ImageRegion' or item['page']['type'] == 'TableRegion' or item['page']['type'] == 'Table' or item['page']['type'] == 'Figure':
            page_number = item['page']['page_number']

            pic_path = file_path + '/' + name + '/pdf_image/' + str(page_number) + '.png'
            img = Image.open(pic_path)
            w = img.width
            h = img.height

            # 图表框的位置
            x0_f = item['rectangle']['x_1']
            y0_f = item['rectangle']['y_1']
            x1_f = item['rectangle']['x_2']
            y1_f = item['rectangle']['y_2']
            width_1 = h * 0.02
            width_2 = h * 0.05
            # 比例参数可调整，根据标题是否能被准确找到
            h1 = y1_f - y0_f
            w1 = x1_f - x0_f

            words = ['图表', '图', '表', 'Exhibit']
            text = ''
            title = ''

            i = 0
            while i < len(contents):
                if contents[i]['page_number'] == page_number:
                    # 文本行的位置
                    x0_t = contents[i]['x0_t']
                    y0_t = contents[i]['y0_t']
                    x1_t = contents[i]['x1_t']
                    y1_t = contents[i]['y1_t']
                    # t = y0_t - y1_t
                    # y0_t -= t
                    # y1_t += t
                    # 都变为以左上角为坐标原点，用矩形框的左上角和右下角来定位
                    if x0_t >= (x0_f - width_1) and y0_t >= (y0_f - width_1) and x1_t <= (x1_f + width_1) and y1_t <= (y1_f + width_1):
                        text += contents[i]['text']
                    # for word in words: #要改 不然有三次重复的
                    if x0_t >= (x0_f - width_2) and y0_t >= (y0_f - width_2) and x1_t <= (x1_f + width_2) and y1_t <= (y1_f + width_2):
                        for word in words:
                            loc = contents[i]['text'].find(word)
                            if loc != -1:
                                if len(contents[i]['text']) == loc + 1:
                                    if contents[i + 1]['text'][0].isdigit():
                                        title += contents[i]['text']
                                        title += contents[i + 1]['text']
                                        title += contents[i + 2]['text']
                                elif contents[i]['text'][loc + 1].isdigit() and len(contents[i]) < 8:
                                    title += contents[i]['text']
                                    title += contents[i + 1]['text']
                                elif contents[i]['text'][loc + 1].isdigit():
                                    title += contents[i]['text']
                i += 1

            match = {'position': {}}
            # match = {'diagram':{}, 'content':{}}
            match['type'] = item['page']['type']
            match['file_name'] = name
            match['page_id'] = page_number
            match['position']['x0_f'] = x0_f
            match['position']['y0_f'] = y0_f
            match['position']['x1_f'] = x1_f
            match['position']['y1_f'] = y1_f
            # 存入图表的位置
            match['Picture_name'] = 'page-' + str(item['page']['page_number']) + '-blockid-' + str(item['page']['id'])
            if match['type'] == 'ImageRegion' or match['type'] == 'Figure':
                match['img_path'] = picture_path + '/' + name + '/pdf_figure/' + match['Picture_name'] + '.png'

            if match['type'] == 'TableRegion' or match['type'] == 'Table':
                match['img_path'] = picture_path + '/' + name + '/pdf_table/' + match['Picture_name'] + '.png'
            # 图表的路径
            match['text'] = text
            match['title'] = title
            # 如果截出的图片长宽占整个页面大于0.6且字数过多，则判断这张图片无效，则滤除
            if (h1 / h > 0.6 and w1 / w > 0.6) and len(text) > 500:
                pass
            else:
                matches.append(match)
    f4_path = file_path + '/' + name + '/pdf_title.json'
    json_file = open(f4_path, mode='w')
    json.dump(matches, json_file, ensure_ascii=False, indent=4)

    # 导入layout parser的predict title并整合
    f5_path = file_path + '/' + name + '/Figure_Title.json'
    try:
        with open(f5_path, 'r') as f:
            predict_title = json.load(f)
    except FileNotFoundError:
        print("There is no predict title file")
        predict_title = []

    aggregations = []
    for item in matches:
        aggregation = {}
        aggregation['Picture_name'] = item['Picture_name']
        aggregation['title'] = item['title']
        aggregation['text'] = item['text']
        aggregation['type'] = item['type']
        aggregation['img_path'] = item['img_path']
        aggregation['file_name'] = item['file_name']
        aggregation['page_id'] = item['page_id']
        aggregation['position'] = item['position']

        for each in predict_title:
            if each['Picture_name'] == item['Picture_name']:
                aggregation['Predict_Title_1'] = each['Predict_Title_1']
                aggregation['Predict_Title_2'] = each['Predict_Title_2']
        aggregations.append(aggregation)

    f6_path = file_path + '/' + name + '/content_aggregation.json'
    json_file = open(f6_path, mode='w')
    json.dump(aggregations, json_file, ensure_ascii=False, indent=4)
    return aggregations


def get_title_ppt(args, name, pdf_path, file_path, picture_path):
    '''
    每一页ppt看作一张图片，选取其中较大的文字作为一整页的标题，存在content_aggregation.json中
    并且所有标题信息return传出去
    '''
    f3_path = picture_path + '/' + name + '/pdf_image'
    files = os.listdir(f3_path)
    p_max = len(files)

    contents = get_text(args, name, pdf_path, file_path, picture_path, p_max)
    p = 1
    results = []
    while p <= p_max:
        result = {}
        title = ''
        text = ''
        f7_path = picture_path + '/' + name + '/pdf_image/' + str(p) + '.png'
        img = Image.open(f7_path)
        h = img.height
        for item in contents:
            if item['page_number'] == p:
                if item['width'] / h >= 0.05:
                    title += item['text']
                else:
                    text += item['text']
        result['page_id'] = p
        result['file_name'] = name
        result['title'] = title
        result['text'] = text
        result['img_path'] = f7_path

        ocr = PaddleOCR(use_angle_cls=True, use_gpu=True, lang='ch', rec_char_dict_path=args.rec_char_dict_path)
        textocr = ''
        ocr_result = ocr.ocr(f7_path, cls=True)
        for line in ocr_result:
            ocr = line[1][0]
            x0_1 = line[0][0][0]
            y0_1 = line[0][0][1]
            x1_1 = line[0][2][0]
            y1_1 = line[0][2][1]
            # ocr是得到的文本结果，xy是矩形左上角和右下角的位置坐标，以左上角为坐标原点
            a = 1  # 标志位
            for item in contents:
                if item['page_number'] == p:
                    if x0_1 > item['x0_t'] - 10 and y0_1 > item['y0_t'] - 10 and x1_1 < item['x1_t'] + 10 and y1_1 < item['y1_t'] + 10:
                        # 对于ocr得出的文本结果，根据位置坐标判断它是否跟get_text抽取的文本有重合，如果有重合则删除
                        a = 0
            if a == 1:
                textocr += ocr
        result['textocr'] = textocr
        results.append(result)
        p += 1
    f6_path = file_path + '/' + name + '/content_aggregation.json'
    json_file = open(f6_path, mode='w')
    json.dump(results, json_file, ensure_ascii=False, indent=4)
    return results
