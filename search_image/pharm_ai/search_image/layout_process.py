import os
import layoutparser as lp
import numpy as np
import cv2
import json
from PIL import Image
from paddleocr import PaddleOCR


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


def get_content_from_ocr(args, image, all_blocks, dectid, page_number):
    # 对layout切出来的文字区域block进行ocr
    for block in all_blocks:
        if block.type == "TextRegion" or block.type == "Text" or block.type == "List" or block.type == "Title":
            # block每个框都有编号
            if block.id == dectid:
                if page_number == page_number:
                    # segment_image为切下来的图片 left,right,top,bottom为图片框周围填充大小
                    segment_image = (block.pad(left=2,
                                               right=2,
                                               top=2,
                                               bottom=2).crop_image(image))
                    ocr = PaddleOCR(use_angle_cls=True,
                                    use_gpu=True,
                                    lang='ch',
                                    rec_char_dict_path=args.rec_char_dict_path)
                    result = ocr.ocr(segment_image,
                                     cls=True,
                                     rec=True,
                                     det=True)
                    text = ''
                    for line in result:
                        text += line[1][0]
                    return text


def create_file(args, pdf_name, image_path):
    num = 1
    path_list = os.listdir(image_path)
    # 按照顺序进行排序 1.png,2.png......
    path_list.sort(key=lambda x: int(x.split('.')[0]))
    results = []
    # 拿一张图片进行处理
    for pic_name in path_list:
        picture_path = os.path.join(image_path, pic_name)
        img = open(os.path.join(image_path, pic_name), 'rb')
        img_new = Image.open(img)
        image = cv2.imread(picture_path)
        image = image[..., ::-1]
        h = img_new.height
        w = img_new.width
        ratio = h / w
        # pdf情况 只有pdf才用到layoutparser
        if ratio > 1.1:
            model = lp.PaddleDetectionLayoutModel(
                model_path=args.paddle_model_path,
                label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}, threshold=0.5)
            # model = lp.Detectron2LayoutModel(config_path=args.config_path, model_path=args.model_path, label_map={1: "TextRegion", 2: "ImageRegion", 3: "TableRegion", 4: "MathsRegion", 5: "SeparatorRegion", 6: "OtherRegion"}, extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8])
            layout = model.detect(image)
            wanted_blocks = lp.Layout([F for F in layout])
            # h, w = img_new.size[:2]
            # 对block框进行排序
            left_interval = lp.Interval(0, w / 2 * 1.05, axis='x').put_on_canvas(image)
            left_blocks = wanted_blocks.filter_by(left_interval, center=True)
            left_blocks = left_blocks.sort(key=lambda b: b.coordinates[1])
            right_blocks = [b for b in wanted_blocks if b not in left_blocks]
            right_blocks.sort(key=lambda b: b.coordinates[1])
            all_blocks = lp.Layout([
                b.set(id=idx)
                for idx, b in enumerate(left_blocks + right_blocks)])
            # layoutresult = lp.draw_box(image, all_blocks, box_width=3, show_element_id=True, show_element_type=True)
            # if not os.path.exists(layout_path):
            #     os.makedirs(layout_path)
            # path_file = (layout_path + '/' + '%s' % count + '.png')
            # count += 1
            # layoutresult.save(path_file, quality=95)

            # 从layout所有框中选想要的框，写到json中
            for block in all_blocks._blocks:
                result = {'page': {}, 'rectangle': {}}
                result['page']['page_number'] = num
                result['page']['id'] = block.id
                result['page']['type'] = block.type
                result['rectangle']['x_1'] = block.block.x_1
                result['rectangle']['x_2'] = block.block.x_2
                result['rectangle']['y_1'] = block.block.y_1
                result['rectangle']['y_2'] = block.block.y_2
                result['rectangle']['height'] = block.block.height
                result['rectangle']['width'] = block.block.width
                # 由于有paddle和detectron两个模型，可以通过修改70行注释进行切换。一份type为TextRegion,type为Text。
                if block.type == "TextRegion" or block.type == 'Text' or block.type == 'Title' or block.type == 'List':
                    dectid = block.id
                    page_number = num
                    text = get_content_from_ocr(args, image, all_blocks,
                                                dectid, page_number)
                    text = str(text)
                    result['page']['data'] = text
                elif block.type == 'TableRegion' or block.type == 'Table':
                    segment_figure_image = (block.pad(
                        left=5, right=5, top=5, bottom=5).crop_image(image))
                    table_img = Image.fromarray(segment_figure_image)
                    pdf_table_path = args.pic_table_save_path + '/' + pdf_name + '/pdf_table'
                    if not os.path.exists(pdf_table_path):
                        os.makedirs(pdf_table_path)

                    path_file = (pdf_table_path + '/' + 'page-' + '%s' % num + '-' + 'blockid-%s' % block.id + '.png')
                    table_img.save(path_file, quality=95)
                    result['page']['data'] = 'Excel'
                    # TableClip(args, pdf_table_path, excel_path)
                elif block.type == 'MathsRegion':
                    result['page']['data'] = 'MathsRegion'
                elif block.type == 'SeparatorRegion':
                    result['page']['data'] = 'SeparatorRegion'
                elif block.type == 'OtherRegion':
                    result['page']['data'] = 'OtherRegion'
                else:
                    result['page']['data'] = 'ImageRegion'
                    segment_figure_image = (block.pad(
                        left=5, right=5, top=5, bottom=5).crop_image(image))
                    img_tr = Image.fromarray(segment_figure_image)
                    pdf_figure_path = args.pic_table_save_path + '/' + pdf_name + '/pdf_figure'
                    if not os.path.exists(pdf_figure_path):
                        os.makedirs(pdf_figure_path)
                    path_file = (pdf_figure_path + '/' + 'page-' + '%s' % num + '-' + 'blockid-%s' % block.id + '.png')
                    # 对面积小的图片进行过滤
                    cut_width = img_tr.size[0]
                    cut_height = img_tr.size[1]
                    # 这里设置了图片宽高小于300就过滤掉
                    if cut_height >= 300 and cut_width >= 300:
                        img_tr.save(path_file, quality=95)
                results.append(result)
            num = num + 1
    json_data = json.dumps(results, indent=1, ensure_ascii=False, cls=NpEncoder)
    json_result_path = args.file_save_path + '/' + pdf_name + '/json_path/'
    if not os.path.exists(json_result_path):
        os.makedirs(json_result_path)
    result_path = json_result_path + pdf_name + '.json'
    f = open(result_path, 'w')
    f.write(json_data)
    f.close()
    return result_path
