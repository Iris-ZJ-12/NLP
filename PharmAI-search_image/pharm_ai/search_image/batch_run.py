import os
import json
# import time
from args_config import parse_args
from pre_process import get_h_and_w_from_pdf_path, pdf_2_img, get_pdf_name_file
from multiprocessing import Pool
from functools import partial


def pre_create(args, file_name):
    # 这个函数主要就是建文件夹
    # path_list = os.listdir(args.pdf_path)
    # path_list.sort(key=lambda x:int(x.split('.')[0]))
    # for file_name in path_list:
    # file = open(os.path.join(pdf_path,file_name),'rb')
    # time.sleep(1)

    pdf_name = get_pdf_name_file(file_name)
    # one_pdf_path = args.pdf_path + '/' + pdf_name + '.pdf'
    file_path = args.file_save_path + '/' + pdf_name
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # 通过ratio对pdf和ppt进行分类
    try:
        height, width = get_h_and_w_from_pdf_path(file_name)
        ratio = height / width
        # print(ratio, os.getpid())
    except Exception as e:
        print(e)
        ratio = 2
        # print(os.getpid())

    # 和paddle layout相关的import要放在多进程里面，否则会报这个错：OSError: (External) CUDA error(3), initialization error.
    from title_process import save_title, get_title_ppt, get_title_pdf
    pictures = []
    files_path = args.file_save_path
    pictures_path = args.pic_table_save_path
    # ppt
    if ratio <= 1.1:
        image_path = args.pic_table_save_path + '/' + pdf_name + '/pdf_image'
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        pdf_2_img(file_name, image_path)

        pic = get_title_ppt(args, pdf_name, file_name, files_path, pictures_path)
        pictures += pic
    # pdf
    else:
        image_path = args.file_save_path + '/' + pdf_name + '/pdf_image'
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        pdf_2_img(file_name, image_path)

        pdf_table_path = args.pic_table_save_path + '/' + pdf_name + '/pdf_table'
        if not os.path.exists(pdf_table_path):
            os.makedirs(pdf_table_path)

        pdf_figure_path = args.pic_table_save_path + '/' + pdf_name + '/pdf_figure'
        if not os.path.exists(pdf_figure_path):
            os.makedirs(pdf_figure_path)

        from layout_process import create_file
        result_path = create_file(args, pdf_name, image_path)
        save_title(args, result_path, pdf_name)

        pic = get_title_pdf(args, pdf_name, file_name, files_path, pictures_path)
        pictures += pic
    return pictures


def muti_read_pdf(args):
    file_name = [os.path.join(args.pdf_path, file_name) for file_name in os.listdir(args.pdf_path)]
    # 固定args
    result = partial(pre_create, args)
    # 同时处理4个文件
    with Pool(4) as p:
        # 对可迭代对象file_name（1份pdf/ppt）进行迭代
        pictures = p.map(result, file_name)
        # 最后的大json(ccc_4_1.json)
        f8_path = args.result_json
        json_file = open(f8_path, mode='w')
        json.dump(pictures, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # 这里不能删掉 会报paddle多进程错
    import multiprocessing as mp
    mp.set_start_method('spawn')
    args = parse_args()
    muti_read_pdf(args)
