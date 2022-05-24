import os
import fitz
from PIL import Image 
from PyPDF2 import PdfFileReader
import json
import numpy as np
from paddleocr import PaddleOCR
from layout_process import createFile
from args_config import parse_args

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

def GetNumPage(pdfpath):
    read = PdfFileReader(pdfpath)
    if read.isEncrypted:
        read.decrypt('')
    page_num = read.getNumPages()
    return page_num

def pdf2img(pdfPath,imagePath):
        idx = 1
        pdfDoc = fitz.open(pdfPath)
        for pg in range(pdfDoc.pageCount):
            page = pdfDoc[pg]
            rotate = int(0) 
            zoom_x = 2             
            zoom_y = 2
            trans = fitz.Matrix(zoom_x, zoom_y).preRotate(rotate)
            try:
                pix = page.getPixmap(matrix=trans, alpha=False)
            except:
                print("imagePath=" + pdfPath)
                continue
            if not os.path.exists(imagePath):  
                os.makedirs(imagePath)         
            pix.writePNG(imagePath + '/' + '%s.png' % idx)
            idx +=1

def GetPdfNamefile(pdfpath):
    pdfpath = pdfpath[pdfpath.rfind("/")+1: pdfpath.rfind(".")]
    return pdfpath

def save_Title(args,json_path,pdfname):
    dataJson = json.load(open(json_path, encoding='UTF-8'))
    results = []
    for i in range(0,len(dataJson)): 
        if dataJson[i]['page']['type']=='Table' or dataJson[i]['page']['type']=='Figure':
            result = {'Picture_name':{},'Predict_Title_1':{},'Predict_Title_2':{}}   
            id = dataJson[i]['page']['id']
            if id == 0:
                if i < len(dataJson)-1:
                    if len(dataJson[i+1]['page']['data'])>30:
                        dataJson[i+1]['page']['data'] = 'TOO Long'
                    result['Picture_name'] = 'page-'+ str(dataJson[i]['page']['pagenumber']) + '-blockid-'+ str(dataJson[i]['page']['id'])
                    result['Predict_Title_1'] = 'None'
                    result['Predict_Title_2'] = dataJson[i+1]['page']['data']
                        # print('page-{}-blockid-{} Predict_Title:{}'.format(dataJson[i]['page']['pagenumber'],dataJson[i]['page']['id'],dataJson[i+1]['page']['data']))    
            elif id > 0 :
                if i > 0 and i < len(dataJson)-1:
                    if len(dataJson[i+1]['page']['data'])>30:
                        dataJson[i+1]['page']['data'] = 'TOO Long'
                    if len(dataJson[i-1]['page']['data'])>30:
                        dataJson[i-1]['page']['data'] = 'TOO Long'    
                    result['Picture_name'] = 'page-'+ str(dataJson[i]['page']['pagenumber']) + '-blockid-'+ str(dataJson[i]['page']['id'])
                    result['Predict_Title_1'] = dataJson[i+1]['page']['data']
                    result['Predict_Title_2'] = dataJson[i-1]['page']['data']
                        # print('page-{}-blockid-{} Predict_Title_1:{}    Predict_Title_2:{}'.format(dataJson[i]['page']['pagenumber'],dataJson[i]['page']['id'], dataJson[i-1]['page']['data'],dataJson[i+1]['page']['data']))
            results.append(result) 
    jsonData = json.dumps(results,indent=1,ensure_ascii=False,cls=NpEncoder)
    jsonresultpath = args.fileSavePath +  pdfname +'/Figure_Title.json'
    f = open(jsonresultpath,'w')                   
    f.write(jsonData)
    f.close()

def gettext(name,pdf_path,json_path):
    f1_path = pdf_path + '/' + name + '.pdf'
    contents=[]
    p = 0
    # 打开pdf文件
    doc=fitz.open(f1_path)
    for page in doc:
        p+=1
        words=page.get_text_words()
        for w in words:
            #位置信息：fitz.Rect(w[:4])
            #w[4]：文本信息
            location_t=fitz.Rect(w[:4])
            text=w[4]
            #print(location_t,text)
            content = {}
            content['pagenumber'] = p
            content['x0_t'] = location_t[0]*2
            content['y0_t'] = location_t[1]*2
            content['x1_t'] = location_t[2]*2
            content['y1_t'] = location_t[3]*2
            content['width'] = location_t[3]*2 - location_t[1]*2
            #左上角为坐标原点
            content['type'] = 'text'
            content['text'] = text
            contents.append(content)

    if not contents:
        print('not parseable pdf')
    else:
        f2_path = json_path + '/' + name + '/pdf_text.json'
        json_file = open(f2_path, mode='w')
        json.dump(contents, json_file, ensure_ascii=False, indent=4)
    return contents

def getpicture(name,pdf_path,json_path):
    picture_path = json_path + '/' + name + '/pdf_image/' + '1.png'
    img = Image.open(picture_path)
    w = img.width
    h = img.height
    ratio = h/w
    # threshold<1.1 to distinguish ppt from pdf
    if ratio <= 1.1:
        gettitleppt(name,pdf_path,json_path,h)
        pass
    if ratio > 1.1:
        gettitlepdf(name,pdf_path,json_path)

def gettitlepdf(name,pdf_path,json_path):
    f3_path = json_path + '/' + name + '/json_path/' + name + '.json'
    with open(f3_path) as json_file:
        figures = json.load(json_file)

    matches = []
    contents=gettext(name,pdf_path,json_path)
    if not contents:
        return

    for item in figures:
        if item['page']['type'] == 'Figure' or item['page']['type'] == 'Table':
            pagenumber = item['page']['pagenumber']
            # 图表框的位置
            x0_f = item['rectangle']['x_1']
            y0_f = item['rectangle']['y_1']
            x1_f = item['rectangle']['x_2']
            y1_f = item['rectangle']['y_2']
            extend_width_1 = 22
            extend_width_2 = 50

            words=['图表', '图', '表', 'Exhibit']
            text=''
            title=''

            for each in contents:
                if each['pagenumber'] == pagenumber:
                    # 文本行的位置
                    x0_t = each['x0_t']
                    y0_t = each['y0_t']
                    x1_t = each['x1_t']
                    y1_t = each['y1_t']     
                    #t = y0_t - y1_t
                    #y0_t -= t
                    #y1_t += t
                    # 都变为以左上角为坐标原点，用矩形框的左上角和右下角来定位
                    if x0_t >= (x0_f-extend_width_1) and y0_t >= (y0_f-extend_width_1) and x1_t <= (x1_f+extend_width_1) and y1_t <= (y1_f+extend_width_1):
                        text+=each['text']
                    # for word in words: #要改 不然有三次重复的
                    if any(word in each['text'] for word in words) and x0_t >= (x0_f-extend_width_2) and y0_t >= (y0_f-extend_width_2) and x1_t <= (x1_f+extend_width_2) and y1_t <= (y1_f+extend_width_2):
                        title+=each['text']

            match={'position':{}}
            # match = {'diagram':{}, 'content':{}}
            match['type'] = item['page']['type']
            match['file_name'] = name
            match['page_id'] = pagenumber
            match['position']['x0_f'] = x0_f
            match['position']['y0_f'] = y0_f
            match['position']['x1_f'] = x1_f
            match['position']['y1_f'] = y1_f
            # 存入图表的位置
            match['Picture_name'] = 'page-' + str(item['page']['pagenumber']) + '-blockid-' + str(item['page']['id'])
            if match['type'] == 'Figure':
                match['img_path'] = json_path + '/' + name + '/pdf_figure/' + match['Picture_name'] + '.png'

            if match['type'] == 'Table':
                match['img_path'] = json_path + '/' + name + '/pdf_table/' + match['Picture_name'] + '.png'
            # 图表的路径
            match['text'] = text
            match['title'] = title
            #match['height'] = h
            #match['width'] = w
            #match['h/w'] = ratio
        
            matches.append(match)

    f4_path = json_path + '/' + name + '/pdf_title.json'
    json_file = open(f4_path, mode='w')
    json.dump(matches, json_file, ensure_ascii=False, indent=4)

    # 导入layout parser的predict title并整合
    f5_path = json_path + '/' + name + '/Figure_Title.json'
    try:
        with open(f5_path, 'r') as f:
            predict_title = json.load(f)
    except FileNotFoundError:
        print("There is no predict title file")
        predict_title = []

    aggregations = []
    for item in matches:
        aggregation = {}
        aggregation['Picture_name']=item['Picture_name']
        aggregation['title']= item['title'] 
        aggregation['text']= item['text']
        aggregation['type']= item ['type']
        aggregation['img_path']= item ['img_path']
        aggregation['file_name']= item['file_name']
        aggregation['page_id']= item['page_id']
        aggregation['position']= item['position']

        for each in predict_title:
            if each['Picture_name'] == item['Picture_name']:
                aggregation['Predict_Title_1']=each['Predict_Title_1']
                aggregation['Predict_Title_2']=each['Predict_Title_2']    

        aggregations.append(aggregation)

    f6_path = json_path + '/' + name + '/content_aggregation.json'
    json_file = open(f6_path, mode='w')
    json.dump(aggregations, json_file, ensure_ascii=False, indent=4)

    f8_path = args.Result_json
    if not os.path.exists(f8_path):
            os.makedirs(f8_path) 
    f8_path = f8_path + '/all.json'
    with open(f8_path,'a',encoding='utf-8') as f:
        json.dump(aggregations, f, ensure_ascii=False, indent=4)

def gettitleppt(name,pdf_path,json_path,h):

    contents=gettext(name,pdf_path,json_path)

    f3_path = json_path + '/' + name + '/json_path/' + name + '.json'
    with open(f3_path) as json_file:
        figures = json.load(json_file)

    p=1
    results=[]
    while p <= figures[-1]['page']['pagenumber']:
        
        result={}
        title=''
        text=''
        for item in contents:
            if item['pagenumber']==p:
                if item['width']/h >= 0.05:
                    title+=item['text']
                else:
                    text+=item['text']
        result['page_id']=p
        result['file_name']=name
        result['title']=title
        result['text']=text
        

        f7_path = json_path + '/' + name + '/pdf_image/' + str(p) + '.png'
            
        ocr = PaddleOCR(use_angle_cls=True, use_gpu=True, rec_char_dict_path=args.rec_char_dict_path)
        textocr=''
        ocrresult = ocr.ocr(f7_path, cls=True)
        for line in ocrresult:
            ocr=line[1][0]
            x0_1=line[0][0][0]
            y0_1=line[0][0][1]
            x1_1=line[0][2][0]
            y1_1=line[0][2][1]
            a=1
            for item in contents:
                if item['pagenumber']==p:
                    if x0_1>item['x0_t']-10 and y0_1>item['y0_t']-10 and x1_1<item['x1_t']+10 and y1_1<item['y1_t']+10:
                        a=0
            if a==1:
                textocr+=ocr
        result['textocr']=textocr
        results.append(result)
        p+=1
    f6_path = json_path + '/' + name + '/content_aggregation.json'
    json_file = open(f6_path, mode='w')
    json.dump(results, json_file, ensure_ascii=False, indent=4)

    f8_path = args.Result_json
    if not os.path.exists(f8_path):
            os.makedirs(f8_path) 
    f8_path = f8_path + '/all.json'
    with open(f8_path,'a',encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def batch_pdf_result(args):
    path_list = os.listdir(args.pdfpath)   
    # path_list.sort(key=lambda x:int(x.split('.')[0]))

    for filename in path_list:
        # file = open(os.path.join(pdfpath,filename),'rb')
        pdfname = GetPdfNamefile(filename)
        one_pdf_path = args.pdfpath + '/'+  pdfname + '.pdf'
        file_path = args.fileSavePath + pdfname 
        if not os.path.exists(file_path):
            os.makedirs(file_path) 
        # ----------------------------------------------------------------
        imagePath = args.fileSavePath + pdfname + '/pdf_image'
        if not os.path.exists(imagePath):
            os.makedirs(imagePath)
        
        pdf2img(one_pdf_path, imagePath)
        # ----------------------------------------------------------------   
        pdf_figure_path = args.picTableSavePath + pdfname + '/pdf_figure'
        if not os.path.exists(pdf_figure_path):
            os.makedirs(pdf_figure_path)
        # ---------------------------------------------------------------- 
        pdf_table_path = args.picTableSavePath + pdfname + '/pdf_table'
        if not os.path.exists(pdf_table_path):
            os.makedirs(pdf_table_path)
        # ---------------------------------------------------------------- 
        # excel_path = args.fileSavePath + pdfname + '/excel_path'
        # if not os.path.exists(excel_path):
        #     os.makedirs(excel_path)
        # ----------------------------------------------------------------
        layoutPath = args.fileSavePath + pdfname +'/pdf_layout_result'
        if not os.path.exists(layoutPath):
            os.makedirs(layoutPath)
        # ----------------------------------------------------------------
        resultpath = createFile(args, imagePath, layoutPath,pdf_figure_path, pdf_table_path, pdfname)
        save_Title(args, resultpath, pdfname)

        pdf_path = args.all_pdfpath
        json_path = args.all_filePath
        filenames=os.listdir(json_path)
        for name in filenames:
            gettext(name,pdf_path,json_path)
            getpicture(name,pdf_path,json_path)
    
if __name__ == "__main__":
    args = parse_args()
    batch_pdf_result(args)
    

