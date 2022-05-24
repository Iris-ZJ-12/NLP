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

def get_content_from_ocr(args, image, all_blocks, dectid, pagenumber):
    for block in all_blocks:
        if block.type == "Text" or block.type =="List" or block.type =="Title" :
            if block.id == dectid:
                if pagenumber == pagenumber:
                    segment_image = (block.pad(left=2, right=2, top=2, bottom=2).crop_image(image))
                    ocr = PaddleOCR(use_angle_cls=True, use_gpu=True, rec_char_dict_path=args.rec_char_dict_path)
                    result = ocr.ocr(segment_image, cls=True, rec=True,det=True)
                    text=''
                    for line in result:
                        text+=line[1][0]
                    return text

def createFile(args,imagePath,layoutpath,pdf_figure_path, pdf_table_path, pdfname):
    count = 1
    num = 1
    path_list = os.listdir(imagePath)   
    path_list.sort(key=lambda x:int(x.split('.')[0]))
    
    results = [] 
    for filename in path_list:
        f1 = open(os.path.join(imagePath,filename),'rb')
        f2 = f1.read()
        f3 = np.frombuffer(f2,dtype=np.uint8)
        f4 = cv2.imdecode(f3, cv2.IMREAD_COLOR)
        img = f4
        image = img[..., ::-1]
        model = lp.PaddleDetectionLayoutModel(model_path = args.paddle_model_path,
        label_map =  {0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"},threshold=0.5)
        layout = model.detect(image)
        
        WantedBlocks = lp.Layout([F for F in layout])
        h, w = image.shape[:2]
        left_interval = lp.Interval(0, w/2*1.05, axis='x').put_on_canvas(image)
        left_blocks = WantedBlocks.filter_by(left_interval, center=True)
        left_blocks = left_blocks.sort(key = lambda b:b.coordinates[1])
        right_blocks = [b for b in WantedBlocks if b not in left_blocks]
        right_blocks.sort(key = lambda b:b.coordinates[1])
        allBlocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])  
        layoutresult = lp.draw_box(image, allBlocks, box_width=3, show_element_id=True, show_element_type=True) 
        if not os.path.exists(layoutpath):  
                os.makedirs(layoutpath)
        pathfile = (layoutpath + '/'+ '%s' % count +'.png')
        count +=1    
        layoutresult.save(pathfile , quality=95)

        for block in allBlocks._blocks:
            result = {'page':{},'rectangle':{}}
            result['page']['pagenumber'] = num
            result['page']['id'] = block.id
            result['page']['type'] = block.type
            result['rectangle']['x_1'] = block.block.x_1
            result['rectangle']['x_2'] = block.block.x_2
            result['rectangle']['y_1'] = block.block.y_1
            result['rectangle']['y_2'] = block.block.y_2
            result['rectangle']['height'] = block.block.height
            result['rectangle']['width'] = block.block.width
                       
            if block.type == 'Text' or block.type =='Title' or block.type =='List' :
                dectid = block.id
                pagenumber = num
                text = get_content_from_ocr(args,image, allBlocks, dectid, pagenumber)
                text = str(text)
                result['page']['data'] = text    

            elif block.type =='Table':
                segment_Figure_image = (block
                            .pad(left=5, right=5, top=5, bottom=5)
                            .crop_image(image))
                table_img = Image.fromarray(segment_Figure_image)

                if not os.path.exists(pdf_table_path):  
                    os.makedirs(pdf_table_path)

                pathfile =(pdf_table_path + '/'+ 'page-'+'%s'% num+'-'+'blockid-%s'%block.id+'.png' )
                table_img.save(pathfile, quality=95)
                result['page']['data'] = 'Excel' 
                # TableClip(args, pdf_table_path, excel_path)       
            else:
                result['page']['data'] = 'Figure'
                segment_Figure_image = (block
                            .pad(left=5, right=5, top=5, bottom=5)
                            .crop_image(image))
                img_tr = Image.fromarray(segment_Figure_image)

                if not os.path.exists(pdf_figure_path):  
                    os.makedirs(pdf_figure_path)
                
                pathfile =(pdf_figure_path + '/'+ 'page-'+'%s'% num+'-'+'blockid-%s'%block.id+'.png' )
                img_tr.save(pathfile, quality=95)
            results.append(result)
        num = num + 1
    jsonData = json.dumps(results,indent=1,ensure_ascii=False, cls=NpEncoder) 

    jsonresultpath = args.fileSavePath + pdfname +'/json_path/'
    if not os.path.exists(jsonresultpath):
        os.makedirs(jsonresultpath)
    resultpath = jsonresultpath + pdfname +'.json'
    f = open(resultpath,'w')                   
    f.write(jsonData)
    f.close()

    return resultpath
