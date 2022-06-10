## requirements


fitz==0.0.1.dev2

layoutparser==0.0.1

numpy==1.19.5

opencv_python==4.5.5.62

paddleocr==2.5.0.3

Pillow==9.1.1

PyPDF2==2.1.0

## args_config需要配置的路径:

all_pdf_path:所有的pdf路径

pdf_path：你想处理的pdf路径

file_save_path：pdf切下来的图片、处理过的json文件、layout的结果存放的位置

pic_table_save_path：用户需要的图片和表格存放的位置

result_json：最终大json存放的位置

paddle_model_path:paddle模型的位置

## 最终输出的结果：

-----pic_table_save_path

-----------pdf1

---------------pdf_figure

---------------pdf_table

-----------pdf2

---------------pdf_figure

---------------pdf_table

-----------ppt1

---------------pdf_image

-----------ppt2

---------------pdf_image

-----result_json

-----------all.json
