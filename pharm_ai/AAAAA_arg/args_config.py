from paddleocr.tools.infer.utility import init_args as infer_args

def init_args():
    parser = infer_args()
    #process batch pdf 
    parser.add_argument("--all_pdfpath",type=str,default='/home/hxf/LayoutParser/Search_img/pdfpath')
    parser.add_argument("--pdfpath",type=str,default='/home/hxf/LayoutParser/Search_img/test_batch_process_4')
    parser.add_argument("--fileSavePath", type=str,default='/home/hxf/LayoutParser/AAA_Test/')
    parser.add_argument("--picTableSavePath", type=str,default='/home/hxf/LayoutParser/BBB_Test/')
    parser.add_argument("--all_filePath", type=str,default='/home/hxf/LayoutParser/AAA_Test')
    parser.add_argument("--Result_json", type=str,default='/home/hxf/LayoutParser/ccc_Result')
    
    # paddle config
    parser.add_argument("--paddle_model_path", type=str,default='/home/hxf/LayoutParser/paddleocr_test/ppyolov2_r50vd_dcn_365e_publaynet')
    # parser.add_argument("--table_model_dir", type=str,default='.local/lib/python3.9/site-packages/paddleocr/inference/en_ppocr_mobile_v2.0_table_structure_infer/')
    # parser.add_argument("--table_char_dict_path",type=str,default='.local/lib/python3.9/site-packages/paddleocr/ppocr/utils/dict/table_structure_dict.txt')
    
    
    ##############################################################
    parser.add_argument("--table_max_len", type=int, default=488)
    parser.add_argument("--table_char_type", type=str, default='en')
    parser.add_argument("--layout_path_model", type=str, default="lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config")
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--label_map_path", type=str, default='./vqa/labels/labels_ser.txt')
    parser.add_argument("--mode",type=str, default='structure', help='structure and vqa is supported')
    ##############################################################

    return parser

def parse_args():
    parser = init_args()
    return parser.parse_args()
