from paddleocr.tools.infer.utility import init_args as infer_args


def init_args():
    parser = infer_args()
    # process batch pdf
    parser.add_argument("--all_pdf_path", type=str, default='./LayoutParser/Search_img/pdfpath')
    parser.add_argument("--pdf_path", type=str, default='./LayoutParser/Search_img/test_batch_process_4')
    parser.add_argument("--file_save_path", type=str, default='./LayoutParser/AAA_4_1')
    parser.add_argument("--pic_table_save_path", type=str, default='./LayoutParser/BBB_4_1')
    parser.add_argument("--result_json", type=str, default='./LayoutParser/CCC_4_1.json')
    # detectron2 config
    parser.add_argument("--config_path", type=str, default='/home/hxf/LayoutParser/MyConfig/config_PrimaLayout_mask_rcnn_R_50_FPN_3x.yaml')
    parser.add_argument("--model_path", type=str, default='/home/hxf/LayoutParser/MyWeights/model_final_PrimaLayout_mask_rcnn_R_50_FPN_3x.pth')
    # paddle config
    parser.add_argument("--paddle_model_path", type=str, default='/home/hxf/LayoutParser/paddleocr_test/ppyolov2_r50vd_dcn_365e_publaynet')
    # parser.add_argument("--table_model_dir", type=str,default='.local/lib/python3.9/site-packages/paddleocr/inference/en_ppocr_mobile_v2.0_table_structure_infer/')
    # parser.add_argument("--table_char_dict_path",type=str,default='.local/lib/python3.9/site-packages/paddleocr/ppocr/utils/dict/table_structure_dict.txt')
    parser.add_argument("--table_max_len", type=int, default=488)
    parser.add_argument("--table_char_type", type=str, default='en')
    parser.add_argument("--layout_path_model", type=str, default="lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config")
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--label_map_path", type=str, default='./vqa/labels/labels_ser.txt')
    parser.add_argument("--mode", type=str, default='structure', help='structure and vqa is supported')

    return parser


def parse_args():
    parser = init_args()
    return parser.parse_args()
