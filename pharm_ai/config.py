class ConfigFilePaths:
    import os
    from pathlib import Path
    target_rec_log_dir = os.path.abspath(os.path.dirname(__file__)) + '/target_recognizer/logs'
    data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")
    model_dir = os.path.join(data_dir, "model")
    ner_dir = os.path.join(data_dir, "ner_data")
    bert_dir = '/Users/gwl/Downloads/bert_pytorch_pretrained/multilingual_cased/'
    bert_dir_remote = '/large_files/pretrained_pytorch/bert_base_multilingual_cased/'
    xlm_r_large = '/Users/gwl/Pharmcube/Data/pytorch_pretrained/xlm-r-large/'
    xlm_r_large_remote = '/large_files/pretrained_pytorch/xlm-r-large/'
    xlm_r_base_remote = '/large_files/pretrained_pytorch/xlm-r-base/'
    t5_large_remote = '/large_files/pretrained_pytorch/t5-large/'
    project_dir = os.path.dirname(os.path.realpath(__file__))
    mbart_remote = '/large_files/pretrained_pytorch/mbart/'
    bart_large_remote = '/large_files/pretrained_pytorch/bart-large/'
    albert_cn_l_qa_remote = '/large_files/pretrained_pytorch/albert-chinese-large-qa/'
    bertserini_bert_base_cmrc = '/large_files/pretrained_pytorch/bertserini-bert-base-cmrc/'
    albert_xxlarge_v1_remote = '/large_files/pretrained_pytorch/albert-xxlarge-v1/'
    albert_xl_v2_squad_v2_remote = '/large_files/pretrained_pytorch/albert-xlarge-v2-squad-v2/'
    mt5_base_remote = '/large_files/pretrained_pytorch/mt5-base/'
    mt5_zh_en_remote = '/large_files/pretrained_pytorch/mt5_zh_en/'
    mt5_small_remote = '/large_files/pretrained_pytorch/mt5-small/'
    longformer_dir_remote = '/large_files/pretrained_pytorch/longformer-base-4096/'
    led_dir_remote = '/large_files/pretrained_pytorch/longformer-encdec-base-16384/'
    zh_en = '/large_files/pretrained_pytorch/matress_models/opus-mt-zh-en'
    mt5_zh_en = '/large_files/pretrained_pytorch/mt5_zh_en/'
    t5_base = '/large_files/pretrained_pytorch/t5-base/'


if __name__ == '__main__':
    cfp = ConfigFilePaths
    print(cfp.project_dir)
