import re
from openpyxl import load_workbook
import pandas as pd
import hanzidentifier as cn
import spacy
from pysbd.utils import PySBDFactory
import json, xmltodict
from rouge import Rouge
from pprint import pprint
from loguru import logger
import os, fnmatch, zipfile
import edit_distance
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import traceback
from bs4 import BeautifulSoup


class Utilfuncs:
    """
    All kinds of convenient methods
    """

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    @staticmethod
    def list_all_files(folder, file_extention):
        res = []
        for file in os.listdir(folder):
            if file.endswith('.'+file_extention):
                res.append(os.path.join(folder, file))
        return res

    @staticmethod
    def unzip_totally(src_folder, dest_folder):
        for root, dirs, files in os.walk(src_folder, dest_folder):
            for filename in fnmatch.filter(files, "*.zip"):
                print("  " + os.path.join(root, filename))
                zipfile.ZipFile(os.path.join(root, filename)).extractall(dest_folder)

    @staticmethod
    def xml2json(xml_path, json_path):
        with open(xml_path) as xml_file:
            data_dict = xmltodict.parse(xml_file.read())
        xml_file.close()
        f = open(json_path, 'w', encoding='utf-8')
        json.dump(data_dict, f, indent=4, ensure_ascii=False)
        print(json_path+' was saved.')
        f.close()

    @staticmethod
    def is_all_ch(text):
        for chr in text:
            if not '\u4e00' <= chr <= '\u9fa5':
                return False
        return True

    @staticmethod
    def to_excel(dataframe, excel_path, sheet_name='default'):
        try:
            book = load_workbook(excel_path)
            writer = pd.ExcelWriter(excel_path, engine='openpyxl')
            writer.book = book
        except:
            writer = pd.ExcelWriter(excel_path, engine='openpyxl')

        dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
        writer.save()

    @staticmethod
    def cut_sent(para):
        # import spacy
        # nlp = spacy.load("en_core_web_sm")
        # doc = nlp(para)
        # sentences = [sent.text for sent in doc.sents]

        import re
        # 输入一个段落，分成句子，可使用split函数来实现
        sentences = re.split('(。|！|\!|\.|？|\?)', para)  # 保留分割符
        new_sents = []
        for i in range(int(len(sentences) / 2)):
            sent = sentences[2 * i] + sentences[2 * i + 1]
            new_sents.append(sent)

        return new_sents

    def tokenize(self, txt, keep_indices=False):

        # raw_tokens = re.split('(\W)', txt)

        doc = self.nlp(txt)
        raw_tokens = []
        for token in doc:
            raw_tokens.append(token.text)
            if token.whitespace_:
                raw_tokens.append(token.whitespace_)

        token_li = []
        substr_en = ''
        for token in raw_tokens:
            if cn.has_chinese(token):
                chars = list(token)
                for char in chars:
                    if cn.has_chinese(char):
                        if substr_en:
                            token_li.append(substr_en)
                            substr_en = ''
                        token_li.append(char)
                    elif char.isalpha():
                        substr_en += char
                    else:
                        if substr_en:
                            token_li.append(substr_en)
                        substr_en = ''
                        token_li.append(char)
            else:
                if substr_en:
                    token_li.append(substr_en)
                    substr_en = ''
                if token != '':
                    token_li.append(token)

        for index, token in enumerate(token_li):
            # words like Michael's were separated as two parts: Michael, 's, here
            # we merge them into one word
            try:
                if token_li[index] == "'s":
                    token = token_li[index-1] + token_li[index]
                    token_li[index] = token
                    del token_li[index-1]
            except:
                continue

        if keep_indices:
            start = 0
            result = []
            for token in token_li:
                end = start + len(token) - 1
                result.append([token, start, end])
                start += len((token))
            df = pd.DataFrame(result, columns=['token', 'start', 'end'])
            return df
        else:
            df = pd.DataFrame({'token': token_li})
            return df

    def cut_para(self, para):
        nlp = spacy.blank('en')
        nlp.add_pipe(PySBDFactory(nlp))
        doc = nlp(para)
        sents = [sent.text for sent in doc.sents]
        return sents

    def tokenize_paragraph(self, para, keep_indices=False):
        sents = self.cut_para(para)
        if keep_indices:
            result_df = pd.DataFrame()
            for sent in sents:
                sent_df = self.tokenize(sent, keep_indices=keep_indices)
                result_df = result_df.append(sent_df)
                result_df = result_df.append({'token': '##sentence##',
                                              'start': -1,
                                              'end': -1}, ignore_index=True)
            return result_df

        result_df = pd.DataFrame()
        for sent in sents:
            sent_df = self.tokenize(sent, keep_indices=keep_indices)
            result_df = result_df.append(sent_df)
            result_df = result_df.append({'token': '##sentence##'}, ignore_index=True)
        return result_df

    @staticmethod
    def check_sentence_boundary_punc(txt):
        import re
        pattern = re.compile('(。|！|\!|\.|？|\?)')
        return bool(pattern.match(txt))

    @staticmethod
    def remove_illegal_chars(string):
        ILLEGAL_CHARACTERS_RE = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]|\xef|\xbf')
        value = ILLEGAL_CHARACTERS_RE.sub('', string)
        return value

    @staticmethod
    def remove_html_tags(raw_str):
        soup = BeautifulSoup(raw_str, features="html.parser")
        return ''.join([s.text.replace('\n', '') for s in soup.contents if hasattr(s, 'text') and s.text])

    @staticmethod
    def clean_space_en(text):
        text = ' '.join(text.split()).strip()
        return text

    @staticmethod
    def clean_space(text):
        """"
        处理多余的空格
        """
        match_regex = re.compile(u'[\u4e00-\u9fa5。\.,，:：《》、\(\)（）]{1} +(?<![a-zA-Z])|\d+ +| +\d+|[a-z A-Z]+')
        should_replace_list = match_regex.findall(text)
        order_replace_list = sorted(should_replace_list, key=lambda i: len(i), reverse=True)
        for i in order_replace_list:
            if i == u' ':
                continue
            new_i = i.strip()
            text = text.replace(i, new_i)
        text = text.replace("\n", "").strip()
        return text

    @staticmethod
    # trues is a list of corresponding actual texts, preds is list of generated texts
    def get_rouge(trues, preds):
        rouge = Rouge()
        scores = rouge.get_scores(preds, trues)
        r1f1_li = []
        r1p_li = []
        r1r_li = []
        r2f1_li = []
        r2p_li = []
        r2r_li = []
        rlf1_li = []
        rlp_li = []
        rlr_li = []
        for kv in scores:
            r1f1 = kv['rouge-1']['f']
            r1f1_li.append(r1f1)
            r1p = kv['rouge-1']['p']
            r1p_li.append(r1p)
            r1r = kv['rouge-1']['r']
            r1r_li.append(r1r)

            r2f1 = kv['rouge-2']['f']
            r2f1_li.append(r2f1)
            r2p = kv['rouge-2']['p']
            r2p_li.append(r2p)
            r2r = kv['rouge-2']['r']
            r2r_li.append(r2r)

            rlf1 = kv['rouge-l']['f']
            rlf1_li.append(rlf1)
            rlp = kv['rouge-l']['p']
            rlp_li.append(rlp)
            rlr = kv['rouge-l']['r']
            rlr_li.append(rlr)

        dt = {'rouge-1_f': r1f1_li, 'rouge-1_precision': r1p_li, 'rouge-1_recall': r1r_li,
              'rouge-2_f': r1f1_li, 'rouge-2_precision': r1p_li, 'rouge-2_recall': r1r_li,
              'rouge-l_f': r1f1_li, 'rouge-l_precision': r1p_li, 'rouge-l_recall': r1r_li}
        df = pd.DataFrame(dt)

        print('*' * 60)
        scores = rouge.get_scores(preds, trues, avg=True)
        pprint(scores)

        return df

    @staticmethod
    def conv_date2en(month, day, year):
        month = str(month)
        day = str(day)
        year = str(year)
        months = ['January', 'February', 'March',
                  'April', 'May', 'June',
                  'July', 'August', 'September',
                  'October', 'Novmber', 'December']

        endings = ['st', 'nd', 'rd'] + 17 * ['th'] \
                  + ['st', 'nd', 'rd'] + 7 * ['th'] \
                  + ['st']
        month_number = int(month)
        day_number = int(day)

        month_name = months[month_number - 1]
        ordinal = day + endings[day_number - 1]
        return month_name + ', ' + ordinal + ', ' + year

    @staticmethod
    def get_edit_distance_ratios(trues, preds):
        res = []
        for t, p in zip(trues, preds):
            sm = edit_distance.SequenceMatcher(t, p)
            r = sm.ratio()
            res.append(r)
        res = pd.DataFrame({'edit_distances': res})
        return res

    @staticmethod
    def fix_torch_multiprocessing():
        """
        This function will close the shared memory of pytorch,
        to fix `OSError: [Errno 12] Cannot allocate memory` ,
        when multiprocessing is used to convert data into transformers features.

        Add this function to the top of `train.py` ,or before loading a transformer model.

        Reference:
        - https://github.com/huaweicloud/dls-example/issues/26#issuecomment-411990039
        - https://github.com/pytorch/fairseq/issues/1171#issuecomment-549345884
        """
        import sys
        import torch
        from torch.utils.data import dataloader
        from torch.multiprocessing.reductions import ForkingPickler
        default_collate_func = dataloader.default_collate

        def default_collate_override(batch):
            dataloader._use_shared_memory = False
            return default_collate_func(batch)

        setattr(dataloader, 'default_collate', default_collate_override)
        for t in torch._storage_classes:
            if sys.version_info[0] == 2:
                if t in ForkingPickler.dispatch:
                    del ForkingPickler.dispatch[t]
            else:
                if t in ForkingPickler._extra_reducers:
                    del ForkingPickler._extra_reducers[t]

    @staticmethod
    def select_best_cuda():
        """Select the cuda device (cuda:0 or cuda:1) that has more memory left"""
        nvmlInit()
        h0 = nvmlDeviceGetHandleByIndex(0)
        info0 = nvmlDeviceGetMemoryInfo(h0)
        h1 = nvmlDeviceGetHandleByIndex(1)
        info1 = nvmlDeviceGetMemoryInfo(h1)
        res_cuda = 0 if info0.free>info1.free else 1
        return res_cuda

    @staticmethod
    def send_email_notification(from_address, password, smtp_server, send_to, notification_text, smtp_port=465):
        """
        :param str from_address: Sender email address. e.g. fanzuquan@outlook.com
        :param str password: Sender email password.
        :param str smtp_server: Sender email SMTP server. e.g. smtp.outlook.com
        :param str send_to: Receiver email address.
        :param str notification_text: sending text.
        :param int smtp_port: SMTP server port. Default: 465.
        """

        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Pharmcube AI notification"
        msg["From"] = from_address
        msg["To"] = send_to
        msg.attach(MIMEText(notification_text, 'plain'))

        use_ssl = smtp_port in [465, 994]
        use_tls = smtp_port==587
        if use_ssl:
            context = ssl.create_default_context()
            server = smtplib.SMTP_SSL(smtp_server, smtp_port, context=context)
        elif use_tls:
            context = ssl.create_default_context()
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls(context=context)
        else:
            server = smtplib.SMTP(smtp_server, smtp_port)
        try:
            server.login(from_address, password)
            server.sendmail(from_address, [send_to], msg.as_string())
            server.quit()
        except:
            print('Notification failed to be sent.')

    @staticmethod
    def email_wrapper(func, send_to, notify_success=True):
        """
        Wrap a function with email notification. If error raised, email notification is sent.
        If execution funished, email notification is sent also (optinal).

        Usage example:

        >>> def my_divide(x):
        >>>     return x/0
        # wraps the excecuting function with email_wrapper:
        >>> my_divide = Utilfuncs.email_wrapper(my_divide, 'fanzuquan@pharmcube.com')
        # excecute the function:
        >>> result = my_divide(2)


        :param func: Function to be wrap.
        :param str send_to: Email receiver address.
        :param bool notify_success: Whether notify successe.
        :return: Function excecuting results.
        """

        sender_email = "pharm_ai_group@163.com"
        sender_password = "SYPZFDNDNIAWQJBL" # This is authorization password, actual password: pharm_ai163
        sender_smtp_server = "smtp.163.com"
        def wrapper(*args, **kwargs):
            try:
                result=func(*args, **kwargs)
            except Exception:
                Utilfuncs.send_email_notification(sender_email, sender_password, sender_smtp_server,
                                                  send_to, traceback.format_exc())
            else:
                if notify_success:
                    success_msg=f"Process `{func.__name__}` finished with exit code 0."
                    if result:
                        # trunk long result
                        result_str = str(result)
                        msg_result = result if len(result_str)<2000 else f"{result_str[:1000]}\n...\n{result_str[-1000:]}"
                        success_msg+=f"\nResult:\n{msg_result}"
                    Utilfuncs.send_email_notification(sender_email, sender_password, sender_smtp_server,
                                                      send_to, success_msg)
                return result
        return wrapper

if __name__ == '__main__':
    def my_divide():
        from numpy import pi
        return pi/0
    my_divide = Utilfuncs.email_wrapper(my_divide, 'fanzuquan@pharmcube.com')
    result = my_divide()