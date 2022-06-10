import io
import os
import ast
import re
import fitz
import string
import requests
import contextlib

from PIL import Image
from Levenshtein import distance
from collections import OrderedDict
from typing import List, Tuple, Optional, Union

month2int = {
    'jan': 1,
    'feb': 2,
    'march': 3,
    'mar': 3,
    'apr': 4,
    'may': 5,
    'jun': 6,
    'jul': 7,
    'aug': 8,
    'sep': 9,
    'oct': 10,
    'nov': 11,
    'dec': 12
}

digit = set([str(i) for i in range(10)])

common_bracket = {
    'left': ["（", "【", "<", "[", "{", "［"],
    'right': ["）", "】", ">", "]", "}", "］"]
}


class Papa:

    def __init__(self, save_dir, ocr_api):
        self.save_dir = save_dir
        self.ocr_api = ocr_api
        self.left_offset = 0
        self.right_offset = 0
        self.limit_side_len = 700
        self.max_height = 1500

    def convert_page2img(
            self,
            pdf_path,
            pdf_f_name,
            del_img,
            page_num=0,
            suffix=''
    ):
        """
        Convert first page to image, save image to self.save_dir if del_img is False
        Return image bytes
        """
        try:
            with contextlib.redirect_stderr(None):
                pdf_doc = fitz.open(pdf_path)
                mat = fitz.Matrix(2, 2).preRotate(0)
                pix = pdf_doc[page_num].getPixmap(matrix=mat, alpha=False)
                image = pix.tobytes()
            # images = convert_from_path(pdf_path, single_file=single_file)
            # buf = io.BytesIO()
            # images[0].save(buf, format='JPEG')
            # image = buf.getvalue()
            if not del_img:
                save_path = f'{self.save_dir}{pdf_f_name}{suffix}.jpg'
                Image.open(io.BytesIO(image)).save(save_path, 'JPEG')
        except Exception as e:
            return None
        return image

    def crop_half_img(
            self,
            image,
            pdf_f_name,
            del_img,
            is_left_half=True,
            suffix=''
    ):
        im = Image.open(io.BytesIO(image))
        width, height = im.size

        # resize if too high
        if height > self.max_height:
            ratio = self.max_height / height
            width = int(width * ratio)
            height = self.max_height
            im = im.resize((width, height))

        # resize if either side is too short
        # if min(width, height) < self.limit_side_len:
        #     if height < width:
        #         ratio = float(self.limit_side_len) / height
        #         width = int(width * ratio)
        #         height = self.limit_side_len
        #     else:
        #         ratio = float(self.limit_side_len) / width
        #         height = int(height * ratio)
        #         width = self.limit_side_len
        #     im = im.resize((width, height))
        top = 0
        bottom = height
        if is_left_half:
            left = 0
            right = width / 2 + self.left_offset
            bottom = height
            half_img = im.crop((left, top, right, bottom))
            half_img_path = f'{self.save_dir}{pdf_f_name}{suffix}-left.jpg'
        else:
            left = width / 2 + self.right_offset
            right = width - 40
            half_img = im.crop((left, top, right, bottom))
            half_img_path = f'{self.save_dir}{pdf_f_name}{suffix}-right.jpg'

        if not del_img:
            half_img.save(half_img_path)
        buf = io.BytesIO()
        half_img.save(buf, format='JPEG')
        half_img = buf.getvalue()
        return half_img

    def ocr_request(
            self,
            image,
            crop,
            pdf_f_name,
            del_img,
            suffix=''
    ):
        if crop == 'left':
            image = self.crop_half_img(image, pdf_f_name, del_img, is_left_half=True, suffix=suffix)
        elif crop == 'right':
            image = self.crop_half_img(image, pdf_f_name, del_img, is_left_half=False, suffix=suffix)
        response = requests.post(self.ocr_api, files={'file': image})
        text = ast.literal_eval(response.text)
        return text

    def request_or_cache(
            self,
            image,
            pdf_f_name,
            crop='left',
            del_img=True,
            cache_path=None,
            suffix=''
    ):
        if cache_path:
            cache_path = f'{cache_path}{suffix}-{crop}.txt'
            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    text = [line.strip() for line in f.readlines()]
            else:
                text = self.ocr_request(image, crop, pdf_f_name, del_img, suffix)
                with open(cache_path, 'w', encoding='utf-8') as f:
                    for line in text:
                        f.write(line + '\n')
        else:
            text = self.ocr_request(image, crop, pdf_f_name, del_img, suffix)
        return text

    def ocr(
            self,
            pdf_path,
            pdf_f_name,
            is_left_half=True,
            both_halves=False,
            full_page=False,
            del_img=True,
            cache_dir=None
    ):
        first_page_img = None
        cache_path = os.path.join(cache_dir, pdf_f_name) if cache_dir else None
        if not cache_path or not os.path.exists(cache_path):
            first_page_img = self.convert_page2img(pdf_path, pdf_f_name, del_img)
            if not first_page_img:
                return None

        if full_page:
            text = self.request_or_cache(first_page_img, pdf_f_name, 'full', del_img, cache_path)
            return text
        else:
            crop = 'left' if is_left_half else 'right'
            left_txt = self.request_or_cache(first_page_img, pdf_f_name, crop, del_img, cache_path)
            if not both_halves:
                return left_txt
            else:
                right_txt = self.request_or_cache(first_page_img, pdf_f_name, 'right', del_img, cache_path)
                return left_txt, right_txt

    def ocr_2nd_left_page(
            self,
            pdf_path,
            pdf_f_name=None,
            del_img=True,
            cache_dir=None
    ):
        if not pdf_f_name:
            pdf_f_name = pdf_path.split("/")[-1].strip('.pdf').strip('.PDF')
        image = None
        cache_path = os.path.join(cache_dir, pdf_f_name) if cache_dir else None
        if not cache_path or not os.path.exists(cache_path):
            image = self.convert_page2img(pdf_path, pdf_f_name, del_img, page_num=1, suffix='2')
            if not image:
                return None
        text = self.request_or_cache(image, pdf_f_name, 'left', del_img, cache_path, suffix='2')
        return text

    def get_all_text(
            self,
            pdf_path,
            pdf_f_name=None,
            is_left_half=True,
            both_halves=False,
            full_page=False,
            del_img=True,
            cache_dir=None
    ):
        if not pdf_f_name:
            pdf_f_name = pdf_path.split("/")[-1].strip('.pdf').strip('.PDF')
        text = self.ocr(
            pdf_path,
            pdf_f_name,
            is_left_half=is_left_half,
            both_halves=both_halves,
            full_page=full_page,
            del_img=del_img,
            cache_dir=cache_dir
        )
        if not text:
            return None if not both_halves else (None, None)
        return text

    def map_brackets(self, line):
        for left_bracket in common_bracket['left']:
            line = line.replace(left_bracket, "(")
        for right_bracket in common_bracket['right']:
            line = line.replace(right_bracket, ")")
        return line


class PapaCn(Papa):

    def __init__(
            self,
            save_dir='/home/clr/disk/playground/outputs/papa/',
            ocr_api='http://localhost/2txt'
    ):
        super(PapaCn, self).__init__(save_dir, ocr_api)
        self.left_offset = 0
        self.right_offset = -5
        self.PATTERN = {
            'application': r'(^|(?<=\D))(\d{12}[x\d]|\d{8}[x\d]?)',
            'date': r'\d{8}$',
            'inid22': r'\d{2,4}.?\d{1,2}.?\d{1,2}',
            'inid45': r'\d{4}年\d{1,2}月\d{1,2}日'
        }

    def clean_ocr(self, ocr_li):
        cleaned = []
        for line in ocr_li:
            line = self.map_brackets(line)
            line = line.replace(" ", "").replace("，", "").replace(".", "").replace("-", "")
            line = line.replace("(", "").replace(")", "")
            line = line.lower()
            cleaned.append(line)
        return cleaned

    def only_two_digit(self, text):
        res = re.match(r'^\d\d$', text)
        if res:
            return res.group()
        return False

    def starts_with_two_digit(self, text):
        res = re.match(r'^\d\d(?=\D)', text)
        if res:
            return res.group()
        return False

    def find_target_text(self, text):
        target_text = []
        inid_code = ''
        for i in range(len(text)):
            if '分案' in text[i]:
                inid_code = self.only_two_digit(text[i - 1])
                if inid_code:
                    target_text.append(text[i - 1])

                if text[i][-1] in digit and text[i][-2] not in digit:
                    text[i] = text[i][:-1]
                target_text.append(text[i])
                if not inid_code:
                    inid_code = self.starts_with_two_digit(text[i])

                for nt in text[i + 1:i + 3]:
                    if nt in digit:
                        continue
                    if not self.starts_with_two_digit(nt) and \
                            not (len(nt) > 4 and '22' in nt[:4] and nt[5] not in digit):
                        target_text.append(nt)
                    else:
                        break
        return inid_code, ''.join(target_text)

    def find_pattern(self, piece, pattern_name, verbose=False):
        pattern = self.PATTERN[pattern_name]
        res = re.search(pattern, piece)
        match = res.group() if res else None
        if verbose:
            print(f'matched {pattern_name} number: {match}')
        return match

    def date_postprocessing(self, date_text):
        if not date_text:
            return ''

        # deal with 年月日 case
        if '年' in date_text:
            yidx = date_text.index('年')
            midx = date_text.index('月')
            year = date_text[:yidx]
            month = date_text[yidx + 1:midx]
            day = date_text[midx + 1:-1]
            return '-'.join([year, month, day])

        # deal with two-digit and 4-digit year
        if date_text.startswith('19') or date_text.startswith('20'):
            year = date_text[:4]
            month_day = date_text[4:]
        else:
            year = date_text[:2]
            month_day = date_text[2:]
            if int(year[0]) < 3:
                year = '20' + year
            else:
                year = '19' + year
        if not 1929 <= int(year) <= 2025:
            return ''

        # split month and day
        if month_day[0] == '.':
            month_day = month_day[1:]
        if '.' in month_day:
            month, day = month_day.split('.')
        else:
            for i in range(1, len(month_day)):
                month = month_day[:i]
                day = month_day[i:]
                if 1 <= int(month) <= 12 and 1 <= int(day) <= 31 and len(day) < 3:
                    break
            else:
                return ''
        return '-'.join([year, month, day])

    def extract_22(self, text):
        date = 'inid22 keyword not found'
        for i in range(len(text)):
            if '申请日' in text[i]:
                piece = self.map_brackets(text[i])
                date = self.find_pattern(piece, 'inid22')
                if not date:
                    piece = self.map_brackets(text[i + 1])
                    date = self.find_pattern(piece, 'inid22')
                date = self.date_postprocessing(date)
                if not date:
                    date = 'inid22 date not found'
                break
        return date

    def extract_45(self, text):
        def match_two_date(p):
            d = self.date_postprocessing(self.find_pattern(p, 'inid45'))
            if not d:
                d = self.date_postprocessing(self.find_pattern(p, 'date'))
            return d

        date = 'inid code 45 not found'
        for i in range(len(text)):
            piece = text[i]
            pre_p = piece.lstrip('1').lstrip('c')
            if pre_p.startswith('43'):
                return 'inid code 43 found'
            if pre_p.startswith('45') or pre_p.startswith('145'):
                date = match_two_date(piece)
                if not date:
                    piece = text[i + 1]
                    date = match_two_date(piece)
                if not date:
                    date = 'inid 45 date not found'
                break
        return date

    def extract_patent_family(self, inid_code, target_text):
        res = {}
        application_num = self.find_pattern(target_text, 'application')
        res['inid_code'] = inid_code
        res['paragraph'] = target_text
        res['application_num'] = application_num
        date = self.find_pattern(target_text, 'date')
        if not date or (date and date in application_num):
            date = ''
        if len(application_num) == 13 and application_num[-4:] == date[:4]:
            res['application_num'] = res['application_num'][:-4]
        res['date'] = self.date_postprocessing(date)
        return res

    def pipeline(self, pdf_path, pdf_f_name=None, left_half=True, res=None, cache_dir=None):
        text = self.get_all_text(
            pdf_path,
            pdf_f_name=pdf_f_name,
            is_left_half=left_half,
            del_img=True,
            cache_dir=cache_dir
        )
        if not text:
            return {'error': 'error when converting to image'}
        if not res:
            res = {}
        cleaned_text = self.clean_ocr(text)
        if 'inid22_date' not in res or res['inid22_date'] == 'inid22 keyword not found':
            res['inid22_date'] = self.extract_22(text)

        if 'inid45_date' not in res or res['inid45_date'] == 'inid code 45 not found':
            res['inid45_date'] = self.extract_45(cleaned_text[:7])

        if 'patent_family' not in res:
            inid_code, target_text = self.find_target_text(cleaned_text)
            if target_text:
                patent_family = self.extract_patent_family(inid_code, target_text)
                res['patent_family'] = patent_family

        # look for right half page if not left
        if left_half and ('patent_family' not in res or
                          res['inid22_date'] == 'inid22 keyword not found' or
                          res['inid45_date'] == 'inid code 45 not found'):
            return self.pipeline(pdf_path, left_half=False, res=res)
        return res


class PapaUs(Papa):

    def __init__(
            self,
            save_dir='/home/clr/disk/playground/outputs/papa/',
            ocr_api='http://localhost/2txt'
    ):
        super(PapaUs, self).__init__(save_dir, ocr_api)
        self.key_word = ['continuation', 'continuationinpart', 'division', 'provisional', 'terminal', 'extend']
        self.PATTERN = {
            'keyword': r'co.{7,9}on|co.{13,15}rt|di.{3,5}on|pr.{6,8}al|te.{3,5}al|\werminal|\wxtend',
            'application': r'(\d{2}/?\d{3},?\d{3})|(((?<=\D,)|(?<=[a-zA-Z:]))\d{1,3},?\d{3}(?!\d))',
            'pct': r'pct/[a-z]{2}[\do]{2,4}/[\do]{5}\d?',
            'pct_pre': r'pct/[a-z]{2}[\do]{7,10}',
            'publication': r'\d{1,2},?\d{3},?\d{3}|re,?\d{2},?\d{3}',
            'date': r'[a-z]{2,5},?\d{1,2},?\d{4}(?!\d)',
            '22date': r'[a-z]{2,5},?\d{1,2},?\d{4}',
            'pct_pub': r'wo\d{2,4}/\d{5,6}(?=\D)',
            'bydays': r'(?<=by)[0-9ol]*(?=day)|(?<=for)[0-9ol]*(?=day)',
        }
        self.left_offset = 5
        self.right_offset = -5

    def clean_ocr(self, ocr_li):
        inid_codes = ('22', '60', '62', '63', '64', '66')
        cleaned = []
        for line in ocr_li:
            line = self.map_brackets(line)
            line = line.replace(" ", "").replace("，", ",").replace(".", ",").replace("-", "").replace("、", ",")
            line = line.replace("：", ":").replace("·", ",").replace("。", ",").replace("ï¼Œ", "").replace(",,", ",")
            line = line.replace("((", "(").lower()
            for code in inid_codes:
                # fix unnormal square bracket
                if line == code + "1" or line == "1" + code or line == "1" + code + "1":
                    line = "(" + code + ")"
                if line.startswith("(" + code + "1"):
                    line = line[:3] + ")" + line[4:]

                w = code + ")"
                if line.startswith(w):
                    line = "(" + line
            if re.match(r'^\d\d\)', line):
                line = '(' + line
            for code in inid_codes:
                w = "(" + code + ")"
                if line.startswith(w):
                    cleaned.append(w)
                    cleaned.append(line[4:].strip('('))
                    break
            else:
                cleaned.append(line)
        return cleaned

    def find_target_text(
            self,
            left: Optional[List[str]],
            right: Optional[List[str]],
            inid_codes: Union[Tuple, str],
            index: int = 0,
            state: int = 0,
            text: Optional[List[str]] = None,
            **kwargs
    ) -> Tuple[List[str], int, List[str], List[str]]:
        """ 
        State Code
        0: not started
        1: found inid code
        2: not finished and preparing to skipping right header
        3: found first () while skipping heading, if second () is found, jump to state 1 to continue collecting

        Returns Tuple[List[str], int, bool], represents target text lines, index number, left page, right page
        """  # color
        if text is None:
            text = []
        if not left:
            return ['-1', ',', '-1'], index, left, right

        if index == len(left):
            if state == 1:
                next_state = 2
            else:
                next_state = 0
            return self.find_target_text(right, None, inid_codes, 0, next_state, text, **kwargs)

        curr = left[index]
        if state == 0:
            # start collecting if triggered
            if type(inid_codes) == tuple:
                if curr[:4] in inid_codes:
                    next_state = 1
                    text.append(curr[1:3] + ",")
                    for i in range(index - 3, index):
                        line = left[i]
                        if line.startswith('co') or \
                                line.startswith('di') or \
                                line.startswith('pa') or \
                                line.startswith('pr') or \
                                line.startswith('reissue'):
                            text.append(''.join(left[i:index]))
                            break
                else:
                    next_state = 0
            else:
                # inid_codes = 'notice'
                if inid_codes in curr[:10]:
                    next_state = 1
                    text.append(curr)
                else:
                    next_state = 0
            return self.find_target_text(left, right, inid_codes, index + 1, next_state, text, **kwargs)
        elif state == 1:
            # check ending condition or continue collecting
            if curr.startswith("(") and \
                    not curr.startswith("(2)") and \
                    not curr.startswith("(*") and \
                    not curr.startswith("(b)") and \
                    not curr.startswith("()"):

                if type(inid_codes) == tuple and len(inid_codes) > 1 and curr.startswith("(con"):
                    # restore cross page information
                    extra_text = self.ocr_2nd_left_page(**kwargs)
                    extra_text = self.clean_ocr(extra_text)
                    start_idx = 99999
                    for i in range(len(extra_text)):
                        if extra_text[i].endswith('data'):
                            start_idx = i + 1
                            break
                    next_state = 1
                    return self.find_target_text(extra_text[start_idx:], None, inid_codes, 0, next_state, text,
                                                 **kwargs)
                else:
                    return text, index, left, right
            else:
                text.append(curr)
                next_state = 1
                return self.find_target_text(left, right, inid_codes, index + 1, next_state, text, **kwargs)
        elif state == 2:
            if "(" in curr:
                next_state = 3
            else:
                next_state = 2
            return self.find_target_text(left, right, inid_codes, index + 1, next_state, text, **kwargs)
        elif state == 3:
            if curr.startswith("("):
                if curr.endswith(":") or left[index - 1].endswith(":"):
                    return self.find_target_text(left, right, inid_codes, index + 2, 1, text, **kwargs)
                else:
                    return self.find_target_text(left, right, inid_codes, index + 1, 1, text, **kwargs)
            return self.find_target_text(left, right, inid_codes, index + 1, 3, text, **kwargs)
        else:
            print(f'unknown state: {state}')

    def find_multiple_target_texts(self, left: List[str], right: List[str], **kwargs) -> List[str]:
        """
        Search for one or two paragraphs that starts with given inid code
        """
        inid_codes = ('(60)', '(62)', '(63)', '(66)')
        target_text1, index, left, right = self.find_target_text(left, right, inid_codes, text=[], **kwargs)
        target_texts = [target_text1]
        if not target_text1[0] == '-1':
            target_text2, _, _, _ = self.find_target_text(left, right, inid_codes, index, text=[], **kwargs)
            if not target_text2[0] == '-1':
                target_texts.append(target_text2)
        target_pieces = [''.join(t) for t in target_texts]
        return target_pieces

    def keyword_correction(self, target_piece: str) -> str:
        """
        Spell correction for OCR
        """
        match = re.findall(self.PATTERN['keyword'], target_piece)
        for m in match:
            for kw in self.key_word:
                if m != kw and distance(m, kw) <= 2:
                    target_piece = target_piece.replace(m, kw)
                    break
        return target_piece

    def split_by_keywords(self, text):
        """
        Use BFS to split sentences by keywords
        """
        li = [text]
        for w in self.key_word[:4]:
            li_ = []
            while li:
                t = li.pop(0)
                if w in t[10:]:
                    i = t.index(w, 10)
                    li_.append(t[:i])
                    li = [t[i:]] + li
                else:
                    li_.append(t)
            li = li_
        li = [t for t in li if len(t) > 8]
        return li

    def find_pattern(self, piece, pattern_name, verbose=False):
        pattern = self.PATTERN[pattern_name]
        match = re.findall(pattern, piece)
        if verbose:
            print(f'matched {pattern_name} number: {match}')
        return match

    def find_date(self, piece, pattern='date'):
        """
        Extract formatted date from piece of text located by regex
        """
        piece = piece.replace('0ct', 'oct').replace('scp', 'sep').replace('jui', 'jul').replace(':', '')
        piece = piece.replace('jur', 'jun').replace('jum', 'jun')
        if 'ap' in piece and 'apr' not in piece:
            piece = piece.replace('ap', 'apr')
        match = self.find_pattern(piece, pattern)
        res = []
        for date_text in match:
            y = m = d = 0
            if date_text:
                y = int(date_text[-4:])
                if not 1929 < y < 2030:
                    continue
                date_text = date_text[:-4]

                # month spell correction
                if len(date_text) > 3 and date_text[3] in string.ascii_lowercase:
                    month_text = date_text[2:5]
                    day_text = date_text[5:]
                else:
                    month_text = date_text[:3]
                    day_text = date_text[3:]
                if not (month_text.startswith('ju') and month_text != 'jun' and month_text != 'jul'):
                    for month in month2int:
                        if (month_text != 'jun' and month_text != 'jul'
                            and month_text != 'may' and month_text != 'mar'
                            and month_text != month and distance(month_text, month) == 1) \
                                or month_text == month:
                            date_text = month + day_text
                            break

                for month in month2int:
                    if month in date_text:
                        m = month2int[month]
                        date_text = date_text.replace(month, "")
                        break

                # if month not found, try to match month with last two characters
                if m == 0:
                    chars = []
                    for c in date_text:
                        if c in string.ascii_lowercase:
                            chars.append(c)
                        else:
                            break
                    cand = "".join(chars[-2:])
                    matched = []
                    for month in month2int:
                        if distance(cand, month) == 1:
                            matched.append(month)
                    if len(matched) == 1:
                        m = month2int[matched[0]]
                        date_text = date_text.replace("".join(chars), "")

                date_text = [c for c in date_text if c in digit]
                if not date_text:
                    continue
                d = int("".join(date_text))
            if 1929 < y < 2030:
                res.append((y, m, d))
        res = ['-'.join(map(str, r)) for r in res]
        return res

    def find_keyword(self, piece):
        for w in ['continued'] + self.key_word[::-1]:
            if w in piece:
                if w == 'continuationinpart':
                    w = 'continuation-in-part'
                return w
        return 'Not found'

    def format(self, fields: Union[str, List[str]]) -> Union[str, List[str]]:
        if type(fields) == str:
            if 'pct' in fields:
                return fields.upper()
            else:
                return fields.replace(',', '').replace('/', '')
        elif type(fields) == list:
            return [self.format(field) for field in fields]
        else:
            raise TypeError(f'Invalid type, expected: Union[str, List[str]], got {type(fields)}')

    def piece_pre_correction(self, piece: str) -> str:
        if 'substitute' in piece:
            piece = piece[:piece.index('substitute')]

        piece = piece.replace('0ct', 'oct').replace('scr', 'ser').replace('sec', 'ser')
        if 'se' in piece and 'ser' not in piece and 'sep' not in piece and 'pct' not in piece:
            piece = piece.replace('se', 'ser')

        if 'pct' in piece:
            slash_index = piece.index('pct') + 3
            if piece[slash_index] != '/':
                piece = piece[:slash_index] + '/' + piece[slash_index:]
        pct_pre = self.find_pattern(piece, 'pct_pre')
        if pct_pre:
            for pre in pct_pre:
                if len(pre) > 14:
                    piece = piece.replace(pre, pre[:10] + '/' + pre[10:])
                else:
                    piece = piece.replace(pre, pre[:8] + '/' + pre[8:])
        return piece

    def confirm_application(self, app, piece):
        res = []
        for a in app:
            for a_ in app:
                if a_ != a and a in a_:
                    break
            else:
                idx = piece.index(a)
                pre = piece[idx - 15:idx]
                if 'app' in pre or 'ser' in pre or 'tion' in pre:
                    res.append(a)
        return res

    def confirm_publication(self, pub, piece):
        res = []
        for p in pub:
            idx = piece.index(p)
            pre = piece[idx - 10:idx]
            if 'pat' in pre:
                res.append(p)
        return res

    def find_all_for_piece(self, piece):
        """
        Find everything needed in one piece of text from patent family paragraph
        """
        piece = self.piece_pre_correction(piece)
        kw = self.find_keyword(piece)
        app = self.find_pattern(piece, 'application')
        app = self.application_postprocessing(app, piece)
        app = self.confirm_application(app, piece)
        pct = self.find_pattern(piece, 'pct')
        pct = self.pct_postprocessing(pct)
        pub = self.find_pattern(piece, 'publication')
        pub = self.confirm_publication(pub, piece)
        date = self.find_date(piece)
        res = {
            'piece': piece,
            'keyword': kw,
            'application_number': self.format(app + pct),
            'publication_number': self.format(pub),
            'date': date
        }
        return res

    def pct_postprocessing(self, pct_result):
        return [r[:6] + r[6:].replace('o', '0') for r in pct_result]

    def application_postprocessing(self, app_result, piece):
        """
        Filter result by keywords
        """
        candidate = []
        for tup in app_result:
            candidate += [t for t in tup if t]
        res = []
        for cand in candidate:
            i = piece.index(cand)
            pre = piece[i - 8:i]
            for month in list(month2int.keys()) + ['pct']:
                if month in pre:
                    break
            else:
                res.append(cand)
        return res

    def extract_patent_family(self, left, right, paragraphs=None, **kwargs):
        if not paragraphs:
            paragraphs = self.find_multiple_target_texts(left, right, **kwargs)
        paragraphs = [self.keyword_correction(t) for t in paragraphs]

        res = []
        for paragraph in paragraphs:
            para_res = {}
            inid_code = paragraph.split(',', 1)[0]
            text = paragraph.split(',', 1)[1]
            para_res['inid_code'] = inid_code
            para_res['paragraph'] = text
            pieces = self.split_by_keywords(text)
            tuples = []
            for piece in pieces:
                tuples.append(self.find_all_for_piece(piece))
            para_res['tuples'] = tuples
            res.append(para_res)
        return res

    def extract_22(self, text, piece=None):
        res = {'piece': '-1', 'date': 'inid_code_22_not_found'}
        if not piece:
            for i in range(len(text)):
                if text[i].startswith('(22)'):
                    piece = text[i - 1] + text[i]
                    for j in range(i + 1, i + 5):
                        if j < len(text) and not text[j].startswith('('):
                            piece += text[j]
                        else:
                            break
                    break
        if piece:
            res['piece'] = piece
            dates = self.find_date(piece, '22date')
            if len(dates) == 1:
                res['date'] = dates[0]
            else:
                # fix US20060100288A1
                if '22' in piece:
                    first_p, second_p = piece.split('22', 1)
                    first_date = self.find_date(first_p, '22date')
                    second_date = self.find_date(second_p, '22date')
                    if len(first_date) > 0 and len(second_date) > 0:
                        res['date'] = second_date[0]
                    else:
                        res['date'] = dates[0]
                else:
                    res['date'] = dates[0]
        return res

    def extract_45(self, text):
        """
        extract date at upper-right corner
        """
        res = {'inid_code': '43', 'piece': ' | '.join(text), 'key_word': '', 'date': ''}
        text = [t.replace('publi', '') for t in text]
        if 'pub' not in ''.join(text):
            res['inid_code'] = '45'
            if 'reissue' not in ''.join(text):
                res['key_word'] = 'date of patent'
            else:
                res['key_word'] = 'date of reissued patent'
            for piece in text:
                date = self.find_date(piece)
                if date:
                    res['date'] = date[0]
                    break
        return res

    def extract_reissue(self, left, right):
        def string_after(s, t):
            u = s[s.index(t) + len(t):]
            return u

        def extract_one_date(p):
            date = self.find_pattern(p, 'date')[0]
            date_norm = self.find_date(date)[0]
            p = string_after(p, date)
            return date_norm, p

        piece = ''.join(self.find_target_text(left, right, ('(64)',))[0])
        res = {'paragraph': piece}
        if piece.startswith('-1') or ('issue' not in piece and 'appl' not in piece and 'patent' not in piece):
            res['paragraph'] = '-1,-1'
            return res
        try:
            patent_num = self.find_pattern(piece, 'publication')[0]
            piece = string_after(piece, patent_num)
            issue_date, piece = extract_one_date(piece)
            match = self.find_pattern(piece, 'application')[0]
            application_num = [m for m in match if m][0]
            piece = string_after(piece, application_num)
            filed_date, piece = extract_one_date(piece)
            res['patent_number'] = self.format(patent_num)
            res['issue_date'] = issue_date
            res['application_number'] = self.format(application_num)
            res['filed_date'] = filed_date

            # if 'pct' in piece:
            #     pct_num = self.find_pattern(piece, 'pct')[0]
            #     piece = string_after(piece, pct_num)
            #     two_date, piece = extract_one_date(piece)
            #     pct_pub_num = self.find_pattern(piece, 'pct_pub')[0]
            #     piece = string_after(piece, pct_pub_num)
            #     pct_pub_date, piece = extract_one_date(piece)
            #     res['pct_num'] = self.format(pct_num)
            #     res['2_date'] = two_date
            #     res['pct_pub_num'] = pct_pub_num
            #     res['pct_pub_date'] = pct_pub_date
        except IndexError:
            res['error'] = 'index error'
        return res

    def extract_pta_td(self, left, right):
        piece = ''.join(self.find_target_text(left, right, 'notice')[0])
        piece = self.keyword_correction(piece)
        res = {'paragraph': piece}

        # special case @US5556942A
        if 'shallnotextend' in piece:
            res['terminal_disclaimer'] = '1'
            pub = self.find_pattern(piece, 'publication')
            pub = self.confirm_publication(pub, piece)
            if pub:
                res['publication_number'] = self.format(pub)
            return res

        if piece.startswith('-1'):
            return res
        if '154' in piece:
            res['154'] = '1'
        if '156' in piece:
            res['156'] = '1'
        match = self.find_pattern(piece, 'bydays')
        if match:
            days = match[0]
            days = days.replace('o', '0').replace('l', '1')
            res['bydays'] = days
        if 'terminal' in piece:
            res['terminal_disclaimer'] = '1'
        date = self.find_date(piece)
        if date:
            res['date'] = date[0]
            res['terminal_disclaimer'] = '1'
        return res

    def extract_early(self, left, right):
        """
        Extract info from patent in 1960s~1970s
        """
        # check format
        start_collecting = False
        target_text = []
        for line in left:
            line = line.replace('ciaim', 'claim').replace('clalm', 'claim')
            if line.startswith('nodrawing') or line.startswith('no,drawing'):
                target_text.append(line)
                start_collecting = True
            elif 'claims' in line:
                break
            elif start_collecting:
                target_text.append(line)
        if not start_collecting:
            return

        target_text = ''.join(target_text)
        if 'filed' in target_text:
            patent_family_text, date_text = target_text.split('filed', 1)
        elif 'thisapp' in target_text:
            patent_family_text, date_text = target_text.split('thisapp', 1)
        else:
            return {'error': 'us early-phase patent text no splitting word.'}

        patent_family = self.extract_patent_family(None, None, [',' + patent_family_text])
        inid22_date = self.extract_22(None, date_text)
        patent_date = self.extract_45(right[1:8])
        res = {
            'inid22_date': inid22_date,
            'patent_date': patent_date,
            'reissue': {'paragraph': '-1,-1'},
            'pta_td': {'paragraph': '-1,-1'},
            'patent_family': patent_family
        }
        return res

    def pipeline(self, pdf_path, pdf_f_name=None, cache_dir=None):
        left, right = self.get_all_text(
            pdf_path,
            pdf_f_name=pdf_f_name,
            both_halves=True,
            cache_dir=cache_dir
        )
        if not left:
            return {'error': 'error when converting to image'}
        left, right = self.clean_ocr(left), self.clean_ocr(right)
        res = self.extract_early(left, right)
        if res:
            return res
        inid22_date = self.extract_22(left + right)
        patent_date = self.extract_45(right[1:8])
        reissue = self.extract_reissue(left, right)
        pta_td = self.extract_pta_td(left, right)
        patent_family = self.extract_patent_family(left, right, pdf_path=pdf_path, cache_dir=cache_dir)
        res = {
            'inid22_date': inid22_date,
            'patent_date': patent_date,
            'reissue': reissue,
            'pta_td': pta_td,
            'patent_family': patent_family
        }
        return res


class PapaEp(Papa):

    def __init__(
            self,
            save_dir='/home/clr/disk/playground/outputs/papa/',
            ocr_api='http://localhost/2txt'
    ):
        super(PapaEp, self).__init__(save_dir, ocr_api)
        self.key_word = ['divisionalapplication', 'art76epc']
        self.PATTERN = {
            'keyword': r'di.{15,19}on|ar.{3,5}pc',
            'application': r'(?<=\D)\d{9}',
            'publication': r'(?<!\D\d\d)\d{7}(?!\d|/)',
            'date': r'(?<=\D)\d\d/?\d\d/?\d\d(?=\D)',
            '22date': r'\d{6,8}'
        }
        self.right_offset = -5

    def clean_ocr(self, ocr_li):
        cleaned = []
        for line in ocr_li:
            line = self.map_brackets(line)
            line = line.replace(" ", "").replace(".", "")
            line = line.replace("(", "").replace(")", "")
            line = line.lower()
            cleaned.append(line)
        return cleaned

    def find_target_text(self, text):
        # find target text for extraction
        # locating places to extract
        target_text = []
        for i, line in enumerate(text):
            if 'divisionalapplication' in line:
                # filter special cases
                if text[i - 1].startswith("thisapplication"):
                    continue
                target_text_piece = [text[i]]
                if line.startswith('divisionalapplication') and 'date' not in text[i - 1]:
                    target_text_piece = [text[i - 1]] + target_text_piece
                target_text += self.find_target_text_last(text, i + 1, target_text_piece)
            elif 'art76' in line:
                target_text_piece = [text[i - 1], text[i]]
                if text[i - 2][0] == '6':
                    target_text_piece = [text[i - 2]] + target_text_piece
                target_text += self.find_target_text_last(text, i + 1, target_text_piece)
        return list(OrderedDict.fromkeys(target_text))

    def find_target_text_last(self, text, i, target_text):
        # find last lines of target text
        if len(text[i]) > 6 and text[i][-1] in digit and 'priority' not in text[i]:
            target_text += [text[i]]
            return self.find_target_text_last(text, i + 1, target_text)
        return target_text

    def keyword_correction(self, target_piece: str) -> str:
        match = re.findall(self.PATTERN['keyword'], target_piece)
        for m in match:
            for kw in self.key_word:
                if m != kw and distance(m, kw) <= 3:
                    target_piece = target_piece.replace(m, kw)
                    break
        return target_piece

    def find_inid_code_and_merge(self, target_text):
        codes = []
        target_text_merged = []
        curr = ''
        for piece in target_text:
            for code in ['60', '62']:
                if piece.startswith(code):
                    if curr:
                        target_text_merged.append(curr + '.')
                    curr = piece
                    codes.append(code)
                    break
            else:
                curr += '.' + piece
        target_text_merged.append(curr + '.')
        if len(codes) == 0 and len(target_text_merged) == 1:
            codes = ['']
        if len(codes) == len(target_text_merged):
            res = [{'inid_code': codes[i], 'paragraph': target_text_merged[i]} for i in range(len(codes))]
        else:
            raise ValueError(f'Something wrong with target text extraction, length of inid \
                codes {len(codes)}, length of target text {len(target_text_merged)}')
        return res

    def find_keyword(self, target_text):
        for kw in self.key_word:
            if kw in target_text:
                if kw == 'divisionalapplication':
                    kw = 'divisional application'
                elif kw == 'art76epc':
                    kw = 'Art.76'
                return kw
        return 'Not found'

    def find_pattern(self, piece, pattern_name, verbose=False):
        pattern = self.PATTERN[pattern_name]
        match = []
        match = re.findall(pattern, piece)
        if verbose:
            print(f'matched {pattern_name} number: {match}')
        return match

    def format_date(self, date_text):
        date_text = date_text.replace('/', '')
        day = date_text[:2]
        month = date_text[2:4]
        year = date_text[4:]
        if len(year) == 2:
            if int(year[0]) < 3:
                year = '20' + year
            else:
                year = '19' + year
        if 1 <= int(month) <= 12 and 1 <= int(day) <= 31 and 1929 < int(year) < 2025:
            date = '-'.join([year, month, day])
        else:
            date = ''
        return date

    def find_all(self, pieces):
        res = []
        for piece in pieces:
            para = piece['paragraph']
            kw = self.find_keyword(para)
            app = self.find_pattern(para, 'application')
            pub = self.find_pattern(para, 'publication')
            date = self.find_pattern(para, 'date')
            res.append({
                'paragraph': piece['paragraph'],
                'inid_code': piece['inid_code'],
                'keyword': kw,
                'application_number': app,
                'publication_number': pub,
                'date': self.format_date(date[0]) if date else ''
            })
        return res

    def extract_22(self, text):
        date = 'inid_code_22_not_found'
        for i in range(len(text)):
            if 'dateoffiling' in text[i]:
                date_text = self.find_pattern(text[i], '22date')
                if not date_text:
                    date_text = self.find_pattern(text[i] + text[i + 1], '22date')
                if date_text:
                    date_text = date_text[0]
                    date = self.format_date(date_text)
                    break
        return date

    def extract_45(self, text):
        date = 'inid_code_45_not_found'
        for i in range(len(text)):
            if text[i].startswith('45'):
                for j in range(i, i + 4):
                    piece = text[j]
                    date_text = self.find_pattern(piece, '22date')
                    if date_text:
                        date = self.format_date(date_text[0])
                        if date:
                            break
                break
        return date

    def pipeline(self, pdf_path, pdf_f_name=None, cache_dir=None):
        left = self.get_all_text(
            pdf_path,
            pdf_f_name=pdf_f_name,
            both_halves=False,
            del_img=True,
            cache_dir=cache_dir
        )
        if not left:
            return {'error': 'error when converting to image'}
        res = {}
        text = self.clean_ocr(left)
        res['inid22_date'] = self.extract_22(text)
        res['inid45_date'] = self.extract_45(text)
        target_text = self.find_target_text(text)
        target_text = [self.keyword_correction(piece) for piece in target_text]
        pieces = self.find_inid_code_and_merge(target_text)
        res['patent_family'] = self.find_all(pieces)
        return res


class PapaJp(Papa):

    def __init__(
            self,
            save_dir='/home/clr/disk/playground/outputs/papa/',
            ocr_api='http://localhost/2txt'
    ):
        super(PapaJp, self).__init__(save_dir, ocr_api)
        self.PATTERN = {
            'application': r'(\d{4}-\d{4,6})|(\D\d{1,2}-\d{5,6}(?=\D))',
            'date_exist': r'原出\w?日',
            'date45_exist': r'行日',
            'date22_exist': r'(?<!原)出\w?日',
            'date': r'(?<!\d|-|/)\d{6,8}(?=\D|$)',
            'date22_early': r'\d{5,6}月\d{1,2}日'
        }

    def clean_ocr(self, ocr_li):
        cleaned = []
        for line in ocr_li:
            line = self.map_brackets(line)
            line = line.replace(" ", "").replace("，", ",").replace(".", "")
            line = line.replace("(", "").replace(")", "").replace("－", '-')
            line = line.lower()
            cleaned.append(line)
        return cleaned

    def find_date(self, text, date_pattern):
        res = {'paragraph': '', 'date': ''}
        for j in range(len(text)):
            if re.search(self.PATTERN[date_pattern], text[j]):
                line = text[j - 1] + text[j] + text[j + 1]
                match = re.findall(self.PATTERN['date'], line)
                if match:
                    year = match[0][:4]
                    month = re.search(r'\d{1,2}(?=月)', line)
                    day = re.search(r'\d{1,2}(?=日)', line)
                    if not year or not month or not day:
                        date = 'something wrong'
                    else:
                        date = year + '-' + month.group() + '-' + day.group()
                    res['paragraph'] = line
                    res['date'] = date
                    break
        return res

    def find_application_num(self, target_piece):
        match = re.search(self.PATTERN['application'], target_piece)
        if match:
            p1, p2 = match.groups()
            if p1:
                application_num = p1
            else:
                application_num = p2
        else:
            application_num = 'something wrong'
        return application_num

    def find_all(self, text):
        res = {'patent_family': {'application_num': '', 'date': '', 'paragraph': ''}}
        for i, line in enumerate(text):
            if '分割' in line:
                target_piece = text[i - 1] + line + text[i + 1]
                res['patent_family']['application_num'] = self.find_application_num(target_piece)
                date_res = self.find_date(text[i + 1:], 'date_exist')
                res['patent_family']['date'] = date_res['date']
                res['patent_family']['paragraph'] = target_piece + ' | ' + date_res['paragraph']
                break
        res['inid22_date'] = self.find_date(text, 'date22_exist')
        res['inid45_date'] = self.find_date(text[:7], 'date45_exist')
        return res

    def extract_22_early(self, line):
        res = {'paragraph': '', 'date': ''}
        match = re.search(self.PATTERN['date22_early'], line)
        if match:
            date_text = match.group()
            year_month, day = date_text.split('月')
            if 1960 < int(year_month[:4]) < 2021 and 0 < int(year_month[4:]) < 13:
                year, month = year_month[:4], year_month[4:]
            else:
                year, month = year_month[1:5], year_month[5:]
            day = day[:-1]
            if not year or not month or not day:
                date = 'something wrong'
            else:
                date = year + '-' + month + '-' + day
            res['paragraph'] = date_text
            res['date'] = date
        return res

    def early_phase_extraction(self, pdf_path, pdf_f_name=None, cache_dir=None):
        text = self.get_all_text(
            pdf_path,
            pdf_f_name=pdf_f_name,
            full_page=True,
            del_img=True,
            cache_dir=cache_dir
        )
        text = self.clean_ocr(text)
        res = {'patent_family': {'application_num': '', 'date': '', 'paragraph': ''}}
        for i, line in enumerate(text):
            if '分割' in line:
                target_piece = text[i - 1] + line
                application_num = self.find_application_num(target_piece)
                res['patent_family']['application_num'] = application_num
                res['patent_family']['paragraph'] = target_piece
                res['inid22_date'] = self.extract_22_early(text[i - 2] + text[i - 1])
                res['inid45_date'] = self.find_date(text[:7], 'date45_exist')
                break
        return res

    def pipeline(self, pdf_path, pdf_f_name=None, cache_dir=None):
        text = self.get_all_text(
            pdf_path,
            pdf_f_name=pdf_f_name,
            both_halves=False,
            del_img=True,
            cache_dir=cache_dir
        )
        if not text:
            return {'error': 'error when converting to image'}
        text = self.clean_ocr(text)
        res = self.find_all(text)
        application_num = res['patent_family']['application_num']
        if not application_num and not res['inid22_date']['date']:
            res = self.early_phase_extraction(pdf_path)
        res['patent_family']['inid_code'] = '62' if application_num and application_num != 'something wrong' else ''
        return res


if __name__ == '__main__':
    pass
