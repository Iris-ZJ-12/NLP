from Levenshtein import distance
from collections import defaultdict
import re
import regex
from PIL import ImageOps

protein_pattern = '(Xaa|xaa|Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val|Glx|Asx){e<=1}'
dna_pattern = 'a|c|t|g|u|n|w'


def padding_resize(image, t_w, t_h):
    """
        padding the image to the target size(t_w, t_h)
        Args:
            image: a input image
            t_w: target width
            t_h: target height
        Returns:
            padded image
    """
    w, h = image.size
    if w > t_w:
        h = h * t_w / w
        w = t_w
    if h > t_h:
        w = w * t_h / h
        h = t_h
    newsize = (int(w), int(h))
    image = image.resize(newsize)
    delta_h = t_h - newsize[1]
    if newsize[1] < 1000:
        padding = (0, 0, 0, int(delta_h))
        return ImageOps.expand(image, padding, fill=(255, 255, 255))
    else:
        return image


def multiple_replace(adict, text):
    """
        Force to Replace frequent wrong letters made by OCR recognition.
        dna_dic = {'七': 't', 'o': 'c', 'q': 'g', 'y': 'g',
           '1': 't', '(': 'c', 'i': 't', 'l': 't', 'd': 'a', 'ä': 'a'}
        prt_dic = {'Pru': 'Pro', 'Gin': 'GLn', 'GIr': 'GLn', 'Tvr': 'Tyr',
           'Aia': 'Ala', 'GLv': 'GLy', '工Te': 'Ile', '工le': 'Ile',
           'IIe': 'Ile', 'Hle': 'Ile', 'Hrg': 'Arg', 'VaI': 'Val',
           'Va1': 'Val', 'Va7': 'Val', 'Lvs': 'Lys', 'Sor': 'Ser'}
    """
    # Create a regular expression from all of the dictionary keys
    regex = re.compile("|".join(map(re.escape, adict.keys())))
    # For each match, look up the corresponding value in the dictionary
    return regex.sub(lambda match: adict[match.group(0)], text)


def swap_prt(seq_string):
    """
        Given a protein string (ex: "Ala Va Gln a his"):
            1. Replace wrong format prt. ex: his --> His
            2. Remove single extra letter. ex: a --> ''
            3. Replace falsely predicted prt based on similarity. ex Va --> Val  Sor --> Ser
    """
    # split the string such as: 'AlAspAspMGluGIy' --> ['Al', 'Asp', 'Asp','Glu','GIy']
    # find all protein when error<=1. (edit distance)
    input_txt = regex.findall(protein_pattern, seq_string)

    # Replace lower letter Protein to capital. ex: arg His met --> Arg His Met
    swap_dict = defaultdict(list)
    Proteins = ['Xaa', 'Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly', 'His', 'Ile',
                'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser', 'Thr', 'Trp', 'Tyr', 'Val']
    for prt in Proteins:
        input_txt = ' '.join(input_txt)
        input_txt = re.sub(prt, prt, input_txt, flags=re.I)
        input_txt = input_txt.split(' ')

    # Find 1 edit distance pair between predicted prt and true prt, and swap with true prt.
    # Ignore edit distance > 1
    # ex: (Alo, Ala): Alo --> Ala
    for i in range(len(input_txt)):
        for prt in Proteins:
            # Not need compute distance, if the input prt in true prt set. Just skip to next input.
            if input_txt[i] in Proteins:
                break
            if distance(input_txt[i], prt) == 1:
                swap_dict[i].append(prt)

    # Swap Wrong prt with true prt, only if exists on candidate.
    # For multiple candidates, just ignore.
    for k, v in swap_dict.items():
        if len(v) == 1:
            input_txt[k] = v[0]

    # Remove single letter
    for i in input_txt:
        if len(i) == 1:
            input_txt.remove(i)
    return ''.join(input_txt)


def fix_b(s):
    """
        fix missing or wrong brackets. ex: <210》 --> <210> or 210> --> <210> or (210> --> <210>
        Args:
            s: a string
        return:
            s: a string
    """
    s = re.sub('》', '>', s)
    s = re.sub('《', '<', s)

    if re.search(r'((?=<).)\d\d\d(.(?<!>)|$)', s):
        i, j = re.search(r'((?=<).)\d\d\d(.(?<!>)|$)', s).span()
        if j - i == 5:
            s = s[:j - 1] + '>' + s[j - 1:]
        else:
            s = s + '>'

    elif re.search(r'(^|(?!<).)\d\d\d(.(?<=>))', s):
        i, j = re.search(r'(^|(?!<).)\d\d\d(.(?<=>))', s).span()
        if j - i == 5:
            s = '<' + s[1:]
        else:
            s = '<' + s
    return s


def full2half(s):
    """
        swap full character to half character
    """
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        num = chr(num)
        n.append(num)
    return ''.join(n)


# def fix_letter(input_txt, dna_dic, prt_dic):
#     state = 0
#     new_list = []
#     i = 0
#     while i < len(input_txt):
#         if '<210>' in input_txt[i] or re.search(r'<([2])\d\d>', input_txt[i]):
#             new_list.append(input_txt[i])
#             i += 1
#             state = 1
#             continue
#
#         if '<400>' in input_txt[i]:
#             new_list.append(input_txt[i])
#             i += 1
#             state = 2
#             if i >= len(input_txt):
#                 continue
#
#             if len(re.findall(protein_pattern, input_txt[i], flags=re.I)) * 3 / len(input_txt[i]) > 0.6:
#                 continue
#             elif len(re.findall(dna_pattern, input_txt[i], flags=re.I)) * 3 / len(input_txt[i]) > 0.6:
#                 continue
#             else:
#                 new_list[-1] = new_list[-1] + input_txt[i]
#                 i += 1
#                 continue
#
#         if state == 1 and '<' not in input_txt[i]:
#             new_list[-1] = new_list[-1] + input_txt[i]
#             i += 1
#             continue
#
#         if state == 2:
#             if re.search(protein_pattern, input_txt[i]):
#                 input_txt[i] = re.sub(r'\d+$', '', input_txt[i])
#                 input_txt[i] = swap_prt(input_txt[i])
#                 input_txt[i] = re.sub(r'[\W]', '', input_txt[i])
#                 input_txt[i] = ''.join(input_txt[i])
#                 input_txt[i] = multiple_replace(prt_dic, input_txt[i])
#             elif re.search(dna_pattern, input_txt[i]):
#                 input_txt[i] = re.sub(r'\d+$', '', input_txt[i])
#                 input_txt[i] = re.sub(r'[\W]', '', input_txt[i])
#                 input_txt[i] = multiple_replace(dna_dic, input_txt[i])
#             else:
#                 pass
#             input_txt[i] = re.sub(r'[\u4e00-\u9FFF]', '', input_txt[i])
#             input_txt[i] = re.sub(r'[\W]', '', input_txt[i])
#             new_list.append(input_txt[i])
#             i += 1
#             continue
#         i += 1
#     return new_list
#
#
# def check_error(input_txt):
#     state = 0
#     i = 0
#     curr_len = 0
#     seq_len = 0
#     seq_id = 0
#     curr_210_index = 0
#     curr_type = ''
#     errors = []
#     while i < len(input_txt):
#         if re.search('<210>', input_txt[i]):
#             try:
#                 seq_id = int(re.findall(r'\d+', re.sub('<210>', '', input_txt[i]))[0])
#             except:
#                 pass
#
#             if curr_type == 'PRT':
#                 if not (curr_len / 3).is_integer():
#                     input_txt[curr_210_index] = input_txt[
#                                                     curr_210_index] + ' 错[Seq_length:{},current_length:{}]'.format(
#                         seq_len, curr_len / 3)
#                     # print("seq_{} PRT has a length error".format(seq_id))
#                     errors.append("seq_{} PRT has a length error".format(seq_id))
#                 elif curr_len / 3 != seq_len:
#                     input_txt[curr_210_index] = input_txt[
#                                                     curr_210_index] + ' 错[Seq_length:{},current_length:{}]'.format(
#                         seq_len, curr_len / 3)
#                     # print("seq_{} PRT has a length error".format(seq_id))
#                     errors.append("seq_{} PRT has a length error".format(seq_id))
#             elif curr_type == 'DNA':
#                 # print(curr_len, seq_len)
#                 if curr_len != seq_len:
#                     input_txt[curr_210_index] = input_txt[
#                                                     curr_210_index] + ' 错[Seq_length:{},current_length:{}]'.format(
#                         seq_len, curr_len)
#                     # print("seq_{} DNA has a length error".format(seq_id))
#                     errors.append("seq_{} DNA has a length error".format(seq_id))
#             curr_210_index = i
#             state = 0
#             i += 1
#             continue
#
#         if re.search('<211>', input_txt[i]):
#             try:
#                 seq_len = int(re.findall(r'\d+', re.sub('<211>', '', input_txt[i]))[0])
#             except:
#                 pass
#             curr_len = 0
#
#         if re.search('<212>', input_txt[i]):
#             if re.search('PRT', input_txt[i]):
#                 curr_type = 'PRT'
#             else:
#                 curr_type = 'DNA'
#
#         if re.search('<400>', input_txt[i]):
#             if seq_id == 0:
#                 try:
#                     seq_id = int(re.findall(r'\d+', re.sub('<400>', '', input_txt[i]))[0])
#                 except:
#                     pass
#             state = 2
#             i += 1
#             continue
#
#         if state == 2:
#             if curr_type == 'PRT':
#                 if re.search(
#                         r'[^Xaa|xaa|Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|'
#                         r'Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val|Glx|Asx|\d]',
#                         input_txt[i]):
#                     # print('seq_{} Found Wrong PRT Letter!'.format(seq_id))
#                     errors.append('seq_{} Found Wrong PRT Letter!'.format(seq_id))
#                     s, e = re.search(
#                         r'[^Xaa|xaa|Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|'
#                         r'Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val|Glx|Asx|\d]',
#                         input_txt[i]).span()
#                     input_txt[i] = input_txt[i][0:s] + '错' + input_txt[i][s:]
#                 if re.search(r'[\d][a-zA-Z]|[a-zA-Z][\d]', input_txt[i]):
#                     s, e = re.search(r'[\d][a-zA-Z]|[a-zA-Z][\d]', input_txt[i]).span()
#                     input_txt[i] = input_txt[i][0:s] + '错' + input_txt[i][s:]
#
#                 curr_len += len(re.sub(r'\d', '', input_txt[i]))
#             else:
#                 if re.search(protein_pattern, input_txt[i]):
#                     pass
#                 else:
#                     if re.search(r'[^a|c|t|g|u|s|\d]', input_txt[i]):
#                         # print('seq_{} Found Wrong DNA Letter!'.format(seq_id))
#                         errors.append('seq_{} Found Wrong DNA Letter!'.format(seq_id))
#                         s, e = re.search(r'[^a|c|t|g|u|s|\d]', input_txt[i]).span()
#                         input_txt[i] = input_txt[i][0:s] + '错' + input_txt[i][s:]
#                     if re.search(r'[\d][a-zA-Z]|[a-zA-Z][\d]', input_txt[i]):
#                         s, e = re.search(r'[\d][a-zA-Z]|[a-zA-Z][\d]', input_txt[i]).span()
#                         input_txt[i] = input_txt[i][0:s] + '错' + input_txt[i][s:]
#                     curr_len += len(re.sub(r'\d', '', input_txt[i]))
#         i += 1
#     return input_txt, errors
