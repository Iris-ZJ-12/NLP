import os
import re
import regex
from seq_clean import *
import json

# Protein symbols
protein = '(Xaa|xaa|Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val|Glx|Asx){e<=1}'
# DNA symbols
dna = 'a|c|t|g|u|n|w'

# Dictionaries that stores letters frequently get wrong
dna_dic = {'七': 't', 'o': 'c', 'q': 'g', 'y': 'g', '1': 't', '8': 'g',
           '(': 'c', 'i': 't', 'l': 't', 'd': 'a', 'ä': 'a'}
prt_dic = {'Pru': 'Pro', 'Gin': 'GLn', 'GIr': 'GLn', 'Tvr': 'Tyr',
           'Aia': 'Ala', 'GLv': 'GLy', '工Te': 'Ile', '工le': 'Ile',
           'IIe': 'Ile', 'Hle': 'Ile', 'Hrg': 'Arg', 'VaI': 'Val',
           'Va1': 'Val', 'Va7': 'Val', 'Lvs': 'Lys', 'Sor': 'Ser'}

# The dictionary for Protein conversion
aaList = {"Ala": "A", "Arg": "R", "Asn": "N",
          "Asp": "D", "Cys": "C", "Gln": "Q",
          "Glu": "E", "Gly": "G", "His": "H",
          "Ile": "I", "Leu": "L", "Lys": "K",
          "Met": "M", "Phe": "F", "Pro": "P",
          "Ser": "S", "Thr": "T", "Trp": "W",
          "Tyr": "Y", "Val": "V", "Glx": "Z",
          "Asx": "B", "Xle": "J", }
aaList.update(dict.fromkeys(["xaa", "Xaa", "Kys", "kys"], "X"))


def clean_sequence(pdf_path):
    """
        Clean unnecessary chars.
        Args:
            pdf_path: List, ocr result's path
        Outputs:
            res: List, cleaned bio-seq
            No210: int
            No400: int
    """
    # load OCR results
    with open(pdf_path) as f:
        data = f.read()
    new_lines = data.split('\n')
    No210 = 0
    No400 = 0
    res = []

    for line in new_lines:
        line = fix_b(line)
        line = full2half(line)
        # remove Chinese Characters
        line = re.sub(r'[\u4e00-\u9FFF]', '', line)
        line = re.sub(' ', '', line)
        if len(line) == 0:
            continue
        # check whether PRT or DNA. Replace wrong letter according to predefined Dict.
        if len(regex.findall(protein, line, flags=re.I)) * 3 / len(line) > 0.7:
            line = re.sub(r'\d+$', '', line)
            line = multiple_replace(prt_dic, line)
            line = swap_prt(line)
        elif len(re.findall(dna, line, flags=re.I)) / len(line) > 0.7:
            line = re.sub(r'\d+$', '', line)
            line = multiple_replace(dna_dic, line)

        # count the number of <210> and <400>, respectively
        if '<210>' in line:
            No210 += 1
        if '<400>' in line:
            No400 += 1
        res.append(line)

    return res, No210, No400


def merge_sequence(clean_seq_list, patentID):
    """
        Use Finite State Auto Machine to Merge multiple sequence lines and multiple Info lines, respectively.
        ex: '<210>1'                  '<210>1<211>DNA'      (bio-info)
            '<211>DNA'        --->    'actgagatactutggggg'  (dna sequence)
            'actgagat'
            'actutggggg'
        state == 0 : initial state
        state == 1 : At Bio-Info lines, merge info lines to one.
        state == 2 : PRT sequence lines, merge prt sequence lines to one.
        state == 3 : DNA sequence lines, merge dna sequence lines to one.
    """
    state = 0
    seq_id = 0
    cur = []
    result = []
    for l in clean_seq_list:
        # check if line is None
        if len(l) == 0:
            continue
        if len(l.split()) == 0:
            continue
        if l[:2] == '<1':
            continue

        # check if we should switch to state 1.
        # append cur to result; switch state;
        # create a new sequence prefix title according to patentID, seq_id.
        if re.search(r'<\d\d\d>', l) and state != 1:
            if cur:
                result.append("".join(cur))
            cur = []
            seq_id += 1
            prefix = f'>{patentID}_{seq_id} ' + l
            cur.append(prefix)
            state = 1
            continue

        # check if we should switch to state 2.
        if len(regex.findall(protein, l, flags=re.I)) * 3 / len(l) > 0.7 and state != 2 and state != 3:
            if cur:
                result.append("".join(cur))
            cur = []
            cur.append(l)
            state = 2
            continue

        # check if we should switch to state 2.
        if len(re.findall(dna, l, flags=re.I)) / len(l) > 0.7 and state != 2 and state != 3:
            if cur:
                result.append("".join(cur))
            cur = []
            cur.append(l)
            state = 3
            continue

        if state == 1:
            cur.append(l)

        if state == 2:
            if len(regex.findall(protein, l, flags=re.I)) * 3 / len(l) > 0.7:
                cur.append(l)

        if state == 3:
            if len(re.findall(dna, l, flags=re.I)) / len(l) > 0.7:
                cur.append(l)

    # Because we only append cur into result when switching to a new state.
    # For the last state will not switching anymore, must manually append last cur.
    # append last cur to res
    if len(cur) != 0:
        result.append("".join(cur))

    return result


def save_sequence(output_dir, merged_seq, patentID):

    dna_name = f'{output_dir}/{patentID}_dna.txt'
    prt_name = f'{output_dir}/{patentID}_prt.txt'
    dna_txt = open(dna_name, 'w')
    prt_txt = open(prt_name, 'w')

    i = 0
    while i + 1 < len(merged_seq):
        if not re.search('DNA|PRT', merged_seq[i]):
            i += 1
            continue
        matches = re.findall('DNA|PRT', merged_seq[i])
        matches = set(matches)
        if len(matches) > 1:
            if regex.search(protein, merged_seq[i + 1]):
                current_seq_type = 'PRT'
            else:
                current_seq_type = 'DNA'
        else:
            current_seq_type = list(matches)[0]

        if current_seq_type == 'PRT':

            if merged_seq[i + 1][0] == '>':
                i += 1
            else:
                # convert Prt symbols to short expression.
                # ex: 'ArgMetPhePro' --> 'RMFP'
                string = multiple_replace(aaList, merged_seq[i + 1])

                # replace wrong letters with X
                # ex: 'RaaMFPddad' --> 'RXMFPXX'
                #    1-3 letters --> X ; 4-6 letters --> XX ; 7-9 letters --> XXX
                if re.search('[a-z]+', string):
                    matches = re.findall('[a-z]+', string)
                    matches = sorted(re.findall('[a-z]+', string), key=len)[::-1]
                    for m in matches:
                        if len(m) < 4:
                            string = re.sub(m, 'X', string)
                            continue
                        if len(m) < 7:
                            string = re.sub(m, 'XX', string)
                            continue
                        if len(m) < 10:
                            string = re.sub(m, 'XXX', string)
                            continue
                # For Info string, remove everything except a-z A-Z, 0-9,<,> and white space.
                # And write it in prt_txt.
                prt_txt.write(re.sub(r'[^\sa-zA-Z0-9<>]', '', merged_seq[i]) + '\n')
                # For PRT string, remove everything except A-Z.
                # And write it in prt_txt.
                prt_txt.write(re.sub('[^A-Z]', '', string) + '\n')
                i += 2
        else:
            if merged_seq[i + 1][0] == '>':
                i += 1
            else:
                dna_txt.write(re.sub(r'[^\sa-zA-Z0-9<>]', '', merged_seq[i]) + '\n')
                dna_txt.write(re.sub('[0-9]', '', merged_seq[i + 1]) + '\n')
                i += 2

    dna_txt.close()
    prt_txt.close()


if __name__ == '__main__':
    errors = {'empty': [], 'diff>20%': [], 'no 210 or 400': []}
    path1 = '/home/zb/disk/pycharm/pharm_ai/poker/20211122055413992/OCR-2021-raw'
    path2 = '/home/zb/disk/pycharm/pharm_ai/poker/CN-NOREAD-SEQ-raw'
    output_dir = './RYZ_CN_raw'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for p in os.listdir(path1):
        if not p.endswith('.txt'):
            continue
        pdf_name = os.path.join(path1, p)
        patentID = p.split('.')[0]
        print(pdf_name)
        clean_seq_list, No210, No400 = clean_sequence(pdf_name)

        if clean_seq_list == 0:
            errors['empty'].append(pdf_name)
            continue
        if min(No400, No210) == 0:
            errors['no 210 or 400'].append(pdf_name)
        elif abs(No400 - No210) / max(No400, No210) > 0.2:
            errors['diff>20%'].append(pdf_name)

        merged_sequence = merge_sequence(clean_seq_list, patentID)
        save_sequence(output_dir, merged_sequence, patentID)
    for p in os.listdir(path2):
        if not p.endswith('.txt'):
            continue
        pdf_name = os.path.join(path2, p)
        patentID = p.split('.')[0]
        print(pdf_name)
        clean_seq_list, No210, No400 = clean_sequence(pdf_name)

        if clean_seq_list == 0:
            errors['empty'].append(pdf_name)
            continue
        if min(No400, No210) == 0:
            errors['no 210 or 400'].append(pdf_name)
        elif abs(No400 - No210) / max(No400, No210) > 0.2:
            errors['diff>20%'].append(pdf_name)
        merged_sequence = merge_sequence(clean_seq_list, patentID)
        save_sequence(output_dir, merged_sequence, patentID)

    with open(os.path.join(output_dir, 'info_list.json'), 'w') as outfile:
        json.dump(errors, outfile, indent=4)