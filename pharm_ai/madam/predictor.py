from pathlib import Path
from pharm_ai.perk.predictor import Ner
from mart.pdf_util.extractor_v3 import PDFTextExtractor
from tqdm import tqdm
import langid
import json
from itertools import chain, repeat
import re

class MadamPredictorV1:
    root = Path(__file__).parent
    version='v1'
    pdf_extractor = PDFTextExtractor()
    tasks = ['drug', 'disease']

    def __init__(self, cuda_device=0):
        self.ner_model = Ner('v2.1', cuda_device, 1)

    def predict(self, pdf_file: Path, return_fulltext=False):
        fulltext:str = self.pdf_extractor.extract_raw_text_from_pdf(pdf_file)
        paragraph = fulltext.split('\n')
        prefix, paras = zip(*[(pre, para) for para in paragraph for pre in self.tasks])
        raw_results = self.ner_model.predict(list(prefix), list(paras))
        res_dic = {}
        for pre in self.tasks:
            _, r = zip(*list(filter(lambda x: x[0]==pre, zip(prefix, raw_results))))
            res_dic[pre]=set(r_ for rr in r for r_ in rr if r_)
        results = (res_dic, fulltext) if return_fulltext else res_dic
        return results

def export_labeling_data(pdf_path='raw_data/ocr_pdf/PDF_xw', cuda_device=0,
                         result_json='results/extracted_v1.json',
                         to_label_jsonl = 'results/madam_tolabel_cn.json'):
    pdf_path = Path(pdf_path)
    res_json_path = Path(result_json)
    if not res_json_path.exists():
        all_pdfs = [f for d in pdf_path.iterdir() for f in d.iterdir() if f.suffix == '.pdf']
        predictor = MadamPredictorV1(cuda_device=cuda_device)
        results = []
        for pdf in tqdm(all_pdfs):
            extract_result, fulltext = predictor.predict(pdf, return_fulltext=True)
            extract_result = {k: list(extract_result[k]) for k in extract_result}
            lang = langid.classify(fulltext)[0]
            results.append({
                'FileName': pdf.name,
                'FullText': fulltext,
                'Language': lang,
                **extract_result
            })
        with res_json_path.open('w') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    else:
        with res_json_path.open('r') as f:
            results = json.load(f)
    exports = []
    for res in results:
        if res['Language']=='zh':
            labels=[]
            for key, val in chain(zip(repeat('drug'), res['drug']), zip(repeat('disease'), res['disease'])):
                for res_re in re.finditer(re.escape(val), res['FullT  gext']):
                    labels.append([res_re.start(), res_re.end(), key])
            labels.sort(key=lambda x:x[0])
            exports.append({'filename':res['FileName'], 'text': res['FullText'], 'labels': labels})
    with open(to_label_jsonl, 'w') as f:
        for item in exports:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    export_labeling_data(cuda_device=7)