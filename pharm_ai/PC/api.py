from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
from pharm_ai.PC.predictor import T5Predictor, PredictorSum
from mart.api_util.ecslogger import ECSLogger
from typing import Dict, List, Tuple
from argparse import ArgumentParser

arg_parser = ArgumentParser()
arg_parser.add_argument("-c", "--cuda-device", type=int, default=-1, help="CUDA device. DEFAULT: -1.")
arg_parser.add_argument("-p", "--port", type=int, default=21310, help="API listening port. DEFAULT: 21310")
arg_parser.add_argument("-v", "--version", type=int, default=4, choices=[3, 4], help="Deploy version. DEFAULT: 4.")
arg_parser.add_argument("-b", "--batch-size", type=int, default=1, help="Batch size. DEFAULT: 1.")

args = arg_parser.parse_args()

description = """
本文档提供**专利文献4分类系统**的接口文档。

给定专利权利要求书文本，首先识别该文本属于从权（`claim-dependent`）还是独权（`claim-independent`）；
若属于独权，进一步判断其类别（`claim_type`）；若属于从权，进一步识别其从属的claims（`claim_ref`）。

另外，给定一个申请人名称，判断其类型。


## 调用接口说明

**请求数据**的格式见相应部分的**Parameters->Request body->Schema**部分，
参考样例详见相应部分的**Parameters->Request body->Example Value**部分。

测试接口请点击相应部分的**Try it out**。

**响应数据**的格式见相应部分的**Responses->Successful Response (code 200)->Description/Schema**部分，
参考样例详见相应部分的**Responses->Successful Response (code 200)->Example Value**部分。

## 更新日志：

- v2:
    - 提高了提取从属claim的速度
    - 解除提取从属claim时长度为100的限制
- v3:
    - 新增“申请人(法人/自然人)分类识别”的接口
    """
description_v4 = """
- v4:
    - 申请人分类v2: 类型由“法人/自然人”细分为更多种类型

## 修复bug:
- 2021-7-1:
    - 修复文本带canceled词但claim_type预测不为#的问题

"""

if args.version == 4:
    p = T5Predictor(version='v4.0', cuda_device=args.cuda_device)
    description += description_v4
    update_model_args = {'eval_batch_size': args.batch_size}
    p.model.args.update_from_dict(update_model_args)
elif args.version == 3:
    p = PredictorSum(subtask_versions=('v1', 'v1', 'v2.6'),
                     cuda_device=(args.cuda_device, args.cuda_device, args.cuda_device))

app = FastAPI(
    title='专利文献4分类系统 \nPC\n(Patent Classifiers)',
    version=f"v{args.version}",
    description=description,
    openapi_tags=[
        {
            "name": "Patent",
            "description": "专利相关接口"
        },
        {
            "name": "Other",
            "description": "其他接口"
        }
    ]
)

ecslogger = ECSLogger('PC', fastapi_app=app)

class PatentInput(BaseModel):
    """
    输入字段包括**ID**（`claim_id`）、**编号**（`claim_num`）、**句子**（`claim_text`）。
    """
    patent_input: Dict[str, List[Tuple[int, str]]] = Field(
        ...,
        example={'1d6dd1414184c5e9fc55515ee65de016': [
            [2, '2. The apparatus of claim 1, wherein the first bevel angle is less than the second bevel angle.'],
            [3,
             '3. The apparatus of claim 1, wherein the first bevel angle is less than 20 degrees and the second bevel '
             'angle is greater than the first bevel angle.'],
            [5,
             '5. The apparatus of claim 1, wherein a ratio of a bevel height to a bevel width is less than about 2.5.'],
            [18,
             '18. The apparatus of claim 12, wherein the beveled surface defines an inside angle of greater than 30 '
             'degrees.'],
            [23,
             '23. A method, comprising: inserting a microneedle into an eye such that a distal edge defined by a '
             'beveled surface of the microneedle does extend through the choroid of the eye, the beveled surface '
             'defining a tip angle of less than about 20 degrees, the beveled surface having a height such that an '
             'opening defined by the beveled surface is within at least one of a suprachoroidal space or a lower '
             'portion of the sclera; and conveying a substance from a cartridge coupled to a proximal end portion of '
             'the microneedle into the suprachoroidal space via the opening defined by the beveled surface.'],
            [24,
             '24. The method of claim 23, wherein the inserting includes inserting the microneedle via a target '
             'location of the surface of the eye, a center line of the microneedle defining an angle with a plane '
             'tangent to the target location of between 80 and 100 degrees.'],
            [26,
             '26. The method of claim 23, wherein the inserting is independent of an angular orientation about a '
             'center line of the microneedle.'],
            [14,
             '14. The apparatus of claim 12, wherein an outer diameter of a shaft portion of the microneedle is '
             'substantially constant and an inner diameter of the microneedle is about 120 µm or less.'],
            [28,
             '28. The method of claim 23, wherein the conveying includes defining, at least in part, '
             'the suprachoroidal space.'],
            [6, '6. The apparatus of claim 1, wherein the first bevel angle is less than about 18 degrees.'],
            [9, '9. The apparatus of claim 1, wherein at least a portion of the bevel surface is curved.'],
            [11,
             '11. The '
             'apparatus of claim 1, wherein an outer diameter of the microneedle is substantially constant and an '
             'inner diameter of the microneedle is about 120 µm or less.'],
            [12,
             '12. An apparatus, comprising: a microneedle having a distal end portion and a proximal end portion, '
             'and defining a lumen, the proximal end portion configured to be coupled to a cartridge to place the '
             'lumen in fluid communication with the cartridge, the proximal end portion including a base surface '
             'configured to contact a surface of a target tissue, the distal end portion including a beveled surface, '
             'the beveled surface defining a tip angle of less than about 20 degrees and a ratio of a bevel height to '
             'a bevel width of less than about 2.5.'],
            [15, '15. The apparatus of claim 12, wherein the tip angle is less than about 18 degrees.'],
            [19, '19. The apparatus of claim 12, wherein the bevel height is less than about 500 lam.'],
            [21,
             '21. The '
             'apparatus '
             'of claim '
             '12, '
             'wherein '
             'the base '
             'surface '
             'substantially circumscribes a shaft portion of the microneedle.'],
            [27,
             '27. The method of claim 23, wherein the inserting includes inserting the microneedle such that a base '
             'of the microneedle contacts a target surface of the eye.'],
            [29,
             '29. The method of claim 23, wherein the substance is at least one of a VEGF, a VEGF inhibitor, '
             'or a combination thereof.'],
            [1,
             '1. An apparatus, comprising: a microneedle having a distal end portion and a proximal end portion, '
             'and defining a lumen, the proximal end portion configured to be coupled to a cartridge to place the '
             'lumen in fluid communication with the cartridge, the proximal end portion including a base surface '
             'configured to contact a surface of a target tissue, the distal end portion including a beveled surface, '
             'the beveled surface defining a first bevel angle and a second bevel angle different than the first '
             'bevel angle.'],
            [4,
             '4. The apparatus of claim 1, wherein: the first bevel angle is a tip angle; and the second bevel angle '
             'is an inside angle.'],
            [8, '8. The apparatus of claim 1, wherein the second bevel angle is greater than about 45 degrees.'],
            [13, '13. The apparatus of claim 12, wherein the microneedle is rigid and is 30 gauge or smaller.'],
            [17, '17. The apparatus of claim 12, wherein at least a portion of the bevel surface is curved.'],
            [7, '7. The apparatus of claim 1, wherein the first bevel angle is less than about 12 degrees.'],
            [10,
             '10. '
             'The '
             'apparatus of claim 1, wherein: a bevel height is less than about 500 µm; and a bevel width is less than '
             'about 320 µm.'],
            [16,
             '16. The apparatus of claim 12, wherein the ratio of the bevel height to the bevel width less than about '
             '2.2.'],
            [20,
             '20. The apparatus of claim 12, further comprising: the cartridge coupled to the proximal end portion of '
             'the microneedle, the cartridge configured to contain at least one of a VEGF, a VEGF inhibitor, '
             'or a combination thereof.'],
            [22,
             '22. The apparatus of claim 12, wherein the base surface is substantially normal to a center line of the '
             'lumen of the microneedle.'],
            [25,
             '25. The method of claim 23, wherein, the inserting includes inserting the microneedle substantially '
             'normal to a target surface of the eye.']],
            '1a445e7cd9be988c0438edd2568cd098': [[8,
                                                  '9. The octapeptide of claim 1, of the formula '
                                                  'pentafluoro-D-Phe-Cys-Tyr-D-Trp-Lys-Val-Cys-Thr-NH2, '
                                                  'or a pharmaceutically acceptable salt thereof.'],
                                                 [10,
                                                  '11· The octapeptide of claim 1 of the formula N-Ac-D- '
                                                  '8-Nal-Cys-Tyr-D-Trp-Lys-Val-Cys-Thr-NH2, '
                                                  'or a pharmaceutically acceptable salt thereof.'],
                                                 [14,
                                                  '15. The octapeptide of claim 1 of the formula D- 8 '
                                                  '-Nal-Cys-Tyr-D-Trp-Lys- a -aminobutyric acid-CysThr-NH2. '
                                                  '-1016. The octapeptide of claim 1 of the formula: '
                                                  'D-PheCys-B-Nal-D-Trp-Lys-Val-Cys-Thr-NH2.'],
                                                 [16,
                                                  '19. The octapeptide of claim 1 of the formula: '
                                                  'D-PheCys-Tyr-D-Trp-Lys-Thr-Cys-Nal-NH2.'],
                                                 [17,
                                                  '20. The octapeptide of claim 1 of the formula: 3-0('
                                                  'ascorbic)-butyryl-D-Nal-Cys-Tyr-D-Trp-Lys-Val-Cys-Thr-NH2. '
                                                  '21· A therapeutic composition capable of inhibiting the '
                                                  'release of growth hormone, insulin, glucagon, or pancreatic '
                                                  'exocrine secretion comprising a therapeutically effective '
                                                  'amount of the compound of claim 1 together with a '
                                                  'pharmaceutically acceptable carrier substance. 22. The '
                                                  'therapeutic composition of claim 21 wherein said composition '
                                                  'is in the form of a pill, tablet, or capsule for oral '
                                                  'administration to a human patient in need of said compound. '
                                                  '23. The therapeutic composition of claim 21 wherein said '
                                                  'composition is in the form of a liquid for oral '
                                                  'administration to a human patient in need of said compound. '
                                                  '24. The therapeutic composition of claim 22, said composition '
                                                  'being coated with a substance capable of protecting said '
                                                  'composition from the gastric acid in the stomach of said '
                                                  'human patient for a period of time sufficient to allow said '
                                                  'composition to pass undisintegrated into the small intestine '
                                                  'of said human patient. -11r 25. The therapeutic composition '
                                                  'of claim 21, said composition being in the form of a cream, '
                                                  'gel, spray, or ointment for application to the skin of a '
                                                  'human patient in need of said compound. 26. The therapeutic '
                                                  'composition of claim 21, said composition being in the form '
                                                  'of a liquid capable of being administered nasally as drops or '
                                                  'spray to a human patient in need of said compound. 27. The '
                                                  'therapeutic composition of claim 21, said composition being '
                                                  'in the form of a liquid for intravenous, subcutaneous, '
                                                  'parenteral, or < intraperitioneal administration to a human '
                                                  'patient in need of said compound. 28. The therapeutic '
                                                  'composition of claim 21, said composition being in the form '
                                                  'of a biodegradable sustained release composititon for '
                                                  'intramuscular administration to a human patient in need of '
                                                  'said compound. 29. An octapeptide as claimed in any one of '
                                                  'claims 1 to 20 for use in medicine. 30. The use of an '
                                                  'octapeptide as claimed in any one of claims 1 to 20 in the '
                                                  'preparation of a medicament for the reduction of growth '
                                                  'hormone, insulin, glucagon, or pancreatic exocrine secretion. '
                                                  '-1231. An octapeptide of the formula given and defined in '
                                                  'claim 1, or a pharmaceutically acceptable salt thereof, '
                                                  'substantially as hereinbefore described. 32. A process for '
                                                  'the preparation of an 5 octapeptide of the formula given and '
                                                  'defined in claim 1, substantially as hereinbefore described. '
                                                  '33. An octapeptide of the formula given and defined in claim '
                                                  '1, whenever prepared by a process claimed in claim 32. 10 34. '
                                                  'A therapeutic composition according to claim'],
                                                 [1,
                                                  "Claims 1. An octapeptide of the formula; A A *3 -CH-CO-Cys-A4 "
                                                  "-D-Trp-Lys-Ag Cys-A?-NH2, wherein each A^ and A2, "
                                                  "independently, is H, C1_12 alkyl, C^_^gphenylalkyl, "
                                                  "R^CO (where R^ is C1_20 alkyl, C2-20 alkenY1' C3-20 alkynYi» "
                                                  "phenyl, naphthyl, or C7_^q phenylalkyl), or R2OCO (where R2 "
                                                  "is alkYl or C7-io Phenylalkyl) r provided that when one of A^ "
                                                  "or A^ is R^CO or R2OCO, the other must be H; Ag is CH2~Ag ("
                                                  "where Ag is pentafluorophenyl, naphthyl, pyridyl, or phenyl); "
                                                  "A4 is o- m- or p-substituted X-Phe (where X is a halogen, H, "
                                                  "NO2, OH, NH2, or C^_g alkyl), pentafluoro-Phe, or β -Nal; Ag "
                                                  "is Thr, Ser, Phe, Val, a -aminobutyric acid, or Ile, "
                                                  "provided that when Ag is phenyl, A^ is H, and A2 is H, "
                                                  "Ag cannot be Val; and A? is Thr, Trp, or 8 -Nal; or a "
                                                  "pharmaceutically acceptable salt thereof."],
                                                 [2,
                                                  '2. The octapeptide of claim 1 wherein is D-β-naphthylalanine.'],
                                                 [11,
                                                  '12. The octapeptide of claim 1 of the formula D- '
                                                  '8-Nal-Cys-pentafluoro-Phe-D-Trp-Lys-Val-Cys-ThrNH2, '
                                                  'or pharmaceutically acceptable salt thereof.'],
                                                 [4, '4. The octapeptide of claim 1 wherein naphthyl.'],
                                                 [12,
                                                  '13. The '
                                                  'octapeptide of claim 1 of the formula D- '
                                                  '8-Nal-Cys-Tyr-D-Trp-Lys-Val-Cys- β -Nal-NH2, or a pharmaceutically '
                                                  'acceptable salt thereof.'],
                                                 [15,
                                                  '17. The octapeptide of claim 1 of the formula: '
                                                  'D-PheCys-Tyr-D-Trp-Lys-Abu-Cys-Nal-NH218. The octapeptide of claim '
                                                  '1 of the formula: D-NalCys-Tyr-D-Trp-Lys-Thr-Cys-Thr-NH2.'],
                                                 [6,
                                                  '6. The octapeptide of claim 1, wherein Ag is pentafluorophenyl. A. '
                                                  'A., I1!2 3 4 5 6 N-CH-CO I A2 A.A, I1!3 N-CH-CO I A„ IS -97. The '
                                                  'octapeptide of claim 1 wherein N—CH—CO is D-Phe or D-Nal and Ας is '
                                                  'Val or Thr. I A2'],
                                                 [3,
                                                  '3. The octapeptide of claim 1 wherein is D-Phe and Ag is a - '
                                                  'aminobutyric acid.'],
                                                 [5, '5. The octapeptide of claim 1, wherein R^ is CHg or C2Hg.'],
                                                 [7,
                                                  '8. The octapeptide of claim 1, of the formula D- '
                                                  'β-Nal-Cys-Tyr-D-Trp-Lys-Val-Cys-Thr- NH2, or a pharmaceutically '
                                                  'acceptable salt thereof.'],
                                                 [9,
                                                  '10. The octapeptide of claim 1 of the formula '
                                                  'D-Phe-Cys-Tyr-D-Trp-Lys- a-aminobutyric acidCys-Thr-NH2, '
                                                  'or a pharmaceutically acceptable salt thereof.'],
                                                 [13,
                                                  '14. The octopeptide of claim 1 of the formulas '
                                                  'D-Phe-Cys-Tyr-D-Trp-Lys-Val-Cys- 8-Nal-NH2, or a pharmaceutically '
                                                  'acceptable salt thereof.'],
                                                 [18, '21, substantially as hereinbefore described.']]
        }
    )


def convert_ecs_input(input: BaseModel) -> Dict:
    input_dic = input.dict()
    result = []
    for esid, v in input_dic['patent_input'].items():
        res_item = {'esid': esid}
        values = [{'claim_num': vv[0], 'claim_text': vv[1]} for vv in v]
        res_item.update({'values': values})
        result.append(res_item)
    return {'patent_input': result}


def convert_ecs_output(output: dict) -> Dict:
    result = []
    for esid, v in output.items():
        res_item = {'esid': esid}
        values = [{'claim_num': vv[0], 'claim_class': vv[1], 'claim_type': vv[2], 'claim_ref': vv[3]} for vv in v]
        res_item.update({'result': values})
        result.append(res_item)
    return {'result': result}


@app.post("/", tags=['Patent'], response_model=Dict[str, List[Tuple[int, str, str, list]]],
          responses={
              200: {
                  "description": """
输出字段包括**ID**（`claim_id`）、**编号**（`claim_num`）、**从权/独权类型**（`claim_class`）、
**独权类型**（`claim_type`）、**从属的claim**（`claim_ref`）。

输出格式：
```
{
    claim_id: [
        [claim_num, claim_class, claim_type, claim_ref]
    ]
}
```
""",
                  "content": {
                      "application/json": {
                          "example": {
                              "1a445e7cd9be988c0438edd2568cd098": [
                                  [
                                      8,
                                      "dependent-claim",
                                      "#",
                                      [
                                          1, 2, 3, 4, 5, 6, 7
                                      ]
                                  ],
                                  [
                                      10,
                                      "dependent-claim",
                                      "#",
                                      [
                                          1
                                      ]
                                  ],
                                  [
                                      14,
                                      "dependent-claim",
                                      "#",
                                      [
                                          1
                                      ]
                                  ],
                                  [
                                      16,
                                      "dependent-claim",
                                      "#",
                                      [
                                          1
                                      ]
                                  ],
                                  [
                                      17,
                                      "independent-claim",
                                      "组合物",
                                      [
                                          "*"
                                      ]
                                  ],
                                  [
                                      1,
                                      "independent-claim",
                                      "化合物",
                                      [
                                          "*"
                                      ]
                                  ],
                                  [
                                      2,
                                      "dependent-claim",
                                      "#",
                                      [
                                          1
                                      ]
                                  ],
                                  [
                                      11,
                                      "dependent-claim",
                                      "#",
                                      [
                                          1
                                      ]
                                  ],
                                  [
                                      4,
                                      "dependent-claim",
                                      "#",
                                      [
                                          1
                                      ]
                                  ],
                                  [
                                      12,
                                      "dependent-claim",
                                      "#",
                                      [
                                          1
                                      ]
                                  ],
                                  [
                                      15,
                                      "dependent-claim",
                                      "#",
                                      [
                                          1
                                      ]
                                  ],
                                  [
                                      6,
                                      "dependent-claim",
                                      "#",
                                      [
                                          1
                                      ]
                                  ],
                                  [
                                      3,
                                      "dependent-claim",
                                      "#",
                                      [
                                          1
                                      ]
                                  ],
                                  [
                                      5,
                                      "dependent-claim",
                                      "#",
                                      [
                                          1
                                      ]
                                  ],
                                  [
                                      7,
                                      "dependent-claim",
                                      "#",
                                      [
                                          1
                                      ]
                                  ],
                                  [
                                      9,
                                      "dependent-claim",
                                      "#",
                                      [
                                          1
                                      ]
                                  ],
                                  [
                                      13,
                                      "dependent-claim",
                                      "#",
                                      [
                                          1
                                      ]
                                  ],
                                  [
                                      18,
                                      "dependent-claim",
                                      "#",
                                      [
                                          21
                                      ]
                                  ]
                              ],
                              "1d6dd1414184c5e9fc55515ee65de016": [
                                  [
                                      2,
                                      "dependent-claim",
                                      "#",
                                      [
                                          1
                                      ]
                                  ],
                                  [
                                      3,
                                      "dependent-claim",
                                      "#",
                                      [
                                          1
                                      ]
                                  ],
                                  [
                                      5,
                                      "dependent-claim",
                                      "#",
                                      [
                                          1
                                      ]
                                  ],
                                  [
                                      18,
                                      "dependent-claim",
                                      "#",
                                      [
                                          12
                                      ]
                                  ],
                                  [
                                      23,
                                      "independent-claim",
                                      "医药用途",
                                      [
                                          "*"
                                      ]
                                  ],
                                  [
                                      24,
                                      "dependent-claim",
                                      "#",
                                      [
                                          23
                                      ]
                                  ],
                                  [
                                      26,
                                      "dependent-claim",
                                      "#",
                                      [
                                          23
                                      ]
                                  ],
                                  [
                                      14,
                                      "dependent-claim",
                                      "#",
                                      [
                                          12
                                      ]
                                  ],
                                  [
                                      28,
                                      "dependent-claim",
                                      "#",
                                      [
                                          23
                                      ]
                                  ],
                                  [
                                      6,
                                      "dependent-claim",
                                      "#",
                                      [
                                          1
                                      ]
                                  ],
                                  [
                                      9,
                                      "dependent-claim",
                                      "#",
                                      [
                                          1
                                      ]
                                  ],
                                  [
                                      11,
                                      "dependent-claim",
                                      "#",
                                      [
                                          1
                                      ]
                                  ],
                                  [
                                      12,
                                      "independent-claim",
                                      "给药装置",
                                      [
                                          "*"
                                      ]
                                  ],
                                  [
                                      15,
                                      "dependent-claim",
                                      "#",
                                      [
                                          12
                                      ]
                                  ],
                                  [
                                      19,
                                      "dependent-claim",
                                      "#",
                                      [
                                          12
                                      ]
                                  ],
                                  [
                                      21,
                                      "dependent-claim",
                                      "#",
                                      [
                                          12
                                      ]
                                  ],
                                  [
                                      27,
                                      "dependent-claim",
                                      "#",
                                      [
                                          23
                                      ]
                                  ],
                                  [
                                      29,
                                      "dependent-claim",
                                      "#",
                                      [
                                          23
                                      ]
                                  ],
                                  [
                                      1,
                                      "independent-claim",
                                      "给药装置",
                                      [
                                          "*"
                                      ]
                                  ],
                                  [
                                      4,
                                      "dependent-claim",
                                      "#",
                                      [
                                          1
                                      ]
                                  ],
                                  [
                                      8,
                                      "dependent-claim",
                                      "#",
                                      [
                                          1
                                      ]
                                  ],
                                  [
                                      13,
                                      "dependent-claim",
                                      "#",
                                      [
                                          12
                                      ]
                                  ],
                                  [
                                      17,
                                      "dependent-claim",
                                      "#",
                                      [
                                          12
                                      ]
                                  ],
                                  [
                                      7,
                                      "dependent-claim",
                                      "#",
                                      [
                                          1
                                      ]
                                  ],
                                  [
                                      10,
                                      "dependent-claim",
                                      "#",
                                      [
                                          1
                                      ]
                                  ],
                                  [
                                      16,
                                      "dependent-claim",
                                      "#",
                                      [
                                          12
                                      ]
                                  ],
                                  [
                                      20,
                                      "dependent-claim",
                                      "#",
                                      [
                                          12
                                      ]
                                  ],
                                  [
                                      22,
                                      "dependent-claim",
                                      "#",
                                      [
                                          12
                                      ]
                                  ],
                                  [
                                      25,
                                      "dependent-claim",
                                      "#",
                                      [
                                          23
                                      ]
                                  ]
                              ]
                          }
                      }
                  }
              }
          })
@ecslogger.log_input_output(convert_ecs_input, convert_ecs_output)
async def predict(patent_input: PatentInput):
    """
    专利权利要求书三分类接口。
    """
    result = p.predict(patent_input.patent_input)
    return result


class PersonInput(BaseModel):
    """
    输入一系列待识别法人/自然人的字符串。
    """
    classification_input: List[str] = Field(
        ...,
        example=['UNIVERSITÄT ZÜRICH',
                 'TargaGenix, Inc',
                 'H-J·金',
                 'ALLIGATOR BIOSCIENCE AB',
                 'MARTNER, Anna',
                 '青海民族大学'] if args.version != 4 else [
            'THE GOVERNMENT OF THE UNITED STATES OF AMERICA REPRESENTED BY THE SECRETARY US DEPARTMENT OF HEALTH AND HUMAN SERVICES',
            '칠드런스 하스피탈 오브 이스턴 온타리오 리서치 인스티튜트 인코포레이티드',
            '耶路撒冷希伯来大学伊森姆研究发展有限公司',
            'MaxCyte Inc',
            'جيوفري هارولد بيكير',
            'ОРГАНИЗАСЬОН ДЕ СЕНТЕЗ МОНДЬЯЛЬ ОРСИМОНД']
    )


class PersonResult(BaseModel):
    """
    输出分类识别结果：
    - v3: 1=**法人**，0=**自然人**。
    - v4: 0=**企业**, 1=**学校/研究机构**, 2=**政府机构**, 3=**医院**, 4=**个人**, 5=**其他**。

    类型0～3对应“法人”， 类型4对应“自然人”。
    """
    classification_result: List[int] = Field(
        ...,
        example=[2, 3, 1, 0, 4, 5] if args.version == 4 else [1, 1, 0, 1, 0, 1]
    )


@app.post("/classification/", tags=['Other'], response_model=PersonResult)
@ecslogger.log_input_output()
async def classify_person(person: PersonInput):
    """
    申请人分类识别接口。
    """
    result = p.predict_person(person.classification_input, return_int=True)
    return {'classification_result': result}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
