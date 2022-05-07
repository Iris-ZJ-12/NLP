# **NER**(Named Entity Recognition)
To **locate** and **classify** named entities(instance of some classes) in a sequence.

---
## 1.sm_util.py  
- Class: **SMUtil**: utility functions for simple_transformers
    - Methods:
        - **ner_predict()**: use NERmodel or dictionary to predict input text
          - *inputs*:
            - __*text*__: a text--a string,can be a sentence or a text composed of some sentences
            - __*model*__: NERModel()
            - __*prepro*__: the rules of tokenization
            - __*rule_ner=None*__: dictionary match class
            - __*len_threhold=900*__: the number of characters in English text or the number of Chinese characters in Chinese 
            text;usually 900. if the number>len_threhold,use dictionary match to predict inputs,else use model to predict inputs       
                              
          - *outputs*: a list of model predictions or dictionary match. every element means the prediction result of a sentence
          
          - *example*:
            - __input a text(a string)__: `'What is central diabetes insipidus. Inhibitor compounds'`
            - __procedure__: 
              - __1__.input a text: `'What is central diabetes insipidus. Inhibitor compounds'`
              - __2__.token text to sentences: `[['What', 'is', 'central', 'diabetes', 'insipidus', '.'], ['Inhibitor', 'compounds']]`
              - __3__.for each sentence: `['What', 'is', 'central', 'diabetes', 'insipidus', '.'],['Inhibitor', 'compounds']` 
              to predict: `[['What', 'O'], ['is', 'O'], ['central', 'b-disease'], ['diabetes', 'b-disease'], 
              ['insipidus', 'b-disease'], ['.', 'O']],[['Inhibitor', 'O'], ['compounds', 'O']]`
              - __4__.get the results of the text
            - __return__: 
            ```
            [  
              [['What', 'O'], ['is', 'O'], ['central', 'b-disease'], ['diabetes', 'b-disease'], ['insipidus', 'b-disease'], ['.', 'O']],     
              [['Inhibitor', 'O'], ['compounds', 'O']],  
            ]
            ```
        
        - **ner2xlsx()**: save the prediction results into a xlsx file
          - *inputs*:
            - __*raw_ner_result*__: the list of all texts prediction results. **note**: the text(a string) can be a sentence or a paragraph
            - __*article_ids*__: the list of every text' article_id. **note**:a text result correspond an article_id
            - __*xlsx_path*__: the path of xlsx file to save
            - __*sheet_name*__: sheet name
            - __*label_li*__: label list. **note**:the label list during training and prediction should be the same
          - *outputs*: a xlsx file to save the predict_results
          
          - *example*:
            - __input a raw_ner_result__: `List[text: List[sentences: List[words result: List[words, *NER_labels]]]]`  
            ```
            [  
              [  
                [['PD-1', 'O', 'b-target'], ['是', 'O', 'O'], ['啥', 'O', 'O'], ['？', 'O', 'O']],  
                [['心', 'b-disease', 'O'], ['脏', 'i-disease', 'O'], ['病', 'i-disease', 'O'], ['咋', 'O', 'O'],['治', 'O', 'O']],  
              ],   
              [  
                [['What', 'O', 'O'], ['is', 'O', 'O'], ['diabetes?', 'O', 'O']],  
              ],  
            ]  
            ```
           
            - __input a list of article_ids__: a text corresponds an article_id
            ```
            [  
              '1',  
              '2',  
            ]
            ```
            
            - __outputs__: a xlsx file to label data
            
        - **recover_ner_entity()**: transform NER labels of words to phrase entities
          - *inputs*:
            - __*raw_result*__: like the raw_ner_result in *ner2xlsx()*
          - *outputs*: phrase entities
          - *example*:
            - **input a raw_result**:
                ```
              [
                 [
                  [['PD-1', 'O', 'b-target'], ['是', 'O', 'O'], ['啥', 'O', 'O'], ['？', 'O', 'O']],
                  [['心', 'b-disease', 'O'], ['脏', 'i-disease', 'O'], ['病', 'i-disease', 'O'], ['咋', 'O', 'O'], ['治', 'O', 'O']],
                 ], 
                 [
                  [['What', 'O', 'O'], ['is', 'O', 'O'], ['diabetes?', 'O', 'O']],
                 ],
              ]
                ```
            - **return**: `List[paragraph: dict[NER_label->sentence_entities: List[entity]]]`
            ```
            [
               {
                  'target': ['PD-1'], 
                  'disease': ['心脏病']
               }, 
               {
               },
            ]
            ```
    
        - **eval_ner()**: columns which must be included test_df: sentence_id, words, labels; predicted_labels is optional
          - *inputs*:
            - __*test_df*__: dataframe(columns:'sentence_id', 'words', 'labels'), data for evaluation
            - __*title*__: name,string
            - __*model*=None__: NERModel
          - *outputs*: evaluations based by the phrase and the word
        
        - **ner_upsampling()**:to solve the class-imbalance problem,up-sampling NER dataset to balance sentences that have all 'O' lables and ones not 
          - *inputs*:
            - __*df*__: dataframe(columns:'sentence_id', 'words', 'labels')
            - __*random_state*__: int,set random state
            - __*shuffle*=True__:  whether to shuffle data at sentence level
          - *outputs*: the processed data for training
          - *example*:
            - **input a df**: df
            - **return**: df
                  
        - **auto_rm_outputs_dir()**: use this method after running `model.train_model(train_df, eval_df)`
        
## 2.prepro.py  
- Class: **Prepro**: for Chinese-English tokenization 
    - Properties:
      - __*ht*__: Chinese sentences tokenization method 
      - __*pst*__: English sentences tokenization method 
    
    - Methods:
        - **tokenize_hybrid_text()**: used for tokenizing one sentence,word tokenization
          - *inputs*:
            - __*text*__: a string,a text to separate to many words
            - __*sep_end_punc=True*__: separate ending punctuation
          - *outputs*: a list of word tokenization
          - *example*:
            - **input a text**: `'What is central diabetes insipidus (CDI) ?'`
            - **return**:` ['What', 'is', 'central', 'diabetes', 'insipidus', '(', 'CDI', ')', '?']`
          
        - **tokenize_hybrid_text_generic()**: used for tokenizing text containing multiple sentences
          - *inputs*:
            - __*text*__: a text containing multiple sentences,a string
          - *outputs*: a list containing many sentences,each sentence is a list containing the separated words
          - *example*:
            - **input a text**: `'What is central diabetes insipidus (CDI)? Tokenizing text containing multiple sentences '`
            - **return**: 
            ```
            [  
              ['What', 'is', 'central', 'diabetes', 'insipidus', '(', 'CDI', ')', '?'],    
              ['Tokenizing', 'text', 'containing', 'multiple', 'sentences'],  
            ]  
            ```
        - **bert_ner_prepro()**: tokenization and prepare the data for prediction
          - *inputs*:
            - __*texts*__: a list of texts, each text may contain one or more sentences.
          - *outputs*: `List[text:List[sentences]]`
          - *example*:
            - **input a list of texts**:
             ```
            [
               'What is central diabetes.  Tokenizing text containing multiple',
               'each paragraph may contain one or more sentences.',
            ]
            ```
            - **return**: `List[text:List[sentences]]`
            ```
            [ 
               [
                  'What is central diabetes .', 
                  'Tokenizing text containing multiple',
               ],  
               [
                  'each paragraph may contain one or more sentences .'
               ],  
            ]
            ```
            
        - **prep_ner_train_dt()**: prepare training data for NER with default labels 'O', save it as a excel file
          - *inputs*: a list of texts, each text may contain one or more sentences.
          - *outputs*: a .xlsx file

## 3.ner_util.rule_ner.py           
- Class: **RuleNER**: dictionary match 
    - *what did it do?*
      - **1**.give the dic and dic_path,it may create the .pkl file to save the dictionary
      - **2**.for each text,it with remove stopwords
      - **3**.for each text,it can label the text
    
    - Init:
      - **dic**: dictionary a python object--dict,every key represent a label class
      - **dic_path**: .pkl file,
      
    - Properties:
      - **prepro**: tokenization method
      - **dic**: self.prep_dic(dic, dic_path), a list
    
    - Methods:        
        - **label_sentence()**: label a text(usually sentence) by using dictionary match
          - *inputs*: 
            - __*sentence*__: a string,a text to separate to many words
          - *outputs*: the prediction results of every word in a text by using dictionary match
          - *note* : this method doesn't use sentences tokenization
           
    - example:        
      - *inputs*:
      ```
            dic = {'disease': ['2型糖尿病', '糖尿病', '高血糖'],'target': ['hepatitis D virus(HDV)', 'hepatitis D virus', 'HDV', 'EDA-FN']}  
            dd = '../../patent_ner/disease_dic-20201104.pkl'  
            r = RuleNER(dic, dd)  
            print(r.label_sentence('hepatitis D virus(HDV)是啥？'))  
      ```  
        
      - *outputs*: 
      ```
      [
         ['hepatitis', 'b-target'], ['D', 'i-target'], ['virus', 'i-target'], ['(', 'i-target'], ['HDV', 'i-target'], [')', 'i-target'], ['是', 'O'], ['啥', 'O'], ['？', 'O']
      ]
  
      ```
          
## 4.Process
### (1)get data from **ES**--to **.xlsx** file
### (2)process data for colleagues to label
- get _**texts**_,_**article_ids**_,_**model**_,the texts is a list of some texts,every text corresponds an article_ids,every text may contain one or more sentences
- for every text in texts, do:`SMUtil.ner_predict(...)` to get the result,and concat the results to a **list**
- use `SMUtil.ner2xlsx(...)` to save the results to a **.xlsx** file
### (3)training
- get **labeled data** from colleagues and save the data as a **.h5** file
- data preprocessing: **divide data set**,**shuffle**,**dropna...**
- use `model.train_model(train_df, eval_df)`,view **metrics** and **tuning**
### (4)eval and predict
- use `SMUtil.eval_ner(...)` to eval, use `SMUtil.ner_predict(...)` to predict
- use `SMUtil.ner2xlsx(...)` to save the results to a **.xlsx** file
### (5)give the results file to colleagues and get the feedback
- if it can be used **online**,turn to **(6)**
- if it can be **better** by training more data,turn to **(1)**
### (6)API Deployment 
