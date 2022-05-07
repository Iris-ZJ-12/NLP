# PharmAI

## Intro
As the name suggests, this repo handles AI topics. It consists of many projects, usually each 
project is self-contained or loosely dependent on other projects. Common project layout shown below, we should apply 
this layout and the naming conventions suggested for any machine learning projects, however this layout may be modified
to accommodate to non-machine-learning or hybrid-style projects.

## Common Project Layout
```
# __init__.py files are omitted below.
.
├── ...
├── PharmAI                    
│   ├── pharm_ai/   
│   │   ├── project-01/
│   │   │   ├── dt.py           # data preprocessing code
│   │   │   ├── api.py          # deploy this project as a RESTful API
│   │   │   ├── train.py        # code to train a model
│   │   │   ├── best_model/     # path specified in the train.py file and auto-generated during 
│   │   │   │                   # training, fine-tuned model stored here which has lowest eval loss.
│   │   │   ├── runs/           # path specified in the train.py file and auto-generated during 
│   │   │   │                   # training, tensorboards stored here.
│   │   │   ├── outputs/        #  path specified in the train.py file and should be deleted if 
│   │   │   │                   # best model saved already.
│   │   │   └── predictor.py    #  after training, use predictor.py to predict unlabeled data             
│   │   ├── project-02/...
│   │   ├── project-03/...
│   │   ├── ...
│   │   ├── util/
│   │   ├── config.py
│   ├── .gitignore
│   ├── README.md
│   └── ...
└── ...
```
   