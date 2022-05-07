import math
import os

import pandas as pd
import wandb
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import losses, datasets, util
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sklearn.utils import shuffle

'''
- This script is used for testing policy multi-class classification task with sentence transformers library.
- One specific model was fine-tuned with BatchSemiHardTripletLoss function and evaluated based on about 26242
labeled data.  
- BinaryClassificationEvaluator.from_input_examples was modified to log accuracy in eval data. At each evaluation
step, all samples in train data would be converted to embedding vectors using current model and calculate the accuracy. 
'''

wandb.init(project="test-sentence-transformer")
os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7,8'
##########################################################
#################### format data #########################
train_batch_size = 150
train_da = pd.read_hdf("/home/ryz/work/from_zj/data/st_semihard_train.h5", "train")
eval_da = pd.read_hdf("/home/ryz/work/from_zj/data/st_semihard_eval.h5", "eval")
eval_da = eval_da[["标题", "政策分类"]]
eval_da = eval_da.drop_duplicates()
### format training data with required InputExample to dataloader
train_samples = []
train_da.apply(lambda x: train_samples.append(InputExample(texts=[x["标题"]], label=x["labels"])), axis=1)
train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=train_batch_size)

### for evaluation data, the format is [text_a,text_b], score
resamples = {}  #### set the sampling ratio or number of each class
labelNum = eval_da["政策分类"].value_counts()
for i in labelNum.index:
    if labelNum[i] > 100:
        resamples[i] = 100  ### each class contained more than 100 would sample 100 samples
    else:
        resamples[i] = labelNum[i] - 1
total_label = eval_da["政策分类"].unique()
total_label_num = len(total_label)


def get_spair(x):
    # np.random.seed(2)
    poss = eval_da[eval_da["政策分类"] == x["政策分类"]].sample(resamples[x["政策分类"]])["标题"].values
    negs = eval_da[eval_da["政策分类"] != x["政策分类"]].groupby("政策分类").sample(1)["标题"].values
    dict_pos = {"text_a": x["标题"], "text_b": poss, "score": 1}
    pos_df = pd.DataFrame(dict_pos, index=range(len(poss)))
    dict_nes = {"text_a": x["标题"], "text_b": negs, "score": 0}
    neg_df = pd.DataFrame(dict_nes, index=range(len(negs)))
    return pd.concat([pos_df, neg_df])


### for each item in raw evaluation data, make positive examples and corresponding negative examples
eval_pairs = []
for i in range(0, eval_da.shape[0]):
    eval_pairs.append(get_spair(eval_da.iloc[i,]))
eval_data = pd.concat(eval_pairs, axis=0).reset_index(drop=True)
eval_data.to_hdf("/home/ryz/work/from_zj/data/st_semihard_eval_triplet.h5", "eval", mode='w')
eval_data = shuffle(eval_data)
### format the generated evaluation data to dev_evaluator
eval_samples = []
eval_data.apply(lambda x: eval_samples.append(InputExample(texts=[x["text_a"], x["text_b"]], label=x["score"])), axis=1)
# dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(eval_samples, batch_size=train_batch_size,name='sts-dev')
dev_evaluator = BinaryClassificationEvaluator.from_input_examples(eval_samples, train_da, eval_da,
                                                                  batch_size=train_batch_size, name='sts-dev')

#######################################################
###### Here we define our SentenceTransformer model ###
model_name = "/large_files/pretrained_pytorch/paraphrase-multilingual-mpnet-base-v2"
# model_name = "/large_files/pretrained_pytorch/paraphrase-multilingual-MiniLM-L12-v2"
max_seq_length = 128
model = SentenceTransformer(model_name)
# train_loss = losses.MultipleNegativesRankingLoss(model)
train_loss = losses.BatchSemiHardTripletLoss(model=model)

num_epochs = 6
model_save_path = "/home/ryz/work/from_zj/test/"
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=int(len(train_dataloader) * 0.1),
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          use_amp=True  ### Set to True, if your GPU supports FP16 operations
          )
### this process was same with evaluator in fit procedure
test_evaluator = BinaryClassificationEvaluator.from_input_examples(eval_samples, train_da, eval_da,
                                                                   batch_size=train_batch_size,
                                                                   name='sts-test')
test_evaluator(model, output_path=model_save_path)

#######################################################
###### evaluate the final accuracy ###
'''calculate the accuracy
get embedding of all train samples and calculate the cosine similarity between
train samples and evaluated samples. The label of trained sample 
which have maximum score would assign to each evaluated sample as the predicted class.'''
model = SentenceTransformer('/home/ryz/work/from_zj/test')
train_embeddings = model.encode(train_da["标题"].tolist())
eval_embeddings = model.encode(eval_da["标题"].tolist())
cosine_scores = util.pytorch_cos_sim(train_embeddings, eval_embeddings)
target_texts = []
target_labels = []
for i in range(len(eval_embeddings)):
    query = train_da.iloc[i, 1]
    query_lab = train_da.iloc[i, 2]
    target = train_da.iloc[cosine_scores[:, i].argmax().numpy().tolist()]
    target_text = target["标题"]
    target_label = target["政策分类"]
    target_texts.append(target_text)
    target_labels.append(target_label)
eval_da["target"] = target_texts
eval_da["target_label"] = target_labels
correct = eval_da[eval_da["政策分类"] == eval_da["target_label"]].shape
wrong = eval_da[eval_da["政策分类"] != eval_da["target_label"]]
wrong.to_csv("/home/ryz/work/from_zj/data/st_mistake_data.txt", sep="\t")
print("correct = ", correct[0], ",acc =", correct[0] / eval_da.shape[0])
