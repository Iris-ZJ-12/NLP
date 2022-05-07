from simpletransformers.classification import ClassificationModel, ClassificationArgs
from pharm_ai.wok.model_config import WokConfig
from pharm_ai.wok.dt import Preprocessor
from sklearn.metrics import classification_report
import os

class WokPredictor:
    def __init__(self, version, n_gpu=1, cuda_device=-1, update_args=dict()):
        if cuda_device==-1:
            select_cuda = cuda_device
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
            select_cuda = 0
        self.version = version
        args = dict(
            n_gpu=n_gpu,
            train_batch_size=8,
            eval_batch_size=8,
            learning_rate=4e-6,
            evaluate_during_training_steps=8
        )
        if update_args:
            args.update(update_args)
        self.preprocessor = Preprocessor.get_preprocessor_class(version)()
        wok_config = WokConfig(version)
        model_args = wok_config.classification_args
        model_args.update_from_dict(args)

        self.label_map = self.preprocessor.get_dataset('label_map')

        self.model = ClassificationModel('bert', wok_config.best_model_dir,
                                         num_labels=len(self.label_map),
                                         args = model_args, cuda_device=select_cuda)
        self.MNC = Preprocessor.get_MNC_list()

    def predict(self, to_predicts):
        result_raw, _ = self.model.predict(to_predicts)
        result = [self.label_map.get(r) for r in result_raw]
        return result

    def eval(self):
        eval_df = self.preprocessor.get_dataset('eval')
        eval_df['labels'] = eval_df['labels'].map(self.label_map.get)
        eval_df['predicts'] = self.predict(eval_df['text'].tolist())

        report = classification_report(eval_df['labels'], eval_df['predicts'], digits=4)
        print(report)

    def predict_MNC(self, text):
        f = lambda x: x in text if isinstance(x,str) else False
        cond = self.MNC.applymap(f).any(axis=1)
        return self.MNC['cn'][cond].tolist()

if __name__ == '__main__':
    predictor = WokPredictor('v1.2', cuda_device=0,
                             update_args={
                                 # 'sliding_window':True,
                                 # 'max_seq_length': 512
                             }
                             )
    text = "Boehringer-Ingelheim aaa 亚力兄"
    print(predictor.predict_MNC(text))


