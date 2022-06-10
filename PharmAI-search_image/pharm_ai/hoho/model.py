from os.path import join
from pharm_ai.config import ConfigFilePaths
from simpletransformers.classification import ClassificationModel


class HohoModel:

    def __init__(self, args):
        self.model = ConfigFilePaths.bert_dir_remote
        self.args = args

        # basic config
        self.args.reprocess_input_data = True
        self.args.use_cached_eval_features = False
        self.args.fp16 = False

        # directories
        project_dir = join(ConfigFilePaths.project_dir, 'hoho')
        self.args.overwrite_output_dir = True
        self.args.output_dir = join(project_dir, 'outputs/')
        self.args.cache_dir = join(project_dir, 'cache/')
        self.args.best_model_dir = join(project_dir, 'best_model/')
        self.args.tensorboard_dir = join(project_dir, 'runs/')

    def get_model(self, train=True, use_cuda=True):
        model_path = self.model if train else self.args.best_model_dir
        return ClassificationModel(model_type='bert', model_name=model_path, args=self.args, use_cuda=use_cuda)
