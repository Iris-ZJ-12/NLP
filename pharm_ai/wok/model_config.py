from pathlib import Path
from simpletransformers.classification import ClassificationArgs
from pharm_ai.config import ConfigFilePaths

class WokConfig:
    root_path = Path(__file__).parent
    def __init__(self, version):
        self.output_dir = self.root_path/'outputs'/version
        self.cache_dir = self.root_path/'cache'/version
        self.best_model_dir = self.root_path/'best_model'/version
        self.bert_dir_remote = ConfigFilePaths.bert_dir_remote
        self.classification_args = ClassificationArgs(
            overwrite_output_dir=True,
            reprocess_input_data=False,
            use_cached_eval_features=True,
            use_multiprocessing=True,
            save_eval_checkpoints=False,
            evaluate_during_training=True,
            evaluate_during_training_verbose=True,
            output_dir=self.output_dir.as_posix(),
            cache_dir=self.cache_dir.as_posix(),
            best_model_dir=self.best_model_dir.as_posix(),
            wandb_project='wok',
            wandb_kwargs={'tags':['classification',version]}
        )