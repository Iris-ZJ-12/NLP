from simpletransformers.classification import ClassificationModel, ClassificationArgs
import os
from pharm_ai.wok.model_config import WokConfig
from pharm_ai.wok.dt import Preprocessor
from pharm_ai.util.utils import Utilfuncs as u
from pharm_ai.util.sm_util import SMUtil


# set training parameters
version='v1.2'
n_gpu=5
cuda_devices = [0,1,2,3,4]

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(c) for c in cuda_devices)
args = dict(
    n_gpu=n_gpu,
    train_batch_size=80,
    eval_batch_size=50,
    learning_rate=4e-5,
    num_train_epochs=15,
    evaluate_during_training_steps=80,
    # sliding_window=True,
    # max_seq_length=512,
    # stride=0.8
)

# config model
preprocessor = Preprocessor.get_preprocessor_class(version)()
df_train = preprocessor.get_dataset('train')
df_eval = preprocessor.get_dataset('eval')

wok_config = WokConfig(version)
model_args = wok_config.classification_args
model_args.update_from_dict(args)

u.fix_torch_multiprocessing()

model = ClassificationModel('bert', wok_config.bert_dir_remote, num_labels=preprocessor.get_num_labels(),
                            args=model_args, cuda_device=-1)
model.train_model(df_train, eval_df=df_eval)

SMUtil.auto_rm_outputs_dir(wok_config.output_dir.as_posix())