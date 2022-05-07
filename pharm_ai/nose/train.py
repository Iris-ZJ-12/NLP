from pharm_ai.nose.dt import NosePreprocessor
from pharm_ai.nose.utils import NoseT5ModelBase, NoseClassificationModelBase


def train(version='v1.0', ddp=True, cuda_devices=None, notification=False, task_id=0,
          pretrained_model='mt5', weight=None, **kwargs):
    """pretrained_model: (mt5, zh_en, cls)."""
    prepro_class = NosePreprocessor.get_preprocessor_class(version)
    prepro = prepro_class()

    train_df = prepro.get_from_h5('train')
    eval_df = prepro.get_from_h5('eval')

    if pretrained_model != 'cls':
        nose_model = NoseT5ModelBase(version, cuda_devices, task_id)
        nose_model.train(train_df, eval_df, ddp=ddp, notification=notification,
                         pretrained_model=pretrained_model, **kwargs)
    else:
        nose_model = NoseClassificationModelBase(version, getattr(prepro, 'prefix', None), cuda_devices, task_id)
        nose_model.train(train_df, eval_df, ddp, notification, **kwargs)


if __name__ == '__main__':
    train('v1.7', ddp=True, cuda_devices=[7, 8], notification=True, task_id=0, pretrained_model='zh_en',
          train_batch_size=40, eval_batch_size=40, evaluate_during_training_steps=10, learning_rate=4e-3,
          num_train_epochs=4)