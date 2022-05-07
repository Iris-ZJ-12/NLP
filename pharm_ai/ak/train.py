from pharm_ai.ak.dt import AkPreprocessor
from pharm_ai.ak.utils import AkT5ModelBase, AkClassificationModelBase


def train(version='v1.0', ddp=True, cuda_devices=None, notification=False, task_id=0,
          pretrained_model='mt5', **kwargs):
    """pretrained_model: (mt5, zh_en, cls)."""
    prepro_class = AkPreprocessor.get_preprocessor_class(version)
    prepro = prepro_class()

    train_df = prepro.get_from_h5('train')
    eval_df = prepro.get_from_h5('eval')

    if pretrained_model != 'cls':
        ak_model = AkT5ModelBase(version, cuda_devices, task_id)
        ak_model.train(train_df, eval_df, ddp=ddp, notification=notification,
                       pretrained_model=pretrained_model, **kwargs)
    else:
        ak_model = AkClassificationModelBase(version, getattr(prepro, 'prefix', None), cuda_devices, task_id)
        ak_model.train(train_df, eval_df, ddp, notification, **kwargs)

if __name__ == '__main__':
    train('v1.4', cuda_devices=[0, 1, 2, 3, 5], notification=True, task_id=1, pretrained_model='zh_en',
          num_train_epochs=5, learning_rate=4e-4, train_batch_size=50, eval_batch_size=25,
          evaluate_during_training_steps=300, max_length=40)