from pharm_ai.prophet.dt import ProphetPreproessor
from pharm_ai.prophet.utils import ProphetT5ModelBase, ProphetClassificationModelBase

def train(version = 'v3.2', ddp=True, cuda_devices = None, notification=False, task_id=0,
          pretrained_model='mt5', **kwargs):
    """pretrained_model: (mt5, zh_en, cls)."""
    prepro_class = ProphetPreproessor.get_preprocessor_class(version)
    prepro = prepro_class()

    train_df = prepro.get_from_h5('train')
    eval_df = prepro.get_from_h5('eval')

    if pretrained_model!='cls':
        prophet_model = ProphetT5ModelBase(version, cuda_devices, task_id)
        prophet_model.train(train_df, eval_df, ddp=ddp, notification=notification,
                            pretrained_model=pretrained_model, **kwargs)
    else:
        prophet_model = ProphetClassificationModelBase(version, prepro.prefix, cuda_devices, task_id)
        prophet_model.train(train_df, eval_df, ddp, notification, **kwargs)

if __name__ == '__main__':
    train(version='v3.6', ddp=True, cuda_devices=[2, 3, 7, 8], notification=True, task_id=1, pretrained_model='zh_en',
          train_batch_size=48,
          eval_batch_size=48,
          logging_steps=30,
          evaluate_during_training_steps=300,
          learning_rate=4e-5,
          num_train_epochs=15,
          max_seq_length=300,
          max_length=50
          )
