import os
import time
from simpletransformers.classification import ClassificationModel, ClassificationArgs


class PolicyIdentifier():
    def __init__(self, version='v1.0'):
        project_dir = os.path.dirname(os.path.realpath(__file__))
        self.version = version
        best_model_dir = os.path.join(project_dir, 'best_model')
        use_cuda = False
        self.args = {"n_gpu": 0, "silent": True}
        self.model = ClassificationModel('bert', best_model_dir, args=self.args, use_cuda=False)

    def predict(self, text):
        predictions, raw_outputs = self.model.predict(text)
        return predictions, raw_outputs
