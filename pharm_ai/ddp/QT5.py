import random
import torch
import warnings
import numpy as np

from transformers.utils.fastT5 import get_onnx_model
from transformers.models.t5 import T5Config, T5ForConditionalGeneration, T5Tokenizer
from transformers.models.mt5 import MT5Config, MT5ForConditionalGeneration
from simpletransformers.t5 import T5Model, T5Args
from simpletransformers.config.utils import sweep_config_to_sweep_values


try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False


MODEL_CLASSES = {
    "t5": (T5Config, T5ForConditionalGeneration),
    "mt5": (MT5Config, MT5ForConditionalGeneration),
}


class QT5Model(T5Model):

    def __init__(
            self,
            model_type,
            model_name,
            args=None,
            tokenizer=None,
            use_cuda=False,
            cuda_device=-1,
            **kwargs,
    ):
        """
                Initializes a T5Model model.

                Args:
                    model_type: The type of model (t5, mt5)
                    model_name: The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
                    args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
                    use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
                    cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
                    **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
                """  # noqa: ignore flake8"

        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, T5Args):
            self.args = args

        if "sweep_config" in kwargs:
            self.is_sweeping = True
            sweep_config = kwargs.pop("sweep_config")
            sweep_values = sweep_config_to_sweep_values(sweep_config)
            self.args.update_from_dict(sweep_values)
        else:
            self.is_sweeping = False

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        if use_cuda:
            warnings.warn("Trying to use quantized model with GPU, switch to cpu automatically.")
            use_cuda = False
        self.device = "cpu"

        self.results = {}

        config_class, model_class = MODEL_CLASSES[model_type]

        if model_name is None:
            raise ValueError(
                "Using Quantized T5 model without specified path."
                "Make sure to quantize first and provide model dir."
            )
        else:
            self.config = config_class.from_pretrained(model_name, **self.args.config)
            self.model = get_onnx_model(model_name)

        if isinstance(tokenizer, T5Tokenizer):
            self.tokenizer = tokenizer
            self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name, truncate=True)

        if self.args.dynamic_quantize:
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )

        if not use_cuda:
            self.args.fp16 = False

        if self.args.special_tokens_list:
            self.tokenizer.add_tokens(
                self.args.special_tokens_list, special_tokens=True
            )
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.args.model_type = model_type
        if model_name is None:
            self.args.model_name = "T5_from_scratch"
        else:
            self.args.model_name = model_name

        if self.args.wandb_project and not wandb_available:
            warnings.warn(
                "wandb_project specified but wandb is not available. Wandb disabled."
            )
            self.args.wandb_project = None


if __name__ == "__main__":
    model = QT5Model("mt5", "/home/clr/disk/playground/outputs/v2.1")
