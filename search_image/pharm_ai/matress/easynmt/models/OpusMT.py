import time
from transformers import MarianMTModel, MarianTokenizer
import torch
from typing import List
import logging

logger = logging.getLogger(__name__)


class OpusMT:

    def __init__(self, easynmt_path: str = None, max_loaded_models: int = 10, device='cuda'):
        self.models = {}
        self.max_loaded_models = max_loaded_models
        self.max_length = None
        self.easynmt_path = easynmt_path
        self.device = device

    def load_model(self, model_name):
        if self.easynmt_path is not None:
            model_name = self.easynmt_path

        if model_name in self.models:
            self.models[model_name]['last_loaded'] = time.time()
            return self.models[model_name]['tokenizer'], self.models[model_name]['model']
        else:
            logger.info("Load model: "+model_name)
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            model.eval()
            model.to(self.device)

            if len(self.models) >= self.max_loaded_models:
                oldest_time = time.time()
                oldest_model = None
                for loaded_model_name in self.models:
                    if self.models[loaded_model_name]['last_loaded'] <= oldest_time:
                        oldest_model = loaded_model_name
                        oldest_time = self.models[loaded_model_name]['last_loaded']
                del self.models[oldest_model]

            self.models[model_name] = {'tokenizer': tokenizer, 'model': model, 'last_loaded': time.time()}
            return tokenizer, model

    def translate_sentences(
            self,
            sentences: List[str],
            source_lang: str,
            target_lang: str,
            device: str,
            beam_size: int = 5,
            **kwargs
    ):
        model_name = 'Helsinki-NLP/opus-mt-{}-{}'.format(source_lang, target_lang)
        # now = time.time()
        tokenizer, model = self.load_model(model_name)
        # print(time.time()-now)

        inputs = tokenizer(sentences, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].to(device)

        # now = time.time()
        with torch.no_grad():
            translated = model.generate(**inputs, num_beams=beam_size, **kwargs)

            output = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        # print(time.time() - now)

        return output


