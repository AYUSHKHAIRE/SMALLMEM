import gc
import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModelForImageTextToText,
    GenerationConfig
)
from config.logger_config import logger

class GemmaManager:
    def __init__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        self.model_path = "/home/ayushkhaire/code/accessweb/web-view/accessweb/browse/models/gemma-3n-transformers-gemma-3n-e2b-it-v1"
        self.dtype = torch.bfloat16
        self.setup()

    def setup(self):
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.text_model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=self.dtype, device_map="auto")
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def ask_query(self, prompt):
        inputs = self.text_tokenizer(prompt, return_tensors="pt").to(self.text_model.device)
        generation_config = GenerationConfig(
            max_new_tokens=20000,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        outputs = self.text_model.generate(**inputs, generation_config=generation_config)
        response = self.text_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        return response