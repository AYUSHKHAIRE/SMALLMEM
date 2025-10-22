import gc
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
    TextIteratorStreamer
)
from threading import Thread
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
        logger.info("Loading Gemma model...")
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.text_model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=self.dtype, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        logger.info("Gemma model loaded successfully.")

    def ask_query(self, prompt):
        inputs = self.text_tokenizer(prompt, return_tensors="pt").to(self.text_model.device)

        generation_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        # 🧠 Create streamer
        streamer = TextIteratorStreamer(self.text_tokenizer, skip_special_tokens=True, skip_prompt=True)

        # Run generation in a background thread to avoid blocking
        thread = Thread(
            target=self.text_model.generate,
            kwargs=dict(**inputs, streamer=streamer, generation_config=generation_config),
        )
        thread.start()

        # 🌀 Yield streamed text as it comes
        for new_text in streamer:
            yield new_text

        thread.join()
