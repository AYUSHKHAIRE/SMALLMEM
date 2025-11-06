import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from load import DataLoader
from RAG.pre_processor import Chunker
from RAG.embeding import Embedder
from LLM.docker_model import dockerModel

class Engine:
    def __init__(self):
        self.embeder = Embedder()
        self.chunker = Chunker()
        self.dockermodel = dockerModel(
            model="ai/granite-4.0-h-tiny:7B",
            hostname="localhost",
            port=12434,
            stream=True,
            system_prompt="You are a helpful assistant."
        )
        self.data = DataLoader().load_processed()

# Example
eg = Engine()
print(eg.data.columns)
