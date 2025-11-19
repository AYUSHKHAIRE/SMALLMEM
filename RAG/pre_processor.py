from langchain.text_splitter import CharacterTextSplitter
from config.logger_config import logger

class Chunker:
    def __init__(self):
        self.text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=400,
            length_function=len,
        )

    def chunk_text(self, text: str):
        chunks = self.text_splitter.split_text(text)
        return chunks