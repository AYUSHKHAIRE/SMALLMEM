import pymupdf4llm

class PDFProcessor:
    def __init__(self):
        self.md = None

    def get_markdown(self, file_path: str):
        """Open a PDF file and load it as a PyMuPDF4LLM document."""
        return pymupdf4llm.to_markdown(file_path)