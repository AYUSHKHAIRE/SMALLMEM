from tree import TreeNode
from pre_processing import TextProcessor
from tqdm import tqdm

class ConcentratedContext:
    def __init__(self,name):
        self.name = name
        self.structure = TreeNode(name)
        self.TP = TextProcessor()

    def __str__(self):
        return self.name

    def build(self, chunks):
        for chk in tqdm(chunks, "Processing chunks ..."):
            processed_text = self.TP.process_text(chk)
            if not processed_text:
                continue
            current = self.structure  # start at root

            # Create chain of keywords
            for keyword in processed_text:
                current = current.find_or_create(keyword)

            # Attach actual chunk at the end of the chain
            content_node = TreeNode(f"CHUNK: {chk[:30]}...")  # or store full text
            content_node.original_text = chk  # store chunk text
            current.add_child(content_node)

    def print(self):
        self.structure.print_tree()