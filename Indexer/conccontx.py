from .tree import TreeNode
from pre_processing.text_processor import TextProcessor
from tqdm import tqdm
from config.logger_config import logger

class ConcentratedContext:
    """
    ConcentratedContext indexes text chunks using a keyword → tree-based structure.
    It does not use embeddings. Retrieval is keyword-based + BFS expansion.
    """

    def __init__(self, name="CCX"):
        self.name = name
        self.structure = TreeNode(name)   # root node
        self.TP = TextProcessor()
        self._built = False

    def __str__(self):
        return f"ConcentratedContext({self.name})"

    # ---------------------------------------------------
    # BUILD TREE
    # ---------------------------------------------------
    def build(self, chunks):
        """
        Build the keyword → chunk tree.
        Each processed keyword creates a path in the tree.
        Leaf nodes contain original text of chunks.
        """
        for chk in tqdm(chunks, desc="Building Concentrated Context Tree"):
            processed_text = self.TP.process_text(chk)
            if not processed_text:
                continue

            current = self.structure  # begin at root

            # create tree path for keywords
            for keyword in processed_text:
                current = current.find_or_create(keyword)

            # finally attach chunk as leaf node
            content_node = TreeNode(f"CHUNK: {chk[:30]}...")
            content_node.original_text = chk
            current.add_child(content_node)

        self._built = True
    logger.info("context build successful")
    # ---------------------------------------------------
    # CHUNK RETRIEVAL
    # ---------------------------------------------------
    def fetch_relevant_chunks(self, query, k=20):
        """
        Retrieve chunks related to query using the tree.
        Keyword match → BFS → collect chunk nodes.
        Returns list of original text chunks.
        """
        if not self._built:
            return []

        keywords = self.TP.process_text(query)
        if not keywords:
            return []

        matched_nodes = []

        # find first keyword that exists in tree
        for key in keywords:
            matched_nodes = self.structure.find_all_nodes_anywhere(key)
            if matched_nodes:
                break

        if not matched_nodes:
            return []

        result = []
        seen = set()

        # BFS from all matched nodes
        queue = matched_nodes[:]

        while queue and len(result) < k:
            node = queue.pop(0)

            for child in node.children:
                if child.data.startswith("CHUNK:"):  # a chunk leaf
                    text = child.original_text
                    if text not in seen:
                        seen.add(text)
                        result.append(text)
                        if len(result) >= k:
                            break
                else:
                    # add branches to BFS queue
                    queue.append(child)

        return result

    # ---------------------------------------------------
    # FORMAT INTO A CONCENTRATED CONTEXT BLOCK
    # ---------------------------------------------------
    def generate(self, context_text: str, query: str, k=20):
        """
        Used by Streamlit:
        - Ignores whatever context_text is passed.
        - Returns a new concentrated context built from the tree.
        """
        chunks = self.fetch_relevant_chunks(query, k=k)

        if not chunks:
            return "No concentrated context found."

        formatted = "\n\n".join(chunks)
        return formatted

    # ---------------------------------------------------
    # UTILITY
    # ---------------------------------------------------
    def print(self):
        self.structure.print_tree()

    def clear(self):
        """Reset the context tree."""
        self.structure = TreeNode(self.name)
        self._built = False
