class TreeNode:
    def __init__(self, data):
        self.data = data
        self.children = []
        self.parent = None

    def get_level(self):
        level = 0
        p = self.parent
        while p:
            level += 1
            p = p.parent
        return level

    def print_tree(self):
        spaces = ' ' * self.get_level() * 3
        prefix = spaces + "|__" if self.parent else ""
        print(prefix + self.data)
        for child in self.children:
            child.print_tree()

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def get_child(self, name):
        for child in self.children:
            if child.data == name:
                return child
        return None

    def find_or_create(self, name):
        child = self.get_child(name)
        if not child:
            child = TreeNode(name)
            self.add_child(child)
        return child

    # OLD: find only first match
    def find_node_anywhere(self, keyword):
        if self.data == keyword:
            return self
        for child in self.children:
            found = child.find_node_anywhere(keyword)
            if found:
                return found
        return None

    # NEW: find ALL nodes where node.data == keyword
    def find_all_nodes_anywhere(self, keyword):
        matches = []
        queue = [self]

        while queue:
            node = queue.pop(0)
            if node.data == keyword:
                matches.append(node)
            queue.extend(node.children)

        return matches
