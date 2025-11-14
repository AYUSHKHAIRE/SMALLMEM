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

    def get_parent_route(self):
        route = []
        p = self
        while p:
            route.append(p.data)
            p = p.parent
        return list(reversed(route))

    def print_tree(self):
        spaces = ' ' * self.get_level() * 3
        prefix = spaces + "|__" if self.parent else ""
        print(prefix + self.data)
        if self.children:
            for child in self.children:
                child.print_tree()

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def search(self, key):
        results = []
        if self.data == key:
            results.append(self.get_parent_route())
        for child in self.children:
            results.extend(child.search(key))
        return results

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

def build_product_tree():
    root = TreeNode("Electronics")

    laptop = TreeNode("Laptop")
    laptop.add_child(TreeNode("Mac"))
    laptop.add_child(TreeNode("Surface"))
    laptop.add_child(TreeNode("Thinkpad"))

    cellphone = TreeNode("Cell Phone")
    cellphone.add_child(TreeNode("iPhone"))
    cellphone.add_child(TreeNode("Google Pixel"))
    cellphone.add_child(TreeNode("Vivo"))
    cellphone.add_child(TreeNode("LG"))

    tv = TreeNode("TV")
    tv.add_child(TreeNode("Samsung"))
    tv.add_child(TreeNode("LG"))

    root.add_child(laptop)
    root.add_child(cellphone)
    root.add_child(tv)

    root.print_tree()
    print(root.search("LG"))

if __name__ == '__main__':
    build_product_tree()