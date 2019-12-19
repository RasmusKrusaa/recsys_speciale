from typing import List


class Node():
    def __init__(self, users: List[int], question: (int, int), ratio: float = None):
        self.users = users
        self.question = question
        self.children = []
        self.ratio = ratio
        self.parent = None

    def add_child(self, obj):
        self.children.append(obj)

    def add_parent(self, obj):
        self.parent = obj

    # TODO: implement traverse with actuals vector. Make sure ids of items are inner ids! I.e. the ones from SVD model

    def pretty_print(self, prefix: str = ''):
        print(f'{prefix}{self.question}')
        for child in self.children:
            child.pretty_print(prefix.replace("|", " ") + "|--")
