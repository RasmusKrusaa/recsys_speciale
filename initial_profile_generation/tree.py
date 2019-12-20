import numpy as np
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


def traverse_tree(tree: Node, u_answers) -> Node:
    """
    Returns the leaf, at which the user ends up in
    """
    if not tree.children or not tree.question:
        return tree

    q1, q2 = tree.question

    if isinstance(u_answers, list):
        ans1 = [r for _, i, r in u_answers if i == q1]
        ans2 = [r for _, i, r in u_answers if i == q2]
        if ans1 and ans2:
            ratio = round((2 * ans1[0]) / ans2[0] - 1) if ans1[0] >= ans2[0] else 1 / (round(2 * ans2[0] / ans1[0] - 1))
        else:
            ratio = 0

    elif isinstance(u_answers, np.ndarray):
        ans1 = u_answers[q1]
        ans2 = u_answers[q2]
        if ans1 != 0 and ans2 != 0:
            ratio = round((2 * ans1) / ans2 - 1) if ans1 >= ans2 else 1 / (round(2 * ans2 / ans1 - 1))
        else:
            ratio = 0

    for child in tree.children:
        if child.ratio == ratio:
            return traverse_tree(child, u_answers)
    # if no children with ratio as user
    return tree
