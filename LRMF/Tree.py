from collections import defaultdict
import numpy as np

class Node(object):
    def __init__(self, users, question, like=None, dislike=None):
        self.users = users                      # Set of users
        self.question = question                # item id used for question
        self.like = like                        # Like child
        self.dislike = dislike                  # Dislike child

    def is_leaf(self):
        if self.like is None and self.dislike is None:
            return True
        return False

    def _questions(self):
        if self.is_leaf():
            return [[]]  # one path: only contains self.value
        paths = []
        for child in [self.like, self.dislike]:
            for path in child._questions():
                paths.append([self.question] + path)
        return paths

    def _local_questions(self):
        if self.is_leaf():
            return [self.local_questions]
        return self.like._groups() + self.dislike._groups()

    def _groups(self):
        if self.is_leaf():
            return [self.users]
        return self.like._groups() + self.dislike._groups()

    def set_transformation(self, transformation):
        self.transformation = transformation

    def set_globals(self, globals):
        self.global_questions = globals

    def set_locals(self, locals):
        self.local_questions = locals


def traverse_a_user(user: int, data, tree: Node):
    if tree.is_leaf():
        return tree

    if ((data['uid'] == user) & (data['iid'] == tree.question)).any():
        return traverse_a_user(user, data, tree.like)
    else:
        return traverse_a_user(user, data, tree.dislike)


def find_user_group(user, tree: Node):
    if tree.is_leaf():
        return tree

    elif user in tree.like.users:
        return find_user_group(user, tree.like)

    else:
        return find_user_group(user, tree.dislike)


def number_of_leaves(root: Node):
    if root.is_leaf():
        return 1
    return number_of_leaves(root.like) + number_of_leaves(root.dislike)


