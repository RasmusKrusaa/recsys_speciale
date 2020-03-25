from collections import defaultdict


class Node(object):
    def __init__(self, users, question, like=None, dislike=None):
        self.question = question    # item id used for question
        self.users = users          # Set of users
        self.like = like            # Like child
        self.dislike = dislike      # Dislike child

    def _is_leaf(self):
        if self.like is None and self.dislike is None:
            return True
        return False

    def _questions(self):
        if self._is_leaf():
            return [[]]  # one path: only contains self.value
        paths = []
        for child in [self.like, self.dislike]:
            for path in child._questions():
                paths.append([self.question] + path)
        return paths

    def _groups(self):
        if self._is_leaf():
            return [self.users]
        return self.like._groups() + self.dislike._groups()

    def groups_and_questions(self):
        res = {}
        for leaf in range(number_of_leaves(self)):
            res[leaf] = {'users': self._groups()[leaf],
                         'questions': self._questions()[leaf]}
        return res


def number_of_leaves(root: Node):
    if root._is_leaf():
        return 1
    return number_of_leaves(root.like) + number_of_leaves(root.dislike)
