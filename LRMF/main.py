import numpy as np


class LRMF():
    def __init__(self, l1, l2):
        self.l1 = l1,
        self.l2 = l2

    def _grow_tree(self, users, items, depth):
        loss = {}
        for item in items:
            loss[item] = np.inf  # evaluate eq. 11

        best_item = min(loss, key=loss.get)  # returns item with lowest loss

        if

def grow_tree(users, items, depth):


if __name__ == '__main__':
    d = {1: 100, 2: 200, 3: 5}
    print(min(d, key=d.get))