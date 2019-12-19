import tree

if __name__ == '__main__':
    root = tree.Node([1, 2, 3], (1,2))
    a = tree.Node([1], (1, 3))
    b = tree.Node([2,3], (2, 3))
    c = tree.Node([2], (1, 3))
    d = tree.Node([3], (3, 4))
    e = tree.Node([1], (2,3))
    root.add_child(a)
    root.add_child(b)
    a.add_child(e)
    b.add_child(c)
    b.add_child(d)
    root.pretty_print('')


