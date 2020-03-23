import collections
import math
import scipy.sparse
import numpy as np
import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for
#!/usr/bin/env python
# -*- coding: utf-8 -*-

@not_implemented_for('multigraph')
def divrank(G, alpha=0.25, d=0.85, personalization=None,
            max_iter=100, tol=1.0e-6, nstart=None, weight='weight',
            dangling=None):
    '''
    Returns the DivRank (Diverse Rank) of the nodes in the graph.
    This code is based on networkx.pagerank.
    Args: (diff from pagerank)
      alpha: controls strength of self-link [0.0-1.0]
      d: the damping factor
    Reference:
      Qiaozhu Mei and Jian Guo and Dragomir Radev,
      DivRank: the Interplay of Prestige and Diversity in Information Networks,
      http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.174.7982
    '''
    if len(G) == 0:
        return {}

    if not G.is_directed():
        D = G.to_directed()
    else:
        D = G

    # Create a copy in (right) stochastic form
    W = nx.stochastic_graph(D, weight=weight)
    N = W.number_of_nodes()

    # self-link (DivRank)
    for n in W.nodes:
        for n_ in W.nodes:
            if n != n_ :
                if n_ in W[n]:
                    W[n][n_][weight] *= alpha
            else:
                if n_ not in W[n]:
                    W.add_edge(n, n_)
                W[n][n_][weight] = 1.0 - alpha

    # Choose fixed starting vector if not given
    if nstart is None:
        x = dict.fromkeys(W, 1.0 / N)
    else:
        # Normalized nstart vector
        s = float(sum(nstart.values()))
        x = dict((k, v / s) for k, v in nstart.items())

    if personalization is None:
        # Assign uniform personalization vector if not given
        p = dict.fromkeys(W, 1.0 / N)
    else:
        missing = set(G) - set(personalization)
        if missing:
            raise NetworkXError('Personalization dictionary '
                                'must have a value for every node. '
                                'Missing nodes %s' % missing)
        s = float(sum(personalization.values()))
        p = dict((k, v / s) for k, v in personalization.items())

    if dangling is None:
        # Use personalization vector if dangling vector not specified
        dangling_weights = p
    else:
        missing = set(G) - set(dangling)
        if missing:
            raise NetworkXError('Dangling node dictionary '
                                'must have a value for every node. '
                                'Missing nodes %s' % missing)
        s = float(sum(dangling.values()))
        dangling_weights = dict((k, v/s) for k, v in dangling.items())
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = d * sum(xlast[n] for n in dangling_nodes)
        for n in x:
            D_t = sum(W[n][nbr][weight] * xlast[nbr] for nbr in W[n])
            for nbr in W[n]:
                #x[nbr] += d * xlast[n] * W[n][nbr][weight]
                x[nbr] += (
                    d * (W[n][nbr][weight] * xlast[nbr] / D_t) * xlast[n]
                )
            x[n] += danglesum * dangling_weights[n] + (1.0 - d) * p[n]

        # check convergence, l1 norm
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N*tol:
            return x
    raise NetworkXError('divrank: power iteration failed to converge '
                        'in %d iterations.' % max_iter)


def divrank_scipy(G, alpha=0.25, d=0.85, personalization=None,
                  max_iter=100, tol=1.0e-6, nstart=None, weight='weight',
                  dangling=None):
    '''
    Returns the DivRank (Diverse Rank) of the nodes in the graph.
    This code is based on networkx.pagerank_scipy
    '''
    N = len(G)
    if N == 0:
        return {}

    nodelist = G.nodes()
    M = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight=weight,
                                  dtype=float)
    S = np.array(M.sum(axis=1)).flatten()
    S[S != 0] = 1.0 / S[S != 0]
    Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
    M = Q * M

    # self-link (DivRank)
    M = scipy.sparse.lil_matrix(M)
    M.setdiag(0.0)
    M = alpha * M
    M.setdiag(1.0 - alpha)
    #print M.sum(axis=1)

    # initial vector
    x = np.repeat(1.0 / N, N)

    # Personalization vector
    if personalization is None:
        p = np.repeat(1.0 / N, N)
    else:
        missing = set(nodelist) - set(personalization)
        if missing:
            raise NetworkXError('Personalization vector dictionary '
                                'must have a value for every node. '
                                'Missing nodes %s' % missing)
        p = scipy.array([personalization[n] for n in nodelist],
                        dtype=float)
        p = p / p.sum()

    # Dangling nodes
    if dangling is None:
        dangling_weights = p
    else:
        missing = set(nodelist) - set(dangling)


        if missing:
            raise NetworkXError('Dangling node dictionary '
                                'must have a value for every node. '
                                'Missing nodes %s' % missing)
        # Convert the dangling dictionary into an array in nodelist order
        dangling_weights = scipy.array([dangling[n] for n in nodelist],
                                       dtype=float)
        dangling_weights /= dangling_weights.sum()
    is_dangling = np.where(S == 0)[0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        D_t =  M * x
        x = (
            d * (x / D_t * M * x + sum(x[is_dangling]) * dangling_weights)
            + (1.0 - d) * p
        )
        # check convergence, l1 norm
        err = np.absolute(x - xlast).sum()
        if err < N * tol:
            return dict(zip(nodelist, map(float, x)))

    raise NetworkXError('divrank_scipy: power iteration failed to converge '
                        'in %d iterations.' % max_iter)


if __name__ == '__main__':

    g = nx.Graph()

    # this network appears in the reference.
    edges = {
        1: [2, 3, 6, 7, 8, 9],
        2: [1, 3, 10, 11, 12],
        3: [1, 2, 15, 16, 17],
        4: [11, 13, 14],
        5: [17, 18, 19, 20],
        6: [1],
        7: [1],
        8: [1],
        9: [1],
        10: [2],
        11: [4],
        12: [2],
        13: [4],
        14: [4],
        15: [3],
        16: [3],
        17: [3, 5],
        18: [5],
        19: [5],
        20: [5]
    }

    for u, vs in edges.items():
        for v in vs:
            g.add_edge(u, v)

    scores = nx.pagerank(g)
    print('# PageRank')
    print('# rank: node score')
    #print sum(scores.values())
    for i, n in enumerate(sorted(scores, key=lambda n: scores[n], reverse=True)):
        print('# {}: {} {}'.format(i+1, n, scores[n]))

    scores = divrank(g)
    print('\n# DivRank')
    # print sum(scores.values())
    print('# rank: node score')
    for i, n in enumerate(sorted(scores, key=lambda n: scores[n], reverse=True)):
        print('# {}: {} {}'.format(i + 1, n, scores[n]))

    scores = divrank_scipy(g)
    print('\n# DivRank (scipy)')
    # print sum(scores.values())
    print('# rank: node score')
    for i, n in enumerate(sorted(scores, key=lambda n: scores[n], reverse=True)):
        print('# {}: {} {}'.format(i + 1, n, scores[n]))

class DivRank():
    """
        Implementation of the paper: DivRank: the Interplay of Prestige and Diversity in Information Networks
        Url: https://dl.acm.org/citation.cfm?id=1835931
    """

    def rank(self, network: np.ndarray, steps = 100, teleport_prob : int = 0.9, follow_link_prob : int = 0.25, epsilon : int = 0.001):
        # teleport_prob is lambda in paper
        # follow_link_prob is alpha in paper
        _, n_vertices = network.shape
        p0 = np.zeros(shape=(n_vertices, n_vertices))
        pt = np.zeros(shape=(n_vertices, n_vertices))

        ranks : dict = {}
        visits : dict = {}
        preference_distribution : dict = {}

        # at step 0 every node has been "visited" once
        for v in range(0, n_vertices):
            visits[v] = 1

        # preference of visiting each node
        sumColikes = 0
        for v in range(0, n_vertices):
            # summing up each row
            preference_distribution[v] = np.sum(network[v])
            # used for normalizing
            sumColikes += np.sum(network[v])
        # normalizing visit distribution
        for v in range(0, n_vertices):
            preference_distribution[v] = \
                preference_distribution[v] / sumColikes

        # transition probabilities at step 0
        for u in range(0, n_vertices):
            for v in range(0, n_vertices):
                if u == v:
                    p0[u, v] = 1 - follow_link_prob
                else:
                    p0[u, v] = follow_link_prob * network[u, v]

        # Before iterations begin, we can compute the probability
        # of being in each node.
        # 1/n_vertices is the probability of being in
        # each node before walk has begun
        for v in range(0, n_vertices):
            sum = 0
            for u in range(0, n_vertices):
                sum += p0[u, v]
            ranks[v] = sum * 1/n_vertices

        print("--- DivRank started ---")
        # iterations begin
        for step in range(0, steps):
            newRanks : dict = {}
            DT : dict = {}

            # filling up DT with equation 4
            for u in range(0, n_vertices):
                sum = 0
                for v in range(0, n_vertices):
                    sum += p0[u, v] * visits[v]

                DT[u] = sum

            # transition probabilities at step T
            for u in range (0, n_vertices):
                for v in range(0, n_vertices):
                    pt[u, v] = \
                        (1 - teleport_prob) * preference_distribution[v] + \
                        teleport_prob * ((p0[u,v] * visits[v]) / DT[u])

            # probabilities of being each node at time T
            for v in range(0, n_vertices):
                sum = 0
                for u in range(0, n_vertices):
                    sum += pt[u, v] * ranks[u]
                newRanks[v] = sum

            # "performing walks" with equation 11
            for v in range(0, n_vertices):
                sum = 0
                for u in range(0, n_vertices):
                    sum += ((p0[u, v] * visits[v]) / DT[u]) * ranks[u]

                newRanks[v] = (1 - teleport_prob) * preference_distribution[v]\
                              + teleport_prob * sum


            # normalizing ranks in order to make sure they sum to 1
            # and not 0.9999 or similar
            factor = 1.0 / math.fsum(ranks.values())
            for k in ranks:
                ranks[k] = ranks[k] * factor

            # visit a random node.
            # Node is selected by using the probabilities of being in each node.
            nodes = np.fromiter(ranks.keys(), dtype=int)
            node = np.random.choice(nodes, 1, p=np.fromiter(ranks.values(), dtype=float))[0]
            visits[node] += 1

            delta = np.linalg.norm((np.fromiter(ranks.values(), dtype=float) - np.fromiter(newRanks.values(), dtype=float)), ord=1)

            if step % 25 == 0:
                print(f'step: {step}. delta: {delta}')

            if delta <= epsilon or step == steps - 1:
                return collections.OrderedDict(newRanks)

            ranks = newRanks