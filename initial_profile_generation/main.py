import numpy as np
import initial_profile_generation.pairwise_comparison as pc

if __name__ == '__main__':
    P = np.genfromtxt('../data/user_profiles.csv', delimiter=',')
    Q = np.genfromtxt('../data/item_profiles.csv', delimiter=',')

    print(Q)

    cluster_centers = pc.cluster_items(Q, 10)

    print(cluster_centers)