import numpy as np
import torch
import matplotlib.pyplot as plt

def get_ts_matrix(ts):
    array = ts.genotype_matrix()

    return torch.from_numpy(array[:, np.sum(array, axis=0) > 1])

def get_distances(ts):
    positions = [variant.position for variant in ts.variants()]
    distances = []

    for i in range(1, len(positions)):
        distance = (positions[i] - positions[i - 1])
        distances.append(distance)
    
    return np.array(distances)


def plot_ts_tmrca(ts, l):
    site_pos = list(ts.variants())[l].site.position

    N = ts.num_samples

    tmrca = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            tmrca[i,j] = ts.at(site_pos).tmrca(i,j)

    plt.title(f"ts TMRCA at site {l}")
    plt.xlabel("haplotype i")
    plt.ylabel("haplotype j")
    
    plt.imshow(tmrca)
    plt.colorbar()
    plt.show()

def plot_model_distances(d, l):
    # plot distance matrix d:
    plt.title(f"LiStax Distance matrix at site {l}")
    plt.xlabel("haplotype i")
    plt.ylabel("haplotype j")

    plt.imshow(d)
    plt.colorbar()
    plt.show()