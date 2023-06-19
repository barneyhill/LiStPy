import numpy as np
import jax.numpy as jnp
import tskit, msprime
import matplotlib.pyplot as plt

def simulate_ts(
    sample_size: int,
    length: int = 100,
    population_size: int = 1000,
    sim_duration = 1E10,
    mutation_rate: float = 1e-6,
    random_seed: int = 1,
) -> tskit.TreeSequence:
    """
    Simulate some data using msprime with recombination and mutation and
    return the resulting tskit TreeSequence.
    Note this method currently simulates with ploidy=1 to minimise the
    update from an older version. We should update to simulate data under
    a range of ploidy values.
    """
    ancestry_ts = msprime.sim_ancestry(
        sample_size,
        population_size=population_size,
        ploidy=1,
        recombination_rate=1E-8,
        sequence_length=length,
        random_seed=random_seed,
        model=msprime.StandardCoalescent(duration=sim_duration)
    )

    # Make sure we generate some data that's not all from the same tree
    assert ancestry_ts.num_trees > 1
    return msprime.sim_mutations(
        ancestry_ts, rate=mutation_rate, random_seed=random_seed, model="binary",
    )

def get_distances(ts):
    positions = [variant.position for variant in ts.variants()]
    distances = []

    for i in range(1, len(positions)):
        distance = (positions[i] - positions[i - 1])
        distances.append(distance)
    
    return np.array(distances)

def ts_to_array(ts, chunks=None, samples=None, phased=False):
    """
    Convert the specified tskit tree sequence into an array.
    Note this just generates haploids for now - see the note above
    in simulate_ts.
    """
    if samples is None:
        samples = ts.samples()

    genotypes = []

    for var in ts.variants(samples=samples):
        genotypes.append(var.genotypes)

    return jnp.array(genotypes)

def plot_ts_tmrca(ts, l):
    site_pos = list(ts.variants())[l].site.position

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

def plot_model_distances(d):
    # plot distance matrix d:
    plt.title("LiStax Distance matrix")
    plt.xlabel("haplotype i")
    plt.ylabel("haplotype j")

    plt.imshow(d)
    plt.colorbar()
    plt.show()