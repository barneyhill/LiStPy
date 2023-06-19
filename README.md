# LiStax
## Minimal Vectorised Li and Stephens implementation in JAX.

This library contains a minimal and vectorised implementation of the Li and Stephens model as described in the (2022 `kalis` paper)[https://arxiv.org/abs/2212.11403]. 

## Goals
- A simple formulation of the Li and Stephens model to showcase both the model itself and the JAX library.
- A general platform to investigate O(N^2) -> O(N) optimisations, (see recent paper)[https://www.biorxiv.org/content/10.1101/2023.05.19.541517v1]
- Automatic differentation upon the HMM parameters.

## TODO
- Testing, currently the results only look plausible - will add tests.

## Example

```
import listax

key = jax.random.PRNGKey(0)

# generate random haplotype matrix L=400 by N=300
# with variants in rows and haplotypes in columns
L, N = 1000, 50

# mutation rate (across all L)
mu = 1e-6
population_size = 100000

ts = listax.utils.simulate_ts(sample_size = N,
    length = L,
    population_size=population_size,
    sim_duration=1E12,
    mutation_rate= mu,
    random_seed = 2)

# (L sites, N haplotypes) haplotype matrix
h = listax.utils.ts_to_array(ts)

# distances between variants
m = listax.utils.get_distances(ts)

model = listax.LiStephens(h, m, gamma=1, N_est=population_size, mu=mu)
p, d = model.run(target_l=10)

listax.utils.plot_ts_tmrca(ts, l=10)
listax.utils.plot_model_distances(d)
```

![image](https://github.com/barneyhill/listax/assets/43707014/7fd991e9-7969-41e5-9122-0ffcd2a9da30)
![image](https://github.com/barneyhill/listax/assets/43707014/c3d43eea-411e-4a97-8717-eb676f3a0611)

The distance matrix seems plausible... testing and many bug fixes coming...

