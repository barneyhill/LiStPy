import numpy as np
import itertools
import msprime 

import lshmm.forward_backward.fb_haploid_samples_variants as fbh_sv
import lshmm.forward_backward.fb_haploid_variants_samples as fbh_vs

import listax.listax as listax

EQUAL_BOTH_HOM = 4
UNEQUAL_BOTH_HOM = 0
BOTH_HET = 7
REF_HOM_OBS_HET = 1
REF_HET_OBS_HOM = 2
MISSING = -1
MISSING_INDEX = 3


class LSBase:
    """Superclass of Li and Stephens tests."""

    def haplotype_emission(self, mu, m):
        # Define the emission probability matrix
        e = np.zeros((m, 2))
        e[:, 0] = mu  # If they match
        e[:, 1] = 1 - mu  # If they don't match

        return e

    def assertAllClose(self, A, B):
        """Assert that all entries of two matrices are 'close'"""
        # assert np.allclose(A, B, rtol=1e-9, atol=0.0)
        assert np.allclose(A, B, rtol=1e-09, atol=1e-08)


    def example_haplotypes(self, ts):

        H = ts.genotype_matrix()
        s = H[:, 0].reshape(1, H.shape[0])
        H = H[:, 1:]

        haplotypes = [s, H[:, -1].reshape(1, H.shape[0])]
        s_tmp = s.copy()
        s_tmp[0, -1] = MISSING
        haplotypes.append(s_tmp)
        s_tmp = s.copy()
        s_tmp[0, ts.num_sites // 2] = MISSING
        haplotypes.append(s_tmp)
        s_tmp = s.copy()
        s_tmp[0, :] = MISSING
        haplotypes.append(s_tmp)

        return H, haplotypes

    def example_parameters_haplotypes(self, ts, seed=42):
        """Returns an iterator over combinations of haplotype, recombination and mutation rates."""
        np.random.seed(seed)
        H, haplotypes = self.example_haplotypes(ts)
        n = H.shape[1]
        m = ts.get_num_sites()

        # Here we have equal mutation and recombination
        r = np.zeros(m) + 0.01
        mu = np.zeros(m) + 0.01
        r[0] = 0

        e = self.haplotype_emission(mu, m)

        for s in haplotypes:
            yield n, m, H, s, e, r

        # Mixture of random and extremes
        rs = [np.zeros(m) + 0.999, np.zeros(m) + 1e-6, np.random.rand(m)]
        mus = [np.zeros(m) + 0.33, np.zeros(m) + 1e-6, np.random.rand(m) * 0.33]

        e = self.haplotype_emission(mu, m)

        for s, r, mu in itertools.product(haplotypes, rs, mus):
            r[0] = 0
            e = self.haplotype_emission(mu, m)
            yield n, m, H, s, e, r

    def verify(self, ts, mu):
        for n, m, H_vs, s, e_vs, r in self.example_parameters_haplotypes(ts):
            e_sv = e_vs.T
            H_sv = H_vs.T

            # variants x samples
            F_vs, c_vs, ll_vs = fbh_vs.forwards_ls_hap(
                n, m, H_vs, s, e_vs, r, norm=False
            )
            B_vs = fbh_vs.backwards_ls_hap(n, m, H_vs, s, e_vs, c_vs, r)
            self.assertAllClose(np.log10(np.sum(F_vs * B_vs, 1)), ll_vs * np.ones(m))
            F_tmp, c_tmp, ll_tmp = fbh_vs.forwards_ls_hap(
                n, m, H_vs, s, e_vs, r, norm=True
            )
            B_tmp = fbh_vs.backwards_ls_hap(n, m, H_vs, s, e_vs, c_tmp, r)
            self.assertAllClose(ll_vs, ll_tmp)
            self.assertAllClose(np.sum(F_tmp * B_tmp, 1), np.ones(m))

            # samples x variants
            F_sv, c_sv, ll_sv = fbh_sv.forwards_ls_hap(
                n, m, H_sv, s, e_sv, r, norm=False
            )
            B_sv = fbh_sv.backwards_ls_hap(n, m, H_sv, s, e_sv, c_sv, r)
            self.assertAllClose(np.log10(np.sum(F_sv * B_sv, 0)), ll_sv * np.ones(m))
            F_tmp, c_tmp, ll_tmp = fbh_sv.forwards_ls_hap(
                n, m, H_sv, s, e_sv, r, norm=True
            )
            B_tmp = fbh_sv.backwards_ls_hap(n, m, H_sv, s, e_sv, c_tmp, r)
            self.assertAllClose(ll_sv, ll_tmp)
            self.assertAllClose(np.sum(F_tmp * B_tmp, 0), np.ones(m))

            # samples x variants agrees with variants x samples
            self.assertAllClose(ll_vs, ll_sv)

            #model = listax.LiStM(H_sv, m, gamma=1, N_est=n, mu=mu)
            #p, d = model.run(target_l=10)

            print(F_sv.shape, B_sv.shape)

            p_lshmm = F_tmp * B_tmp
            p_lshmm /= np.sum(p_lshmm, 0)

            print(p_lshmm)
            print(p_lshmm.shape)

        # Define a bunch of very small tree-sequences for testing a collection of parameters on
    def test_simple_n_10_no_recombination(self):
        mu = 0.5
        ts = msprime.simulate(
            10, recombination_rate=0, mutation_rate=mu, random_seed=42
        )
        assert ts.num_sites > 3
        self.verify(ts, mu)

    def test_simple_n_6(self):
        mu=7
        ts = msprime.simulate(6, recombination_rate=2, mutation_rate=mu, random_seed=42)
        assert ts.num_sites > 5
        self.verify(ts, mu)

    def test_simple_n_8(self):
        mu=5
        ts = msprime.simulate(8, recombination_rate=2, mutation_rate=5, random_seed=42)
        assert ts.num_sites > 5
        self.verify(ts, mu)

    def test_simple_n_8_high_recombination(self):
        mu=5

        ts = msprime.simulate(8, recombination_rate=20, mutation_rate=5, random_seed=42)
        assert ts.num_trees > 15
        assert ts.num_sites > 5
        self.verify(ts, mu)

    def test_simple_n_16(self):
        mu=5

        ts = msprime.simulate(16, recombination_rate=2, mutation_rate=5, random_seed=42)
        assert ts.num_sites > 5
        self.verify(ts, mu)


