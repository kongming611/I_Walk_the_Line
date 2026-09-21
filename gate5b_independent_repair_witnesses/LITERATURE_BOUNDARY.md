# Gate-5B literature boundary

This note records what the implementation is and is not claiming.

1. Eulig et al., *Toward Falsifying Causal Graphs Using a Permutation-Based
   Test*, enumerates graph-implied local Markov conditions and compares their
   falsification behavior with a permutation baseline. It does not establish
   a minimum number of edge repairs. Gate-5B therefore treats repair-signature
   coverage as the candidate methodological contribution, not as an already
   established novelty claim.

   Primary source: <https://arxiv.org/pdf/2305.09565>

2. Peters et al., *Causal Discovery with Continuous Additive Noise Models*,
   gives the mechanism assumption used by `RR_ANM`: under suitable additive
   noise conditions, causal direction/DAG structure can become identifiable
   beyond ordinary Markov-equivalence information. Gate-5B reports `RR_ANM`
   separately and does not present it as a distribution-free Markov-only
   certificate.

   Primary source: <https://www.jmlr.org/papers/volume15/peters14a/peters14a.pdf>

3. Shah and Peters, *The Hardness of Conditional Independence Testing and the
   Generalised Covariance Measure*, establishes important limits on universal
   nonparametric CI testing. RCIT and the residual tests are consequently
   treated as empirically evaluated tests under the frozen synthetic families,
   not as universally finite-sample-valid tests.

   Primary source: <https://www.imstat.org/publications/aos/aos_48_3/aos_48_3.pdf>

The experiment's evidence boundary is therefore: ground-truth-free at
certificate construction time, empirically valid on the frozen mechanism
families if the gate passes, and assumption-specific for `RR_ANM`.
