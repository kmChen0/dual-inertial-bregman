# Algorithm sources

The baselines are local implementations of the published algorithms, not imported upstream reference implementations. These references identify the algorithms; they do not imply endorsement of implementation details or a license grant for this repository.

- **AEPG:** R. Merchav, S. Sabach and M. Teboulle, “The adaptive integrated mean: a novel perspective for acceleration and its extension to Bregman first-order methods,” *Mathematical Programming* (2026), DOI: 10.1007/s10107-026-02423-4. The experiment uses its Shannon-entropy specialization.
- **ABPG-g:** F. Hanzely, P. Richtárik and L. Xiao, “Accelerated Bregman Proximal Gradient Methods for Relatively Smooth Convex Optimization,” *Computational Optimization and Applications* 79 (2021), 405–440. The experiment uses gain adaptation, with parameters documented in `PROTOCOL.md`.
- **V-iBPPA:** L. Yang and K.-C. Toh, “Bregman Proximal Point Algorithm Revisited: A New Inexact Version and Its Inertial Variant,” *SIAM Journal on Optimization* 32(3) (2022), 1523–1554, DOI: 10.1137/20M1360748. The entropy exponent-two practical rule is heuristic, as explained in Remark 5.2 of that work.
- **FISTA:** A. Beck and M. Teboulle, “A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems,” *SIAM Journal on Imaging Sciences* 2(1) (2009), 183–202.
- **Gradient restart:** B. O'Donoghue and E. Candès, “Adaptive Restart for Accelerated Gradient Schemes,” *Foundations of Computational Mathematics* 15 (2015), 715–732.

The DI and energy-safeguarded DI implementations correspond to the accompanying manuscript. `SOURCE_PROVENANCE.json` records which experiment snapshots were used to prepare the package. It is a provenance record, not a runtime dependency list.
