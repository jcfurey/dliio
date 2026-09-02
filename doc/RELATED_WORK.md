# Related Work — Degeneracy in LiDAR(-Inertial) Odometry

Focused companion to `doc/REFERENCES.md` (the full citation card) and
`doc/FINDINGS_2026-06-22.md` (the experimental arc). This file collects the
prior art specific to **one problem**: a geometrically degenerate environment
(the 06042026 featureless tunnel) leaves a pose axis unobservable, the gate
holds the IMU prior on it, and that prior dead-reckons into a km-scale runaway.

It exists to (1) credit the work the degeneracy gate and governor descend from,
and (2) put the governor's math next to the closest published formulations so we
know what is standard, what is reinvented, and what is genuinely ours.

> Citations already in `doc/REFERENCES.md` are cross-linked, not repeated in
> full. Zhang 2016 (the gate's origin) and COIN-LIO (the intensity term) live
> there; this card adds the degeneracy-mitigation literature around them.

---

## 1. The field's decomposition (and ours)

Every method below answers the same three questions in some order — the same
three our FINDINGS arc worked through:

1. **Detect** — *which* directions are ill-constrained this scan?
2. **Inform** — can a complementary modality (intensity, vision, point-to-point)
   *observe* the weak axis and rescue it?
3. **Contain** — when no modality can, how do you keep the unobservable axis
   from corrupting the estimate?

Parts 1–6 of FINDINGS closed (2) for this rig; the governor is our answer to (3).

---

## 2. Detection — which axis is degenerate

| Reference | Venue | Link | What it grounds / how it relates |
|---|---|---|---|
| Zhang, Kaess, Singh — *On Degeneracy of Optimization-Based State Estimation Problems* | ICRA 2016 | [PDF](https://frc.ri.cmu.edu/~zhangji/publications/ICRA_2016.pdf) · [IEEE](https://ieeexplore.ieee.org/document/7487211/) | **The gate's origin** (already in REFERENCES.md). Degeneracy factor = min eigenvalue of the Hessian; *solution remapping* projects the update onto well-constrained directions and **holds the prior** on degenerate ones. Our gate's "erase the GICP correction along held eigen-directions" *is* solution remapping. |
| Tuna, Nubert, Nava, Khattak, Hutter — *X-ICP: Localizability-Aware LiDAR Registration for Robust Localization in Extreme Environments* | IEEE T-RO 2024 (vol. 40, 452–471) | [arXiv:2211.16335](https://arxiv.org/abs/2211.16335) | **Closest detection+mitigation cousin.** Ternary per-direction judgment (fully / partially / non-localizable) from the localizability *contribution* of correspondences against the optimization's principal directions; non-localizable directions are then frozen by a hard equality constraint *inside* the solve (§4). |
| Hatleskog, Alexis — *Probabilistic Degeneracy Detection for Point-to-Plane Error Minimization* | RA-L 2024 (vol. 9, no. 12) | [arXiv:2410.10784](https://arxiv.org/pdf/2410.10784) | **Already implemented here** as the soft probabilistic gate (`probGateKeepFraction`). First-principles keep-fraction = confidence the eigenvalue clears a noise floor, replacing a hard threshold; easier parameterization, better generalization. |
| *Informed, Constrained, Aligned: A Field Analysis on Degeneracy-Aware Point Cloud Registration in the Wild* | 2024 | [arXiv:2408.11809](https://arxiv.org/abs/2408.11809) | A **taxonomy + field benchmark** sorting all mitigations into *informed* (add a modality — our Part B), *constrained* (X-ICP-style solve-time freezing), and *aligned* (regularize toward a prior). Use this to position the governor in related work. |
| *LP-ICP: General Localizability-Aware Point Cloud Registration for Robust Localization in Extreme Unstructured Environments* | 2025 | [arXiv:2501.02580](https://arxiv.org/pdf/2501.02580) | Extends X-ICP's localizability analysis to point-to-line/point-to-plane; restates X-ICP's ternary thresholds and the hard-equality freeze on non-localizable directions. |

## 3. Inform — adding observability to the weak axis (the levers we tried)

| Reference | Venue | Link | Relation to our FINDINGS |
|---|---|---|---|
| Pfreundschuh et al. — *COIN-LIO: Complementary Intensity-Augmented LiDAR Inertial Odometry* | ICRA 2024 | [arXiv:2310.01235](https://arxiv.org/abs/2310.01235) · [code](https://github.com/ethz-asl/COIN-LIO) | **The paper our LiDAR-image term reimplements** (also in REFERENCES.md). Selects intensity patches *complementary to the degenerate geometric directions*. Our Parts 2–6 independently confirm its gains are texture-dependent: reflectivity here is along-axis **aliased** and near-IR is **shot-noise-limited**, so the term is a no-op for this tunnel. |
| *GenZ-ICP: Generalizable and Degeneracy-Robust LiDAR Odometry Using an Adaptive Weighting* (COCEL, POSTECH) | RA-L 2025 | [arXiv:2411.06766](https://arxiv.org/abs/2411.06766) · [code](https://github.com/cocel-postech/genz-icp) | **The cheapest unexplored lever.** No extra modality: adaptively blends point-to-plane and point-to-point metrics by local geometry to keep long-corridor optimization well-posed. Open-source (ROS1/2). |
| *LION: Lidar-Inertial Observability-Aware Navigator for Vision-Denied Environments* (CoSTAR / DARPA SubT) | FSR 2021 | [arXiv:2102.03443](https://arxiv.org/pdf/2102.03443) | Uses an **observability metric to modulate LiDAR-vs-IMU trust** in a fixed-lag smoother — the same instinct as the governor "bound the pose, let the observer pull velocity back from the IMU prior." |
| *DARE-SLAM: Degeneracy-Aware and Resilient Loop Closing in Perceptually-Degraded Environments* (CoSTAR) | JINT 2021 | [arXiv:2102.05117](https://arxiv.org/pdf/2102.05117) | Excludes unobservable regions from loop-closure search. Relevant only if a back-end is added; the degeneracy estimate would gate place recognition too. |

## 4. Contain — bound / de-weight the failure (where the governor lives)

| Reference | Venue | Link | Relation to the governor |
|---|---|---|---|
| Lee, Marsim, Myung — *LODESTAR: Degeneracy-Aware LiDAR-Inertial Odometry with Adaptive Schmidt-Kalman Filter and Data Exploitation* | RA-L 2025 (KAIST) | [arXiv:2511.09142](https://arxiv.org/abs/2511.09142) | **Closest published cousin to the governor.** Classifies states active/fixed by condition number (threshold 1.5); fixed states get **zero Kalman gain** and anchor the active states *through cross-covariance*. That is our two-part move — hold the degenerate axis + let its (co)variance mark it untrusted — done as a principled Schmidt-Kalman update instead of an explicit clamp + rank-1 inflation. |
| Nubert et al. — *Informed, Constrained, Aligned* (above) | 2024 | [arXiv:2408.11809](https://arxiv.org/abs/2408.11809) | The governor is a fourth bucket the taxonomy doesn't quite name: *constrained at the output*, not in the solve. |
| *Leap-SLAM: Degeneracy-Mitigated Robust SLAM with Leapfrogging Multi-Robot Collaboration in Tunnels* | Information Fusion 2026 | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1566253526004173) | The only result found that names our exact environment; mitigates via multi-robot leapfrogging (out of scope for a single platform, but confirms tunnels remain an open single-agent problem). |

---

## 5. Lineage — what the governor inherits, reinvents, and owns

| Governor component (`degeneracy_governor.h`, `odom.cc`, `nano_gicp.cc`) | Closest prior art | Verdict |
|---|---|---|
| Gate holds IMU prior on degenerate eigen-directions (erases the GICP correction) | Zhang 2016 *solution remapping*; X-ICP *non-localizable → not updated* | **Inherited** — cite Zhang as origin, X-ICP as the modern equivalent |
| Rank-1 covariance inflation on held axes (`covPosVar`/`covRotVar`) so downstream de-weights them | LODESTAR fixed-state covariance anchoring (Schmidt-Kalman) | **Reinvented** — a cruder explicit add vs. LODESTAR's principled cross-covariance coupling |
| **Per-scan OUTPUT displacement clamp vs. the previous pose** (`governPose`, `maxStep{Trans,Rot}`) | — (nearest is X-ICP's *in-solve* equality constraint) | **Ours** — a cheap post-solve geometric fail-safe, not a constrained solver |
| Velocity recovered via the geometric observer instead of a ZUPT | LION observability-aware trust modulation | **Convergent** — same instinct, different mechanism |
| Intensity / near-IR photometric term | COIN-LIO | **Reimplements**; our negative result for this rig is the contribution |

---

## 6. Math comparison — `governPose` vs. the constrained / Schmidt-Kalman formulations

### 6.1 X-ICP (constrain *inside* the solve) vs. the governor (clamp the *output*)

X-ICP categorizes each principal direction of the registration Hessian and, for a
**non-localizable** direction `vₖ`, adds a **hard equality constraint** forcing
zero update along that eigenvector — the pose in that direction stays at the
initial estimate (the prior). The constraint acts on the optimization update
`dx` in the Hessian eigenbasis, *before* a step is ever taken.

The governor reaches the same end state (no motion the data can't support along a
held axis) but from the **opposite side of the solve**:

```
gate (nano_gicp):   erase correction along held vₖ   → prior is held (≈ X-ICP freeze)
GICP solve:         runs free on the held axis        → can still drift via the prior
governPose:         clamp |T_new − T_prev · vₖ| ≤ cap → post-hoc geometric bound
```

- **X-ICP** prevents the bad update; the correspondences themselves are
  constrained, so the solver never "sees" the unobservable direction.
- **Governor** lets the gate hold the prior, lets GICP converge unconstrained,
  then bounds the *displacement of the final pose* projected onto each held
  world-frame eigenvector. The clamp is exact and order-independent because the
  held directions are orthonormal eigenvectors of one self-adjoint block:

  ```
  dp += vₖ · ( clamp(vₖ·dp, ±cap) − vₖ·dp )     # vₖ·vⱼ = 0 ⇒ independent per axis
  ```

**Trade-off to record:** X-ICP's in-solve constraint also improves the
*correspondences* along the weak axis; the governor does not — it only bounds the
output. In exchange the governor is solver-agnostic, default-off, bit-identical
when disabled, and lives in a pure unit-testable free function. It is the
*aligned/constrained-at-output* point in the design space, deliberately cheaper
than a constrained solver.

### 6.2 LODESTAR (Schmidt-Kalman) vs. the governor's covariance inflation

LODESTAR partitions states into *updating* (`u`) and *fixed* (`f`) by Jacobian
condition number (`χ(H) = σ_max/σ_min`, threshold `1.5`). Fixed states take
**zero gain** and influence the updating states purely through the
**cross-covariance** blocks of a Joseph-form update:

```
K = [Kᵤ; 0]            # fixed-state gain is zero
P = [[P_uu, P_uf],
     [P_fu, P_ff]]     # P_ff held; P_uf / P_fu couple the anchor in
```

The governor's covariance handling is the **explicit, rank-1 analogue** of this:
for each held world-frame direction `d` it adds `covVar · d dᵀ` onto the
position / rotation block of the published `/odom` covariance, marking that axis
untrusted for any downstream consumer:

```
Σ_extra = Σ_k  covVar · d_k d_kᵀ        # rank-1 per held axis, added to the 6×6 cov
```

- **LODESTAR**: principled — the held axis's distrust propagates *into the
  estimator* via cross-covariance, so the filter itself de-weights it.
- **Governor**: pragmatic — the inflation is *informational only* (it tells
  downstream nodes "don't trust this axis"); it does not feed back into our own
  geometric observer. The observer is corrected separately, by the pose clamp
  pulling the velocity estimate.

**Takeaway for a future iteration:** if we ever want the held-axis distrust to
act *inside* DLIO's own estimator rather than only flag it downstream, LODESTAR's
Schmidt-Kalman coupling is the formulation to adopt — it would subsume both the
clamp and the inflation into one update. For now the clamp + flag is the smaller,
testable, default-off change, and that was the point of the pivot.

---

## 7. Recommended next experiment

Given Parts 1–6 closed the *inform* lever for this rig and the governor handles
*contain*, the cheapest untried option is **GenZ-ICP's adaptive point-to-plane /
point-to-point weighting** — it targets long-corridor degeneracy with no extra
modality and is open-source. Worth an A/B before committing to containment-only,
and it composes with the governor (GenZ-ICP improves the solve; the governor
remains the fail-safe). See §3.

---

## Credits

This file credits the authors above for the ideas the gate and governor build on.
The DLIO architecture itself, GICP, the contracting observer, and nanoflann are
credited in `doc/REFERENCES.md`. Any text here that paraphrases a paper's method
is attributed inline with a link to the original; please preserve those links if
this section is edited.
