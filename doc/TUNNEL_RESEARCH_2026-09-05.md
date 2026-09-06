# Curved-tunnel odometry: evidence, literature, and next experiments

Research date: **2026-09-05**. Implementation baseline: dliio
`e6c70db9a42194ecc70b1b428e7d87177c72ca65`; workspace experiment record:
`96ddc7de12a372036ff369ed0c876e32d0dd69c4`.

The subsequent [diagnostic implementation and replay](TUNNEL_DIAGNOSTICS.md)
tests the first recommendation below. Its measured rejection breakdown and
geometry results update the hypotheses in this research snapshot.

**The next experiment should measure coupled yaw/translation ambiguity and
explain the long gaps in usable surface texture.** Increasing intensity weight
alone is poorly justified: the current measurement is intermittent, and its
translation-only direction selector can miss a weak combination of rotation
and translation. A separate finer-geometry experiment can test whether the
current filtering discards useful surface detail.

These are research conclusions and proposed experiments, not additional
estimator changes or a finding that the remaining sloshing has been solved.
The [reference card](REFERENCES.md) records primary sources and reading scope;
[tunnel_references.bib](tunnel_references.bib) provides exportable citations.

## 1. What this recording establishes

The evidence is the longest September capture,
`bags/2025_09_09.13_56_34.139/ros2`: 1105.50 seconds, approximately 18 minutes
25 seconds. It contains a moving-head VLP-16, IMU, gimbal measurements, and
roughly 1 Hz stereo images. The earlier
[Ouster tunnel findings](TUNNEL_FINDINGS.md) and
[water-pool research](ROBUSTNESS_RESEARCH.md) concern another rig and failure
case. Their conclusions cannot establish what happened here.

| Local observation | What it establishes | What remains unknown |
|---|---|---|
| Native point fields are `x/y/z/t/i`; `i` preserves 8-bit values divided by 255 | One recorded return-strength channel is available | Exact upstream radiometric processing and this scanner's calibration |
| IMU timing remained continuous near the catastrophic failure | No observed IMU gap explains that event | Bias error and observer response during slower sloshing |
| Registration once accepted a trial after 5,723 correspondences became zero | A concrete solver acceptance defect | Accuracy of otherwise finite, converged registrations |
| Corrected geometry and texture runs both complete all 3,669 sweeps | The fix removes that observed catastrophic failure | Along-tunnel trajectory accuracy |
| With matched slower observer gains, texture reduces peak estimated speed from 6.50 to 4.51 m/s and estimated travel from 830.64 to 748.89 m | The patch term changes estimated excursions | Whether those changes better match actual motion |
| Texture is accepted on 316 sweeps, or 8.6%; the longest unsupported interval is 132.27 s | Appearance provides only intermittent constraints in this configuration | Individual causes within the aggregate rejection stages |
| Two corrected texture replays produce identical compared outputs | Reproducibility for that input, build, and configuration | Robustness across recordings, initializations, or independent motion truth |

The [initial investigation](../../../docs/dliio-tunnel-investigation.md) and
[corrected surface-texture experiment](../../../docs/dliio-surface-texture.md)
contain the measurements, profiles, regressions, and replay instructions.
The original-gain compatibility run also completes, but differs in both
observer gains and the older pointwise intensity term; it is not the isolated
geometry-versus-patch comparison above.

## 2. What “intensity” means on this rig

The VLP-16 manual describes its return-strength byte as factory-calibrated
reflectivity, with diffuse returns in 0–100 and retroreflective returns in
101–255. That is consistent with a normalized byte in `i`, but does not prove
the complete Exyn conversion chain or this unit's calibration. The inspected
manufacturer-hosted Rev. F is marked DRAFT.
[VLP-16 manual, §6.1, p. 34](https://data.ouster.io/downloads/velodyne/user-manual/vlp-16-user-manual-revf.pdf)

There is still only **one** recorded channel. Ouster's separate signal-photon
and calibrated-reflectivity fields describe a different interface.
[Ouster field definitions](https://static.ouster.dev/sensor-docs/image_route1/image_route2/sensor_data/sensor-data.html)

Consequently, the proposed radiometric audit should begin with native `i`,
not an unconditional range-squared correction. Repeated observations of the
same surface should measure residual dependence on range, incidence, laser
row, and head phase. For this moving head, a physical range calculation needs
the optical origin at each return's acquisition time; the vehicle origin
after sweep assembly is insufficient. Preserve that provenance before
testing a sensor-specific correction. General radiometric models explain why
sensor processing must be established first.
[Kashani et al., 2015](https://www.mdpi.com/1424-8220/15/11/28099)

The existing pipeline corrects head motion per return, assembles three scans,
and applies vehicle-motion deskew. The resulting vehicle-frame cloud is
unorganized. A spherical projection that assumes a stationary Ouster scanner
does not preserve the original acquisition geometry here. The current 3D
surface patches avoid that requirement; an image-based experiment would need
an explicit acquisition model and retained row/time metadata.

## 3. Why the curve deserves a coupled-motion diagnostic

McDermott explicitly discusses ambiguity between along-track translation and
yaw in a curved tunnel, as well as finite ambiguities caused by repeated
structures. This closely matches the reported symptom, but is a hypothesis
for this recording, not a diagnosis from the current telemetry.
[McDermott, 2025, §2.8, pp. 46–47](https://dl.tufts.edu/downloads/w9505f71j)

The local [registration implementation](../src/nano_gicp/nano_gicp.cc) examines
the rotation and translation 3×3 Hessian blocks separately for its degeneracy
gate. The new surface matcher selects its search direction from the
geometry-only translation block. Good information in each block, with the
other variables held fixed, does not establish good information jointly.

### A small counterexample

This is an illustrative local model, not a fitted model of the bag. Let
`u = R * delta_yaw` and `v = delta_along_curve`, both in meters, and suppose a
geometric residual depends on `r = v - u`. Its normal matrix is:

```text
                [ 1  -1 ]
H = J^T J =     [       ]       eigenvalues: 0, 2
                [-1   1 ]

H_uu = 1, H_vv = 1             weak direction: [u, v] proportional to [1, 1]
```

Each diagonal block appears constrained. Moving and turning together leaves
the residual unchanged. The example can be embedded in a six-coordinate
system with the other four coordinates well constrained.

### Proposed measurements

1. **Express the geometry Jacobians about the scan center.** The current
   left-update Jacobian uses transformed points; a distant world origin can
   introduce a large rotation lever arm. For a point `p` and scan center `c`,
   the centered form is `J = [-skew(p - c), I]`, with correspondingly transformed
   pose increments. Verify invariance to a change of world origin.
2. **Keep all cross terms and declare a scale.** Rotation and translation have
   different units. For example, use coordinates `[L * delta_theta, delta_t]`
   with a documented characteristic length `L`, transform the Jacobians
   accordingly, and report sensitivity to `L`. A prior-whitened analysis is
   another option only if the prior covariance is defensible. An arbitrary
   raw 6×6 condition-number threshold is not sufficient.
3. **Compare conditional and coupled weakness.** Record the full scaled
   geometry spectrum and weak vectors alongside the existing block ratios.
   Also inspect `H_tt - H_tr * pinv(H_rr) * H_rt`, the translation information
   after rotation is allowed to adjust. Use a consistent rank tolerance and
   stable factorization; near-singular blocks require care.
4. **Relate the weak mode to actual corrections.** Log the yaw and local
   along-curve components of scan-to-prior corrections, observer innovations,
   and bias estimates. Project the texture residual's Jacobian onto that mode.
   Handle eigenvector sign changes and nearly repeated eigenvalues before
   interpreting a time series. Local tangent estimates derived from odometry
   are diagnostic coordinates, not independent truth.
5. **Inspect finite ambiguity as well as local curvature.** On selected cached
   sweeps, sample a small joint yaw/along-curve cost surface. Show both the
   fixed-support objective and a separately labeled rematched objective.
   A locally healthy Hessian can coexist with another plausible finite
   alignment. Retain the corrected fixed-support LM acceptance rule.

X-ICP supports scan-centered analysis and correspondence-level contributions,
but it too separates rotation and translation because of their units. It
must not be cited as an existing solution to the specific cross-block issue
above. The coupled diagnostic is a proposed local extension.
[X-ICP, §V-A](https://arxiv.org/html/2211.16335v4)

If weak mixed modes coincide with the reported sloshing while the translation
ratio stays healthy, that supports changing the detector and texture search
coordinates. If they do not, investigate association changes, patch lifetime,
and observer innovations before replacing the gate.

## 4. Handling weak geometry does not create a missing measurement

Solution remapping suppresses corrections in weak directions. A field
comparison of degeneracy-aware registration emphasizes the importance of a
useful prior and the dependence of results on method and tuning. These are
reasons to measure how the prior and correction interact, rather than assume
one gate setting generalizes to every tunnel.
[Zhang et al., 2016](https://www.cs.cmu.edu/~kaess/pub/Zhang16icra.pdf),
[Tuna et al., 2025, §VI](https://arxiv.org/html/2408.11809v3)

Two distinctions matter when citing methods already represented in dliio:

| Published idea | Local implementation distinction | Research implication |
|---|---|---|
| Noise-aware directional confidence | `probGate` uses configured scalar noise floors; full point/normal noise propagation is deferred | Do not interpret its output as a validated posterior probability |
| GenZ-ICP plane/point blending | The paper uses the planar-correspondence fraction; [genz_weight.h](../include/nano_gicp/genz_weight.h) uses translation-Hessian conditioning | Label this an inspired variant and evaluate it on its own terms |

The first distinction follows the noise model in
[Hatleskog & Alexis](https://arxiv.org/html/2410.10784v2); the second follows
[GenZ-ICP, §III-E](https://arxiv.org/html/2411.06766v1).

Giving point-to-point residuals more weight can make a numerical system
stiffer without establishing a stable physical correspondence along an
otherwise uniform wall. The covariance literature warns that ignoring ICP
rematching can invalidate point-to-point covariance estimates. Accordingly,
neither a better-conditioned matrix nor inverse-Hessian entries alone prove
that the missing motion is measured.
[Bonnabel et al., 2016](https://arxiv.org/abs/1410.7632)

The already-fixed acceptance defect remains a separate prerequisite:
compare each LM trial against its base associations and objective settings.
FastGICP's implementation makes that separation between linearization and
trial error evaluation. A new degeneracy method must preserve it.
[FastGICP optimizer](https://github.com/koide3/fast_gicp/blob/master/include/fast_gicp/gicp/impl/lsq_registration_impl.hpp),
[FastGICP error evaluation](https://github.com/koide3/fast_gicp/blob/master/include/fast_gicp/gicp/impl/fast_gicp_impl.hpp)

## 5. Make texture support measurable, then extend it

COIN-LIO selects appearance information complementary to weak geometric
directions and processes intensity images to improve brightness consistency.
PG-LIO uses normalized photometric factors and a sliding-window estimator.
These motivate useful patch selection and longer-lived references; the
current 3D pairwise matcher is an adaptation, not a reproduction of either
system. Their released image paths also assume Ouster-style input.
[COIN-LIO methods](https://arxiv.org/html/2310.01235v4),
[PG-LIO methods](https://arxiv.org/html/2506.18583v1),
[COIN-LIO code](https://github.com/ethz-asl/COIN-LIO),
[MIMOSA code](https://github.com/ntnu-arl/mimosa)

Mean subtraction and contrast normalization make NCC useful under a local
positive brightness gain and offset. They cannot distinguish repeated
patterns or recover absent samples. An optimization treatment is available
in Woodford's preprint; it does not establish the covariance of this spatial
matcher.
[Woodford, v3, 2022](https://arxiv.org/abs/1810.04320v3)

### What the saved counters already show

A new audit of `fixed-texture-full.csv` gives the following scan counts.
Times use the first registered scan midpoint as zero. The 360–500 s window
matches the earlier experiment report; it is not an independently labeled
ground-truth failure interval.

| Condition | Whole replay (3,669 scans) | 360–500 s (466 scans) | Interior gap (210 scans) |
|---|---:|---:|---:|
| Translation eigenvalue ratio below 0.15 | 3,354 | 466 | 210 |
| At least one candidate patch | 3,339 | 466 | 210 |
| At least eight candidate patches | 3,331 | 466 | 210 |
| At least eight supported patches | 1,557 | 250 | 61 |
| At least eight unique matches | 417 | 60 | 5 |
| Accepted consensus | 316 | 50 | 0 |

The **132.27 s gap is at startup**, before the first accepted texture scan.
The longest gap between two accepted scans is **63.30 s, from 671.37 to
734.67 s**. Every one of its 210 interior scans produces candidate patches.
Simply widening the translation weakness gate would not address that gap.
In the 360–500 s window, candidate extraction also runs on every scan.

The counters are not individual rejection labels. `candidates` counts patches
after reference geometry, sampling, and contrast checks. `supported` additionally
requires passing the reference repetition test and obtaining valid, sufficiently
contrasted current samples at every search shift. `unique` adds the correlation,
peak-boundary, peak-separation, and curvature tests. Thus a drop before
`supported` cannot yet be attributed specifically to missing coverage, weak
contrast, or repetition. The 101 scans with at least eight unique matches but
no acceptance fail the subsequent agreement checks.

The input SHA-256 is
`e3c8a46dca6868245258647033e95f1363a95d95d8b5cc81525bb26ae670ff66`.
The detailed audit is in the ignored workspace artifact
`local/dliio-tunnel-research/texture-support-audit.json`. The following reproduces
the main counts and intervals from the workspace root without rerunning DLIO:

```bash
python3 - <<'PY'
import csv
from pathlib import Path

path = Path('local/dliio-surface-texture/fixed-texture-full.csv')
rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(path.open())]
t0 = rows[0]['scan_stamp']
accepted = [i for i, r in enumerate(rows) if r['texture_valid'] > 0]
a, b = max(zip(accepted, accepted[1:]),
           key=lambda p: rows[p[1]]['scan_stamp'] - rows[p[0]]['scan_stamp'])
print('first acceptance:', rows[accepted[0]]['scan_stamp'] - t0)
print('longest interior interval:', rows[a]['scan_stamp'] - t0,
      rows[b]['scan_stamp'] - t0, rows[b]['scan_stamp'] - rows[a]['scan_stamp'])
groups = {'all': rows,
          '360-500': [r for r in rows if 360 <= r['scan_stamp'] - t0 < 500],
          'gap interior': rows[a + 1:b]}
for label, group in groups.items():
    print(label, 'scans:', len(group), 'ratio < 0.15:',
          sum(0 <= r['texture_weak_ratio'] < 0.15 for r in group))
    for field, minimum in [('texture_candidates', 1), ('texture_candidates', 8),
                           ('texture_supported', 8), ('texture_unique', 8),
                           ('texture_valid', 1)]:
        print(field, '>=', minimum, sum(r[field] >= minimum for r in group))
PY
```

### Follow-up measurements and reference lifetime

The saved counters narrow the problem but do not resolve it. Additional
logging should distinguish:

- **No search:** no suitable geometric weak axis, two weak translation axes,
  invalid deskew/reference, or excessive frame gap.
- **No supported patch:** insufficient local planarity or contrast, spacing
  rejection, or incomplete sampling anywhere in the search window.
- **No unique measurement:** self-similar reference, weak peak, ambiguous
  alternative, boundary optimum, or insufficient patch agreement.

Plot those stages against head phase, range, patch location, and time.
Measure useful-patch lifetime and whether the same physical regions return
after a head revolution. Audit the requirement that *every* search position
has the same complete support before relaxing it: otherwise missing returns
can manufacture a correlation peak.

If stable patches persist beyond adjacent sweeps, test a bounded local
reference set. A 2–5 second lifetime is an initial experiment range, not a
literature-derived setting; motion and visibility may be better expiration
criteria. Store patches in corrected coordinates, check geometric visibility,
retain overlap and uniqueness tests, and discard inconsistent references.

Repeated observations of one patch share pose and appearance errors.
Extending its life must not count it as fresh independent information on
every scan, or shrink uncertainty merely with the number of correlated
patches. Local keyframes can reduce frame-to-frame drift accumulation but
remain estimated references; they do not establish an absolute along-tunnel
position.

Only after this audit should the weight change. The current `100.0` multiplier
is experimental, not a calibrated covariance. Increased trust must follow
measured repeatability and agreement, not just a weak geometry score. Preserve
flat, ramp, periodic, missing-coverage, and conflicting-motion controls; add
intensity-shuffled replay as an appearance negative control.

## 6. Test whether finer geometry survives the tunnel

BIEVR-LIO represents fine surface geometry with oriented height images and
samples using map information. Its limitations include point density and
initialization; the authors still find intensity useful in a strongly
degenerate tunnel. This motivates testing retained detail here without
assuming every smooth-looking tunnel contains enough geometry.
[BIEVR-LIO, methods and §V](https://arxiv.org/html/2604.14421v2)

The local geometry path uses 20 cm voxels while texture has a separate 6 cm
cloud. On identical cached input, compare the current geometry control with
a smaller geometric voxel size and, separately, sampling that favors useful
surface variation. Keep the observer and texture settings fixed for each
comparison. Measure correspondence survival, mixed-mode information, pose
corrections, memory, and runtime; many more correlated points do not by
themselves provide many more independent constraints.

An improvement would support the hypothesis that preprocessing discards
useful wall/floor detail. Failure would narrow that hypothesis but would not
prove physical unobservability: sampling, covariance estimation, and initial
alignment could still limit access to the detail.

## 7. Fusion and newer work: useful follow-ups, different commitments

| Direction | Why keep it on the list | What is needed before applying it here |
|---|---|---|
| Direct visual fusion | FAST-LIVO2 is a relevant published visual/LiDAR system | The archived stereo stream is only about 1 Hz; verify overlap, exposure, calibration, and timing first |
| Historical-state fusion | LODESTAR uses covariance-aware historical-state information through a Schmidt-Kalman design | A coherent state/covariance implementation; old poses are not independent ground truth |
| Doppler velocity | LiDAR-radar-inertial work provides a distinct velocity-measurement route | A future radar recording; no such measurement is available in this bag |

These entries are supported by the inspected abstracts and author material,
not a completed port review:
[FAST-LIVO2](https://arxiv.org/abs/2408.14035),
[LODESTAR](https://arxiv.org/abs/2511.09142),
[Degradation Resilient LiDAR-Radar-Inertial Odometry](https://arxiv.org/abs/2403.05332).

Two 2026 publications warrant a subsequent full-method read:
[GIF-LIO](https://doi.org/10.1109/TIM.2026.3671940) and
[adaptive photometric weighting](https://www.mdpi.com/2079-9292/15/17/3970).
The latter was published September 3, only two days before this survey.
The publisher-indexed material is enough to identify relevant work, not to
endorse its weighting rules or transfer reported gains to Exyn. IGE-LIO is
also retained as a [metadata-verified lead](https://doi.org/10.1109/TIM.2024.3427795).

There is no present evidence requiring replacement of DLIO's observer with
an EKF or factor graph. First compare scan priors, corrections, innovations,
and bias trajectories using the now-correct scan/velocity timing. Then change
one observer gain group at a time if those traces support an observer-response
hypothesis. A stronger prior can suppress visible motion while also suppressing
real motion, so smoothness alone cannot select the gain.

## 8. Experiment order and decision criteria

All proposals below keep the solver acceptance fix. Use the same cache,
initialization, sweep order, build identification, and warm-up; retain the
full recording as well as selected diagnostic intervals. Tests that alter
raw-field provenance require a new derived cache without editing the source
capture.

| Priority | Experiment | Controlled comparison | Evidence needed to proceed |
|---|---|---|---|
| 1 | Mixed yaw/translation and rejection telemetry | Instrumented replay versus retained baseline | Pose parity for logging alone; coupled weakness or a specific rejection mechanism coincides with excursions |
| 2 | Appearance persistence | Pairwise references versus bounded local patches | More useful coverage and shorter unsupported gaps, while negative controls still reject and independent error does not worsen |
| 3 | Fine geometric detail | Current voxels versus finer geometry, texture off | Better supported constraints and independent error at an acceptable measured runtime |
| 4 | Radiometric audit | Native normalized byte versus a correction fitted on separate observations | Better held-out patch repeatability across head phase/range, without flat-region false matches |
| 5 | Coupled constraint or noise-aware gate | Current method versus one clearly specified extension | Correct behavior on coupled-motion synthetic scenes and improved held-out trajectory evidence |
| 6 | Observer response | One gain group at a time after diagnosing innovations | Less unsupported correction without lost real translation or turn motion |

For every candidate, record accepted constraints, longest unsupported interval,
rejection causes, geometric overlap, finite/converged status, bias/innovation
traces, and compute cost. A repeat with shuffled intensity should remove the
appearance benefit; a stationary-output estimator must not score well simply
because it is smooth.

### Independent validation

September speed, travel, loop appearance, and map shape are consistency checks.
They cannot yield absolute trajectory error without an independent reference.
A known endpoint or loop return checks one relationship and can hide large
interior drift. Useful additional evidence includes surveyed positions or
chainage, independently located landmarks, and held-out revisits that were
not also used to fit the estimator.

Public data can complement the private bag:

- **ENWIDE:** tunnel and other weak-geometry sequences with a total-station
  prism position reference. Apply the prism-to-sensor lever arm and time
  convention. A TUM-formatted file does not turn that position reference into
  measured orientation. It uses Ouster, so it tests the general method rather
  than this moving-head acquisition pipeline.
  [Official ENWIDE data and calibration](https://projects.asl.ethz.ch/datasets/enwide/)
- **GEODE:** include an Alpha/VLP-16 degenerate sequence to reduce the sensor
  mismatch, then compare another geometry or sensor platform. Follow the
  platform-specific ground-truth frames and calibration instructions. Its
  mounting is still different from the Exyn rotating head.
  [Official GEODE instructions](https://github.com/PengYu-Team/GEODE_dataset)

Use absolute and segment-relative errors where independent truth exists.
Metric LiDAR-inertial estimates should not receive a fitted Sim(3) scale.
Use translation plus yaw alignment when both trajectories share a known
gravity direction, or document why an SE(3) frame alignment is needed.
Report positional error separately when orientation truth is unavailable.
This follows the evaluation tutorial's distinction between observable state,
alignment, and error definitions.
[Zhang & Scaramuzza, 2018](https://www.ifi.uzh.ch/dam/jcr:89d3db14-37b1-431d-94c3-8be9f37466d3/IROS18_Zhang.pdf)

Choose segment lengths before tuning, for example 10 m and 50 m where the
reference permits, and keep another September recording or public sequence
held out. Report motion retention as well as error and runtime. Replaying the
same tuned recording identically establishes reproducibility; a second
recording and an independent reference establish different things.

## 9. Research provenance and immediate next step

Relevant manuscript sections, author implementations, publisher metadata,
manufacturer field definitions, and dataset instructions were inspected.
Abstract-only and metadata-only leads are labeled in [REFERENCES.md](REFERENCES.md).
This pass did not reproduce external papers, download benchmark capture
payloads, establish September ground truth, or run the proposed new ablations.
Downloaded research PDFs and metadata are kept under the workspace's ignored
`local/dliio-tunnel-research/`; the bibliography links their public sources.

The immediate implementation task supported by this research is **diagnostic
logging for the scan-centered, scaled full geometry system and detailed
surface-match rejection reasons**, with pose parity checked against the
retained corrected replay. Those measurements should decide whether the next
estimator change is a coupled search, longer-lived patches, finer geometry,
or observer tuning.
