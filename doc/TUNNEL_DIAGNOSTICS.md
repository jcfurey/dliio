# Moving-head tunnel diagnostics

This implements the first experiment from the
[curved-tunnel research memo](TUNNEL_RESEARCH_2026-09-05.md): log coupled
geometric information and individual texture rejection stages without
changing the estimator's objective, gates, matches, or observer gains.

Both complete instrumented replays reproduce **all 44 shared non-timing
CSV columns exactly** across **3,669 scans**, including pose, observer state,
velocity, biases, convergence, and the existing texture measurements. Its
largest measured texture bottleneck is **missing spatial support in the
current sweep during the displacement search**. This is an estimator
consistency result; trajectory accuracy is still unverified.

Research documentation was committed at dliio `86cebfc` and workspace
`e10419f` before this implementation. The input, retained parameter profile,
and original results are described in the workspace
[surface-texture report](../../../docs/dliio-surface-texture.md).

## Geometry measurement and scope

`odom/gicp/geometryDiagnostics/enabled` defaults to `false`.
`odom/gicp/geometryDiagnostics/lengthScale` defaults to **5 m** and accepts
0.01–1000 m. Both settings require a restart; an atomic live update that
includes either setting is rejected before other parameters are queued.
The scale is a declared analysis convention, not an estimated tunnel radius.

The [analyzer](../src/nano_gicp/geometry_diagnostics.cc) reads the **initial
geometry-only normal matrix at the registration prior**, before adding
appearance constraints or damping. It transforms a copy into coordinates
`[L * rotation_about_scan_center, center_translation]`, expressed along the
target/world axes. The solver's matrix and update remain untouched.

For world-left increments `[w, v]` and centered increments `[w, dc]`,
`v = dc + center × w`. Applying that change of coordinates and the length
scale gives the Jacobian `[-skew(point - center) / L, I]`. This removes the
arbitrary world-origin lever arm while retaining rotation/translation cross
terms. Tests cover a translated world origin, a rotated world frame, length
units, a coupled null mode, and singular or invalid matrices.

The ROS diagnostic values and appended replay columns include:

| Quantity | Meaning |
|---|---|
| `geometry_valid`, `geometry_length` | Whether this scan has a valid analysis, and the declared length in meters |
| `geometry_h_i_j` | 21 upper-triangular entries of the symmetric centered/scaled 6×6 matrix; ordering is rotations 0–2, translations 3–5 |
| `geometry_eig_0` … `_5` | Ascending eigenvalues of that matrix |
| `geometry_mode_0` … `_5` | Its unit weakest eigenvector; largest-magnitude component has positive sign |
| `geometry_rot_ratio`, `geometry_trans_ratio` | Minimum/maximum eigenvalue within each centered diagonal block |
| `geometry_schur_ratio` | Minimum translation eigenvalue after rotation can adjust, divided by the original translation maximum |
| `geometry_schur_retained` | Same Schur minimum divided by the original translation minimum; measures the change in the weakest translation information |
| `geometry_half_length_ratio`, `geometry_double_length_ratio` | Full-system minimum/maximum ratio at `L/2` and `2L` |
| `geometry_correction_0` … `_5` | Final GICP correction relative to its initial prior: scaled rotation vector and scan-center displacement |
| `geometry_correction_projection` | Signed projection of that finite correction onto the initial weakest mode |
| `geometry_texture_projection` | Absolute projection of an accepted unit center-translation residual's Jacobian onto that mode; −1 without a texture measurement |
| `prior_x/y/z`, `prior_qx/qy/qz/qw`, `bgx/y/z` | Registration prior and observer gyro-bias snapshot; existing CSV fields retain accelerometer biases |

The Schur calculation uses an eigenvalue-based pseudoinverse of the rotation
block with relative rank tolerance `1e-9`. Invalid/unavailable ratios use −1.
All-zero geometry is unavailable, rather than reported as confident. These
values inherit GICP's covariance regularization and floating-point products;
they are **not calibrated information, a pose covariance, or an independent
motion reference**.

Eigenvector sign is conventional, and a nearly repeated eigenvalue makes a
single vector unstable. The analysis reports continuous component shares and
world-Z rotation per meter of weak-mode translation. It reports the count
with `lambda_1 > 1.2 * lambda_0` as a descriptive separation check, not an
estimator threshold. No binary “mixed mode” classification is used: a broad
curve can couple a small yaw increment to a substantial translation.

The final rotation vector and center displacement are diagnostic coordinates,
not the full SE(3) logarithm. GICP's correction is recorded before any later
node-level governor/fuse action; those mechanisms are disabled in the retained
experiment. The matrix describes the prior linearization, whereas the older
`geo_rot`/`geo_trans` gate margins describe the solver's later gate evaluations.

## Texture rejection accounting

The existing matching decisions are unchanged. Each examined reference
anchor records the **first** failed check. The counters obey these exact
identities on every scan:

```text
examined = nonfinite + spacing + neighborhood + nonplanar + axis_normal
         + reference_support + reference_contrast + candidates
candidates = repeated + source_support + source_contrast + supported
supported = boundary + low_correlation + ambiguous + flat_peak + unique
```

`source_support` means a spatial interpolation failed at some point in the
search, which still requires three neighbors within 12 cm and a common,
complete 21-sample patch at every displacement. It does not distinguish
sampling holes, motion/deskew error, surface orientation, or an inadequate
search neighborhood. Counts are conditional on earlier checks passing;
they are not independent failure probabilities.

ROS diagnostics expose a readable status. The numeric CSV `texture_status`
codes are stable in the following order:

| Code | Status | Interpretation |
|---:|---|---|
| 0 | disabled | Weight disabled, or the initial seed scan did not run registration |
| 1 | no_source | No prepared current texture cloud; inspect deskew/lifecycle diagnostics |
| 2 | no_reference | No previous texture reference |
| 3 | frame_gap | A previous frame exists but is outside the permitted interval |
| 4 | invalid_input | Insufficient cloud size or invalid matcher arguments |
| 5 | invalid_geometry | Geometry did not yield a usable translation spectrum |
| 6 | translation_strong | Weakest translation ratio did not trigger a search |
| 7 | multiple_weak_translations | A one-dimensional search cannot separate the weak translations |
| 8 | too_few_unique | Fewer than eight unique patch matches survived |
| 9 | shift_disagreement | Robust patch-shift disagreement exceeds 8 cm |
| 10 | too_few_inliers | Fewer than eight patches support the consensus |
| 11 | low_consensus | Fewer than 60% of unique matches support the consensus |
| 12 | accepted | A texture constraint was supplied |

Statuses 1–4 are evaluated when the geometric gate admits a match attempt.
The search-status field therefore reports the first applicable stage, not
every potentially unavailable input on a scan.

## Full-recording texture findings

The replay has 330,278 candidate patches. Of those:

| Outcome before uniqueness testing | Patches | Share of candidates |
|---|---:|---:|
| Missing current-sweep spatial support | 276,322 | 83.7% |
| Repeated reference pattern | 14,752 | 4.5% |
| Insufficient current-sweep contrast | 10,396 | 3.1% |
| Fully supported across the search | 28,808 | 8.7% |

Of the 28,808 supported patches, 11,407 fail the correlation threshold, 3,610
peak at the search boundary, 3,024 are ambiguous, and 10,767 are unique.
Across scans, 2,922 have too few unique matches, 17 fail shift disagreement,
84 have too few inliers, and 316 are accepted. Another 314 scans do not trigger
the translation gate and 15 have multiple weak translations. The first seed
scan has no registration measurement.

During 360–500 s, 45,156 of 53,375 candidates (84.6%) fail current spatial
support. In the longest interior unsupported interval, **671.37–734.67 s**,
13,300 of 17,593 candidates (75.6%) fail support. Every one of those 210 scans
produces candidates; 205 then have too few unique matches and five have too
few consensus inliers. Increasing the weakness-gate threshold would not admit
additional searches in either interval.

The first 132.27 s precede the first accepted texture measurement. That startup
gap must not be described as the mid-tunnel event. Neither the 360–500 s
analysis window nor the longest interior gap is an independently labeled
trajectory-error interval.

## Geometry findings and interpretation

In the final matrix replay, the centered geometry analysis is valid for all
3,668 registration scans. At `L = 5 m`, the full-system minimum/maximum ratio
has median 0.00457, compared with 0.04373 for the translation block. These
ratios describe different coordinate spaces and should not share a threshold.

The weakest full-system mode is dominated by world-X/Y rotation in the
360–500 s window: the median total rotation share is 90.9%, but its median
world-Z rotation share is only 0.0224%. A single weakest eigenvector therefore
does not answer the proposed yaw/translation question. The recorded matrix
allows inspection of other modes and a conditional 4×4 subsystem
`[L * world-Z rotation, xyz translation]`, holding world-X/Y rotation fixed.
That conditional view is not an IMU covariance update.

The saved matrix reproduces the logged six-dimensional spectrum in the
independent Python analysis. Its conditional yaw/translation view yields:

| Median quantity | Whole replay | 360–500 s | 671.37–734.67 s interior gap |
|---|---:|---:|---:|
| Translation information retained after all rotations can adjust | 92.31% | 99.46% | 98.67% |
| Translation information retained after only world-Z rotation can adjust | 98.71% | 99.97% | 99.81% |
| Conditional yaw/translation minimum/maximum eigenvalue ratio, `L = 5 m` | 0.00992 | 0.00773 | 0.01560 |

“Retained” compares minimum translation eigenvalues before and after the
specified Schur complement; it does not mean that 99.97% of a true position
measurement is available. Translation is already weak throughout 360–500 s
under the existing gate. The local matrix does not identify additional hidden
yaw coupling as its dominant failure there. It also does not rule out finite curved-tunnel ambiguity,
association changes, or errors in the prior. The next texture experiment
should target the measured support failure before increasing appearance trust.

## Reproduction and validation

Build an isolated installation with the workspace
[replay instructions](../../../docs/dliio-tunnel-investigation.md#validation-and-reproduction),
including the standalone harness. Use the immutable cache and retained texture
profile. After sourcing the isolated installation, add:

```text
-p odom/gicp/geometryDiagnostics/enabled:=true
-p odom/gicp/geometryDiagnostics/lengthScale:=5.0
```

Then run:

```bash
python3 src/dliio/scripts/analyze_tunnel_diagnostics.py \
  local/dliio-tunnel-diagnostics/diagnostic-matrix-full.csv \
  --baseline local/dliio-surface-texture/fixed-texture-full.csv \
  --output local/dliio-tunnel-diagnostics/report-matrix --plot
```

The analysis validates timestamps, counter identities, acceptance/status
agreement, eigenvector norms, and consistency between the saved matrix and
spectrum. With `--baseline`, it compares every shared estimator/texture column,
excluding wall-clock compute/wait timing and the intentionally changed
geometry diagnostic columns. Any difference returns a failing exit status
and is detailed in `metrics.json`.

The synthetic tests additionally check known texture absence, periodic
appearance, missing coverage, search boundaries, competing motions, exact
registration parity with geometry diagnostics toggled, and atomic parameter
rejection. The final matrix implementation passed **23 CTest targets / 394 cases**;
the parent archive's **84 tests** and documentation/syntax checks also passed.
The standalone analysis was checked against independent and coupled synthetic
matrices, as well as the complete recorded counter identities and spectrum.

Build hashes, logs, CSVs, metric reports, and PNG/SVG plots are retained under
ignored `local/dliio-tunnel-diagnostics/`. The initial spectrum-only replay is
`diagnostic-full.csv`; `diagnostic-matrix-full.csv` also records the centered matrix.
The latter covers all 1100.99 s between registered scan midpoints and took
195.10 s of wall time in this run. This is not an isolated throughput benchmark.
The [final plot](../../../local/dliio-tunnel-diagnostics/report-matrix/diagnostics.png)
and [metrics](../../../local/dliio-tunnel-diagnostics/report-matrix/metrics.json)
retain the full traces and the exact baseline comparison.
The source capture and normal workspace installation are preserved.

## Next controlled experiment

Measure support at zero displacement and across the full search interval,
including head phase and patch location. This should separate a poorly
sampled current patch from a patch rejected only at the extremes of the
search. Compare support at the prior and at the accepted geometric correction
as a diagnostic, without treating the latter as independent truth.

If support is the limiting factor, test an acquisition-aware patch/support
policy on the same cached inputs and the existing flat, periodic, and
missing-coverage controls. Longer-lived references alone cannot create
missing samples in the current sweep. Any use of accumulated current data
must track shared points and pose errors so that it cannot match a patch to
itself or count old measurements as new independent information.
