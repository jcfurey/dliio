# Observable surface constraints

An ordinary loop still requires all six pose directions to pass the existing
geometry gates. A caller may explicitly propose a `surface_subspace` loop when
repeated geometry supports only some directions. Automatic retrieval does not
silently convert its rejected ordinary loops into these factors.

The measurement remains `Z_ij`, mapping original observation `j` into `i`.
Its error is `e = Log(Z_ij^-1 T_i^-1 T_j)`, ordered as translation then rotation
in the right local tangent. The optional object contains:

- `projection`: a finite symmetric, idempotent 6×6 matrix `P`, rank 3–5.
- `characteristic_length`: source RMS range `L`, in metres.
- `eigenvalue_ratio`: declared normalized-information cutoff, 0.001–0.1.

With `D = diag(1,1,1,L,L,L)`, the native residual is `D^-1 P D e`.
The declared covariance must be `sigma² D^-2`, with `covariance_kind: assumed`.
Consequently the squared error is `||P D e||² / sigma²`. Discarded directions
carry exactly zero information; no large artificial variance approximates them.
The native boundary permutes both covariance and projection to GTSAM's order.
Odometry projections must remain identity and its consecutive chain stays full
rank, preserving the numerical gauge and graph connectivity.

`propose_subspace` uses original fitting geometry to compute normalized
point-to-plane information. Forward and reverse correspondences use local
normals and distinct target support. The reverse derivative includes rotation
of the source normal. The backend independently recomputes the requested
projection before admitting it. Each direction of both fitting and held
geometry must support every retained mode. Existing overlap, inlier distance,
surface residual, temporal exclusion, quality and bounded-window checks remain
in force. Stationarity and independent refits are checked in retained modes.

A projected loop requires explicit fitting and held observation IDs on both
visits. Fitting may use **one original observation per node**, avoiding a rigid
average of neighbouring observations when their relative odometry is uncertain.
Each visit still requires at least **two disjoint held observations**. Ordinary
window-loop support retains its minimum of two fitting observations. Anchor,
ordering, span and registration-quality checks apply to every support record.

After solving, both unrobustified projected errors and absolute projected
translation/rotation bounds must pass. Reports include the unprojected error
so unconstrained sliding is visible. A projected factor does not certify
agreement in discarded directions. Evaluate actual reconstructed surfaces and
overlap coverage separately; low projected graph error is not a map-quality
certificate or independent place recognition.

The native registration `fit` also accepts `observable_ratio`, zero by default.
Nonzero values in 0.001–0.1 normalize rotation by local RMS range and truncate
weak directions during each fitting step. This prevents refinement from walking
along an unobservable corridor axis. Existing full fitting remains unchanged
with the default. The optional mode currently requires the native backend.

The information spectrum selects geometry directions; it is not a calibrated
measurement covariance. Fitting and held windows share frontend calibration,
deskew and motion history, and neighbouring factors may share observations.
The assumed noise must not be interpreted as independent survey accuracy.
See the [ICP covariance analysis](https://arxiv.org/abs/1410.7632) and existing
[uncertainty contract](POSE_GRAPH.md).

`reweight_loops` changes noise through the same atomic graph/map transaction as
other graph actions. Each `updates` entry has exactly `id`, `covariance`,
`covariance_kind`, and `covariance_model`; a request also supplies `reason`.
IDs must be unique and active. Transforms, support and original provenance are
preserved. Loading an archive replays noise amendments alongside additions and
removals, rejecting changes inconsistent with the request ledger. Rejected
noise trials leave the map and active factor values unchanged.

Native tests compare projected optimization against independent numerical
least squares, finite-difference both surface derivatives, exercise the exact
nullspace, reject unsupported modes and bad held geometry, and verify durable
commit, reload, noise amendment, removal and original-record preservation.
